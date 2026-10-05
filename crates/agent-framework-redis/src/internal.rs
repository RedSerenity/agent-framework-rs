//! Shared, crate-private plumbing used by every type in this crate.

use std::sync::Arc;

use agent_framework_core::error::{Error, Result};
use agent_framework_core::types::Message;
use redis::aio::MultiplexedConnection;
use redis::{AsyncCommands, AsyncConnectionConfig, ConnectionAddr, ConnectionInfo};
use tokio::sync::OnceCell;

pub use redis::auth::{BasicAuth, StreamingCredentialsProvider};

/// A Redis connection source plus a lazily-established multiplexed async
/// connection.
///
/// [`redis::aio::MultiplexedConnection`] is a cheap, `Clone`-able handle onto
/// a single background connection (commands from many clones are
/// multiplexed over one socket), so [`LazyConnection::get`] hands out clones
/// rather than requiring callers to serialize on a lock. An *owned*
/// connection is only opened on the first call to `get`, mirroring the
/// Python store's behavior of not touching the network until a store
/// operation actually runs. A *borrowed* connection (the caller handed one
/// in) is used as-is — the caller owns its lifetime, as upstream's borrowed
/// `redis_client` (#8883).
pub(crate) struct LazyConnection {
    client: Option<redis::Client>,
    config: AsyncConnectionConfig,
    cell: OnceCell<MultiplexedConnection>,
}

impl LazyConnection {
    /// Parse `url` into a [`redis::Client`] (no network I/O yet).
    pub(crate) fn open(url: &str) -> Result<Self> {
        let client = redis::Client::open(url).map_err(|e| {
            Error::Configuration(format!("invalid Redis URL '{}': {e}", redact_url(url)))
        })?;
        Self::from_client(client, AsyncConnectionConfig::new())
    }

    /// Build a lazy connection from explicit connection info (no I/O yet).
    pub(crate) fn open_info(info: ConnectionInfo, config: AsyncConnectionConfig) -> Result<Self> {
        let client = redis::Client::open(info)
            .map_err(|e| Error::Configuration(format!("invalid Redis connection info: {e}")))?;
        Self::from_client(client, config)
    }

    fn from_client(client: redis::Client, config: AsyncConnectionConfig) -> Result<Self> {
        if !client.get_connection_info().addr().is_supported() {
            return Err(Error::Configuration(
                "this Redis address needs TLS, which requires the `tls` feature of \
                 agent-framework-redis"
                    .into(),
            ));
        }
        Ok(Self {
            client: Some(client),
            config,
            cell: OnceCell::new(),
        })
    }

    /// Wrap a caller-owned connection. No I/O; the connection is used as-is.
    pub(crate) fn borrowed(connection: MultiplexedConnection) -> Self {
        Self {
            client: None,
            config: AsyncConnectionConfig::new(),
            cell: OnceCell::new_with(Some(connection)),
        }
    }

    /// Whether the connection was supplied by the caller.
    pub(crate) fn is_borrowed(&self) -> bool {
        self.client.is_none()
    }

    /// The database number an owned connection selects, if known.
    pub(crate) fn db(&self) -> Option<i64> {
        self.client
            .as_ref()
            .map(|c| c.get_connection_info().redis_settings().db())
    }

    /// Whether an owned connection is configured for RESP3.
    pub(crate) fn resp3(&self) -> bool {
        self.client.as_ref().is_some_and(|c| {
            c.get_connection_info().redis_settings().protocol() == redis::ProtocolVersion::RESP3
        })
    }

    /// Return a clone of the shared multiplexed connection, establishing it
    /// on first use.
    pub(crate) async fn get(&self) -> Result<MultiplexedConnection> {
        self.cell
            .get_or_try_init(|| async {
                match &self.client {
                    Some(client) => client
                        .get_multiplexed_async_connection_with_config(&self.config)
                        .await
                        .map_err(map_redis_err),
                    None => Err(Error::service("borrowed Redis connection is missing")),
                }
            })
            .await
            .cloned()
    }
}

/// Hide the password in a Redis URL before it reaches an error message.
pub(crate) fn redact_url(url: &str) -> String {
    match (url.find("://"), url.rfind('@')) {
        (Some(scheme), Some(at)) if at > scheme => {
            format!("{}://***@{}", &url[..scheme], &url[at + 1..])
        }
        _ => url.to_string(),
    }
}

/// Map a [`redis::RedisError`] onto the framework's [`Error::Service`] variant.
pub(crate) fn map_redis_err(e: redis::RedisError) -> Error {
    Error::service(format!("redis error: {e}"))
}

/// Delegates [`StreamingCredentialsProvider`] to a shared trait object, since
/// `AsyncConnectionConfig::set_credentials_provider` wants an owned `P`.
struct SharedCredentials(Arc<dyn StreamingCredentialsProvider>);

impl StreamingCredentialsProvider for SharedCredentials {
    fn subscribe(
        &self,
    ) -> std::pin::Pin<
        Box<dyn futures::Stream<Item = redis::RedisResult<BasicAuth>> + Send + 'static>,
    > {
        self.0.subscribe()
    }
}

/// Explicit host-based connection settings, the Rust shape of upstream's
/// `host` / `port` / `ssl` / `username` / `credential_provider` arguments.
#[derive(Clone, Default)]
pub(crate) struct HostSettings {
    pub(crate) host: Option<String>,
    pub(crate) port: Option<u16>,
    pub(crate) ssl: Option<bool>,
    pub(crate) username: Option<String>,
    pub(crate) password: Option<String>,
    pub(crate) credential_provider: Option<Arc<dyn StreamingCredentialsProvider>>,
}

/// Upstream's default port: 6380, the Azure Cache for Redis TLS port.
pub(crate) const DEFAULT_HOST_PORT: u16 = 6380;

impl HostSettings {
    /// Whether any host-shaped setting was supplied.
    pub(crate) fn is_set(&self) -> bool {
        self.host.is_some() || self.credential_provider.is_some()
    }

    /// Resolve into a [`LazyConnection`]. `host` is required.
    pub(crate) fn connect(&self) -> Result<LazyConnection> {
        let host = self.host.clone().ok_or_else(|| {
            Error::Configuration("host is required when using credential_provider".into())
        })?;
        if host.is_empty() {
            return Err(Error::Configuration("host must be non-empty".into()));
        }
        let port = self.port.unwrap_or(DEFAULT_HOST_PORT);
        let addr = if self.ssl.unwrap_or(true) {
            ConnectionAddr::TcpTls {
                host,
                port,
                insecure: false,
                tls_params: None,
            }
        } else {
            ConnectionAddr::Tcp(host, port)
        };
        let mut redis_settings = redis::RedisConnectionInfo::default();
        if let Some(username) = &self.username {
            redis_settings = redis_settings.set_username(username);
        }
        if let Some(password) = &self.password {
            redis_settings = redis_settings.set_password(password);
        }
        let info = redis::IntoConnectionInfo::into_connection_info(addr)
            .map_err(|e| Error::Configuration(format!("invalid Redis address: {e}")))?
            .set_redis_settings(redis_settings);
        let mut config = AsyncConnectionConfig::new();
        if let Some(provider) = &self.credential_provider {
            config = config.set_credentials_provider(SharedCredentials(provider.clone()));
        }
        LazyConnection::open_info(info, config)
    }
}

// region: list-backed history helpers shared by the history providers

/// Read a list of JSON-serialized messages.
///
/// `tail` limits the read to the last N entries (`LRANGE -N -1`); `Some(0)`
/// reads nothing without a round trip, because `LRANGE key -0 -1` is the
/// whole list. With `skip_malformed`, an entry that is not a valid message
/// is logged and skipped instead of failing the read.
pub(crate) async fn read_messages(
    conn: &LazyConnection,
    key: &str,
    tail: Option<usize>,
    skip_malformed: bool,
) -> Result<Vec<Message>> {
    if tail == Some(0) {
        return Ok(Vec::new());
    }
    let start: isize = match tail {
        Some(n) => -(isize::try_from(n).unwrap_or(isize::MAX)),
        None => 0,
    };
    let mut c = conn.get().await?;
    let raw: Vec<String> = c.lrange(key, start, -1).await.map_err(map_redis_err)?;
    let mut out = Vec::with_capacity(raw.len());
    for entry in raw {
        if skip_malformed && entry.is_empty() {
            continue;
        }
        match serde_json::from_str::<Message>(&entry) {
            Ok(message) => out.push(message),
            Err(e) if skip_malformed => {
                tracing::warn!(key, error = %e, "skipping malformed chat history entry");
            }
            Err(e) => return Err(e.into()),
        }
    }
    Ok(out)
}

/// Append messages atomically (`MULTI` + `RPUSH`…), then trim the list to
/// the most recent `max` entries when a limit is set.
pub(crate) async fn push_messages(
    conn: &LazyConnection,
    key: &str,
    messages: &[Message],
    max: Option<usize>,
) -> Result<()> {
    if messages.is_empty() {
        return Ok(());
    }
    let mut payloads = Vec::with_capacity(messages.len());
    for message in messages {
        payloads.push(serde_json::to_string(message)?);
    }
    let mut c = conn.get().await?;
    let mut pipe = redis::pipe();
    pipe.atomic();
    for payload in payloads {
        pipe.rpush(key, payload);
    }
    let _: () = pipe.query_async(&mut c).await.map_err(map_redis_err)?;
    if let Some(max) = max {
        let len: usize = c.llen(key).await.map_err(map_redis_err)?;
        if len > max {
            let start = -(isize::try_from(max).unwrap_or(isize::MAX));
            let _: () = c.ltrim(key, start, -1).await.map_err(map_redis_err)?;
        }
    }
    Ok(())
}

/// `LLEN key`.
pub(crate) async fn list_len(conn: &LazyConnection, key: &str) -> Result<usize> {
    let mut c = conn.get().await?;
    c.llen(key).await.map_err(map_redis_err)
}

/// `DEL key`.
pub(crate) async fn delete_key(conn: &LazyConnection, key: &str) -> Result<()> {
    let mut c = conn.get().await?;
    let _: () = c.del(key).await.map_err(map_redis_err)?;
    Ok(())
}

/// `PING`, as a boolean.
pub(crate) async fn ping(conn: &LazyConnection) -> bool {
    let Ok(mut c) = conn.get().await else {
        return false;
    };
    redis::cmd("PING")
        .query_async::<String>(&mut c)
        .await
        .is_ok()
}

// endregion

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn redact_url_hides_credentials() {
        assert_eq!(
            redact_url("redis://user:secret@host:6379/0"),
            "redis://***@host:6379/0"
        );
        assert_eq!(redact_url("redis://host:6379"), "redis://host:6379");
    }

    #[test]
    fn host_settings_require_a_host() {
        let settings = HostSettings {
            username: Some("u".into()),
            ..Default::default()
        };
        assert!(settings.connect().is_err());
    }

    #[test]
    fn plain_host_settings_build_without_io() {
        let settings = HostSettings {
            host: Some("127.0.0.1".into()),
            port: Some(6379),
            ssl: Some(false),
            username: Some("default".into()),
            password: Some("pw".into()),
            credential_provider: None,
        };
        let conn = settings.connect().expect("plain TCP needs no TLS feature");
        assert!(!conn.is_borrowed());
        assert_eq!(conn.db(), Some(0));
    }

    #[cfg(not(feature = "tls"))]
    #[test]
    fn tls_without_the_feature_is_a_configuration_error() {
        let settings = HostSettings {
            host: Some("cache.example".into()),
            ..Default::default()
        };
        let err = settings.connect().err().expect("TLS is the default");
        assert!(err.to_string().contains("tls"), "{err}");
    }
}
