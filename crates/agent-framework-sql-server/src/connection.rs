//! Connections and the per-operation transaction — upstream's `_Client`.
//!
//! Every operation opens a connection, runs in one transaction, commits or
//! rolls back, and closes the connection, exactly as upstream's
//! `_Client._run_sync` does. The I/O sits behind two small traits
//! ([`Connector`] / [`Session`]) so the transaction and cleanup semantics —
//! including the committed-cleanup case — are tested offline against a
//! scripted fake; the production implementation is [`TiberiusConnector`].

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use agent_framework_azure::{DefaultAzureCredential, ManagedIdentityCredential, TokenCredential};
use agent_framework_core::error::{Error, Result};
use agent_framework_core::settings::SecretString;
use tiberius::{AuthMethod, Client, ColumnData, Config, FromSql};
use tokio::net::TcpStream;
use tokio::sync::RwLock;
use tokio_util::compat::{Compat, TokioAsyncWriteCompatExt};

use crate::sql::{Cell, NullKind, SqlParam};

/// The Microsoft Entra audience for Azure SQL.
pub const SQL_SERVER_SCOPE: &str = "https://database.windows.net/.default";

/// The message prefix of the error [`is_committed_cleanup`] recognizes.
pub(crate) const COMMITTED_CLEANUP_PREFIX: &str =
    "SQL Server transaction committed, but connection cleanup failed";

/// Whether `error` reports a transaction that **committed** but whose
/// connection then failed to close — upstream's
/// `SqlServerCommittedCleanupException`.
///
/// The write is durable: do not retry it automatically, especially with
/// generated keys, which would insert a second row. Every other error from
/// this crate means the transaction rolled back (or never started).
pub fn is_committed_cleanup(error: &Error) -> bool {
    matches!(error, Error::Service(message) if message.starts_with(COMMITTED_CLEANUP_PREFIX))
}

/// A boxed, `Send` future borrowing a session.
pub(crate) type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// One open connection.
#[async_trait::async_trait]
pub(crate) trait Session: Send {
    /// Run one parameterized statement, returning the rows of its first
    /// result set (empty for a statement with none).
    async fn query(&mut self, sql: &str, params: &[SqlParam]) -> Result<Vec<Vec<Cell>>>;
    /// Run a parameterless batch (`BEGIN TRANSACTION`, `COMMIT`, ...).
    async fn batch(&mut self, sql: &str) -> Result<()>;
    /// Close the connection.
    async fn close(self: Box<Self>) -> Result<()>;
}

/// Opens connections.
#[async_trait::async_trait]
pub(crate) trait Connector: Send + Sync {
    async fn connect(&self) -> Result<Box<dyn Session>>;
}

/// The shared client: a connector plus a close gate. Operations hold the
/// read side, so [`ClientHandle::close`] waits for in-flight operations, as
/// upstream's `close()` waits for its worker.
pub(crate) struct ClientHandle {
    connector: Arc<dyn Connector>,
    closed: RwLock<bool>,
}

impl ClientHandle {
    pub(crate) fn new(connector: Arc<dyn Connector>) -> Self {
        Self {
            connector,
            closed: RwLock::new(false),
        }
    }

    pub(crate) async fn ensure_open(&self) -> Result<()> {
        if *self.closed.read().await {
            return Err(Error::other("The SQL Server client is closed."));
        }
        Ok(())
    }

    pub(crate) async fn close(&self) {
        *self.closed.write().await = true;
    }

    /// Run `operation` in one transaction on a fresh connection —
    /// upstream's `_Client._run_sync`.
    ///
    /// * Success: commit, then close. A close failure **after** the commit
    ///   is the committed-cleanup error ([`is_committed_cleanup`]).
    /// * Failure (of the operation or the commit): roll back, close, and
    ///   return the original error; a rollback or close failure on this path
    ///   is logged rather than masking it.
    pub(crate) async fn run<T, F>(&self, operation: F) -> Result<T>
    where
        T: Send,
        F: for<'a> FnOnce(&'a mut dyn Session) -> BoxFuture<'a, Result<T>> + Send,
    {
        let closed = self.closed.read().await;
        if *closed {
            return Err(Error::other("The SQL Server client is closed."));
        }
        let mut session = self.connector.connect().await?;
        let outcome = async {
            session
                .batch("SET XACT_ABORT ON; BEGIN TRANSACTION")
                .await?;
            let result = operation(session.as_mut()).await?;
            session.batch("COMMIT TRANSACTION").await?;
            Ok::<T, Error>(result)
        }
        .await;
        match outcome {
            Ok(result) => match session.close().await {
                Ok(()) => Ok(result),
                Err(error) => Err(Error::service(format!(
                    "{COMMITTED_CLEANUP_PREFIX}; do not retry this operation automatically: \
                     {error}"
                ))),
            },
            Err(error) => {
                if let Err(rollback) = session
                    .batch("IF @@TRANCOUNT > 0 ROLLBACK TRANSACTION")
                    .await
                {
                    tracing::warn!(error = %rollback, "SQL Server rollback failed after an operation error");
                }
                if let Err(cleanup) = session.close().await {
                    tracing::warn!(
                        error = %cleanup,
                        "SQL Server connection cleanup also failed after an operation error"
                    );
                }
                Err(error)
            }
        }
    }
}

// region: connection strings and authentication

/// How a connection authenticates beyond what the connection string's own
/// keys say.
#[derive(Clone)]
pub(crate) enum Auth {
    /// Whatever the connection string says (SQL login, or none).
    ConnectionString,
    /// A Microsoft Entra access token from this credential, for
    /// [`SQL_SERVER_SCOPE`].
    Token(Arc<dyn TokenCredential>),
}

/// Split an ADO.NET connection string into `(key, raw segment)` pairs,
/// honouring `"..."`, `'...'`, and `{...}` quoting so a `;` inside a
/// password does not split it.
fn ado_segments(connection_string: &str) -> Vec<(String, String)> {
    let mut segments = Vec::new();
    let mut current = String::new();
    let mut quote: Option<char> = None;
    for ch in connection_string.chars() {
        match quote {
            Some(q) => {
                current.push(ch);
                if ch == q {
                    quote = None;
                }
            }
            None => match ch {
                '"' | '\'' => {
                    quote = Some(ch);
                    current.push(ch);
                }
                '{' => {
                    quote = Some('}');
                    current.push(ch);
                }
                ';' => {
                    segments.push(std::mem::take(&mut current));
                }
                _ => current.push(ch),
            },
        }
    }
    segments.push(current);
    segments
        .into_iter()
        .filter(|s| !s.trim().is_empty())
        .map(|segment| {
            let key = segment
                .split_once('=')
                .map(|(k, _)| k.trim().to_ascii_lowercase())
                .unwrap_or_default();
            (key, segment)
        })
        .collect()
}

fn segment_value(segment: &str) -> String {
    let value = segment.split_once('=').map(|(_, v)| v.trim()).unwrap_or("");
    let unquoted = value
        .strip_prefix('{')
        .and_then(|v| v.strip_suffix('}'))
        .or_else(|| value.strip_prefix('"').and_then(|v| v.strip_suffix('"')))
        .or_else(|| value.strip_prefix('\'').and_then(|v| v.strip_suffix('\'')))
        .unwrap_or(value);
    unquoted.to_string()
}

/// Resolve the `Authentication=` key the way upstream's driver does, for
/// the passwordless modes upstream's README documents:
///
/// * `ActiveDirectoryDefault` → [`DefaultAzureCredential`] (environment,
///   workload identity, managed identity, Azure CLI);
/// * `ActiveDirectoryMSI` / `ActiveDirectoryManagedIdentity` →
///   [`ManagedIdentityCredential`], with `UID` / `User ID` as a
///   user-assigned identity's client id;
/// * `SqlPassword` or absent → the connection string's own login.
///
/// Returns the tiberius configuration (with the `Authentication` key, and
/// for managed identity the `UID`, removed) and the token credential, if
/// any. An explicit `credential` overrides the connection string.
pub(crate) fn parse_connection_string(
    connection_string: &SecretString,
    credential: Option<Arc<dyn TokenCredential>>,
) -> Result<(Config, Auth)> {
    let segments = ado_segments(connection_string.expose_secret());
    let mode = segments
        .iter()
        .find(|(key, _)| key == "authentication")
        .map(|(_, segment)| segment_value(segment).to_ascii_lowercase());
    let user = segments
        .iter()
        .find(|(key, _)| key == "uid" || key == "user id" || key == "user")
        .map(|(_, segment)| segment_value(segment));
    let mut drop_user = false;
    let auth = match (credential, mode.as_deref()) {
        (Some(credential), _) => {
            drop_user = true;
            Auth::Token(credential)
        }
        (None, None | Some("sqlpassword")) => Auth::ConnectionString,
        (None, Some("activedirectorydefault")) => {
            Auth::Token(Arc::new(DefaultAzureCredential::new(SQL_SERVER_SCOPE)))
        }
        (None, Some("activedirectorymsi" | "activedirectorymanagedidentity")) => {
            drop_user = true;
            let mut credential = ManagedIdentityCredential::new(SQL_SERVER_SCOPE);
            if let Some(client_id) = user.filter(|u| !u.is_empty()) {
                credential = credential.with_client_id(client_id);
            }
            Auth::Token(Arc::new(credential))
        }
        (None, Some(other)) => {
            return Err(Error::Configuration(format!(
                "SQL Server Authentication={other} is not supported by this connector; use \
                 ActiveDirectoryDefault, ActiveDirectoryMSI, SqlPassword, or pass a \
                 TokenCredential with `with_token_credential`."
            )))
        }
    };
    let remaining: Vec<&str> = segments
        .iter()
        .filter(|(key, _)| {
            key != "authentication"
                && !(drop_user && (key == "uid" || key == "user id" || key == "user"))
        })
        .map(|(_, segment)| segment.as_str())
        .collect();
    // Never echo the tiberius parse error verbatim: it can quote the value.
    let config = Config::from_ado_string(&remaining.join(";")).map_err(|_| {
        Error::Configuration(
            "connection_string is not a valid ADO.NET SQL Server connection string.".into(),
        )
    })?;
    Ok((config, auth))
}

// endregion

// region: tiberius

/// The production [`Connector`]: TDS over tokio, TLS through rustls.
pub(crate) struct TiberiusConnector {
    pub(crate) config: Config,
    pub(crate) auth: Auth,
    pub(crate) query_timeout: Option<Duration>,
}

fn driver_error(error: tiberius::error::Error) -> Error {
    Error::service(format!("SQL Server operation failed: {error}"))
}

#[async_trait::async_trait]
impl Connector for TiberiusConnector {
    async fn connect(&self) -> Result<Box<dyn Session>> {
        let mut config = self.config.clone();
        if let Auth::Token(credential) = &self.auth {
            let token = credential.get_token_for_scope(SQL_SERVER_SCOPE).await?;
            config.authentication(AuthMethod::aad_token(token));
        }
        let client = connect_with_redirect(config).await?;
        Ok(Box::new(TiberiusSession {
            client,
            query_timeout: self.query_timeout,
        }))
    }
}

/// Connect, following one Azure SQL gateway redirect.
async fn connect_with_redirect(mut config: Config) -> Result<Client<Compat<TcpStream>>> {
    for _ in 0..2 {
        let tcp = TcpStream::connect(config.get_addr())
            .await
            .map_err(|e| Error::service(format!("SQL Server connection failed: {e}")))?;
        tcp.set_nodelay(true)
            .map_err(|e| Error::service(format!("SQL Server connection failed: {e}")))?;
        match Client::connect(config.clone(), tcp.compat_write()).await {
            Ok(client) => return Ok(client),
            Err(tiberius::error::Error::Routing { host, port }) => {
                config.host(&host);
                config.port(port);
            }
            Err(error) => return Err(driver_error(error)),
        }
    }
    Err(Error::service(
        "SQL Server connection failed: too many redirects",
    ))
}

struct TiberiusSession {
    client: Client<Compat<TcpStream>>,
    query_timeout: Option<Duration>,
}

/// Bound one statement by the configured `query_timeout`; `0` disables it.
/// A timed-out statement fails the operation, whose connection is then
/// dropped — the server rolls back an uncommitted transaction on
/// disconnect.
async fn bounded<T>(limit: Option<Duration>, future: impl Future<Output = Result<T>>) -> Result<T> {
    match limit {
        Some(limit) if !limit.is_zero() => tokio::time::timeout(limit, future)
            .await
            .map_err(|_| Error::service("SQL Server operation failed: query timed out"))?,
        _ => future.await,
    }
}

impl tiberius::ToSql for SqlParam {
    fn to_sql(&self) -> ColumnData<'_> {
        use std::borrow::Cow;
        match self {
            SqlParam::Null(kind) => match kind {
                NullKind::Str => ColumnData::String(None),
                NullKind::I64 => ColumnData::I64(None),
                NullKind::F64 => ColumnData::F64(None),
                NullKind::Bool => ColumnData::Bit(None),
                NullKind::Uuid => ColumnData::Guid(None),
                NullKind::Bytes => ColumnData::Binary(None),
                NullKind::Date => ColumnData::Date(None),
                NullKind::DateTime => ColumnData::DateTime2(None),
            },
            SqlParam::Str(v) => ColumnData::String(Some(Cow::Borrowed(v.as_str()))),
            SqlParam::I64(v) => ColumnData::I64(Some(*v)),
            SqlParam::F64(v) => ColumnData::F64(Some(*v)),
            SqlParam::Bool(v) => ColumnData::Bit(Some(*v)),
            SqlParam::Uuid(v) => ColumnData::Guid(Some(*v)),
            SqlParam::Bytes(v) => ColumnData::Binary(Some(Cow::Borrowed(v.as_slice()))),
            SqlParam::Date(v) => v.to_sql(),
            SqlParam::DateTime(v) => v.to_sql(),
        }
    }
}

fn cell(data: &ColumnData<'static>) -> Result<Cell> {
    let unsupported = || Error::service("SQL Server returned a value of an unsupported type.");
    Ok(match data {
        ColumnData::U8(v) => v.map_or(Cell::Null, |v| Cell::I64(i64::from(v))),
        ColumnData::I16(v) => v.map_or(Cell::Null, |v| Cell::I64(i64::from(v))),
        ColumnData::I32(v) => v.map_or(Cell::Null, |v| Cell::I64(i64::from(v))),
        ColumnData::I64(v) => v.map_or(Cell::Null, Cell::I64),
        ColumnData::F32(v) => v.map_or(Cell::Null, |v| Cell::F64(f64::from(v))),
        ColumnData::F64(v) => v.map_or(Cell::Null, Cell::F64),
        ColumnData::Bit(v) => v.map_or(Cell::Null, Cell::Bool),
        ColumnData::String(v) => v.as_ref().map_or(Cell::Null, |v| Cell::Str(v.to_string())),
        ColumnData::Guid(v) => v.map_or(Cell::Null, Cell::Uuid),
        ColumnData::Binary(v) => v.as_ref().map_or(Cell::Null, |v| Cell::Bytes(v.to_vec())),
        ColumnData::Date(_) => time::Date::from_sql(data)
            .map_err(driver_error)?
            .map_or(Cell::Null, Cell::Date),
        ColumnData::DateTime2(_) | ColumnData::DateTime(_) | ColumnData::SmallDateTime(_) => {
            time::PrimitiveDateTime::from_sql(data)
                .map_err(driver_error)?
                .map_or(Cell::Null, Cell::DateTime)
        }
        _ => return Err(unsupported()),
    })
}

#[async_trait::async_trait]
impl Session for TiberiusSession {
    async fn query(&mut self, sql: &str, params: &[SqlParam]) -> Result<Vec<Vec<Cell>>> {
        let limit = self.query_timeout;
        let client = &mut self.client;
        let run = async move {
            let refs: Vec<&dyn tiberius::ToSql> =
                params.iter().map(|p| p as &dyn tiberius::ToSql).collect();
            let stream = client.query(sql, &refs).await.map_err(driver_error)?;
            let rows = stream.into_first_result().await.map_err(driver_error)?;
            rows.into_iter()
                .map(|row| row.into_iter().map(|data| cell(&data)).collect())
                .collect::<Result<Vec<Vec<Cell>>>>()
        };
        bounded(limit, run).await
    }

    async fn batch(&mut self, sql: &str) -> Result<()> {
        let limit = self.query_timeout;
        let client = &mut self.client;
        let run = async move {
            client
                .simple_query(sql)
                .await
                .map_err(driver_error)?
                .into_results()
                .await
                .map_err(driver_error)?;
            Ok(())
        };
        bounded(limit, run).await
    }

    async fn close(self: Box<Self>) -> Result<()> {
        self.client.close().await.map_err(driver_error)
    }
}

// endregion

#[cfg(test)]
pub(crate) mod fake {
    //! A scripted [`Connector`] that records every statement.

    use super::*;
    use std::collections::VecDeque;
    use std::sync::Mutex;

    #[derive(Debug, Clone, PartialEq)]
    pub(crate) enum Event {
        Connect,
        Batch(String),
        Query(String, Vec<SqlParam>),
        Close,
    }

    #[derive(Default)]
    pub(crate) struct Script {
        pub(crate) events: Vec<Event>,
        /// Responses to `query`, in order; an exhausted queue answers with
        /// no rows.
        pub(crate) responses: VecDeque<Result<Vec<Vec<Cell>>>>,
        pub(crate) fail_connect: bool,
        pub(crate) fail_commit: bool,
        pub(crate) fail_rollback: bool,
        pub(crate) fail_close: bool,
    }

    #[derive(Clone, Default)]
    pub(crate) struct FakeConnector(pub(crate) Arc<Mutex<Script>>);

    impl FakeConnector {
        pub(crate) fn respond(&self, rows: Vec<Vec<Cell>>) {
            self.0.lock().unwrap().responses.push_back(Ok(rows));
        }
        pub(crate) fn fail_next(&self, message: &str) {
            self.0
                .lock()
                .unwrap()
                .responses
                .push_back(Err(Error::service(message.to_string())));
        }
        pub(crate) fn events(&self) -> Vec<Event> {
            self.0.lock().unwrap().events.clone()
        }
        pub(crate) fn queries(&self) -> Vec<(String, Vec<SqlParam>)> {
            self.events()
                .into_iter()
                .filter_map(|e| match e {
                    Event::Query(sql, params) => Some((sql, params)),
                    _ => None,
                })
                .collect()
        }
        pub(crate) fn set(&self, f: impl FnOnce(&mut Script)) {
            f(&mut self.0.lock().unwrap());
        }
    }

    struct FakeSession(Arc<Mutex<Script>>);

    #[async_trait::async_trait]
    impl Connector for FakeConnector {
        async fn connect(&self) -> Result<Box<dyn Session>> {
            let mut script = self.0.lock().unwrap();
            if script.fail_connect {
                return Err(Error::service("SQL Server connection failed: refused"));
            }
            script.events.push(Event::Connect);
            Ok(Box::new(FakeSession(Arc::clone(&self.0))))
        }
    }

    #[async_trait::async_trait]
    impl Session for FakeSession {
        async fn query(&mut self, sql: &str, params: &[SqlParam]) -> Result<Vec<Vec<Cell>>> {
            let mut script = self.0.lock().unwrap();
            script
                .events
                .push(Event::Query(sql.to_string(), params.to_vec()));
            script.responses.pop_front().unwrap_or(Ok(Vec::new()))
        }

        async fn batch(&mut self, sql: &str) -> Result<()> {
            let mut script = self.0.lock().unwrap();
            script.events.push(Event::Batch(sql.to_string()));
            if sql.starts_with("COMMIT") && script.fail_commit {
                return Err(Error::service("SQL Server operation failed: commit failed"));
            }
            if sql.contains("ROLLBACK") && script.fail_rollback {
                return Err(Error::service(
                    "SQL Server operation failed: rollback failed",
                ));
            }
            Ok(())
        }

        async fn close(self: Box<Self>) -> Result<()> {
            let mut script = self.0.lock().unwrap();
            script.events.push(Event::Close);
            if script.fail_close {
                return Err(Error::service("SQL Server operation failed: close failed"));
            }
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::fake::{Event, FakeConnector};
    use super::*;

    fn handle(fake: &FakeConnector) -> ClientHandle {
        ClientHandle::new(Arc::new(fake.clone()))
    }

    async fn select_one(handle: &ClientHandle) -> Result<usize> {
        handle
            .run(|session| Box::pin(async move { Ok(session.query("SELECT 1", &[]).await?.len()) }))
            .await
    }

    const BEGIN: &str = "SET XACT_ABORT ON; BEGIN TRANSACTION";
    const ROLLBACK: &str = "IF @@TRANCOUNT > 0 ROLLBACK TRANSACTION";

    #[tokio::test]
    async fn a_successful_operation_commits_then_closes() {
        let fake = FakeConnector::default();
        fake.respond(vec![vec![Cell::I64(1)]]);
        assert_eq!(select_one(&handle(&fake)).await.unwrap(), 1);
        assert_eq!(
            fake.events(),
            vec![
                Event::Connect,
                Event::Batch(BEGIN.into()),
                Event::Query("SELECT 1".into(), vec![]),
                Event::Batch("COMMIT TRANSACTION".into()),
                Event::Close,
            ]
        );
    }

    #[tokio::test]
    async fn a_driver_error_rolls_back_and_closes() {
        let fake = FakeConnector::default();
        fake.fail_next("SQL Server operation failed: boom");
        let error = select_one(&handle(&fake)).await.unwrap_err();
        assert!(error.to_string().contains("boom"));
        assert!(!is_committed_cleanup(&error));
        let events = fake.events();
        assert_eq!(events[events.len() - 2], Event::Batch(ROLLBACK.into()));
        assert_eq!(events.last(), Some(&Event::Close));
        assert!(!events.contains(&Event::Batch("COMMIT TRANSACTION".into())));
    }

    #[tokio::test]
    async fn a_commit_failure_rolls_back_and_closes() {
        let fake = FakeConnector::default();
        fake.set(|s| s.fail_commit = true);
        let error = select_one(&handle(&fake)).await.unwrap_err();
        assert!(error.to_string().contains("commit failed"));
        assert!(!is_committed_cleanup(&error));
        let events = fake.events();
        assert_eq!(events[events.len() - 2], Event::Batch(ROLLBACK.into()));
        assert_eq!(events.last(), Some(&Event::Close));
    }

    #[tokio::test]
    async fn a_close_failure_after_commit_is_distinguishable() {
        let fake = FakeConnector::default();
        fake.set(|s| s.fail_close = true);
        let error = select_one(&handle(&fake)).await.unwrap_err();
        assert!(is_committed_cleanup(&error), "{error}");
        assert!(error.to_string().contains("do not retry"));
        assert!(fake
            .events()
            .contains(&Event::Batch("COMMIT TRANSACTION".into())));
    }

    #[tokio::test]
    async fn cleanup_failures_after_an_error_preserve_the_original() {
        let fake = FakeConnector::default();
        fake.set(|s| {
            s.fail_close = true;
            s.fail_rollback = true;
        });
        fake.fail_next("SQL Server operation failed: original");
        let error = select_one(&handle(&fake)).await.unwrap_err();
        assert!(error.to_string().contains("original"), "{error}");
        assert!(!is_committed_cleanup(&error));
    }

    #[tokio::test]
    async fn a_connection_failure_runs_nothing() {
        let fake = FakeConnector::default();
        fake.set(|s| s.fail_connect = true);
        let error = select_one(&handle(&fake)).await.unwrap_err();
        assert!(error.to_string().contains("connection failed"));
        assert!(fake.events().is_empty());
    }

    #[tokio::test]
    async fn a_closed_client_refuses_operations() {
        let fake = FakeConnector::default();
        let handle = handle(&fake);
        handle.close().await;
        assert!(handle.ensure_open().await.is_err());
        assert!(select_one(&handle).await.is_err());
        assert!(fake.events().is_empty());
    }

    #[test]
    fn connection_strings_split_on_unquoted_semicolons() {
        let segments = ado_segments("Server=x;Password={a;b};User ID='c;d';;");
        assert_eq!(
            segments.iter().map(|(k, _)| k.as_str()).collect::<Vec<_>>(),
            vec!["server", "password", "user id"]
        );
        assert_eq!(segment_value(&segments[1].1), "a;b");
        assert_eq!(segment_value(&segments[2].1), "c;d");
    }

    #[test]
    fn authentication_modes_resolve_to_credentials() {
        let parse = |s: &str| parse_connection_string(&SecretString::new(s), None);
        let (_, auth) = parse("Server=tcp:h,1433;Database=d;User ID=u;Password=p").unwrap();
        assert!(matches!(auth, Auth::ConnectionString));
        let (_, auth) =
            parse("Server=h;Database=d;Authentication=ActiveDirectoryDefault;Encrypt=yes").unwrap();
        assert!(matches!(auth, Auth::Token(_)));
        let (_, auth) = parse("Server=h;Authentication=ActiveDirectoryMSI;UID=client-id").unwrap();
        assert!(matches!(auth, Auth::Token(_)));
        let (_, auth) = parse("Server=h;Authentication=SqlPassword;UID=u;PWD=p").unwrap();
        assert!(matches!(auth, Auth::ConnectionString));
        assert!(parse("Server=h;Authentication=ActiveDirectoryInteractive").is_err());
        let credential: Arc<dyn TokenCredential> =
            Arc::new(agent_framework_azure::StaticTokenCredential::new("t"));
        let (_, auth) =
            parse_connection_string(&SecretString::new("Server=h;UID=u;PWD=p"), Some(credential))
                .unwrap();
        assert!(matches!(auth, Auth::Token(_)));
    }

    #[test]
    fn a_bad_connection_string_error_does_not_echo_secrets() {
        let error = parse_connection_string(
            &SecretString::new("Server=tcp:h,notaport;Password=hunter2"),
            None,
        )
        .err();
        if let Some(error) = error {
            assert!(!error.to_string().contains("hunter2"));
        }
    }
}
