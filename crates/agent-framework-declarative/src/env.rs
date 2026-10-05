//! Shell-style environment-variable interpolation for spec string fields.
//!
//! Every string in a parsed agent or Rust-native workflow spec is passed
//! through [`interpolate`], which expands `${VAR}` and `${VAR:-default}`
//! placeholders. This is a convention of this port; upstream instead
//! evaluates PowerFx `=Env.VAR` expressions, which this crate also supports
//! (agent fields with safe mode off, and the `Env` scope of upstream-format
//! workflows — which are not `${}`-interpolated).
//!
//! * `${VAR}` — expands to the value of `VAR`; errors if `VAR` is unset.
//! * `${VAR:-default}` — expands to `VAR` if set (and non-empty per POSIX
//!   `:-`), otherwise the literal `default` (itself interpolated).
//! * `$${` — an escape producing a literal `${`.

use crate::error::{DeclarativeError, Result};

/// A source of environment-variable values.
///
/// The default source reads the process environment; tests inject a fixed map
/// for hermetic behavior.
pub trait EnvSource {
    /// Look up a variable, returning `None` if it is not set.
    fn get(&self, key: &str) -> Option<String>;
}

/// An [`EnvSource`] backed by [`std::env::var`].
#[derive(Debug, Clone, Copy, Default)]
pub struct ProcessEnv;

impl EnvSource for ProcessEnv {
    fn get(&self, key: &str) -> Option<String> {
        std::env::var(key).ok()
    }
}

impl<F> EnvSource for F
where
    F: Fn(&str) -> Option<String>,
{
    fn get(&self, key: &str) -> Option<String> {
        self(key)
    }
}

/// Expand `${VAR}` / `${VAR:-default}` placeholders in `input`.
///
/// Returns [`DeclarativeError::MissingEnvVar`] when a `${VAR}` without a default
/// is unset, and [`DeclarativeError::MalformedPlaceholder`] for an unterminated
/// `${`.
pub fn interpolate(input: &str, env: &dyn EnvSource) -> Result<String> {
    if !input.contains('$') {
        return Ok(input.to_string());
    }
    let bytes = input.as_bytes();
    let mut out = String::with_capacity(input.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'$' {
            // Escape: "$${" -> literal "${".
            if i + 1 < bytes.len() && bytes[i + 1] == b'$' {
                out.push('$');
                i += 2;
                continue;
            }
            if i + 1 < bytes.len() && bytes[i + 1] == b'{' {
                // Find the matching closing brace, honoring nested `${...}`.
                let start = i + 2;
                let Some(rel_end) = find_matching_close(&input[start..]) else {
                    return Err(DeclarativeError::MalformedPlaceholder {
                        value: input.to_string(),
                        reason: "unterminated '${' placeholder".to_string(),
                    });
                };
                let end = start + rel_end;
                let expr = &input[start..end];
                out.push_str(&expand_placeholder(expr, input, env)?);
                i = end + 1;
                continue;
            }
        }
        // Not a placeholder: copy the char (respecting UTF-8 boundaries).
        let ch = input[i..].chars().next().unwrap();
        out.push(ch);
        i += ch.len_utf8();
    }
    Ok(out)
}

/// Given the text immediately after an opening `${`, return the byte offset of
/// the matching `}`, accounting for nested `${...}` in default values.
fn find_matching_close(s: &str) -> Option<usize> {
    let bytes = s.as_bytes();
    let mut depth = 0usize;
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'$' && i + 1 < bytes.len() && bytes[i + 1] == b'{' {
            depth += 1;
            i += 2;
        } else if bytes[i] == b'}' {
            if depth == 0 {
                return Some(i);
            }
            depth -= 1;
            i += 1;
        } else {
            i += 1;
        }
    }
    None
}

/// Expand the contents of a single `${...}` placeholder.
fn expand_placeholder(expr: &str, whole: &str, env: &dyn EnvSource) -> Result<String> {
    if let Some((name, default)) = expr.split_once(":-") {
        let name = name.trim();
        match env.get(name) {
            Some(v) if !v.is_empty() => Ok(v),
            // POSIX `:-` uses the default when unset *or* empty.
            _ => interpolate(default, env),
        }
    } else {
        let name = expr.trim();
        if name.is_empty() {
            return Err(DeclarativeError::MalformedPlaceholder {
                value: whole.to_string(),
                reason: "empty '${}' placeholder".to_string(),
            });
        }
        env.get(name)
            .ok_or_else(|| DeclarativeError::MissingEnvVar(name.to_string()))
    }
}

/// Recursively interpolate every string in a [`serde_yaml::Value`] tree.
///
/// Mapping keys are interpolated too (they are almost always literals, but this
/// keeps behavior uniform). Applied to the whole document before typed
/// deserialization so interpolation reaches every string field.
pub fn interpolate_value(value: &mut serde_yaml::Value, env: &dyn EnvSource) -> Result<()> {
    match value {
        serde_yaml::Value::String(s) => {
            *s = interpolate(s, env)?;
        }
        serde_yaml::Value::Sequence(seq) => {
            for item in seq.iter_mut() {
                interpolate_value(item, env)?;
            }
        }
        serde_yaml::Value::Mapping(map) => {
            // Rebuild the mapping so interpolated keys are applied.
            let mut rebuilt = serde_yaml::Mapping::with_capacity(map.len());
            for (k, v) in std::mem::take(map) {
                let mut k = k;
                let mut v = v;
                interpolate_value(&mut k, env)?;
                interpolate_value(&mut v, env)?;
                rebuilt.insert(k, v);
            }
            *map = rebuilt;
        }
        _ => {}
    }
    Ok(())
}
