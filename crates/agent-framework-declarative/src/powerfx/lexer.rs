//! Tokenizer for the PowerFx subset.
//!
//! Lexical rules follow the PowerFx language reference (en-US locale):
//!
//! * numbers use `.` as the decimal separator and may carry an exponent
//!   (`1.5e3`);
//! * text literals are double-quoted and the only escape is a doubled quote
//!   (`"say ""hi"""`) — PowerFx has **no** backslash escapes;
//! * single quotes delimit *identifiers* (`Local.'my name'`), with `''`
//!   escaping a quote, exactly like PowerFx;
//! * `$"…{expr}…"` is an interpolated string, where `{{` / `}}` produce
//!   literal braces;
//! * `//` line comments and `/* … */` block comments are skipped.

use super::{PowerFxError, PowerFxErrorKind};

/// A lexical token.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Tok {
    /// A numeric literal.
    Number(f64),
    /// A text literal (escapes already resolved).
    Str(String),
    /// An interpolated string `$"…"`.
    Interp(Vec<InterpTok>),
    /// A bare identifier (may be a keyword such as `And`).
    Ident(String),
    /// A single-quoted identifier — never treated as a keyword.
    QuotedIdent(String),
    /// An operator or punctuation symbol.
    Op(&'static str),
    /// End of input.
    Eof,
}

/// One segment of an interpolated string.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum InterpTok {
    /// Literal text.
    Text(String),
    /// An embedded expression, already tokenized (terminated by [`Tok::Eof`]).
    Expr(Vec<Spanned>),
}

/// A token with its character offset in the formula.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Spanned {
    pub tok: Tok,
    pub pos: usize,
}

/// Multi-character operators first so the longest match wins.
const OPERATORS: &[&str] = &[
    "<>", "<=", ">=", "&&", "||", "+", "-", "*", "/", "^", "&", "%", "=", "<", ">", "!", ".", ",",
    "(", ")", "[", "]", "{", "}", ":", ";", "@",
];

struct Lexer<'a> {
    chars: &'a [char],
    pos: usize,
}

/// Tokenize a whole formula. The returned vector always ends with [`Tok::Eof`].
pub(crate) fn tokenize(formula: &str) -> Result<Vec<Spanned>, PowerFxError> {
    let chars: Vec<char> = formula.chars().collect();
    let mut lexer = Lexer {
        chars: &chars,
        pos: 0,
    };
    let mut out = Vec::new();
    loop {
        let tok = lexer.next_token()?;
        let done = tok.tok == Tok::Eof;
        out.push(tok);
        if done {
            return Ok(out);
        }
    }
}

fn syntax(pos: usize, msg: impl Into<String>) -> PowerFxError {
    PowerFxError::new(
        PowerFxErrorKind::Syntax,
        format!("{} (at character {pos})", msg.into()),
    )
}

impl Lexer<'_> {
    fn peek(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }

    fn peek_at(&self, offset: usize) -> Option<char> {
        self.chars.get(self.pos + offset).copied()
    }

    fn starts_with(&self, s: &str) -> bool {
        s.chars()
            .enumerate()
            .all(|(i, c)| self.peek_at(i) == Some(c))
    }

    fn skip_trivia(&mut self) -> Result<(), PowerFxError> {
        loop {
            match self.peek() {
                Some(c) if c.is_whitespace() => self.pos += 1,
                Some('/') if self.peek_at(1) == Some('/') => {
                    while let Some(c) = self.peek() {
                        if c == '\n' || c == '\r' {
                            break;
                        }
                        self.pos += 1;
                    }
                }
                Some('/') if self.peek_at(1) == Some('*') => {
                    let start = self.pos;
                    self.pos += 2;
                    loop {
                        match self.peek() {
                            None => return Err(syntax(start, "unterminated block comment")),
                            Some('*') if self.peek_at(1) == Some('/') => {
                                self.pos += 2;
                                break;
                            }
                            Some(_) => self.pos += 1,
                        }
                    }
                }
                _ => return Ok(()),
            }
        }
    }

    fn next_token(&mut self) -> Result<Spanned, PowerFxError> {
        self.skip_trivia()?;
        let pos = self.pos;
        let Some(c) = self.peek() else {
            return Ok(Spanned { tok: Tok::Eof, pos });
        };
        let tok = if c.is_ascii_digit() {
            self.number()?
        } else if c == '"' {
            self.pos += 1;
            Tok::Str(self.string_body(pos)?)
        } else if c == '$' && self.peek_at(1) == Some('"') {
            self.pos += 2;
            Tok::Interp(self.interpolated(pos)?)
        } else if c == '\'' {
            self.pos += 1;
            Tok::QuotedIdent(self.quoted_ident(pos)?)
        } else if c.is_alphabetic() || c == '_' {
            let start = self.pos;
            while let Some(c) = self.peek() {
                if c.is_alphanumeric() || c == '_' {
                    self.pos += 1;
                } else {
                    break;
                }
            }
            Tok::Ident(self.chars[start..self.pos].iter().collect())
        } else if let Some(op) = OPERATORS.iter().find(|op| self.starts_with(op)) {
            self.pos += op.chars().count();
            Tok::Op(op)
        } else {
            return Err(syntax(pos, format!("unexpected character {c:?}")));
        };
        Ok(Spanned { tok, pos })
    }

    fn number(&mut self) -> Result<Tok, PowerFxError> {
        let start = self.pos;
        while self.peek().is_some_and(|c| c.is_ascii_digit()) {
            self.pos += 1;
        }
        if self.peek() == Some('.') && self.peek_at(1).is_some_and(|c| c.is_ascii_digit()) {
            self.pos += 1;
            while self.peek().is_some_and(|c| c.is_ascii_digit()) {
                self.pos += 1;
            }
        }
        if matches!(self.peek(), Some('e') | Some('E')) {
            let save = self.pos;
            self.pos += 1;
            if matches!(self.peek(), Some('+') | Some('-')) {
                self.pos += 1;
            }
            if self.peek().is_some_and(|c| c.is_ascii_digit()) {
                while self.peek().is_some_and(|c| c.is_ascii_digit()) {
                    self.pos += 1;
                }
            } else {
                self.pos = save;
            }
        }
        let text: String = self.chars[start..self.pos].iter().collect();
        text.parse::<f64>()
            .map(Tok::Number)
            .map_err(|_| syntax(start, format!("invalid number literal {text:?}")))
    }

    /// Read a `"…"` body (opening quote already consumed).
    fn string_body(&mut self, start: usize) -> Result<String, PowerFxError> {
        let mut s = String::new();
        loop {
            match self.peek() {
                None => return Err(syntax(start, "unterminated text literal")),
                Some('"') if self.peek_at(1) == Some('"') => {
                    s.push('"');
                    self.pos += 2;
                }
                Some('"') => {
                    self.pos += 1;
                    return Ok(s);
                }
                Some(c) => {
                    s.push(c);
                    self.pos += 1;
                }
            }
        }
    }

    fn quoted_ident(&mut self, start: usize) -> Result<String, PowerFxError> {
        let mut s = String::new();
        loop {
            match self.peek() {
                None => return Err(syntax(start, "unterminated quoted identifier")),
                Some('\'') if self.peek_at(1) == Some('\'') => {
                    s.push('\'');
                    self.pos += 2;
                }
                Some('\'') => {
                    self.pos += 1;
                    if s.is_empty() {
                        return Err(syntax(start, "empty quoted identifier"));
                    }
                    return Ok(s);
                }
                Some(c) => {
                    s.push(c);
                    self.pos += 1;
                }
            }
        }
    }

    /// Read an interpolated string body (`$"` already consumed).
    fn interpolated(&mut self, start: usize) -> Result<Vec<InterpTok>, PowerFxError> {
        let mut parts = Vec::new();
        let mut text = String::new();
        loop {
            match self.peek() {
                None => return Err(syntax(start, "unterminated interpolated string")),
                Some('"') if self.peek_at(1) == Some('"') => {
                    text.push('"');
                    self.pos += 2;
                }
                Some('"') => {
                    self.pos += 1;
                    if !text.is_empty() {
                        parts.push(InterpTok::Text(std::mem::take(&mut text)));
                    }
                    return Ok(parts);
                }
                Some('{') if self.peek_at(1) == Some('{') => {
                    text.push('{');
                    self.pos += 2;
                }
                Some('}') if self.peek_at(1) == Some('}') => {
                    text.push('}');
                    self.pos += 2;
                }
                Some('{') => {
                    let brace = self.pos;
                    self.pos += 1;
                    if !text.is_empty() {
                        parts.push(InterpTok::Text(std::mem::take(&mut text)));
                    }
                    let mut tokens = Vec::new();
                    let mut depth = 0usize;
                    loop {
                        let t = self.next_token()?;
                        match &t.tok {
                            Tok::Eof => {
                                return Err(syntax(brace, "unterminated interpolation hole"))
                            }
                            Tok::Op("{") => depth += 1,
                            Tok::Op("}") if depth == 0 => {
                                tokens.push(Spanned {
                                    tok: Tok::Eof,
                                    pos: t.pos,
                                });
                                break;
                            }
                            Tok::Op("}") => depth -= 1,
                            _ => {}
                        }
                        tokens.push(t);
                    }
                    parts.push(InterpTok::Expr(tokens));
                }
                Some('}') => return Err(syntax(self.pos, "unmatched '}' in interpolated string")),
                Some(c) => {
                    text.push(c);
                    self.pos += 1;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toks(s: &str) -> Vec<Tok> {
        tokenize(s).unwrap().into_iter().map(|t| t.tok).collect()
    }

    #[test]
    fn numbers_strings_and_operators() {
        assert_eq!(
            toks("1.5 + 2e2 <> \"a\"\"b\""),
            vec![
                Tok::Number(1.5),
                Tok::Op("+"),
                Tok::Number(200.0),
                Tok::Op("<>"),
                Tok::Str("a\"b".into()),
                Tok::Eof
            ]
        );
    }

    #[test]
    fn identifiers_and_quoted_identifiers() {
        assert_eq!(
            toks("Local.'my var'"),
            vec![
                Tok::Ident("Local".into()),
                Tok::Op("."),
                Tok::QuotedIdent("my var".into()),
                Tok::Eof
            ]
        );
    }

    #[test]
    fn comments_are_skipped() {
        assert_eq!(
            toks("1 // c\n + /* b */ 2"),
            vec![Tok::Number(1.0), Tok::Op("+"), Tok::Number(2.0), Tok::Eof]
        );
    }

    #[test]
    fn interpolation_holes_are_tokenized() {
        let t = toks("$\"a{{ {x & \"}\"} b\"");
        let Tok::Interp(parts) = &t[0] else {
            panic!("expected interpolation, got {t:?}")
        };
        assert_eq!(parts[0], InterpTok::Text("a{ ".into()));
        assert!(matches!(&parts[1], InterpTok::Expr(e) if e.len() == 4));
        assert_eq!(parts[2], InterpTok::Text(" b".into()));
    }

    #[test]
    fn errors_for_unterminated_literals() {
        assert!(tokenize("\"abc").is_err());
        assert!(tokenize("'abc").is_err());
        assert!(tokenize("$\"a{1").is_err());
        assert!(tokenize("/* x").is_err());
        assert!(tokenize("#").is_err());
    }
}
