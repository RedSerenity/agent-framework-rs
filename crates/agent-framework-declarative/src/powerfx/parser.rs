//! Recursive-descent / precedence-climbing parser producing an [`Expr`] tree.
//!
//! Operator precedence mirrors the PowerFx `TexlParser` (lowest first):
//! `||`/`Or` < `&&`/`And` < `in`/`exactin` < comparisons (`= <> < <= > >=`)
//! < `&` < `+ -` < `* /` < prefix (`-`, `!`, `Not`) < `^` < postfix `%`
//! < primary (member access, calls).

use super::lexer::{tokenize, InterpTok, Spanned, Tok};
use super::{ExpressionLimits, PowerFxError, PowerFxErrorKind};

/// A parsed PowerFx expression.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Expr {
    Number(f64),
    Text(String),
    Bool(bool),
    Interp(Vec<InterpPart>),
    /// A bare (or single-quoted) identifier.
    Name(String),
    /// The `ThisRecord` keyword.
    ThisRecord,
    Member(Box<Expr>, String),
    Call(String, Vec<Expr>),
    Unary(UnaryOp, Box<Expr>),
    Binary(BinaryOp, Box<Expr>, Box<Expr>),
    Record(Vec<(String, Expr)>),
    Table(Vec<Expr>),
}

/// A segment of an interpolated string.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum InterpPart {
    Text(String),
    Expr(Expr),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum UnaryOp {
    Neg,
    Not,
    Percent,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BinaryOp {
    Or,
    And,
    In,
    ExactIn,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    Concat,
    Add,
    Sub,
    Mul,
    Div,
    Pow,
}

const PREC_OR: u8 = 1;
const PREC_AND: u8 = 2;
const PREC_IN: u8 = 3;
const PREC_COMPARE: u8 = 4;
const PREC_CONCAT: u8 = 5;
const PREC_ADD: u8 = 6;
const PREC_MUL: u8 = 7;
const PREC_PREFIX: u8 = 8;
const PREC_POWER: u8 = 9;

/// Parse `formula` (without the leading `=`) into an expression tree,
/// enforcing the length and nesting limits.
pub(crate) fn parse(formula: &str, limits: &ExpressionLimits) -> Result<Expr, PowerFxError> {
    let length = formula.chars().count();
    if length > limits.max_expression_length {
        return Err(PowerFxError::new(
            PowerFxErrorKind::Limit,
            format!(
                "expression is {length} characters long, exceeding the maximum of {}",
                limits.max_expression_length
            ),
        ));
    }
    let tokens = tokenize(formula)?;
    let mut parser = Parser {
        tokens,
        pos: 0,
        depth: 0,
        max_depth: limits.max_depth,
    };
    let expr = parser.expr(0)?;
    parser.expect_eof()?;
    Ok(expr)
}

struct Parser {
    tokens: Vec<Spanned>,
    pos: usize,
    depth: usize,
    max_depth: usize,
}

fn syntax(pos: usize, msg: impl Into<String>) -> PowerFxError {
    PowerFxError::new(
        PowerFxErrorKind::Syntax,
        format!("{} (at character {pos})", msg.into()),
    )
}

impl Parser {
    fn peek(&self) -> &Tok {
        &self.tokens[self.pos.min(self.tokens.len() - 1)].tok
    }

    fn peek_next(&self) -> &Tok {
        &self.tokens[(self.pos + 1).min(self.tokens.len() - 1)].tok
    }

    fn here(&self) -> usize {
        self.tokens[self.pos.min(self.tokens.len() - 1)].pos
    }

    fn bump(&mut self) -> Tok {
        let tok = self.peek().clone();
        if self.pos < self.tokens.len() - 1 {
            self.pos += 1;
        }
        tok
    }

    fn eat_op(&mut self, op: &str) -> bool {
        if matches!(self.peek(), Tok::Op(o) if *o == op) {
            self.bump();
            true
        } else {
            false
        }
    }

    fn expect_op(&mut self, op: &str) -> Result<(), PowerFxError> {
        if self.eat_op(op) {
            Ok(())
        } else {
            Err(syntax(
                self.here(),
                format!("expected '{op}', found {}", describe(self.peek())),
            ))
        }
    }

    fn expect_eof(&mut self) -> Result<(), PowerFxError> {
        match self.peek() {
            Tok::Eof => Ok(()),
            other => Err(syntax(
                self.here(),
                format!(
                    "unexpected {} after the end of the expression",
                    describe(other)
                ),
            )),
        }
    }

    fn enter(&mut self) -> Result<(), PowerFxError> {
        self.depth += 1;
        if self.depth > self.max_depth {
            return Err(PowerFxError::new(
                PowerFxErrorKind::Limit,
                format!(
                    "expression nesting exceeds the maximum depth of {}",
                    self.max_depth
                ),
            ));
        }
        Ok(())
    }

    /// The binary operator at the cursor and its precedence, if any.
    fn binary_op(&self) -> Option<(BinaryOp, u8)> {
        Some(match self.peek() {
            Tok::Op("||") => (BinaryOp::Or, PREC_OR),
            Tok::Op("&&") => (BinaryOp::And, PREC_AND),
            Tok::Ident(s) if s == "Or" => (BinaryOp::Or, PREC_OR),
            Tok::Ident(s) if s == "And" => (BinaryOp::And, PREC_AND),
            Tok::Ident(s) if s == "in" => (BinaryOp::In, PREC_IN),
            Tok::Ident(s) if s == "exactin" => (BinaryOp::ExactIn, PREC_IN),
            Tok::Op("=") => (BinaryOp::Eq, PREC_COMPARE),
            Tok::Op("<>") => (BinaryOp::Ne, PREC_COMPARE),
            Tok::Op("<") => (BinaryOp::Lt, PREC_COMPARE),
            Tok::Op("<=") => (BinaryOp::Le, PREC_COMPARE),
            Tok::Op(">") => (BinaryOp::Gt, PREC_COMPARE),
            Tok::Op(">=") => (BinaryOp::Ge, PREC_COMPARE),
            Tok::Op("&") => (BinaryOp::Concat, PREC_CONCAT),
            Tok::Op("+") => (BinaryOp::Add, PREC_ADD),
            Tok::Op("-") => (BinaryOp::Sub, PREC_ADD),
            Tok::Op("*") => (BinaryOp::Mul, PREC_MUL),
            Tok::Op("/") => (BinaryOp::Div, PREC_MUL),
            Tok::Op("^") => (BinaryOp::Pow, PREC_POWER),
            _ => return None,
        })
    }

    fn expr(&mut self, min_prec: u8) -> Result<Expr, PowerFxError> {
        self.enter()?;
        let mut left = self.unary()?;
        while let Some((op, prec)) = self.binary_op() {
            if prec < min_prec {
                break;
            }
            self.bump();
            let right = self.expr(prec + 1)?;
            left = Expr::Binary(op, Box::new(left), Box::new(right));
        }
        self.depth -= 1;
        Ok(left)
    }

    fn unary(&mut self) -> Result<Expr, PowerFxError> {
        let op = match self.peek() {
            Tok::Op("-") => Some(UnaryOp::Neg),
            Tok::Op("!") => Some(UnaryOp::Not),
            Tok::Ident(s) if s == "Not" && !matches!(self.peek_next(), Tok::Op("(")) => {
                Some(UnaryOp::Not)
            }
            _ => None,
        };
        if let Some(op) = op {
            self.bump();
            let operand = self.expr(PREC_PREFIX)?;
            return Ok(Expr::Unary(op, Box::new(operand)));
        }
        self.postfix()
    }

    fn postfix(&mut self) -> Result<Expr, PowerFxError> {
        let mut expr = self.primary()?;
        loop {
            if self.eat_op(".") {
                let name = match self.bump() {
                    Tok::Ident(s) | Tok::QuotedIdent(s) => s,
                    other => {
                        return Err(syntax(
                            self.here(),
                            format!(
                                "expected a field name after '.', found {}",
                                describe(&other)
                            ),
                        ))
                    }
                };
                expr = Expr::Member(Box::new(expr), name);
            } else if self.eat_op("%") {
                expr = Expr::Unary(UnaryOp::Percent, Box::new(expr));
            } else {
                return Ok(expr);
            }
        }
    }

    fn primary(&mut self) -> Result<Expr, PowerFxError> {
        let pos = self.here();
        match self.bump() {
            Tok::Number(n) => Ok(Expr::Number(n)),
            Tok::Str(s) => Ok(Expr::Text(s)),
            Tok::Interp(parts) => {
                let mut out = Vec::with_capacity(parts.len());
                for part in parts {
                    match part {
                        InterpTok::Text(t) => out.push(InterpPart::Text(t)),
                        InterpTok::Expr(tokens) => {
                            let mut sub = Parser {
                                tokens,
                                pos: 0,
                                depth: self.depth,
                                max_depth: self.max_depth,
                            };
                            let e = sub.expr(0)?;
                            sub.expect_eof()?;
                            out.push(InterpPart::Expr(e));
                        }
                    }
                }
                Ok(Expr::Interp(out))
            }
            Tok::Ident(name) => {
                if matches!(self.peek(), Tok::Op("(")) {
                    self.bump();
                    let args = self.list(")")?;
                    return Ok(Expr::Call(name, args));
                }
                match name.as_str() {
                    "true" => Ok(Expr::Bool(true)),
                    "false" => Ok(Expr::Bool(false)),
                    "ThisRecord" => Ok(Expr::ThisRecord),
                    "And" | "Or" | "in" | "exactin" => Err(syntax(
                        pos,
                        format!("expected an operand, found operator '{name}'"),
                    )),
                    _ => Ok(Expr::Name(name)),
                }
            }
            Tok::QuotedIdent(name) => Ok(Expr::Name(name)),
            Tok::Op("(") => {
                self.enter()?;
                let e = self.expr(0)?;
                self.expect_op(")")?;
                self.depth -= 1;
                Ok(e)
            }
            Tok::Op("[") => {
                self.enter()?;
                let items = self.list("]")?;
                self.depth -= 1;
                Ok(Expr::Table(items))
            }
            Tok::Op("{") => {
                self.enter()?;
                let mut fields = Vec::new();
                if !self.eat_op("}") {
                    loop {
                        let name = match self.bump() {
                            Tok::Ident(s) | Tok::QuotedIdent(s) | Tok::Str(s) => s,
                            other => {
                                return Err(syntax(
                                    self.here(),
                                    format!("expected a field name, found {}", describe(&other)),
                                ))
                            }
                        };
                        self.expect_op(":")?;
                        let value = self.expr(0)?;
                        fields.push((name, value));
                        if self.eat_op(",") {
                            continue;
                        }
                        self.expect_op("}")?;
                        break;
                    }
                }
                self.depth -= 1;
                Ok(Expr::Record(fields))
            }
            other => Err(syntax(
                pos,
                format!("expected an operand, found {}", describe(&other)),
            )),
        }
    }

    /// Parse a comma-separated list terminated by `close` (already past the
    /// opening delimiter).
    fn list(&mut self, close: &str) -> Result<Vec<Expr>, PowerFxError> {
        let mut items = Vec::new();
        if self.eat_op(close) {
            return Ok(items);
        }
        loop {
            items.push(self.expr(0)?);
            if self.eat_op(",") {
                continue;
            }
            self.expect_op(close)?;
            return Ok(items);
        }
    }
}

fn describe(tok: &Tok) -> String {
    match tok {
        Tok::Number(n) => format!("number {n}"),
        Tok::Str(_) | Tok::Interp(_) => "a text literal".to_string(),
        Tok::Ident(s) | Tok::QuotedIdent(s) => format!("identifier '{s}'"),
        Tok::Op(o) => format!("'{o}'"),
        Tok::Eof => "the end of the expression".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(s: &str) -> Expr {
        parse(s, &ExpressionLimits::default()).unwrap()
    }

    #[test]
    fn precedence_of_arithmetic_and_comparison() {
        // 1 + 2 * 3 = 7  →  (1 + (2*3)) = 7
        let e = p("1 + 2 * 3 = 7");
        let Expr::Binary(BinaryOp::Eq, l, _) = e else {
            panic!()
        };
        assert!(matches!(*l, Expr::Binary(BinaryOp::Add, _, _)));
    }

    #[test]
    fn and_binds_tighter_than_or() {
        let e = p("a Or b And c");
        let Expr::Binary(BinaryOp::Or, _, r) = e else {
            panic!()
        };
        assert!(matches!(*r, Expr::Binary(BinaryOp::And, _, _)));
    }

    #[test]
    fn concat_binds_tighter_than_comparison() {
        let e = p("\"a\" & x = \"ab\"");
        assert!(matches!(e, Expr::Binary(BinaryOp::Eq, _, _)));
    }

    #[test]
    fn prefix_not_and_bang() {
        assert!(matches!(p("!x"), Expr::Unary(UnaryOp::Not, _)));
        assert!(matches!(p("Not x"), Expr::Unary(UnaryOp::Not, _)));
        assert!(matches!(p("Not(x)"), Expr::Call(ref n, _) if n == "Not"));
    }

    #[test]
    fn records_tables_members_and_calls() {
        let e = p("First([{a: 1}, {a: 2}]).a");
        let Expr::Member(inner, field) = e else {
            panic!()
        };
        assert_eq!(field, "a");
        assert!(matches!(*inner, Expr::Call(ref n, ref args) if n == "First" && args.len() == 1));
    }

    #[test]
    fn syntax_errors_are_reported() {
        let limits = ExpressionLimits::default();
        for bad in ["1 +", "(1", "{a 1}", "f(1,", "1 2", "", "And", "a."] {
            let err = parse(bad, &limits).unwrap_err();
            assert_eq!(err.kind(), PowerFxErrorKind::Syntax, "{bad}: {err}");
        }
    }

    #[test]
    fn length_and_depth_limits() {
        let limits = ExpressionLimits {
            max_expression_length: 5,
            max_depth: 64,
        };
        assert_eq!(
            parse("123456", &limits).unwrap_err().kind(),
            PowerFxErrorKind::Limit
        );
        let limits = ExpressionLimits {
            max_expression_length: 10_000,
            max_depth: 10,
        };
        let deep = format!("{}1{}", "(".repeat(20), ")".repeat(20));
        assert_eq!(
            parse(&deep, &limits).unwrap_err().kind(),
            PowerFxErrorKind::Limit
        );
        assert!(parse("((1))", &limits).is_ok());
    }
}
