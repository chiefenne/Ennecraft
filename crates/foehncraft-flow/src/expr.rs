//! Expressions in the case file (D53).
//!
//! Boundary values, the body force and the viscosity can be given as
//! strings of a small language instead of numbers:
//!
//! ```toml
//! [constants]
//! u_max = 1.5
//! h = 0.41
//!
//! [tables.ramp]            # piecewise linear, clamped at the ends
//! x = [0.0, 0.5, 10.0]
//! y = [0.0, 1.0, 1.0]
//!
//! [boundary.inlet]
//! type = "velocity-inlet"
//! velocity = ["4 * u_max * y * (h - y) / h^2 * ramp(t)", 0.0, 0.0]
//! ```
//!
//! The language: numbers, the variables `x`, `y`, `z` (m, the face or cell
//! centroid) and `t` (s, simulated time), the constants `pi` and `e`, the
//! case's `[constants]`, the operators `+ - * / % ^` (also `**`),
//! comparisons `< <= > >= == !=`, `and`/`&&`, `or`/`||`, `not`/`!`
//! (true is 1, false 0), the functions `sin cos tan asin acos atan sinh
//! cosh tanh exp log log10 sqrt abs floor ceil round sign` (one argument),
//! `atan2 pow min max hypot step` (two), `clamp(a, lo, hi)` and
//! `if(condition, a, b)`, and every `[tables.<name>]` as a function of one
//! argument. `^` binds tighter than a unary minus (`-x^2` is −x²) and is
//! right-associative.
//!
//! Each string is parsed and checked when the case is read: syntax, names
//! (with a "did you mean"), the number of arguments, and the variables the
//! key allows (a steady case has no `t`; a mass flow is one number for the
//! boundary, so it takes only `t`; the viscosity takes no `t`, as properties
//! do not change during a run, D100). Then it is compiled into a program of
//! the stack machine in `foehncraft_kernels::expr`, with the parts that
//! depend on no variable folded into constants by the same machine. The
//! program, not the string, is what the solver evaluates, on the CPU and
//! the GPU alike, with the in-house functions of D93 (6).
//!
//! The Python API (D61) builds the same strings from expression objects,
//! so a case built in Python and its TOML give the same programs.

use crate::diag::{self, Diagnostic};
use anyhow::Result;
use foehncraft_kernels::expr::{self as k, fn1, fn2, op, word};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// A number, or an expression string (D53).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(untagged)]
pub enum Scalar {
    /// A number.
    Number(f64),
    /// An expression of x, y, z, t and the case constants.
    Expr(String),
}

impl Default for Scalar {
    fn default() -> Self {
        Scalar::Number(0.0)
    }
}

impl From<f64> for Scalar {
    fn from(v: f64) -> Self {
        Scalar::Number(v)
    }
}

impl Scalar {
    /// The number, when it is one.
    pub fn number(&self) -> Option<f64> {
        match self {
            Scalar::Number(v) => Some(*v),
            Scalar::Expr(_) => None,
        }
    }

    /// True for an expression string.
    pub fn is_expr(&self) -> bool {
        matches!(self, Scalar::Expr(_))
    }

    /// The text of the value (a number as TOML writes it).
    pub fn text(&self) -> String {
        match self {
            Scalar::Number(v) => format!("{v:?}"),
            Scalar::Expr(s) => s.clone(),
        }
    }
}

impl fmt::Display for Scalar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Scalar::Number(v) => write!(f, "{v}"),
            Scalar::Expr(s) => write!(f, "\"{s}\""),
        }
    }
}

/// Three numbers or expressions (a vector value).
pub type Vector = [Scalar; 3];

/// A vector of numbers as a [`Vector`].
pub fn vector(v: [f64; 3]) -> Vector {
    v.map(Scalar::Number)
}

/// The numbers of a vector, when all three are numbers.
pub fn numbers(v: &Vector) -> Option<[f64; 3]> {
    Some([v[0].number()?, v[1].number()?, v[2].number()?])
}

/// Variables an expression may use, as a bit set.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Vars(pub u32);

impl Vars {
    /// x.
    pub const X: Vars = Vars(1);
    /// y.
    pub const Y: Vars = Vars(2);
    /// z.
    pub const Z: Vars = Vars(4);
    /// t.
    pub const T: Vars = Vars(8);
    /// x, y, z.
    pub const SPACE: Vars = Vars(7);
    /// x, y, z, t.
    pub const ALL: Vars = Vars(15);
    /// None.
    pub const NONE: Vars = Vars(0);

    /// The union.
    pub const fn with(self, o: Vars) -> Vars {
        Vars(self.0 | o.0)
    }

    /// Every variable of `o` is in `self`.
    pub const fn contains(self, o: Vars) -> bool {
        self.0 & o.0 == o.0
    }

    /// Uses one of the space variables.
    pub const fn space(self) -> bool {
        self.0 & 7 != 0
    }

    /// Uses t.
    pub const fn time(self) -> bool {
        self.0 & 8 != 0
    }

    /// The names, for messages.
    pub fn names(self) -> String {
        let n: Vec<&str> = [("x", 1), ("y", 2), ("z", 4), ("t", 8)]
            .iter()
            .filter(|(_, b)| self.0 & b != 0)
            .map(|(n, _)| *n)
            .collect();
        if n.is_empty() {
            "no variable".into()
        } else {
            n.join(", ")
        }
    }
}

/// A table of the case (`[tables.<name>]`): piecewise linear in its
/// argument, constant beyond the ends.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(default, deny_unknown_fields)]
pub struct TableSection {
    /// The abscissae, strictly increasing (inline table).
    pub x: Vec<f64>,
    /// The ordinates (inline table).
    pub y: Vec<f64>,
    /// A file of two columns (argument, value) instead of `x` and `y`,
    /// relative to the case file; `#` starts a comment, a non-numeric
    /// first line is a header.
    pub file: Option<PathBuf>,
    /// The values read from `file`.
    #[serde(skip)]
    #[schemars(skip)]
    pub data: Option<Arc<TableData>>,
}

/// The abscissae, ordinates and SHA-256 of a table read from a file.
pub type TableData = (Vec<f64>, Vec<f64>, String);

impl TableSection {
    /// The abscissae and ordinates (from the file when it has one).
    pub fn points(&self) -> (&[f64], &[f64]) {
        match &self.data {
            Some(d) => (&d.0, &d.1),
            None => (&self.x, &self.y),
        }
    }

    /// SHA-256 of the file, if the table comes from one.
    pub fn sha256(&self) -> Option<&str> {
        self.data.as_ref().map(|d| d.2.as_str())
    }

    /// Reads the file of a file table.
    pub fn load(&mut self, name: &str) -> Result<()> {
        let Some(path) = &self.file else {
            return Ok(());
        };
        let t = crate::bc::ProfileTable::read(path).map_err(|e| {
            diag::err(
                Diagnostic::new(diag::EXPR_TABLE, format!("[tables.{name}]: {e}"))
                    .key(format!("tables.{name}.file")),
            )
        })?;
        if !t.is_normal() {
            return Err(diag::err(
                Diagnostic::new(
                    diag::EXPR_TABLE,
                    format!(
                        "[tables.{name}]: {} has {} columns; a table has two (argument, value)",
                        path.display(),
                        t.values[0].len() + 1
                    ),
                )
                .key(format!("tables.{name}.file")),
            ));
        }
        let ys = t.values.iter().map(|v| v[0]).collect();
        self.data = Some(Arc::new((t.s, ys, t.sha256)));
        Ok(())
    }

    fn validate(&self, name: &str) -> Result<()> {
        let key = format!("tables.{name}");
        let fail = |m: String| diag::err(Diagnostic::new(diag::EXPR_TABLE, m).key(key.clone()));
        if self.file.is_some() {
            if !self.x.is_empty() || !self.y.is_empty() {
                return Err(fail(format!("[tables.{name}] has both a file and x, y")));
            }
            return Ok(());
        }
        let (x, y) = (&self.x, &self.y);
        if x.len() < 2 || x.len() != y.len() {
            return Err(fail(format!(
                "[tables.{name}] needs x and y of the same length, at least 2 (has {} and {})",
                x.len(),
                y.len()
            )));
        }
        if x.iter().chain(y).any(|v| !v.is_finite()) {
            return Err(fail(format!(
                "[tables.{name}] holds a value that is not finite"
            )));
        }
        if let Some(i) = x.windows(2).position(|w| w[1] <= w[0]) {
            return Err(fail(format!(
                "[tables.{name}] x must increase strictly ({} after {})",
                x[i + 1],
                x[i]
            )));
        }
        Ok(())
    }
}

/// Names that the language reserves.
const VARIABLES: &[(&str, u32)] = &[
    ("x", k::VAR_X),
    ("y", k::VAR_Y),
    ("z", k::VAR_Z),
    ("t", k::VAR_T),
];
const BUILTIN_CONSTANTS: &[(&str, f64)] =
    &[("pi", std::f64::consts::PI), ("e", std::f64::consts::E)];
const FUNCTIONS1: &[(&str, u32)] = &[
    ("sin", fn1::SIN),
    ("cos", fn1::COS),
    ("tan", fn1::TAN),
    ("asin", fn1::ASIN),
    ("acos", fn1::ACOS),
    ("atan", fn1::ATAN),
    ("sinh", fn1::SINH),
    ("cosh", fn1::COSH),
    ("tanh", fn1::TANH),
    ("exp", fn1::EXP),
    ("log", fn1::LOG),
    ("log10", fn1::LOG10),
    ("sqrt", fn1::SQRT),
    ("abs", fn1::ABS),
    ("floor", fn1::FLOOR),
    ("ceil", fn1::CEIL),
    ("round", fn1::ROUND),
    ("sign", fn1::SIGN),
    ("erf", fn1::ERF),
    ("erfc", fn1::ERFC),
];
const FUNCTIONS2: &[(&str, u32)] = &[
    ("atan2", fn2::ATAN2),
    ("pow", fn2::POW),
    ("min", fn2::MIN),
    ("max", fn2::MAX),
    ("hypot", fn2::HYPOT),
    ("step", fn2::STEP),
];
/// Solution fields that later parts of the expressions bring.
const FIELDS: &[&str] = &[
    "u", "v", "w", "p", "T", "rho", "mu", "k", "omega", "epsilon", "nut", "Y",
];
/// Words of the language that are not names.
const KEYWORDS: &[&str] = &["and", "or", "not", "if", "clamp"];

/// Every name an expression can call or read, for the hints.
fn all_names(env: &Env) -> Vec<&str> {
    let mut v: Vec<&str> = VARIABLES.iter().map(|x| x.0).collect();
    v.extend(BUILTIN_CONSTANTS.iter().map(|x| x.0));
    v.extend(FUNCTIONS1.iter().map(|x| x.0));
    v.extend(FUNCTIONS2.iter().map(|x| x.0));
    v.extend(["if", "clamp"]);
    v.extend(env.constants.keys().map(String::as_str));
    v.extend(env.tables.keys().map(String::as_str));
    v
}

/// True when `name` may name a constant or a table of the case.
pub fn reserved(name: &str) -> bool {
    VARIABLES.iter().any(|x| x.0 == name)
        || BUILTIN_CONSTANTS.iter().any(|x| x.0 == name)
        || FUNCTIONS1.iter().any(|x| x.0 == name)
        || FUNCTIONS2.iter().any(|x| x.0 == name)
        || FIELDS.contains(&name)
        || KEYWORDS.contains(&name)
}

/// The constants and tables an expression can refer to.
#[derive(Debug, Clone, Default)]
pub struct Env {
    /// `[constants]`, evaluated.
    pub constants: BTreeMap<String, f64>,
    /// `[tables]`: abscissae and ordinates.
    pub tables: BTreeMap<String, (Vec<f64>, Vec<f64>)>,
}

impl Env {
    /// The environment of the case's `[constants]` and `[tables]` (each
    /// constant may use the ones defined before it in name order is not
    /// required: they are resolved in dependency order, a cycle is an
    /// error).
    pub fn new(
        constants: &BTreeMap<String, Scalar>,
        tables: &BTreeMap<String, TableSection>,
    ) -> Result<Self> {
        let mut env = Env::default();
        for (name, t) in tables {
            check_name(name, "tables")?;
            t.validate(name)?;
            let (x, y) = t.points();
            env.tables.insert(name.clone(), (x.to_vec(), y.to_vec()));
        }
        for name in constants.keys() {
            check_name(name, "constants")?;
            if env.tables.contains_key(name) {
                return Err(diag::err(
                    Diagnostic::new(
                        diag::EXPR_NAME,
                        format!("`{name}` is both a constant and a table"),
                    )
                    .key(format!("constants.{name}")),
                ));
            }
        }
        // resolve in dependency order: repeat over the open ones
        let mut open: Vec<&String> = constants.keys().collect();
        while !open.is_empty() {
            let before = open.len();
            let mut still = Vec::new();
            for name in open {
                let v = &constants[name];
                match v {
                    Scalar::Number(x) => {
                        env.constants.insert(name.clone(), *x);
                    }
                    Scalar::Expr(s) => {
                        let deps = names_in(s);
                        if deps
                            .iter()
                            .any(|d| constants.contains_key(d) && !env.constants.contains_key(d))
                        {
                            still.push(name);
                            continue;
                        }
                        let p = compile(s, &env, Vars::NONE, &format!("constants.{name}"))?;
                        env.constants.insert(name.clone(), p.eval_at([0.0; 3], 0.0));
                    }
                }
                let x = env.constants[name];
                if !x.is_finite() {
                    return Err(diag::err(
                        Diagnostic::new(
                            diag::EXPR_VALUE,
                            format!("[constants] {name} = {x} is not finite"),
                        )
                        .key(format!("constants.{name}")),
                    ));
                }
            }
            if still.len() == before {
                let names: Vec<&str> = still.iter().map(|s| s.as_str()).collect();
                return Err(diag::err(
                    Diagnostic::new(
                        diag::EXPR_NAME,
                        format!(
                            "[constants] refer to each other in a cycle: {}",
                            names.join(", ")
                        ),
                    )
                    .key("constants"),
                ));
            }
            open = still;
        }
        Ok(env)
    }
}

fn check_name(name: &str, section: &str) -> Result<()> {
    let ok = name
        .chars()
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_');
    if !ok || reserved(name) {
        return Err(diag::err(
            Diagnostic::new(
                diag::EXPR_NAME,
                if ok {
                    format!("[{section}] `{name}` is a reserved name of the expressions")
                } else {
                    format!("[{section}] `{name}` is not a name (letters, digits and _, not starting with a digit)")
                },
            )
            .key(format!("{section}.{name}")),
        ));
    }
    Ok(())
}

/// The identifiers of an expression (for the order of the constants).
fn names_in(s: &str) -> Vec<String> {
    match lex(s) {
        Ok(toks) => toks
            .into_iter()
            .filter_map(|t| match t.tok {
                Tok::Name(n) => Some(n),
                _ => None,
            })
            .collect(),
        Err(_) => Vec::new(),
    }
}

/// A compiled expression.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Program {
    /// The text it was compiled from.
    pub source: String,
    /// Instruction words of `foehncraft_kernels::expr`.
    pub code: Vec<u32>,
    /// Constants of the program.
    pub consts: Vec<f64>,
    /// Table data of the program.
    pub tables: Vec<f64>,
    /// The variables it reads.
    pub uses: Vars,
}

impl Program {
    /// A constant program.
    pub fn constant(v: f64) -> Self {
        Self {
            source: format!("{v:?}"),
            code: vec![word(op::CONST, 0)],
            consts: vec![v],
            tables: Vec::new(),
            uses: Vars::NONE,
        }
    }

    /// The value at point `x` and time `t`.
    pub fn eval_at(&self, x: [f64; 3], t: f64) -> f64 {
        k::eval(
            &self.code,
            &self.consts,
            &self.tables,
            &[x[0], x[1], x[2], t],
        )
    }

    /// The value when it depends on no variable.
    pub fn constant_value(&self) -> Option<f64> {
        (self.uses == Vars::NONE).then(|| self.eval_at([0.0; 3], 0.0))
    }
}

/// A [`Scalar`] compiled: a number stays a number (its value bit for bit).
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    /// A number.
    Number(f64),
    /// A program.
    Program(Arc<Program>),
}

impl Value {
    /// The value at point `x` and time `t`.
    #[inline]
    pub fn at(&self, x: [f64; 3], t: f64) -> f64 {
        match self {
            Value::Number(v) => *v,
            Value::Program(p) => p.eval_at(x, t),
        }
    }

    /// The variables it reads.
    pub fn uses(&self) -> Vars {
        match self {
            Value::Number(_) => Vars::NONE,
            Value::Program(p) => p.uses,
        }
    }

    /// The number, when it depends on no variable (folded expressions too).
    pub fn constant(&self) -> Option<f64> {
        match self {
            Value::Number(v) => Some(*v),
            Value::Program(p) => p.constant_value(),
        }
    }
}

/// Compiles a scalar for the key `key` that allows the variables `allowed`.
pub fn value(s: &Scalar, env: &Env, allowed: Vars, key: &str) -> Result<Value> {
    match s {
        Scalar::Number(v) => Ok(Value::Number(*v)),
        Scalar::Expr(src) => Ok(Value::Program(Arc::new(compile(src, env, allowed, key)?))),
    }
}

/// Compiles three scalars.
pub fn vector_value(v: &Vector, env: &Env, allowed: Vars, key: &str) -> Result<[Value; 3]> {
    Ok([
        value(&v[0], env, allowed, &format!("{key}[0]"))?,
        value(&v[1], env, allowed, &format!("{key}[1]"))?,
        value(&v[2], env, allowed, &format!("{key}[2]"))?,
    ])
}

// ---------------------------------------------------------------- lexer

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Num(f64),
    Name(String),
    Op(&'static str),
    LParen,
    RParen,
    Comma,
}

#[derive(Debug, Clone)]
struct Token {
    tok: Tok,
    pos: usize,
}

const OPS: &[&str] = &[
    "**", "<=", ">=", "==", "!=", "&&", "||", "+", "-", "*", "/", "%", "^", "<", ">", "!",
];

fn lex(s: &str) -> std::result::Result<Vec<Token>, (usize, String)> {
    let b = s.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
        let c = b[i] as char;
        if c.is_whitespace() {
            i += 1;
            continue;
        }
        let pos = i;
        if c.is_ascii_digit() || (c == '.' && b.get(i + 1).is_some_and(|d| d.is_ascii_digit())) {
            let mut j = i;
            while j < b.len() && (b[j].is_ascii_digit() || b[j] == b'.') {
                j += 1;
            }
            if j < b.len() && (b[j] == b'e' || b[j] == b'E') {
                let mut k = j + 1;
                if k < b.len() && (b[k] == b'+' || b[k] == b'-') {
                    k += 1;
                }
                if k < b.len() && b[k].is_ascii_digit() {
                    while k < b.len() && b[k].is_ascii_digit() {
                        k += 1;
                    }
                    j = k;
                }
            }
            let text = &s[i..j];
            let v: f64 = text
                .parse()
                .map_err(|_| (pos, format!("`{text}` is not a number")))?;
            out.push(Token {
                tok: Tok::Num(v),
                pos,
            });
            i = j;
            continue;
        }
        if c.is_ascii_alphabetic() || c == '_' {
            let mut j = i;
            while j < b.len() && (b[j].is_ascii_alphanumeric() || b[j] == b'_') {
                j += 1;
            }
            out.push(Token {
                tok: Tok::Name(s[i..j].to_string()),
                pos,
            });
            i = j;
            continue;
        }
        let t = match c {
            '(' => Some(Tok::LParen),
            ')' => Some(Tok::RParen),
            ',' => Some(Tok::Comma),
            _ => None,
        };
        if let Some(t) = t {
            out.push(Token { tok: t, pos });
            i += 1;
            continue;
        }
        if let Some(o) = OPS.iter().find(|o| s[i..].starts_with(**o)) {
            out.push(Token {
                tok: Tok::Op(o),
                pos,
            });
            i += o.len();
            continue;
        }
        return Err((pos, format!("unexpected character `{c}`")));
    }
    Ok(out)
}

// ---------------------------------------------------------------- parser

#[derive(Debug, Clone)]
enum Node {
    Num(f64),
    Var(u32),
    Un(u32, Box<Node>),
    Bin(u32, Box<Node>, Box<Node>),
    Fn1(u32, Box<Node>),
    Fn2(u32, Box<Node>, Box<Node>),
    Tri(u32, Box<Node>, Box<Node>, Box<Node>),
    Table(String, Box<Node>),
}

impl Node {
    fn uses(&self) -> Vars {
        match self {
            Node::Num(_) => Vars::NONE,
            Node::Var(v) => Vars(1 << v),
            Node::Un(_, a) | Node::Fn1(_, a) | Node::Table(_, a) => a.uses(),
            Node::Bin(_, a, b) | Node::Fn2(_, a, b) => a.uses().with(b.uses()),
            Node::Tri(_, a, b, c) => a.uses().with(b.uses()).with(c.uses()),
        }
    }
}

struct Parser<'a> {
    toks: Vec<Token>,
    i: usize,
    env: &'a Env,
    src: &'a str,
}

type PResult<T> = std::result::Result<T, (usize, String)>;

impl Parser<'_> {
    fn peek(&self) -> Option<&Tok> {
        self.toks.get(self.i).map(|t| &t.tok)
    }

    fn pos(&self) -> usize {
        self.toks.get(self.i).map_or(self.src.len(), |t| t.pos)
    }

    fn is_op(&self, o: &str) -> bool {
        matches!(self.peek(), Some(Tok::Op(x)) if *x == o)
    }

    fn is_word(&self, w: &str) -> bool {
        matches!(self.peek(), Some(Tok::Name(x)) if x == w)
    }

    fn expect(&mut self, t: Tok, what: &str) -> PResult<()> {
        if self.peek() == Some(&t) {
            self.i += 1;
            Ok(())
        } else {
            Err((self.pos(), format!("expected {what}")))
        }
    }

    fn or(&mut self) -> PResult<Node> {
        let mut a = self.and()?;
        while self.is_op("||") || self.is_word("or") {
            self.i += 1;
            let b = self.and()?;
            a = Node::Bin(op::OR, Box::new(a), Box::new(b));
        }
        Ok(a)
    }

    fn and(&mut self) -> PResult<Node> {
        let mut a = self.not()?;
        while self.is_op("&&") || self.is_word("and") {
            self.i += 1;
            let b = self.not()?;
            a = Node::Bin(op::AND, Box::new(a), Box::new(b));
        }
        Ok(a)
    }

    fn not(&mut self) -> PResult<Node> {
        if self.is_op("!") || self.is_word("not") {
            self.i += 1;
            let a = self.not()?;
            return Ok(Node::Un(op::NOT, Box::new(a)));
        }
        self.cmp()
    }

    fn cmp(&mut self) -> PResult<Node> {
        let a = self.add()?;
        let o = match self.peek() {
            Some(Tok::Op("<")) => op::LT,
            Some(Tok::Op("<=")) => op::LE,
            Some(Tok::Op(">")) => op::GT,
            Some(Tok::Op(">=")) => op::GE,
            Some(Tok::Op("==")) => op::EQ,
            Some(Tok::Op("!=")) => op::NE,
            _ => return Ok(a),
        };
        self.i += 1;
        let b = self.add()?;
        if matches!(
            self.peek(),
            Some(Tok::Op("<" | "<=" | ">" | ">=" | "==" | "!="))
        ) {
            return Err((
                self.pos(),
                "comparisons do not chain; write `a < x and x < b`".into(),
            ));
        }
        Ok(Node::Bin(o, Box::new(a), Box::new(b)))
    }

    fn add(&mut self) -> PResult<Node> {
        let mut a = self.mul()?;
        loop {
            let o = match self.peek() {
                Some(Tok::Op("+")) => op::ADD,
                Some(Tok::Op("-")) => op::SUB,
                _ => return Ok(a),
            };
            self.i += 1;
            let b = self.mul()?;
            a = Node::Bin(o, Box::new(a), Box::new(b));
        }
    }

    fn mul(&mut self) -> PResult<Node> {
        let mut a = self.unary()?;
        loop {
            let o = match self.peek() {
                Some(Tok::Op("*")) => op::MUL,
                Some(Tok::Op("/")) => op::DIV,
                Some(Tok::Op("%")) => op::MOD,
                _ => return Ok(a),
            };
            self.i += 1;
            let b = self.unary()?;
            a = Node::Bin(o, Box::new(a), Box::new(b));
        }
    }

    fn unary(&mut self) -> PResult<Node> {
        if self.is_op("-") {
            self.i += 1;
            let a = self.unary()?;
            return Ok(match a {
                Node::Num(v) => Node::Num(-v),
                a => Node::Un(op::NEG, Box::new(a)),
            });
        }
        if self.is_op("+") {
            self.i += 1;
            return self.unary();
        }
        self.power()
    }

    fn power(&mut self) -> PResult<Node> {
        let a = self.primary()?;
        if self.is_op("^") || self.is_op("**") {
            self.i += 1;
            let b = self.unary()?;
            return Ok(Node::Bin(op::POW, Box::new(a), Box::new(b)));
        }
        Ok(a)
    }

    fn args(&mut self, name: &str, n: usize) -> PResult<Vec<Node>> {
        let at = self.pos();
        self.expect(Tok::LParen, &format!("`(` after `{name}`"))?;
        let mut v = Vec::new();
        if self.peek() != Some(&Tok::RParen) {
            loop {
                v.push(self.or()?);
                if self.peek() == Some(&Tok::Comma) {
                    self.i += 1;
                    continue;
                }
                break;
            }
        }
        self.expect(Tok::RParen, "`)` or `,`")?;
        if v.len() != n {
            return Err((
                at,
                format!(
                    "`{name}` takes {n} argument{}, not {}",
                    if n == 1 { "" } else { "s" },
                    v.len()
                ),
            ));
        }
        Ok(v)
    }

    fn primary(&mut self) -> PResult<Node> {
        let pos = self.pos();
        let Some(t) = self.peek().cloned() else {
            return Err((pos, "the expression ends too early".into()));
        };
        match t {
            Tok::Num(v) => {
                self.i += 1;
                Ok(Node::Num(v))
            }
            Tok::LParen => {
                self.i += 1;
                let a = self.or()?;
                self.expect(Tok::RParen, "`)`")?;
                Ok(a)
            }
            Tok::Name(name) => {
                self.i += 1;
                let call = self.peek() == Some(&Tok::LParen);
                if call {
                    if let Some(&(_, f)) = FUNCTIONS1.iter().find(|x| x.0 == name) {
                        let mut a = self.args(&name, 1)?;
                        return Ok(Node::Fn1(f, Box::new(a.remove(0))));
                    }
                    if let Some(&(_, f)) = FUNCTIONS2.iter().find(|x| x.0 == name) {
                        let mut a = self.args(&name, 2)?;
                        let b = a.remove(1);
                        return Ok(Node::Fn2(f, Box::new(a.remove(0)), Box::new(b)));
                    }
                    if name == "if" || name == "clamp" {
                        let mut a = self.args(&name, 3)?;
                        let (c, b) = (a.remove(2), a.remove(1));
                        let o = if name == "if" { op::SELECT } else { op::CLAMP };
                        return Ok(Node::Tri(
                            o,
                            Box::new(a.remove(0)),
                            Box::new(b),
                            Box::new(c),
                        ));
                    }
                    if self.env.tables.contains_key(&name) {
                        let mut a = self.args(&name, 1)?;
                        return Ok(Node::Table(name, Box::new(a.remove(0))));
                    }
                } else {
                    if let Some(&(_, v)) = VARIABLES.iter().find(|x| x.0 == name) {
                        return Ok(Node::Var(v));
                    }
                    if let Some(&(_, v)) = BUILTIN_CONSTANTS.iter().find(|x| x.0 == name) {
                        return Ok(Node::Num(v));
                    }
                    if let Some(&v) = self.env.constants.get(&name) {
                        return Ok(Node::Num(v));
                    }
                }
                if FIELDS.contains(&name.as_str()) {
                    return Err((
                        pos,
                        format!(
                            "`{name}` is a solution field; fields in expressions come with the \
                             physics that solves them (this version takes x, y, z and t)"
                        ),
                    ));
                }
                let names = all_names(self.env);
                let hint = match crate::bc::did_you_mean(&name, names.iter().copied()) {
                    Some(c) => format!(" (did you mean `{c}`?)"),
                    None => String::new(),
                };
                let what = if call {
                    "function or table"
                } else if self.env.tables.contains_key(&name)
                    || FUNCTIONS1.iter().any(|x| x.0 == name)
                    || FUNCTIONS2.iter().any(|x| x.0 == name)
                {
                    return Err((pos, format!("`{name}` is a function: write `{name}(…)`")));
                } else {
                    "name"
                };
                Err((pos, format!("unknown {what} `{name}`{hint}")))
            }
            Tok::Op(o) => Err((pos, format!("unexpected `{o}`"))),
            Tok::RParen => Err((pos, "unexpected `)`".into())),
            Tok::Comma => Err((pos, "unexpected `,`".into())),
        }
    }
}

// ---------------------------------------------------------------- compiler

struct Emitter<'a> {
    p: Program,
    depth: usize,
    max: usize,
    tables: BTreeMap<String, u32>,
    env: &'a Env,
}

impl Emitter<'_> {
    fn push(&mut self, w: u32, delta: isize) {
        self.p.code.push(w);
        self.depth = (self.depth as isize + delta) as usize;
        self.max = self.max.max(self.depth);
    }

    fn constant(&mut self, v: f64) {
        let i = match self
            .p
            .consts
            .iter()
            .position(|c| c.to_bits() == v.to_bits())
        {
            Some(i) => i,
            None => {
                self.p.consts.push(v);
                self.p.consts.len() - 1
            }
        };
        self.push(word(op::CONST, i as u32), 1);
    }

    fn table(&mut self, name: &str) -> u32 {
        if let Some(&o) = self.tables.get(name) {
            return o;
        }
        let (x, y) = &self.env.tables[name];
        let off = self.p.tables.len() as u32;
        self.p.tables.push(x.len() as f64);
        self.p.tables.extend_from_slice(x);
        self.p.tables.extend_from_slice(y);
        self.tables.insert(name.to_string(), off);
        off
    }

    fn emit(&mut self, n: &Node) {
        // fold the parts without variables, with the machine itself
        if !matches!(n, Node::Num(_)) && n.uses() == Vars::NONE {
            let mut sub = Emitter {
                p: Program::default(),
                depth: 0,
                max: 0,
                tables: BTreeMap::new(),
                env: self.env,
            };
            sub.emit_raw(n);
            let v = sub.p.eval_at([0.0; 3], 0.0);
            self.constant(v);
            return;
        }
        self.emit_raw(n);
    }

    fn emit_raw(&mut self, n: &Node) {
        match n {
            Node::Num(v) => self.constant(*v),
            Node::Var(v) => self.push(word(op::VAR, *v), 1),
            Node::Un(o, a) => {
                self.emit(a);
                self.push(word(*o, 0), 0);
            }
            Node::Bin(o, a, b) => {
                self.emit(a);
                self.emit(b);
                self.push(word(*o, 0), -1);
            }
            Node::Fn1(f, a) => {
                self.emit(a);
                self.push(word(op::FN1, *f), 0);
            }
            Node::Fn2(f, a, b) => {
                self.emit(a);
                self.emit(b);
                self.push(word(op::FN2, *f), -1);
            }
            Node::Tri(o, a, b, c) => {
                self.emit(a);
                self.emit(b);
                self.emit(c);
                self.push(word(*o, 0), -2);
            }
            Node::Table(name, a) => {
                let off = self.table(name);
                self.emit(a);
                self.push(word(op::TABLE, off), 0);
            }
        }
    }
}

/// The syntax error message with a caret under the position.
fn located(src: &str, pos: usize, msg: &str) -> String {
    let col = src[..pos.min(src.len())].chars().count();
    format!("{msg}\n  {src}\n  {}^", " ".repeat(col))
}

/// Parses, checks and compiles `src` for the key `key` (for messages),
/// which allows the variables `allowed`.
pub fn compile(src: &str, env: &Env, allowed: Vars, key: &str) -> Result<Program> {
    let fail = |code: &'static str, m: String| diag::err(Diagnostic::new(code, m).key(key));
    let toks = lex(src)
        .map_err(|(p, m)| fail(diag::EXPR_SYNTAX, format!("{key}: {}", located(src, p, &m))))?;
    if toks.is_empty() {
        return Err(fail(
            diag::EXPR_SYNTAX,
            format!("{key}: an empty expression"),
        ));
    }
    let mut parser = Parser {
        toks,
        i: 0,
        env,
        src,
    };
    let node = parser.or().map_err(|(p, m)| {
        let code = if m.starts_with("unknown") || m.contains("solution field") {
            diag::EXPR_NAME
        } else {
            diag::EXPR_SYNTAX
        };
        fail(code, format!("{key}: {}", located(src, p, &m)))
    })?;
    if parser.i < parser.toks.len() {
        let p = parser.pos();
        return Err(fail(
            diag::EXPR_SYNTAX,
            format!(
                "{key}: {}",
                located(src, p, "unexpected text after the expression")
            ),
        ));
    }
    let uses = node.uses();
    if !allowed.contains(uses) {
        let extra = Vars(uses.0 & !allowed.0);
        let why = if extra.time() && !allowed.time() && allowed.space() {
            " (`t` needs a transient case, and properties do not change in time)"
        } else if extra.space() && !allowed.space() {
            " (this value is one number for the whole boundary or case)"
        } else {
            ""
        };
        return Err(fail(
            diag::EXPR_VARIABLE,
            format!(
                "{key}: `{src}` uses {}, but this key takes {}{why}",
                extra.names(),
                allowed.names()
            ),
        ));
    }
    let mut e = Emitter {
        p: Program::default(),
        depth: 0,
        max: 0,
        tables: BTreeMap::new(),
        env,
    };
    e.emit(&node);
    if e.max > k::STACK {
        return Err(fail(
            diag::EXPR_SYNTAX,
            format!(
                "{key}: the expression nests too deeply (stack {} > {})",
                e.max,
                k::STACK
            ),
        ));
    }
    let mut p = e.p;
    p.source = src.to_string();
    p.uses = uses;
    Ok(p)
}

/// Resolves the paths of the file tables against the case directory.
pub fn resolve_paths(tables: &mut BTreeMap<String, TableSection>, dir: &Path) {
    for t in tables.values_mut() {
        if let Some(f) = &t.file {
            t.file = Some(dir.join(f));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env() -> Env {
        let mut c = BTreeMap::new();
        c.insert("h".to_string(), Scalar::Number(0.41));
        c.insert("u_max".to_string(), Scalar::Expr("2 * half".into()));
        c.insert("half".to_string(), Scalar::Number(0.75));
        let mut t = BTreeMap::new();
        t.insert(
            "ramp".to_string(),
            TableSection {
                x: vec![0.0, 1.0],
                y: vec![0.0, 2.0],
                ..Default::default()
            },
        );
        Env::new(&c, &t).unwrap()
    }

    fn ev(s: &str, x: [f64; 3], t: f64) -> f64 {
        compile(s, &env(), Vars::ALL, "k").unwrap().eval_at(x, t)
    }

    #[test]
    fn values() {
        assert_eq!(ev("1 + 2 * 3", [0.0; 3], 0.0), 7.0);
        assert_eq!(ev("-2^2", [0.0; 3], 0.0), -4.0);
        assert_eq!(ev("2^3^2", [0.0; 3], 0.0), 512.0);
        assert_eq!(ev("2**-1", [0.0; 3], 0.0), 0.5);
        assert_eq!(ev("u_max", [0.0; 3], 0.0), 1.5);
        assert_eq!(ev("ramp(t)", [0.0; 3], 0.25), 0.5);
        assert_eq!(ev("if(x > 1 and y < 1, 3, 4)", [2.0, 0.0, 0.0], 0.0), 3.0);
        assert_eq!(ev("if(not (x > 1), 3, 4)", [2.0, 0.0, 0.0], 0.0), 4.0);
        assert_eq!(ev("clamp(x, 0, 1)", [5.0, 0.0, 0.0], 0.0), 1.0);
        assert_eq!(ev("max(x, y) + min(x, y)", [1.0, 2.0, 0.0], 0.0), 3.0);
        assert_eq!(ev("-7 % 3", [0.0; 3], 0.0), 2.0);
        assert_eq!(ev("1.5e2 + .5", [0.0; 3], 0.0), 150.5);
        let y = 0.1;
        let want = 4.0 * 1.5 * y * (0.41 - y) / (0.41 * 0.41);
        let got = ev("4 * u_max * y * (h - y) / h^2", [0.0, y, 0.0], 0.0);
        assert!((got - want).abs() < 1e-15, "{got} {want}");
    }

    #[test]
    fn folding_and_uses() {
        let p = compile("sin(pi / 2) * 3 + x", &env(), Vars::ALL, "k").unwrap();
        assert_eq!(p.uses, Vars::X);
        // sin(pi/2)*3 folded into one constant
        assert_eq!(p.code.len(), 3);
        assert_eq!(
            compile("2 * pi", &env(), Vars::NONE, "k")
                .unwrap()
                .constant_value(),
            Some(2.0 * std::f64::consts::PI)
        );
    }

    #[test]
    fn errors() {
        let e = |s: &str, a: Vars| {
            format!(
                "{:#}",
                compile(s, &env(), a, "boundary.in.velocity[0]").unwrap_err()
            )
        };
        assert!(e("1 +", Vars::ALL).contains("ends too early"));
        assert!(e("sinn(x)", Vars::ALL).contains("did you mean `sin`"));
        assert!(e("u_mx * 2", Vars::ALL).contains("did you mean `u_max`"));
        assert!(e("atan2(x)", Vars::ALL).contains("takes 2 arguments"));
        assert!(e("x * t", Vars::SPACE).contains("uses t"));
        assert!(e("p + 1", Vars::ALL).contains("solution field"));
        assert!(e("1 < x < 2", Vars::ALL).contains("do not chain"));
        assert!(e("2 $ 3", Vars::ALL).contains("unexpected character"));
        assert!(e("sin", Vars::ALL).contains("is a function"));
        let err = compile("x +* 2", &env(), Vars::ALL, "k").unwrap_err();
        assert_eq!(diag::of_error(&err).code, diag::EXPR_SYNTAX);
    }

    #[test]
    fn constants_in_cycle_and_reserved() {
        let mut c = BTreeMap::new();
        c.insert("a".to_string(), Scalar::Expr("b + 1".into()));
        c.insert("b".to_string(), Scalar::Expr("a + 1".into()));
        assert!(format!("{:#}", Env::new(&c, &BTreeMap::new()).unwrap_err()).contains("cycle"));
        let mut c = BTreeMap::new();
        c.insert("sin".to_string(), Scalar::Number(1.0));
        assert!(format!("{:#}", Env::new(&c, &BTreeMap::new()).unwrap_err()).contains("reserved"));
    }

    #[test]
    fn scalar_toml() {
        #[derive(Deserialize, Serialize)]
        struct S {
            a: Scalar,
            b: Scalar,
            c: Scalar,
        }
        let s: S = toml::from_str("a = 1\nb = 2.5\nc = \"x + 1\"").unwrap();
        assert_eq!(s.a, Scalar::Number(1.0));
        assert_eq!(s.b, Scalar::Number(2.5));
        assert_eq!(s.c, Scalar::Expr("x + 1".into()));
        let back = toml::to_string(&s).unwrap();
        assert!(
            back.contains("a = 1.0") && back.contains("c = \"x + 1\""),
            "{back}"
        );
    }
}
