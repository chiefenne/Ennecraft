//! The evaluator of compiled expressions (D53).
//!
//! The case file and the Python API give boundary values, sources and
//! properties as expressions; `foehncraft-flow::expr` parses and checks
//! them when the case is read and compiles each into a short program for a
//! stack machine. This module runs such a program for one point. It is a
//! kernel body like the others: the same source on the CPU and the GPU, no
//! allocation, a fixed stack, and only the in-house functions of
//! [`foehncraft_core::math`], so a value is bitwise the same on every
//! backend (D31, D93).
//!
//! A program is a slice of instruction words: the operation in the low 8
//! bits ([`op`]) and an argument in the high 24 bits (the index of a
//! constant, a variable or a table, or the function number of
//! [`op::FN1`]/[`op::FN2`]). Logical values are 1.0 and 0.0.

use foehncraft_core::math as m;

/// Largest stack depth of a program (checked by the compiler).
pub const STACK: usize = 32;

/// Instruction codes.
pub mod op {
    /// Push constant `arg`.
    pub const CONST: u32 = 0;
    /// Push variable `arg` ([`super::VAR_X`] …).
    pub const VAR: u32 = 1;
    /// −a.
    pub const NEG: u32 = 2;
    /// a + b.
    pub const ADD: u32 = 3;
    /// a − b.
    pub const SUB: u32 = 4;
    /// a · b.
    pub const MUL: u32 = 5;
    /// a / b.
    pub const DIV: u32 = 6;
    /// a mod b = a − b ⌊a/b⌋ (the sign of b).
    pub const MOD: u32 = 7;
    /// a^b.
    pub const POW: u32 = 8;
    /// a < b.
    pub const LT: u32 = 9;
    /// a ≤ b.
    pub const LE: u32 = 10;
    /// a > b.
    pub const GT: u32 = 11;
    /// a ≥ b.
    pub const GE: u32 = 12;
    /// a = b.
    pub const EQ: u32 = 13;
    /// a ≠ b.
    pub const NE: u32 = 14;
    /// a and b.
    pub const AND: u32 = 15;
    /// a or b.
    pub const OR: u32 = 16;
    /// not a.
    pub const NOT: u32 = 17;
    /// if(c, a, b): a where c ≠ 0, else b (both are evaluated).
    pub const SELECT: u32 = 18;
    /// One-argument function `arg` ([`super::fn1`]).
    pub const FN1: u32 = 19;
    /// Two-argument function `arg` ([`super::fn2`]).
    pub const FN2: u32 = 20;
    /// clamp(a, lo, hi).
    pub const CLAMP: u32 = 21;
    /// Piecewise-linear table at offset `arg` of the table data, clamped
    /// to its end values.
    pub const TABLE: u32 = 22;
}

/// One-argument functions of [`op::FN1`].
pub mod fn1 {
    /// sin.
    pub const SIN: u32 = 0;
    /// cos.
    pub const COS: u32 = 1;
    /// tan.
    pub const TAN: u32 = 2;
    /// asin.
    pub const ASIN: u32 = 3;
    /// acos.
    pub const ACOS: u32 = 4;
    /// atan.
    pub const ATAN: u32 = 5;
    /// sinh.
    pub const SINH: u32 = 6;
    /// cosh.
    pub const COSH: u32 = 7;
    /// tanh.
    pub const TANH: u32 = 8;
    /// exp.
    pub const EXP: u32 = 9;
    /// Natural logarithm.
    pub const LOG: u32 = 10;
    /// Logarithm to base 10.
    pub const LOG10: u32 = 11;
    /// Square root.
    pub const SQRT: u32 = 12;
    /// |a|.
    pub const ABS: u32 = 13;
    /// ⌊a⌋.
    pub const FLOOR: u32 = 14;
    /// ⌈a⌉.
    pub const CEIL: u32 = 15;
    /// Nearest integer, halves away from zero.
    pub const ROUND: u32 = 16;
    /// −1, 0, 1.
    pub const SIGN: u32 = 17;
    /// The error function.
    pub const ERF: u32 = 18;
    /// The complementary error function 1 − erf.
    pub const ERFC: u32 = 19;
}

/// Two-argument functions of [`op::FN2`].
pub mod fn2 {
    /// atan2(y, x).
    pub const ATAN2: u32 = 0;
    /// pow(a, b).
    pub const POW: u32 = 1;
    /// min(a, b).
    pub const MIN: u32 = 2;
    /// max(a, b).
    pub const MAX: u32 = 3;
    /// hypot(a, b) = √(a² + b²).
    pub const HYPOT: u32 = 4;
    /// step(a, b) = 1 where a ≥ b, else 0.
    pub const STEP: u32 = 5;
}

/// Index of the variable x (m) in the variables of a point.
pub const VAR_X: u32 = 0;
/// y (m).
pub const VAR_Y: u32 = 1;
/// z (m).
pub const VAR_Z: u32 = 2;
/// Simulated time t (s).
pub const VAR_T: u32 = 3;
/// Number of variables of a point.
pub const N_VARS: usize = 4;

/// One instruction word.
#[inline(always)]
pub const fn word(op: u32, arg: u32) -> u32 {
    op | (arg << 8)
}

/// a mod b with the sign of b.
#[inline(always)]
fn modulo(a: f64, b: f64) -> f64 {
    a - b * m::floor(a / b)
}

#[inline(always)]
fn truth(c: bool) -> f64 {
    if c { 1.0 } else { 0.0 }
}

fn call1(f: u32, a: f64) -> f64 {
    match f {
        fn1::SIN => m::sin(a),
        fn1::COS => m::cos(a),
        fn1::TAN => m::tan(a),
        fn1::ASIN => m::asin(a),
        fn1::ACOS => m::acos(a),
        fn1::ATAN => m::atan(a),
        fn1::SINH => m::sinh(a),
        fn1::COSH => m::cosh(a),
        fn1::TANH => m::tanh(a),
        fn1::EXP => m::exp(a),
        fn1::LOG => m::ln(a),
        fn1::LOG10 => m::log10(a),
        fn1::SQRT => m::sqrt(a),
        fn1::ABS => m::abs(a),
        fn1::FLOOR => m::floor(a),
        fn1::CEIL => m::ceil(a),
        fn1::ROUND => m::round(a),
        fn1::SIGN => m::sign(a),
        fn1::ERF => m::erf(a),
        fn1::ERFC => m::erfc(a),
        _ => f64::NAN,
    }
}

fn call2(f: u32, a: f64, b: f64) -> f64 {
    match f {
        fn2::ATAN2 => m::atan2(a, b),
        fn2::POW => m::pow(a, b),
        // NaN-propagating, unlike f64::min
        fn2::MIN => {
            if a.is_nan() || b.is_nan() {
                f64::NAN
            } else if b < a {
                b
            } else {
                a
            }
        }
        fn2::MAX => {
            if a.is_nan() || b.is_nan() {
                f64::NAN
            } else if b > a {
                b
            } else {
                a
            }
        }
        fn2::HYPOT => m::sqrt(a * a + b * b),
        fn2::STEP => truth(a >= b),
        _ => f64::NAN,
    }
}

/// Piecewise-linear interpolation in the table at `tables[off..]`:
/// `n`, then `n` increasing abscissae, then `n` ordinates; clamped to the
/// end values outside.
fn table(tables: &[f64], off: usize, x: f64) -> f64 {
    let n = tables[off] as usize;
    let xs = &tables[off + 1..off + 1 + n];
    let ys = &tables[off + 1 + n..off + 1 + 2 * n];
    if x.is_nan() {
        return f64::NAN;
    }
    if x <= xs[0] {
        return ys[0];
    }
    if x >= xs[n - 1] {
        return ys[n - 1];
    }
    // the interval xs[lo] ≤ x < xs[lo + 1]
    let (mut lo, mut hi) = (0usize, n - 1);
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if xs[mid] <= x {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let w = (x - xs[lo]) / (xs[hi] - xs[lo]);
    ys[lo] + w * (ys[hi] - ys[lo])
}

/// Runs `code` for one point with the variables `vars` (x, y, z, t).
/// `consts` and `tables` are the constants and table data of the program.
/// The compiler guarantees a valid program (stack depth, indices); a
/// malformed one gives NaN.
pub fn eval(code: &[u32], consts: &[f64], tables: &[f64], vars: &[f64]) -> f64 {
    let mut st = [0.0f64; STACK];
    let mut sp = 0usize;
    for &w in code {
        let (o, arg) = (w & 0xFF, (w >> 8) as usize);
        match o {
            op::CONST | op::VAR => {
                if sp >= STACK {
                    return f64::NAN;
                }
                st[sp] = if o == op::CONST {
                    consts[arg]
                } else {
                    vars[arg]
                };
                sp += 1;
            }
            op::NEG | op::NOT | op::FN1 | op::TABLE => {
                if sp < 1 {
                    return f64::NAN;
                }
                let a = st[sp - 1];
                st[sp - 1] = match o {
                    op::NEG => -a,
                    op::NOT => truth(a == 0.0),
                    op::FN1 => call1(arg as u32, a),
                    _ => table(tables, arg, a),
                };
            }
            op::SELECT | op::CLAMP => {
                if sp < 3 {
                    return f64::NAN;
                }
                let (a, b, c) = (st[sp - 3], st[sp - 2], st[sp - 1]);
                sp -= 2;
                st[sp - 1] = if o == op::SELECT {
                    if a != 0.0 { b } else { c }
                } else if a.is_nan() {
                    a
                } else if a < b {
                    b
                } else if a > c {
                    c
                } else {
                    a
                };
            }
            _ => {
                if sp < 2 {
                    return f64::NAN;
                }
                let (a, b) = (st[sp - 2], st[sp - 1]);
                sp -= 1;
                st[sp - 1] = match o {
                    op::ADD => a + b,
                    op::SUB => a - b,
                    op::MUL => a * b,
                    op::DIV => a / b,
                    op::MOD => modulo(a, b),
                    op::POW => m::pow(a, b),
                    op::LT => truth(a < b),
                    op::LE => truth(a <= b),
                    op::GT => truth(a > b),
                    op::GE => truth(a >= b),
                    op::EQ => truth(a == b),
                    op::NE => truth(a != b),
                    op::AND => truth(a != 0.0 && b != 0.0),
                    op::OR => truth(a != 0.0 || b != 0.0),
                    op::FN2 => call2(arg as u32, a, b),
                    _ => f64::NAN,
                };
            }
        }
    }
    if sp == 1 { st[0] } else { f64::NAN }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_programs() {
        // 2 x + sin(t)
        let code = [
            word(op::CONST, 0),
            word(op::VAR, VAR_X),
            word(op::MUL, 0),
            word(op::VAR, VAR_T),
            word(op::FN1, fn1::SIN),
            word(op::ADD, 0),
        ];
        let v = eval(&code, &[2.0], &[], &[1.5, 0.0, 0.0, 0.0]);
        assert_eq!(v, 3.0);
        // if(x > 1, 10, 20)
        let code = [
            word(op::VAR, VAR_X),
            word(op::CONST, 0),
            word(op::GT, 0),
            word(op::CONST, 1),
            word(op::CONST, 2),
            word(op::SELECT, 0),
        ];
        assert_eq!(
            eval(&code, &[1.0, 10.0, 20.0], &[], &[2.0, 0.0, 0.0, 0.0]),
            10.0
        );
        assert_eq!(
            eval(&code, &[1.0, 10.0, 20.0], &[], &[0.0, 0.0, 0.0, 0.0]),
            20.0
        );
        // a table
        let t = [3.0, 0.0, 1.0, 3.0, 0.0, 10.0, 30.0];
        let code = [word(op::VAR, VAR_X), word(op::TABLE, 0)];
        for (x, want) in [(-1.0, 0.0), (0.5, 5.0), (2.0, 20.0), (9.0, 30.0)] {
            assert_eq!(eval(&code, &[], &t, &[x, 0.0, 0.0, 0.0]), want);
        }
        // malformed: stack underflow
        assert!(eval(&[word(op::ADD, 0)], &[], &[], &[0.0; 4]).is_nan());
    }
}
