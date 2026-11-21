//! Elementary functions written in Foehncraft (requirement D93 (6)).
//!
//! The expressions of the case file (D53) and the property forms (D95) call
//! these instead of the platform's maths library, so a value is the same on
//! every CPU and GPU that runs the same binary: each function is a fixed
//! sequence of IEEE additions, multiplications and divisions (Rust never
//! contracts them into fused operations), and the crate is `no_std`, so the
//! same source compiles for the device.
//!
//! The methods are the classical ones of the literature: argument reduction
//! after Cody & Waite (*Software Manual for the Elementary Functions*,
//! 1980) with the constant split into a short leading part and a tail, and
//! truncated Taylor series on the reduced interval, evaluated by Horner's
//! rule (Muller, *Elementary Functions*, 3rd ed., 2016, ch. 11). The errors
//! are a few units in the last place (ulp) over the ranges the solver uses
//! (the tests compare with the platform library); [`sqrt`] is correctly
//! rounded. The trigonometric reduction is accurate for |x| up to about
//! 10⁵; beyond that the result is still deterministic, but loses digits.

/// ln 2, leading part with 32 significant bits (k·LN2_HI is exact for
/// |k| < 2²⁰).
const LN2_HI: f64 = f64::from_bits(0x3FE6_2E42_FEE0_0000);
/// ln 2 − LN2_HI.
const LN2_LO: f64 = f64::from_bits(0x3DEA_39EF_3579_3C76);
/// 1 / ln 2.
const INV_LN2: f64 = core::f64::consts::LOG2_E;
/// π/2 in three parts with 33 significant bits each (Cody & Waite).
const PIO2_1: f64 = f64::from_bits(0x3FF9_21FB_5440_0000);
const PIO2_2: f64 = f64::from_bits(0x3DD0_B461_1A60_0000);
const PIO2_3: f64 = f64::from_bits(0x3BA3_198A_2E03_7073);
const FRAC_2_PI: f64 = core::f64::consts::FRAC_2_PI;
const FRAC_PI_2: f64 = core::f64::consts::FRAC_PI_2;
const FRAC_PI_6: f64 = core::f64::consts::FRAC_PI_6;
const PI: f64 = core::f64::consts::PI;
const SQRT_3: f64 = 1.732_050_807_568_877_2;
/// tan(π/12) = 2 − √3.
const TAN_PI_12: f64 = 0.267_949_192_431_122_7;

/// 2^k as a float for −1022 ≤ k ≤ 1023.
#[inline(always)]
fn pow2i(k: i64) -> f64 {
    f64::from_bits(((k + 1023) as u64) << 52)
}

/// x · 2^k, also when the result is subnormal or overflows.
fn scale2(x: f64, k: i64) -> f64 {
    if k > 1023 {
        let k2 = (k - 1023).min(1023);
        x * pow2i(1023) * pow2i(k2)
    } else if k < -1022 {
        let k2 = (k + 1022).max(-1022);
        x * pow2i(-1022) * pow2i(k2)
    } else {
        x * pow2i(k)
    }
}

/// |x|.
#[inline(always)]
pub fn abs(x: f64) -> f64 {
    f64::from_bits(x.to_bits() & !(1u64 << 63))
}

/// x rounded toward zero.
pub fn trunc(x: f64) -> f64 {
    if x.is_nan() || abs(x) >= 4_503_599_627_370_496.0 || x == 0.0 {
        return x;
    }
    let t = (x as i64) as f64;
    if t == 0.0 && x < 0.0 { -0.0 } else { t }
}

/// The largest integer ≤ x.
pub fn floor(x: f64) -> f64 {
    let t = trunc(x);
    if t > x { t - 1.0 } else { t }
}

/// The smallest integer ≥ x.
pub fn ceil(x: f64) -> f64 {
    let t = trunc(x);
    if t < x { t + 1.0 } else { t }
}

/// x rounded to the nearest integer, halves away from zero.
pub fn round(x: f64) -> f64 {
    let t = trunc(x);
    let d = x - t;
    if d >= 0.5 {
        t + 1.0
    } else if d <= -0.5 {
        t - 1.0
    } else {
        t
    }
}

/// −1, 0 or 1 by the sign of x (NaN stays NaN).
pub fn sign(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        x
    }
}

/// Square root, correctly rounded (integer square root of the significand).
pub fn sqrt(x: f64) -> f64 {
    if x.is_nan() || x < 0.0 {
        return f64::NAN;
    }
    if x == 0.0 || x.is_infinite() {
        return x;
    }
    let bits = x.to_bits();
    let mut e = ((bits >> 52) & 0x7FF) as i64;
    let mut m = bits & ((1u64 << 52) - 1);
    if e == 0 {
        // subnormal: normalise
        let shift = m.leading_zeros() as i64 - 11;
        m <<= shift;
        e = 1 - shift;
        m &= (1u64 << 52) - 1;
    }
    // x = M · 2^E with an integer significand M of 53 bits
    let mut mm = (m | (1u64 << 52)) as u128;
    let mut ee = e - 1023 - 52;
    if ee & 1 != 0 {
        mm <<= 1;
        ee -= 1;
    }
    // q = ⌊√(M·2⁵²)⌋ has 53 bits; round to nearest from the remainder
    let n = mm << 52;
    let mut q = isqrt(n);
    let r = n - q * q;
    if r > q {
        q += 1;
    }
    scale2(q as f64, ee / 2 - 26)
}

/// ⌊√n⌋ by Newton's iteration on integers.
fn isqrt(n: u128) -> u128 {
    if n < 2 {
        return n;
    }
    let bits = 128 - n.leading_zeros();
    let mut x = 1u128 << bits.div_ceil(2);
    loop {
        let y = (x + n / x) >> 1;
        if y >= x {
            return x;
        }
        x = y;
    }
}

/// e^x.
pub fn exp(x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    if x > 709.782_712_893_384 {
        return f64::INFINITY;
    }
    if x < -745.133_219_101_941_2 {
        return 0.0;
    }
    // x = k ln 2 + r, |r| ≤ ln 2 / 2
    let k = floor(x * INV_LN2 + 0.5);
    let r = (x - k * LN2_HI) - k * LN2_LO;
    // e^r by its Taylor series to r¹⁴/14! (truncation < 1e-18 relative)
    let mut p = 1.0 / 87_178_291_200.0;
    for d in [
        6_227_020_800.0,
        479_001_600.0,
        39_916_800.0,
        3_628_800.0,
        362_880.0,
        40_320.0,
        5_040.0,
        720.0,
        120.0,
        24.0,
        6.0,
        2.0,
        1.0,
        1.0,
    ] {
        p = p * r + 1.0 / d;
    }
    scale2(p, k as i64)
}

/// Natural logarithm.
pub fn ln(x: f64) -> f64 {
    if x.is_nan() || x < 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return f64::NEG_INFINITY;
    }
    if x.is_infinite() {
        return x;
    }
    let mut x = x;
    let mut e = 0i64;
    if x < f64::MIN_POSITIVE {
        x *= pow2i(54);
        e -= 54;
    }
    let bits = x.to_bits();
    e += ((bits >> 52) & 0x7FF) as i64 - 1023;
    let mut m = f64::from_bits((bits & ((1u64 << 52) - 1)) | (1023u64 << 52));
    if m > core::f64::consts::SQRT_2 {
        m *= 0.5;
        e += 1;
    }
    // ln(1 + f) = 2 atanh(s), s = f / (2 + f), |s| ≤ 0.172
    let f = m - 1.0;
    let s = f / (2.0 + f);
    let z = s * s;
    let mut r = 2.0 / 23.0;
    for k in [21.0, 19.0, 17.0, 15.0, 13.0, 11.0, 9.0, 7.0, 5.0, 3.0] {
        r = r * z + 2.0 / k;
    }
    let r = r * z;
    // 2s + sR = f − s(f − R), as 2s = f − s f
    let lnm = f - s * (f - r);
    let k = e as f64;
    (k * LN2_LO + lnm) + k * LN2_HI
}

/// Logarithm to base 10.
pub fn log10(x: f64) -> f64 {
    ln(x) * core::f64::consts::LOG10_E
}

/// x^y. Integer exponents up to 1024 in magnitude are computed by repeated
/// squaring (exact powers stay exact), half-integer ones as xⁿ√x; otherwise
/// e^(y ln x), whose relative error grows with |y ln x| (about 1e-15 for
/// the exponents of property laws); NaN for a negative base.
pub fn pow(x: f64, y: f64) -> f64 {
    if y == 0.0 {
        return 1.0;
    }
    if x.is_nan() || y.is_nan() {
        return f64::NAN;
    }
    if trunc(y) == y && abs(y) <= 1024.0 {
        let mut n = abs(y) as u32;
        let mut b = x;
        let mut r = 1.0;
        while n > 0 {
            if n & 1 == 1 {
                r *= b;
            }
            b *= b;
            n >>= 1;
        }
        return if y < 0.0 { 1.0 / r } else { r };
    }
    if x < 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return if y > 0.0 { 0.0 } else { f64::INFINITY };
    }
    // half-integer exponents (T^1.5 of the Sutherland law): x^n √x
    let h = y - 0.5;
    if trunc(h) == h && abs(h) <= 1024.0 {
        return pow(x, h) * sqrt(x);
    }
    exp(y * ln(x))
}

/// sin r and cos r for |r| ≤ π/4 (Taylor series to r¹⁷ and r¹⁸).
fn sin_cos_reduced(r: f64) -> (f64, f64) {
    let z = r * r;
    // sin r = r (1 − z/3! + z²/5! − …)
    let mut s = -1.0 / 6_402_373_705_728_000.0; // −1/19!
    for d in [
        355_687_428_096_000.0,
        -1_307_674_368_000.0,
        6_227_020_800.0,
        -39_916_800.0,
        362_880.0,
        -5_040.0,
        120.0,
        -6.0,
    ] {
        s = s * z + 1.0 / d;
    }
    let sin = r + r * z * s;
    // cos r = 1 − z/2! + z²/4! − …
    let mut c = 1.0 / 2_432_902_008_176_640_000.0; // 1/20!
    for d in [
        -6_402_373_705_728_000.0,
        20_922_789_888_000.0,
        -87_178_291_200.0,
        479_001_600.0,
        -3_628_800.0,
        40_320.0,
        -720.0,
        24.0,
    ] {
        c = c * z + 1.0 / d;
    }
    let cos = (1.0 - 0.5 * z) + z * z * c;
    (sin, cos)
}

/// The quadrant k mod 4 and the reduced argument r = x − kπ/2.
fn reduce_pio2(x: f64) -> (i64, f64) {
    let k = round(x * FRAC_2_PI);
    let r = ((x - k * PIO2_1) - k * PIO2_2) - k * PIO2_3;
    ((k as i64) & 3, r)
}

/// sin x.
pub fn sin(x: f64) -> f64 {
    if !x.is_finite() {
        return f64::NAN;
    }
    if abs(x) < 7.450_580_596_923_828e-9 {
        return x;
    }
    let (q, r) = reduce_pio2(x);
    let (s, c) = sin_cos_reduced(r);
    match q {
        0 => s,
        1 => c,
        2 => -s,
        _ => -c,
    }
}

/// cos x.
pub fn cos(x: f64) -> f64 {
    if !x.is_finite() {
        return f64::NAN;
    }
    let (q, r) = reduce_pio2(x);
    let (s, c) = sin_cos_reduced(r);
    match q {
        0 => c,
        1 => -s,
        2 => -c,
        _ => s,
    }
}

/// tan x.
pub fn tan(x: f64) -> f64 {
    if !x.is_finite() {
        return f64::NAN;
    }
    let (q, r) = reduce_pio2(x);
    let (s, c) = sin_cos_reduced(r);
    if q & 1 == 0 { s / c } else { -c / s }
}

/// atan t for 0 ≤ t ≤ tan(π/12) (Taylor series to t³³).
fn atan_small(t: f64) -> f64 {
    // atan t = t + t z P(z), P(z) = Σ_{n=1}^{16} (−1)ⁿ zⁿ⁻¹ / (2n + 1)
    let z = t * t;
    let mut p = 1.0 / 33.0;
    let mut n = 15;
    while n >= 1 {
        let c = 1.0 / (2 * n + 1) as f64;
        p = p * z + if n % 2 == 0 { c } else { -c };
        n -= 1;
    }
    t + t * z * p
}

/// atan x.
pub fn atan(x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    let a = abs(x);
    let r = if a > 1.0 {
        FRAC_PI_2 - atan_unit(1.0 / a)
    } else {
        atan_unit(a)
    };
    if x < 0.0 { -r } else { r }
}

/// atan t for 0 ≤ t ≤ 1.
fn atan_unit(t: f64) -> f64 {
    if t > TAN_PI_12 {
        // atan t = π/6 + atan((t√3 − 1)/(t + √3))
        let u = (t * SQRT_3 - 1.0) / (t + SQRT_3);
        if u < 0.0 {
            FRAC_PI_6 - atan_small(-u)
        } else {
            FRAC_PI_6 + atan_small(u)
        }
    } else {
        atan_small(t)
    }
}

/// atan2(y, x): the angle of (x, y) in (−π, π].
pub fn atan2(y: f64, x: f64) -> f64 {
    if x.is_nan() || y.is_nan() {
        return f64::NAN;
    }
    if x == 0.0 {
        return if y > 0.0 {
            FRAC_PI_2
        } else if y < 0.0 {
            -FRAC_PI_2
        } else if x.is_sign_negative() {
            if y.is_sign_negative() { -PI } else { PI }
        } else {
            y
        };
    }
    let a = atan(y / x);
    if x > 0.0 {
        a
    } else if y < 0.0 || (y == 0.0 && y.is_sign_negative()) {
        a - PI
    } else {
        a + PI
    }
}

/// asin x (NaN outside [−1, 1]).
pub fn asin(x: f64) -> f64 {
    if !(-1.0..=1.0).contains(&x) {
        return f64::NAN;
    }
    atan2(x, sqrt((1.0 - x) * (1.0 + x)))
}

/// acos x (NaN outside [−1, 1]).
pub fn acos(x: f64) -> f64 {
    if !(-1.0..=1.0).contains(&x) {
        return f64::NAN;
    }
    atan2(sqrt((1.0 - x) * (1.0 + x)), x)
}

/// cos(⅓ arccos x) for x in [−1, 1] (NaN outside): the trigonometric
/// root of the realizable k-ε eddy viscosity (Shih et al. 1995).
pub fn cos_acos3(x: f64) -> f64 {
    cos(acos(x) / 3.0)
}

/// d/dx cos(⅓ arccos x) = sin(θ/3)/(3 sin θ), θ = arccos x: 1/9 at x = 1
/// (the limit), finite inside (−1, 1]; at x = −1 the bound 1/(3·10⁻³⁰⁰)
/// of the vanishing denominator.
pub fn cos_acos3_derivative(x: f64) -> f64 {
    let t = acos(x);
    let s = sin(t);
    if s < 1e-8 && x > 0.0 {
        // sin(θ/3)/(3 sin θ) = 1/9 (1 + 4θ²/27 + …)
        return (1.0 + 4.0 * t * t / 27.0) / 9.0;
    }
    sin(t / 3.0) / (3.0 * s.max(1e-300))
}

/// sinh x for |x| < 0.5 (Taylor series to x¹⁷).
fn sinh_small(x: f64) -> f64 {
    let z = x * x;
    let mut p = 1.0 / 355_687_428_096_000.0; // 1/17!
    for d in [
        1_307_674_368_000.0,
        6_227_020_800.0,
        39_916_800.0,
        362_880.0,
        5_040.0,
        120.0,
        6.0,
    ] {
        p = p * z + 1.0 / d;
    }
    x + x * z * p
}

/// sinh x.
pub fn sinh(x: f64) -> f64 {
    if abs(x) < 0.5 {
        return sinh_small(x);
    }
    let e = exp(abs(x));
    let r = 0.5 * e - 0.5 / e;
    if x < 0.0 { -r } else { r }
}

/// cosh x.
pub fn cosh(x: f64) -> f64 {
    let e = exp(abs(x));
    0.5 * e + 0.5 / e
}

/// tanh x.
pub fn tanh(x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    let a = abs(x);
    let r = if a > 22.0 {
        1.0
    } else if a < 0.5 {
        let s = sinh_small(a);
        s / sqrt(1.0 + s * s)
    } else {
        let e = exp(2.0 * a);
        1.0 - 2.0 / (e + 1.0)
    };
    if x < 0.0 { -r } else { r }
}

// ---------------------------------------------------------------------------
// the error function (exact solutions of transient conduction, stage 7d6)

/// 1/√π.
const FRAC_1_SQRT_PI: f64 = 0.564_189_583_547_756_3;
/// Below this |x| the series of erf, above it the continued fraction of
/// erfc.
const ERF_SPLIT: f64 = 0.75;

/// e^(−x²) for x ≥ 0 with x² taken in two parts: x = x_h + x_l with x_h
/// on a grid of 2⁻¹², so x_h² is exact and e^(−x²) = e^(−x_h²) e^(−x_l (x
/// + x_h)) does not carry the rounding of x² (which grows with x²).
fn exp_minus_square(x: f64) -> f64 {
    let xh = trunc(x * 4096.0) / 4096.0;
    exp(-xh * xh) * exp(-(x - xh) * (x + xh))
}

/// erf x for 0 ≤ x < [`ERF_SPLIT`]: the series of positive terms
/// erf x = (2/√π) e^(−x²) Σ 2ⁿ x^(2n+1)/(1·3···(2n+1)) (Abramowitz &
/// Stegun 1964, 7.1.6); 30 terms are far beyond the last digit there.
fn erf_series(x: f64) -> f64 {
    let x2 = x * x;
    let (mut term, mut sum) = (x, x);
    for n in 1..30 {
        term *= 2.0 * x2 / (2 * n + 1) as f64;
        sum += term;
    }
    2.0 * FRAC_1_SQRT_PI * exp(-x2) * sum
}

/// erfc x for x ≥ [`ERF_SPLIT`]: Laplace's continued fraction
/// erfc x = e^(−x²)/√π · 1/(x + (1/2)/(x + 1/(x + (3/2)/(x + …))))
/// (Abramowitz & Stegun 1964, 7.1.14), evaluated from the tail with
/// 200/x² + 10 levels (at most 400), which converges to the last digit.
fn erfc_fraction(x: f64) -> f64 {
    let n = (200.0 / (x * x)) as usize + 10;
    let mut f = x;
    for k in (1..=n.min(400)).rev() {
        f = x + 0.5 * k as f64 / f;
    }
    exp_minus_square(x) * FRAC_1_SQRT_PI / f
}

/// The error function erf x = (2/√π) ∫₀ˣ e^(−t²) dt.
pub fn erf(x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    let a = abs(x);
    let r = if a < ERF_SPLIT {
        erf_series(a)
    } else if a < 6.0 {
        1.0 - erfc_fraction(a)
    } else {
        1.0
    };
    if x < 0.0 { -r } else { r }
}

/// The complementary error function erfc x = 1 − erf x, accurate also
/// where it is small (to about 27, where it leaves the doubles).
pub fn erfc(x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    let a = abs(x);
    let r = if a < ERF_SPLIT {
        1.0 - erf_series(a)
    } else if a < 27.3 {
        erfc_fraction(a)
    } else {
        0.0
    };
    if x < 0.0 { 2.0 - r } else { r }
}

// ---------------------------------------------------------------------------
// switches of the turbulence models (stage 6 specification, section 2.6)

/// Default relative width w of the smooth switches [`smin`] and [`smax`].
pub const SMOOTH_WIDTH: f64 = 1e-3;

/// min(a, b), the published hard form of a model's switch (a select, no
/// branch on the GPU). NaN in either argument gives NaN.
#[inline(always)]
pub fn pmin(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else if b < a {
        b
    } else {
        a
    }
}

/// max(a, b), the published hard form (see [`pmin`]).
#[inline(always)]
pub fn pmax(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else if b > a {
        b
    } else {
        a
    }
}

/// |a|, the published hard form.
#[inline(always)]
pub fn pabs(a: f64) -> f64 {
    abs(a)
}

/// The smooth minimum of the linearisation and the adjoint (D59 (3)):
///
/// smin(a, b; w) = ½ (a + b − √((a − b)² + (w m)²)),  m = |a| |b| / (|a| + |b|),
///
/// C¹ in a and b away from a = 0 and b = 0, below min(a, b) by at most
/// w m / 2 ≤ w min(|a|, |b|) / 2, and equal to it for w = 0. `w` is the
/// relative width ([`SMOOTH_WIDTH`] by default). The scale m is half the
/// harmonic mean, so the width follows the smaller operand: a floor such
/// as max(x, 10⁻¹⁰) stays within w·10⁻¹⁰ of the floor for any large x
/// (with the arithmetic mean ½ (|a| + |b|) it grew like w² |x|/16, which
/// stalled the SST flat plate, stage 6 specification 2.6).
#[inline(always)]
pub fn smin(a: f64, b: f64, w: f64) -> f64 {
    let s = abs(a) + abs(b);
    let m = if s > 0.0 { abs(a) * abs(b) / s } else { 0.0 };
    let d = abs(a - b);
    let e = w * m;
    // ½(a + b − √(d² + e²)) = min(a, b) − ½ e²/(√(d² + e²) + d), without
    // the cancellation of the first form when |a| ≫ |b|
    let r = sqrt(d * d + e * e) + d;
    if r > 0.0 {
        a.min(b) - 0.5 * e * e / r
    } else {
        a.min(b)
    }
}

/// The smooth maximum, smax(a, b; w) = −smin(−a, −b; w): above
/// max(a, b) by at most w m / 2.
#[inline(always)]
pub fn smax(a: f64, b: f64, w: f64) -> f64 {
    -smin(-a, -b, w)
}

/// The smooth |a| of the linearisation, √(a² + (w s)²) with the scale s =
/// `scale` of the quantity (|a| has no scale of its own at a = 0): C¹,
/// ≥ |a| and above it by at most w s.
#[inline(always)]
pub fn sabs(a: f64, w: f64, scale: f64) -> f64 {
    let e = w * scale;
    sqrt(a * a + e * e)
}

/// A switch of a model: the published hard form, or the smooth one of
/// width `w` (`turbulence.smooth_limits`, section 2.6 of the stage 6
/// specification). The kernels take it by value, so the choice is one
/// select per evaluation and no data-dependent branch.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Switches {
    /// Relative width of the smooth form; 0 gives the hard form.
    pub w: f64,
}

impl Switches {
    /// The published hard form.
    pub const HARD: Switches = Switches { w: 0.0 };
    /// The smooth form with the default width.
    pub const SMOOTH: Switches = Switches { w: SMOOTH_WIDTH };

    /// min(a, b) in the chosen form.
    #[inline(always)]
    pub fn min(self, a: f64, b: f64) -> f64 {
        if self.w > 0.0 {
            smin(a, b, self.w)
        } else {
            pmin(a, b)
        }
    }

    /// max(a, b) in the chosen form.
    #[inline(always)]
    pub fn max(self, a: f64, b: f64) -> f64 {
        if self.w > 0.0 {
            smax(a, b, self.w)
        } else {
            pmax(a, b)
        }
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;

    /// Distance in units of the last place.
    fn ulps(a: f64, b: f64) -> u64 {
        if a == b {
            return 0;
        }
        let (a, b) = (a.to_bits() as i64, b.to_bits() as i64);
        (a - b).unsigned_abs()
    }

    fn sweep(lo: f64, hi: f64, n: usize) -> impl Iterator<Item = f64> {
        (0..=n).map(move |i| lo + (hi - lo) * i as f64 / n as f64 + 1e-7 * (i % 7) as f64)
    }

    fn check(
        name: &str,
        f: fn(f64) -> f64,
        g: fn(f64) -> f64,
        xs: impl Iterator<Item = f64>,
        tol: u64,
    ) {
        for x in xs {
            let (a, b) = (f(x), g(x));
            if b.is_nan() {
                assert!(a.is_nan(), "{name}({x}) = {a}, want NaN");
                continue;
            }
            let rel_floor = if b.abs() < 1e-300 { 0 } else { tol };
            assert!(
                ulps(a, b) <= rel_floor || (a - b).abs() <= 1e-300,
                "{name}({x}) = {a:e}, platform {b:e}, {} ulp",
                ulps(a, b)
            );
        }
    }

    #[test]
    fn sqrt_is_correctly_rounded() {
        for x in sweep(1e-3, 1e6, 20_000).chain([2.0, 3.0, 1e-310, 5e-324, 1.7e308, 0.5, 4.0]) {
            assert_eq!(sqrt(x).to_bits(), x.sqrt().to_bits(), "sqrt({x})");
        }
        assert!(sqrt(-1.0).is_nan());
        assert_eq!(sqrt(0.0), 0.0);
    }

    #[test]
    fn rounding_functions() {
        for x in [
            -2.5,
            -2.0,
            -1.5,
            -0.3,
            0.0,
            0.3,
            1.5,
            2.5,
            1e17,
            -1e17,
            4.5e15 + 0.5,
        ] {
            assert_eq!(floor(x), x.floor(), "floor {x}");
            assert_eq!(ceil(x), x.ceil(), "ceil {x}");
            assert_eq!(trunc(x), x.trunc(), "trunc {x}");
            assert_eq!(round(x), x.round(), "round {x}");
        }
    }

    #[test]
    fn exp_and_ln() {
        check("exp", exp, f64::exp, sweep(-740.0, 709.0, 50_000), 2);
        check("exp", exp, f64::exp, sweep(-1.0, 1.0, 20_000), 2);
        check("ln", ln, f64::ln, sweep(1e-6, 1e6, 50_000), 2);
        check("ln", ln, f64::ln, sweep(0.5, 2.0, 20_000), 2);
        check("ln", ln, f64::ln, [1e-310, 5e-324, 1.7e308].into_iter(), 2);
        check("log10", log10, f64::log10, sweep(1e-6, 1e6, 5_000), 3);
        assert_eq!(exp(0.0), 1.0);
        assert_eq!(ln(1.0), 0.0);
        assert!(ln(-1.0).is_nan());
    }

    #[test]
    fn trigonometric() {
        check("sin", sin, f64::sin, sweep(-100.0, 100.0, 50_000), 2);
        check("cos", cos, f64::cos, sweep(-100.0, 100.0, 50_000), 2);
        check("tan", tan, f64::tan, sweep(-1.5, 1.5, 20_000), 3);
        check("atan", atan, f64::atan, sweep(-50.0, 50.0, 50_000), 2);
        check("asin", asin, f64::asin, sweep(-1.0, 1.0, 20_000), 3);
        check("acos", acos, f64::acos, sweep(-1.0, 1.0, 20_000), 3);
        for (y, x) in [
            (1.0, 1.0),
            (1.0, -1.0),
            (-1.0, -1.0),
            (-1.0, 1.0),
            (0.0, -1.0),
            (2.0, 0.0),
            (0.3, 7.0),
        ] {
            assert!(ulps(atan2(y, x), y.atan2(x)) <= 2, "atan2({y}, {x})");
        }
    }

    #[test]
    fn hyperbolic_and_pow() {
        check("sinh", sinh, f64::sinh, sweep(-20.0, 20.0, 20_000), 3);
        check("cosh", cosh, f64::cosh, sweep(-20.0, 20.0, 20_000), 3);
        check("tanh", tanh, f64::tanh, sweep(-20.0, 20.0, 20_000), 3);
        assert_eq!(pow(2.0, 10.0), 1024.0);
        assert_eq!(pow(3.0, -2.0), 1.0 / 9.0);
        assert_eq!(pow(-2.0, 3.0), -8.0);
        assert!(pow(-2.0, 0.5).is_nan());
        for x in sweep(0.1, 100.0, 2_000) {
            assert!(ulps(pow(x, 1.5), x.powf(1.5)) <= 4, "pow({x}, 1.5)");
            let d = ulps(pow(x, 0.7), x.powf(0.7));
            assert!(d <= 16, "pow({x}, 0.7): {d} ulp");
        }
    }
    #[test]
    fn error_function() {
        // (x, erf x, erfc x) of the C library (glibc, within 1 ulp)
        for (x, e, c) in [
            (0.0, 0.0, 1.0),
            (0.1, 0.1124629160182849, 0.8875370839817152),
            (0.3, 0.3286267594591274, 0.6713732405408726),
            (0.5, 0.5204998778130465, 0.4795001221869535),
            (0.7, 0.6778011938374184, 0.32219880616258156),
            (0.74, 0.7046780778547458, 0.2953219221452542),
            (0.76, 0.7175367528055908, 0.28246324719440913),
            (1.0, 0.8427007929497149, 0.15729920705028513),
            (1.3, 0.9340079449406524, 0.06599205505934755),
            (1.5, 0.9661051464753108, 0.033894853524689274),
            (2.0, 0.9953222650189527, 0.004677734981047265),
            (2.5, 0.999593047982555, 0.0004069520174449589),
            (3.0, 0.9999779095030014, 2.2090496998585438e-05),
            (4.0, 0.9999999845827421, 1.541725790028002e-08),
            (5.0, 0.9999999999984626, 1.5374597944280351e-12),
            (6.0, 1.0, 2.1519736712498916e-17),
            (8.0, 1.0, 1.1224297172982928e-29),
            (10.0, 1.0, 2.088487583762545e-45),
            (15.0, 1.0, 7.212994172451206e-100),
            (20.0, 1.0, 5.3958656116079005e-176),
            (26.0, 1.0, 5.663192408856143e-296),
        ] {
            assert!(ulps(erf(x), e) <= 4, "erf({x}) = {:e}, want {e:e}", erf(x));
            assert!(
                ulps(erfc(x), c) <= 6,
                "erfc({x}) = {:e}, want {c:e}",
                erfc(x)
            );
            assert_eq!(erf(-x), -erf(x));
            assert!(
                (erfc(-x) - (1.0 + e)).abs() <= 2.0 * f64::EPSILON,
                "erfc(-{x})"
            );
        }
        // erf + erfc = 1 and the derivative (2/√π) e^(−x²) over the range
        for x in sweep(0.0, 6.0, 6_000) {
            assert!((erf(x) + erfc(x) - 1.0).abs() <= 2.0 * f64::EPSILON, "{x}");
            let h = 1e-5;
            let d = (erf(x + h) - erf(x - h)) / (2.0 * h);
            assert!(
                (d - 2.0 * FRAC_1_SQRT_PI * exp(-x * x)).abs() < 1e-9,
                "erf'({x})"
            );
        }
        assert!(erf(f64::NAN).is_nan() && erfc(f64::NAN).is_nan());
        assert_eq!(erf(f64::INFINITY), 1.0);
        assert_eq!(erfc(f64::INFINITY), 0.0);
        assert_eq!(erfc(f64::NEG_INFINITY), 2.0);
    }

    #[test]
    fn hard_switches_are_min_max_abs() {
        assert_eq!(pmin(1.0, 2.0), 1.0);
        assert_eq!(pmin(2.0, -1.0), -1.0);
        assert_eq!(pmax(1.0, 2.0), 2.0);
        assert_eq!(pmax(-3.0, -4.0), -3.0);
        assert_eq!(pabs(-2.5), 2.5);
        assert!(pmin(f64::NAN, 1.0).is_nan() && pmax(1.0, f64::NAN).is_nan());
        assert_eq!(Switches::HARD.min(3.0, 1.0), 1.0);
        assert_eq!(Switches::HARD.max(3.0, 1.0), 3.0);
    }

    #[test]
    fn smooth_switches_are_bounded_and_c1() {
        let w = SMOOTH_WIDTH;
        for &(a, b) in &[
            (1.0f64, 2.0f64),
            (2.0, 2.0),
            (-1.0, 3.0),
            (1e-3, 1.1e-3),
            (5.0, -5.0),
            (0.0, 0.0),
            (-1e3, 1e-10),
            (1e3, 1e-10),
        ] {
            let s = a.abs() + b.abs();
            let m = if s > 0.0 { a.abs() * b.abs() / s } else { 0.0 };
            let lo = smin(a, b, w);
            let hi = smax(a, b, w);
            // error bound w m / 2 ≤ w min(|a|, |b|)/2 (the spec), and on
            // the safe side
            assert!(m <= a.abs().min(b.abs()));
            assert!(
                lo <= a.min(b) && a.min(b) - lo <= 0.5 * w * m + 1e-15,
                "smin({a}, {b}) = {lo}"
            );
            assert!(
                hi >= a.max(b) && hi - a.max(b) <= 0.5 * w * m + 1e-15,
                "smax({a}, {b}) = {hi}"
            );
        }
        // w = 0 is the hard form
        assert_eq!(smin(1.0, 2.0, 0.0), 1.0);
        assert_eq!(smax(1.0, 2.0, 0.0), 2.0);
        // C¹: the derivative in a is continuous across a = b (central
        // differences on both sides agree with the value ½ at a = b)
        let b = 1.0;
        let h = 1e-7;
        let der = |a: f64| (smin(a + h, b, w) - smin(a - h, b, w)) / (2.0 * h);
        assert!(
            (der(b) - 0.5).abs() < 1e-3,
            "d smin/da at a = b: {}",
            der(b)
        );
        assert!((der(b - 0.05) - 1.0).abs() < 1e-3 && der(b + 0.05).abs() < 1e-3);
        // smooth |a| is ≥ |a| and within w scale
        assert!(sabs(0.0, w, 1.0) == w && sabs(-2.0, w, 1.0) >= 2.0);
        assert!(sabs(-2.0, w, 1.0) - 2.0 <= w);
        assert_eq!(Switches::SMOOTH.min(1.0, 2.0), smin(1.0, 2.0, w));
    }
}
