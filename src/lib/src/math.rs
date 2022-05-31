// code borrowed from https://github.com/statrs-dev/statrs

use std::f64;

/// Constant value for `ln(pi)`
pub const LN_PI: f64 = 1.1447298858494001741434273513530587116472948129153;

/// Constant value for `ln(2 * sqrt(e / pi))`
pub const LN_2_SQRT_E_OVER_PI: f64 = 0.6207822376352452223455184457816472122518527279025978;

/// Constant value for `2 * sqrt(e / pi)`
pub const TWO_SQRT_E_OVER_PI: f64 = 1.8603827342052657173362492472666631120594218414085755;

/// Auxiliary variable when evaluating the `gamma_ln` function
const GAMMA_R: f64 = 10.900511;

/// Polynomial coefficients for approximating the `gamma_ln` function
const GAMMA_DK: &[f64] = &[
  2.48574089138753565546e-5,
  1.05142378581721974210,
  -3.45687097222016235469,
  4.51227709466894823700,
  -2.98285225323576655721,
  1.05639711577126713077,
  -1.95428773191645869583e-1,
  1.70970543404441224307e-2,
  -5.71926117404305781283e-4,
  4.63399473359905636708e-6,
  -2.71994908488607703910e-9,
];

/// The maximum factorial representable
/// by a 64-bit floating point without
/// overflowing
pub const MAX_FACTORIAL: usize = 170;

// Initialization for pre-computed cache of 171 factorial
// values 0!...170!
lazy_static! {
  static ref FCACHE: [f64; MAX_FACTORIAL + 1] = {
    let mut fcache = [1.0; MAX_FACTORIAL + 1];
    fcache.iter_mut().enumerate().skip(1).fold(1.0, |acc, (i, elt)| {
      let fac = acc * i as f64;
      *elt = fac;
      fac
    });
    fcache
  };
}

/// Computes the logarithm of the gamma function
/// with an accuracy of 16 floating point digits.
/// The implementation is derived from
/// "An Analysis of the Lanczos Gamma Approximation",
/// Glendon Ralph Pugh, 2004 p. 116
pub fn ln_gamma(x: f64) -> f64 {
  if x < 0.5 {
    let s = GAMMA_DK
      .iter()
      .enumerate()
      .skip(1)
      .fold(GAMMA_DK[0], |s, t| s + t.1 / (t.0 as f64 - x));

    LN_PI
      - (f64::consts::PI * x).sin().ln()
      - s.ln()
      - LN_2_SQRT_E_OVER_PI
      - (0.5 - x) * ((0.5 - x + GAMMA_R) / f64::consts::E).ln()
  } else {
    let s = GAMMA_DK
      .iter()
      .enumerate()
      .skip(1)
      .fold(GAMMA_DK[0], |s, t| s + t.1 / (x + t.0 as f64 - 1.0));

    s.ln() + LN_2_SQRT_E_OVER_PI + (x - 0.5) * ((x - 0.5 + GAMMA_R) / f64::consts::E).ln()
  }
}

/// Computes the gamma function with an accuracy
/// of 16 floating point digits. The implementation
/// is derived from "An Analysis of the Lanczos Gamma Approximation",
/// Glendon Ralph Pugh, 2004 p. 116
pub fn gamma(x: f64) -> f64 {
  if x < 0.5 {
    let s = GAMMA_DK
      .iter()
      .enumerate()
      .skip(1)
      .fold(GAMMA_DK[0], |s, t| s + t.1 / (t.0 as f64 - x));

    f64::consts::PI
      / ((f64::consts::PI * x).sin() * s * TWO_SQRT_E_OVER_PI * ((0.5 - x + GAMMA_R) / f64::consts::E).powf(0.5 - x))
  } else {
    let s = GAMMA_DK
      .iter()
      .enumerate()
      .skip(1)
      .fold(GAMMA_DK[0], |s, t| s + t.1 / (x + t.0 as f64 - 1.0));

    s * TWO_SQRT_E_OVER_PI * ((x - 0.5 + GAMMA_R) / f64::consts::E).powf(x - 0.5)
  }
}

/// Computes the logarithmic factorial function `x -> ln(x!)`
/// for `x >= 0`.
///
/// # Remarks
///
/// Returns `0.0` if `x <= 1`
pub fn ln_factorial(x: u64) -> f64 {
  let x = x as usize;
  FCACHE.get(x).map_or_else(|| ln_gamma(x as f64 + 1.0), |&fac| fac.ln())
}

/// Computes the binomial coefficient `n choose k`
/// where `k` and `n` are non-negative values.
///
/// # Remarks
///
/// Returns `0.0` if `k > n`
pub fn binomial(n: u64, k: u64) -> f64 {
  if k > n {
    0.0
  } else {
    (0.5 + (ln_factorial(n) - ln_factorial(k) - ln_factorial(n - k)).exp()).floor()
  }
}
