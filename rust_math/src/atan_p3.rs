use std::f64::consts::PI;

use num_traits::{AsPrimitive, PrimInt};

use crate::atan::{atan2_impl, atan_impl};

/// ```rust
/// use rust_math::atan_p3::*;
/// const EXP: u32 = i32::BITS / 2 - 1;
/// let (a, b) = calc_default_p3_k::<i32>(EXP);
/// assert_eq!(a, 2552);
/// assert_eq!(b, 692);
/// ```
pub fn calc_default_p3_k<T>(exp: u32) -> (T, T)
where
    T: 'static + Copy,
    f64: AsPrimitive<T>,
{
    let k: f64 = 2.0_f64.powi(exp as i32);
    let a = (0.2447 / PI * k).round_ties_even();
    let b = (0.0663 / PI * k).round_ties_even();
    (a.as_(), b.as_())
}

fn atan_p3_impl<T>(x: T, x_abs: T, k: T, a: T, b: T) -> T
where
    T: 'static + Copy + PrimInt,
    i8: AsPrimitive<T>,
{
    x * (k / 4.as_() - (x_abs - k) * (a + x_abs * b / k) / k)
}

/// ```rust
/// use std::f64::consts::PI;
/// use approx::assert_abs_diff_eq;
/// use rust_math::atan_p3::*;
/// const EXP: u32 = i32::BITS / 2 - 1;
/// const K: i32 = 2_i32.pow(EXP);
/// let (a, b) = calc_default_p3_k::<i32>(EXP);
/// let result = atan_p3(1000 * K / 1732, K, a, b);
/// assert_abs_diff_eq!(
///     PI / 6.0,
///     result as f64 * PI / K.pow(2) as f64,
///     epsilon = 0.0016,
/// );
/// ```
pub fn atan_p3(x: i32, k: i32, a: i32, b: i32) -> i32 {
    atan_impl(x, k, |x, x_abs| atan_p3_impl(x, x_abs, k, a, b))
}

/// ```rust
/// use std::f64::consts::PI;
/// use approx::assert_abs_diff_eq;
/// use rust_math::atan_p3::*;
/// const EXP: u32 = i32::BITS / 2 - 1;
/// const K: i32 = 2_i32.pow(EXP);
/// let result = atan_p3_default(1000 * K / 1732);
/// assert_abs_diff_eq!(
///     PI / 6.0,
///     result as f64 * PI / K.pow(2) as f64,
///     epsilon = 0.0016,
/// );
/// ```
pub fn atan_p3_default(x: i32) -> i32 {
    const EXP: u32 = i32::BITS / 2 - 1;
    const K: i32 = 2_i32.pow(EXP);
    let (a, b) = calc_default_p3_k(EXP);
    atan_p3(x, K, a, b)
}

/// ```rust
/// use std::f64::consts::PI;
/// use approx::assert_abs_diff_eq;
/// use rust_math::atan_p3::*;
/// const EXP: u32 = i32::BITS / 2 - 1;
/// const K: i32 = 2_i32.pow(EXP);
/// let (a, b) = calc_default_p3_k::<i32>(EXP);
/// let result = atan2_p3(1000, 1732, K, a, b);
/// assert_abs_diff_eq!(
///     PI / 6.0,
///     result as f64 * PI / K.pow(2) as f64,
///     epsilon = 0.0016,
/// );
/// ```
pub fn atan2_p3(y: i32, x: i32, k: i32, a: i32, b: i32) -> i32 {
    atan2_impl(y, x, k, |x| atan_p3_impl(x, x, k, a, b))
}

/// ```rust
/// use std::f64::consts::PI;
/// use approx::assert_abs_diff_eq;
/// use rust_math::atan_p3::*;
/// const EXP: u32 = i32::BITS / 2 - 1;
/// let result = atan2_p3_default(1000, 1732);
/// assert_abs_diff_eq!(
///     PI / 6.0,
///     result as f64 * PI / 2_i32.pow(2 * EXP) as f64,
///     epsilon = 0.0016,
/// );
/// ```
pub fn atan2_p3_default(y: i32, x: i32) -> i32 {
    const EXP: u32 = i32::BITS / 2 - 1;
    const K: i32 = 2_i32.pow(EXP);
    let (a, b) = calc_default_p3_k(EXP);
    atan2_p3(y, x, K, a, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_overflow() {
        const EXP: u32 = i32::BITS / 2 - 1;
        const K: i32 = 2_i32.pow(EXP);
        let (a, b) = calc_default_p3_k::<i32>(EXP);
        assert!(a > 0);
        assert!(b > 0);
        let x = [-K, 0, K];
        let x_abs = [0, K];
        for p in x {
            for q in x_abs {
                for r in x_abs {
                    let _ = p * (K / 4 - (q - K) * (a + r * b / K) / K);
                }
            }
        }
    }

    #[test]
    fn test_atan_p3_default() {
        const EXP: u32 = i32::BITS / 2 - 1;
        const K: i32 = 2_i32.pow(EXP);

        let (a, b) = calc_default_p3_k(EXP);
        for degrees in -89..90 {
            let x = {
                let x = (degrees as f64).to_radians().tan();
                (x * K as f64).round_ties_even() as i32
            };
            let expected = atan_p3(x, K, a, b);
            let actual = atan_p3_default(x);
            assert_eq!(expected, actual);
        }
    }

    #[test]
    fn test_atan2_p3_default() {
        use std::i32::{MAX, MIN};

        fn f(x: i32, y: i32) {
            let expected = (y as f64).atan2(x as f64);
            let actual = {
                let actual = atan2_p3_default(y, x);
                actual as f64 * PI / 2.0_f64.powi(i32::BITS as i32 / 2 * 2 - 2)
            };
            let error = actual - expected;
            assert!(
                error.abs() < 0.0016,
                "error: {error}, y: {y}, x: {x}, actual: {actual}, expected: {expected}"
            );
        }

        {
            let values = [0, MAX, MIN, MAX];
            for x in values {
                for y in values {
                    f(x, y);
                }
            }
        }

        fn g(degrees: i32) {
            const SCALE: f64 = 1000000.0;
            let radians = (degrees as f64).to_radians();
            let x = SCALE * radians.cos();
            let y = SCALE * radians.sin();
            f(x as i32, y as i32);
        }
        for degrees in -195..-180 {
            g(degrees);
        }
        for degrees in -179..180 {
            g(degrees);
        }
        for degrees in 181..195 {
            g(degrees);
        }
    }
}
