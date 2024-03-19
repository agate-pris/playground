use std::f64::consts::PI;

use num_traits::{AsPrimitive, PrimInt};

use crate::atan::{atan2_impl, atan_impl};

/// ```rust
/// use rust_math::atan_p2::*;
/// const EXP: u32 = i32::BITS / 2 - 1;
/// let result = calc_default_p2_k::<i32>(EXP);
/// assert_eq!(result, 2847);
/// ```
pub fn calc_default_p2_k<T>(exp: u32) -> T
where
    T: 'static + Copy,
    f64: AsPrimitive<T>,
{
    let k = 2.0_f64.powi(exp as i32);
    (0.273 / PI * k).round_ties_even().as_()
}

fn atan_p2_impl<T>(x: T, x_abs: T, k: T, a: T) -> T
where
    T: 'static + Copy + PrimInt,
    i8: AsPrimitive<T>,
{
    x * (k / 4.as_() + a * (k - x_abs) / k)
}

/// ```rust
/// use std::f64::consts::PI;
/// use rust_math::atan_p2::*;
/// const EXP: u32 = i32::BITS / 2 - 1;
/// const K: i32 = 2_i32.pow(EXP);
/// let a = calc_default_p2_k::<i32>(EXP);
/// let result = atan_p2(1732 * K / 1000, K, a);
/// approx::relative_eq!(
///     PI / 6.0,
///     result as f64 * PI / K.pow(2) as f64
/// );
/// ```
pub fn atan_p2(x: i32, k: i32, a: i32) -> i32 {
    atan_impl(x, k, |x, x_abs| atan_p2_impl(x, x_abs, k, a))
}

/// ```rust
/// use std::f64::consts::PI;
/// use rust_math::atan_p2::*;
/// const EXP: u32 = i32::BITS / 2 - 1;
/// const K: i32 = 2_i32.pow(EXP);
/// let result = atan_p2_default(1732 * K / 1000);
/// approx::relative_eq!(
///     PI / 6.0,
///     result as f64 * PI / K.pow(2) as f64
/// );
/// ```
pub fn atan_p2_default(x: i32) -> i32 {
    const EXP: u32 = i32::BITS / 2 - 1;
    const K: i32 = 2_i32.pow(EXP);
    let a = calc_default_p2_k(EXP);
    atan_p2(x, K, a)
}

/// ```rust
/// use std::f64::consts::PI;
/// use rust_math::atan_p2::*;
/// const EXP: u32 = i32::BITS / 2 - 1;
/// const K: i32 = 2_i32.pow(EXP);
/// let a = calc_default_p2_k::<i32>(EXP);
/// let result = atan2_p2(1732, 1000, K, a);
/// approx::relative_eq!(
///     PI / 6.0,
///     result as f64 * PI / K.pow(2) as f64
/// );
/// ```
pub fn atan2_p2(y: i32, x: i32, k: i32, a: i32) -> i32 {
    atan2_impl(y, x, k, |x| atan_p2_impl(x, x, k, a))
}

/// ```rust
/// use std::f64::consts::PI;
/// use rust_math::atan_p2::*;
/// const EXP: u32 = i32::BITS / 2 - 1;
/// let result = atan2_p2_default(1732, 1000);
/// approx::relative_eq!(
///     PI / 6.0,
///     result as f64 * PI / 2_i32.pow(2 * EXP) as f64
/// );
/// ```
pub fn atan2_p2_default(y: i32, x: i32) -> i32 {
    const EXP: u32 = i32::BITS / 2 - 1;
    const K: i32 = 2_i32.pow(EXP);
    let a = calc_default_p2_k(EXP);
    atan2_p2(y, x, K, a)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_overflow() {
        const EXP: u32 = i32::BITS / 2 - 1;
        const K: i32 = 2_i32.pow(EXP);
        let a = calc_default_p2_k::<i32>(EXP);
        assert!(a > 0);
        let x = [-K, 0, K];
        let x_abs = [0, K];
        for p in x {
            for q in x_abs {
                let _ = p * (K / 4 + a * (K - q) / K);
            }
        }
    }

    #[test]
    fn test_atan_p2() {
        const EXP: u32 = i32::BITS / 2 - 1;
        const K: i32 = 2_i32.pow(EXP);

        fn f(x: i32) -> f64 {
            let a = calc_default_p2_k::<i32>(EXP);
            let expected = (x as f64 / K as f64).atan();
            let actual = {
                let actual = atan_p2(x, K, a);
                actual as f64 * PI / 2.0_f64.powi(2 * EXP as i32)
            };
            let error = actual - expected;
            assert!(
                error.abs() < 0.0039,
                "error: {error}, x: {x}, expected: {expected}, actual: {actual}"
            );
            error
        }

        f(0);
        f(i32::MAX);
        f(i32::MIN);

        let mut min = std::f64::INFINITY;
        let mut max = std::f64::NEG_INFINITY;
        for degrees in -89..90 {
            let x = (degrees as f64).to_radians().tan();
            let x = (x * K as f64).round_ties_even() as i32;
            let e = f(x);
            min = min.min(e);
            max = max.max(e);
        }
        println!("min: {min}, max: {max}");
    }

    #[test]
    fn test_atan_p2_default() {
        const EXP: u32 = i32::BITS / 2 - 1;
        const K: i32 = 2_i32.pow(EXP);
        let a = calc_default_p2_k(EXP);
        for degrees in -89..90 {
            let x = {
                let x = (degrees as f64).to_radians().tan();
                (x * K as f64).round_ties_even() as i32
            };
            let expected = atan_p2(x, K, a);
            let actual = atan_p2_default(x);
            assert_eq!(expected, actual);
        }
    }

    #[test]
    fn test_atan2_p2_default() {
        fn f(x: i32, y: i32) {
            let expected = (y as f64).atan2(x as f64);
            let actual = {
                let actual = atan2_p2_default(y, x);
                actual as f64 * PI / 2.0_f64.powi(i32::BITS as i32 / 2 * 2 - 2)
            };
            let error = actual - expected;
            assert!(
                error.abs() < 0.0039,
                "y: {y}, x: {x}, actual: {actual}, expected: {expected}"
            );
        }
        f(0, 0);
        f(0, std::i32::MAX);
        f(0, std::i32::MIN);
        f(std::i32::MAX, 0);
        f(std::i32::MAX, std::i32::MAX);
        f(std::i32::MAX, std::i32::MIN);
        f(std::i32::MIN, 0);
        f(std::i32::MIN, std::i32::MAX);
        f(std::i32::MIN, std::i32::MIN);

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
