use std::{
    f64::consts::{FRAC_2_PI, FRAC_PI_2, FRAC_PI_4},
    ops::Mul,
};

use num_traits::{AsPrimitive, PrimInt, Signed};

use crate::bits::Bits;

pub trait Angle: Bits + AsPrimitive<f64> + AsPrimitive<i8> + PrimInt + Signed {
    const DEFAULT_RIGHT: Self;
}

macro_rules! impl_angle {
    ($($t:ty),*) => {
        $(
            impl Angle for $t {
                const DEFAULT_RIGHT: Self = (2 as Self).pow(Self::BITS / 2 - 1);
            }
        )*
    };
}

// i64 and i128 is not supported
// because the coefficients cannot be calculated
// with sufficient precision.
impl_angle!(i8, i16, i32);

fn square<T: PrimInt>(b: T, denom: T) -> T {
    b.pow(2) / denom
}

fn repeat<T: PrimInt + Signed>(t: T, length: T) -> T {
    let rem = t % length;
    if rem.is_negative() {
        rem + length
    } else {
        rem
    }
}

pub fn calc_default_right<T>() -> T
where
    T: 'static + Copy + Bits + PrimInt,
    i8: AsPrimitive<T>,
{
    let base = 2.as_();
    base.pow(T::BITS / 2 - 1)
}

fn calc_full<T>(right: T) -> T
where
    T: 'static + Copy + Mul<Output = T>,
    i8: AsPrimitive<T>,
{
    right * 4.as_()
}

fn calc_quadrant<T>(x: T, right: T) -> i8
where
    T: AsPrimitive<i8> + PrimInt + Signed,
    i8: AsPrimitive<T>,
{
    (repeat(x, calc_full(right)) / right).as_()
}

fn odd_cos_impl<T>(x: T, right: T) -> T
where
    T: 'static + PrimInt,
    i8: AsPrimitive<T>,
{
    (x % calc_full(right)) + right
}

fn even_sin_impl<T>(x: T, right: T) -> T
where
    T: 'static + PrimInt,
    i8: AsPrimitive<T>,
{
    (x % calc_full(right)) - right
}

/// x
pub fn sin_p1<T>(x: T, right: T) -> T
where
    T: AsPrimitive<i8> + PrimInt + Signed,
    i8: AsPrimitive<T>,
{
    let rem = repeat(x, right);
    match calc_quadrant(x, right) {
        1 => -rem + right,
        3 => rem - right,
        2 => -rem,
        0 => rem,
        _ => unreachable!(),
    }
}

pub fn cos_p1<T>(x: T, right: T) -> T
where
    T: AsPrimitive<i8> + PrimInt + Signed,
    i8: AsPrimitive<T>,
{
    sin_p1(odd_cos_impl(x, right), right)
}

fn even_cos_impl<T, F>(x: T, right: T, f: F) -> T
where
    T: AsPrimitive<i8> + PrimInt + Signed,
    F: Fn(T, T) -> T,
    i8: AsPrimitive<T>,
{
    let rem = repeat(x, right);
    let k = right.pow(2);
    match calc_quadrant(x, right) {
        1 => -k + f(right - rem, right),
        3 => k - f(right - rem, right),
        2 => -k + f(rem, right),
        0 => k - f(rem, right),
        _ => unreachable!(),
    }
}

/// 1 - x ^ 2
pub fn cos_p2<T>(x: T, right: T) -> T
where
    T: AsPrimitive<i8> + PrimInt + Signed,
    i8: AsPrimitive<T>,
{
    even_cos_impl(x, right, |z, _| z.pow(2))
}

pub fn sin_p2<T>(x: T, right: T) -> T
where
    T: AsPrimitive<i8> + PrimInt + Signed,
    i8: AsPrimitive<T>,
{
    cos_p2(even_sin_impl(x, right), right)
}

fn sin_p3_cos_p4_impl<T>(a: T, b: T, z_2: T, right: T) -> T
where
    T: PrimInt,
{
    a - z_2 * b / right
}

/// 1 + k - k * x ^ 2
fn sin_p3_impl<T>(k: T, x: T, right: T) -> T
where
    T: AsPrimitive<i8> + PrimInt + Signed,
    i8: AsPrimitive<T>,
{
    let z = sin_p1(x, right);
    sin_p3_cos_p4_impl(right + k, k, square(z, right), right) * z
}

/// 1.5 * x - 0.5 * x ^ 3
pub fn sin_p3<T>(x: T, right: T) -> T
where
    T: AsPrimitive<i8> + PrimInt + Signed,
    i8: AsPrimitive<T>,
{
    // 1.5 * x - 0.5 * x ^ 3
    // = (1.5 - 0.5 * x ^ 2) * x
    sin_p3_impl(right / 2.as_(), x, right)
}

pub fn cos_p3<T>(x: T, right: T) -> T
where
    T: AsPrimitive<i8> + PrimInt + Signed,
    i8: AsPrimitive<T>,
{
    sin_p3(odd_cos_impl(x, right), right)
}

fn cos_p4_sin_p5_impl<T>(a: T, b: T, z: T, right: T) -> T
where
    T: PrimInt,
{
    let z_2 = square(z, right);
    sin_p3_cos_p4_impl(a, b, z_2, right) * z_2
}

/// (k + 1) * z ^ 2 - k * z ^ 4
fn cos_p4_impl<T>(k: T, z: T, right: T) -> T
where
    T: PrimInt,
{
    cos_p4_sin_p5_impl(k + right, k, z, right)
}

/// 1 - pi / 4
fn cos_p4_k<T>(right: T) -> T
where
    T: AsPrimitive<f64>,
    f64: AsPrimitive<T>,
{
    let right: f64 = right.as_();
    ((1.0 - FRAC_PI_4) * right).round_ties_even().as_()
}

/// 1 - a * z ^ 2 + (a - 1) * z ^ 4
/// a = 1 - pi / 4
pub fn cos_p4<T>(x: T, right: T) -> T
where
    T: PrimInt + AsPrimitive<f64> + AsPrimitive<i8> + Signed,
    f64: AsPrimitive<T>,
    i8: AsPrimitive<T>,
{
    even_cos_impl(x, right, |z, right| {
        cos_p4_impl(cos_p4_k::<T>(right), z, right)
    })
}

pub fn sin_p4<T>(x: T, right: T) -> T
where
    T: PrimInt + AsPrimitive<f64> + AsPrimitive<i8> + Signed,
    f64: AsPrimitive<T>,
    i8: AsPrimitive<T>,
{
    cos_p4(even_sin_impl(x, right), right)
}

/// 5 * (1 - 3 / pi)
fn cos_p4o_k<T>(right: T) -> T
where
    T: AsPrimitive<f64>,
    f64: AsPrimitive<T>,
{
    let right: f64 = right.as_();
    (5.0 * (1.0 - 1.5 * FRAC_2_PI) * right)
        .round_ties_even()
        .as_()
}

/// 1 - a * z ^ 2 + (a - a) * z ^ 4
/// a = 5 * (1 - 3 / pi)
pub fn cos_p4o<T>(x: T, right: T) -> T
where
    T: AsPrimitive<f64> + AsPrimitive<i8> + PrimInt + Signed,
    f64: AsPrimitive<T>,
    i8: AsPrimitive<T>,
{
    even_cos_impl(x, right, |z, right| {
        cos_p4_impl(cos_p4o_k::<T>(right), z, right)
    })
}

pub fn sin_p4o<T>(x: T, right: T) -> T
where
    T: AsPrimitive<f64> + AsPrimitive<i8> + PrimInt + Signed,
    f64: AsPrimitive<T>,
    i8: AsPrimitive<T>,
{
    cos_p4o(even_sin_impl(x, right), right)
}

/// k * x - (2 * k - 2.5) * x ^ 3 + (k - 1.5) * x ^ 5
fn sin_p5_impl<T>(k: T, x: T, right: T) -> T
where
    T: AsPrimitive<i8> + PrimInt + Signed,
    i8: AsPrimitive<T>,
{
    let z = sin_p1(x, right);
    let a = k * 2.as_() - right * 5.as_() / 2.as_();
    let b = k - right * 3.as_() / 2.as_();
    (k - cos_p4_sin_p5_impl(a, b, z, right) / right) * z
}

/// pi / 2
fn sin_p5_k<T>(right: T) -> T
where
    T: AsPrimitive<f64>,
    f64: AsPrimitive<T>,
{
    let right: f64 = right.as_();
    (FRAC_PI_2 * right).round_ties_even().as_()
}

/// a * x - c * x ^ 3 + c * x ^ 5
/// a = pi / 2
/// b = pi - 2.5
/// c = pi / 2 - 1.5
pub fn sin_p5<T>(x: T, right: T) -> T
where
    T: AsPrimitive<f64> + AsPrimitive<i8> + PrimInt + Signed,
    f64: AsPrimitive<T>,
    i8: AsPrimitive<T>,
{
    sin_p5_impl(sin_p5_k::<T>(right), x, right)
}

pub fn cos_p5<T>(x: T, right: T) -> T
where
    T: AsPrimitive<f64> + AsPrimitive<i8> + PrimInt + Signed,
    f64: AsPrimitive<T>,
    i8: AsPrimitive<T>,
{
    sin_p5(odd_cos_impl(x, right), right)
}

/// 4 * (3 / pi - 9 / 16)
fn sin_p5o_k<T>(right: T) -> T
where
    T: AsPrimitive<f64>,
    f64: AsPrimitive<T>,
{
    let right: f64 = right.as_();
    (4.0 * (1.5 * FRAC_2_PI - 9.0 / 16.0) * right)
        .round_ties_even()
        .as_()
}

/// a * x - c * x ^ 3 + c * x ^ 5
/// a = 4 * (3 / pi - 9 / 16)
/// b = 2 * a - 2.5
/// c = a - 1.5
pub fn sin_p5o<T>(x: T, right: T) -> T
where
    T: AsPrimitive<f64> + AsPrimitive<i8> + PrimInt + Signed,
    f64: AsPrimitive<T>,
    i8: AsPrimitive<T>,
{
    sin_p5_impl(sin_p5o_k::<T>(right), x, right)
}

pub fn cos_p5o<T>(x: T, right: T) -> T
where
    T: AsPrimitive<f64> + AsPrimitive<i8> + PrimInt + Signed,
    f64: AsPrimitive<T>,
    i8: AsPrimitive<T>,
{
    sin_p5o(odd_cos_impl(x, right), right)
}

#[cfg(test)]
mod tests {
    use std::{fmt::Display, ops::Range};

    use super::*;

    #[test]
    fn test_repeat() {
        assert_eq!(repeat(-11, 10), 9);
        assert_eq!(repeat(-10, 10), 0);
        assert_eq!(repeat(-9, 10), 1);
        assert_eq!(repeat(-1, 10), 9);
        assert_eq!(repeat(0, 10), 0);
        assert_eq!(repeat(1, 10), 1);
        assert_eq!(repeat(9, 10), 9);
        assert_eq!(repeat(10, 10), 0);
        assert_eq!(repeat(11, 10), 1);
    }

    #[test]
    fn test_default_right() {
        assert_eq!(calc_default_right::<i8>(), 2_i8.pow(i8::BITS / 2 - 1));
        assert_eq!(calc_default_right::<i16>(), 2_i16.pow(i16::BITS / 2 - 1));
        assert_eq!(calc_default_right::<i32>(), 2_i32.pow(i32::BITS / 2 - 1));
    }

    #[test]
    fn test_cos_p4_k() {
        assert_eq!(2, cos_p4_k::<i8>(calc_default_right::<i8>()));
        assert_eq!(27, cos_p4_k::<i16>(calc_default_right::<i16>()));
        assert_eq!(7032, cos_p4_k::<i32>(calc_default_right::<i32>()));
    }

    #[test]
    fn test_sin_p5_k() {
        assert_eq!(13, sin_p5_k::<i8>(calc_default_right::<i8>()));
        assert_eq!(201, sin_p5_k::<i16>(calc_default_right::<i16>()));
        assert_eq!(51472, sin_p5_k::<i32>(calc_default_right::<i32>()));
    }

    #[test]
    fn test_cos_p4o_k() {
        assert_eq!(2, cos_p4o_k::<i8>(calc_default_right::<i8>()));
        assert_eq!(29, cos_p4o_k::<i16>(calc_default_right::<i16>()));
        assert_eq!(7384, cos_p4o_k::<i32>(calc_default_right::<i32>()));
    }

    #[test]
    fn test_sin_p5o_k() {
        assert_eq!(13, sin_p5o_k::<i8>(calc_default_right::<i8>()));
        assert_eq!(201, sin_p5o_k::<i16>(calc_default_right::<i16>()));
        assert_eq!(51437, sin_p5o_k::<i32>(calc_default_right::<i32>()));
    }

    fn compare_sin_cos_f64<Actual, T>(actual: Actual, expected: fn(f64) -> f64, margin: f64, one: T)
    where
        Actual: Fn(T, T) -> T,
        T: Display + AsPrimitive<f64> + Bits + PrimInt + Signed,
        Range<T>: Iterator<Item = T>,
        i8: AsPrimitive<T>,
    {
        const SCALE: f64 = 2_i32.pow(12) as f64;

        let zero: T = 0.as_();
        let right = calc_default_right::<T>();
        let straight = right * 2.as_();
        let frac_pi_straight = {
            let right: f64 = right.as_();
            FRAC_PI_2 / right
        };
        let frac_scale_one = {
            let one: f64 = one.as_();
            SCALE / one
        };

        for x in -straight..straight {
            let actual = actual(x, right);
            let expected = {
                let x: f64 = x.as_();
                expected(frac_pi_straight * x)
            };

            // Check that it is exactly 1, -1 or 0
            // on the coordinate axis,
            // otherwise that the sign is correct.
            if x % right == zero {
                assert!(
                    actual == zero || actual == one || actual == -one,
                    "actual: {actual}",
                );
            } else {
                assert_eq!(0.0 < expected, zero < actual);
                assert_eq!(0.0 > expected, zero > actual);
            }

            // Check that the value is close to the expected value.
            {
                let actual = {
                    let actual: f64 = actual.as_();
                    frac_scale_one * actual
                };
                let expected = SCALE * expected;
                let diff = expected - actual;
                assert!(
                    diff.abs() < margin,
                    "x: {x}, expected: {expected}, actual: {actual}"
                );
            }

            // The value can be greater than 1 or less than -1
            //assert!(actual.abs() <= one, "actual: {actual}, one: {one}");
        }
    }

    fn compare_sin_f64<F, T>(f: F, one: T, margin: f64)
    where
        F: Copy + Fn(T, T) -> T,
        T: Display + AsPrimitive<f64> + Bits + PrimInt + Signed,
        Range<T>: Iterator<Item = T>,
        i8: AsPrimitive<T>,
    {
        compare_sin_cos_f64(f, f64::sin, margin, one);
    }

    fn compare_cos_f64<F, T>(f: F, one: T, margin: f64)
    where
        F: Copy + Fn(T, T) -> T,
        T: Display + AsPrimitive<f64> + Bits + PrimInt + Signed,
        Range<T>: Iterator<Item = T>,
        i8: AsPrimitive<T>,
    {
        compare_sin_cos_f64(f, f64::cos, margin, one);
    }

    #[test]
    fn test_sin_p1() {
        const MARGIN: f64 = 862.264;
        compare_sin_f64(sin_p1, calc_default_right::<i8>(), MARGIN);
        compare_sin_f64(sin_p1, calc_default_right::<i16>(), MARGIN);
        compare_sin_f64(sin_p1, calc_default_right::<i32>(), MARGIN);
    }

    #[test]
    fn test_cos_p1() {
        const MARGIN: f64 = 862.264;
        compare_cos_f64(cos_p1, calc_default_right::<i8>(), MARGIN);
        compare_cos_f64(cos_p1, calc_default_right::<i16>(), MARGIN);
        compare_cos_f64(cos_p1, calc_default_right::<i32>(), MARGIN);
    }

    #[test]
    fn test_cos_p2() {
        const MARGIN: f64 = 229.416;
        compare_cos_f64(cos_p2, calc_default_right::<i16>().pow(2), MARGIN);
        compare_cos_f64(cos_p2, calc_default_right::<i32>().pow(2), MARGIN);
    }

    #[test]
    fn test_sin_p2() {
        const MARGIN: f64 = 229.416;
        compare_sin_f64(sin_p2, calc_default_right::<i16>().pow(2), MARGIN);
        compare_sin_f64(sin_p2, calc_default_right::<i32>().pow(2), MARGIN);
    }

    #[test]
    fn test_sin_p3() {
        compare_sin_f64(sin_p3, calc_default_right::<i32>().pow(2), 82.0);
    }

    #[test]
    fn test_cos_p3() {
        compare_cos_f64(cos_p3, calc_default_right::<i32>().pow(2), 82.0);
    }

    #[test]
    fn test_cos_p4() {
        compare_cos_f64(cos_p4, calc_default_right::<i32>().pow(2), 11.5464);
    }

    #[test]
    fn test_sin_p4() {
        compare_sin_f64(sin_p4, calc_default_right::<i32>().pow(2), 11.5464);
    }

    #[test]
    fn test_cos_p4o() {
        compare_cos_f64(cos_p4o, calc_default_right::<i32>().pow(2), 4.80746);
    }

    #[test]
    fn test_sin_p4o() {
        compare_sin_f64(sin_p4o, calc_default_right::<i32>().pow(2), 4.80746);
    }

    #[test]
    fn test_sin_p5() {
        compare_sin_f64(sin_p5, calc_default_right::<i32>().pow(2), 1.73715);
    }

    #[test]
    fn test_cos_p5() {
        compare_cos_f64(cos_p5, calc_default_right::<i32>().pow(2), 1.73715);
    }

    #[test]
    fn test_sin_p5o() {
        compare_sin_f64(sin_p5o, calc_default_right::<i32>().pow(2), 0.925201);
    }

    #[test]
    fn test_cos_p5o() {
        compare_cos_f64(cos_p5o, calc_default_right::<i32>().pow(2), 0.925201);
    }
}
