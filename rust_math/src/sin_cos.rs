use std::{
    f64::consts::{FRAC_2_PI, FRAC_PI_2, FRAC_PI_4},
    ops::{Div, Mul},
};

use num_traits::{AsPrimitive, PrimInt, Signed};

use crate::bits::Bits;

const RIGHT_I32_DEFAULT: i32 = 2_i32.pow(i32::BITS / 2 - 1);

fn square<T>(b: T, denom: T) -> T
where
    T: Copy + Mul<Output = T> + Div<Output = T>,
{
    b * b / denom
}

fn repeat<T>(t: T, length: T) -> T
where
    T: Copy + Signed,
{
    let rem = t % length;
    if rem.is_negative() {
        rem + length
    } else {
        rem
    }
}

/// ```rust
/// use rust_math::sin_cos::*;
/// assert_eq!(calc_default_right::<i8 >(),     8);
/// assert_eq!(calc_default_right::<i16>(),   128);
/// assert_eq!(calc_default_right::<i32>(), 32768);
/// ```
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

/// 1 - pi / 4
fn cos_p4_k<T>(right: T) -> T
where
    T: AsPrimitive<f64>,
    f64: AsPrimitive<T>,
{
    let right: f64 = right.as_();
    ((1.0 - FRAC_PI_4) * right).round_ties_even().as_()
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

/// a - b * z ^ 2
fn sin_p3_cos_p4_impl<T: PrimInt>(a: T, b: T, z_2: T, right: T) -> T {
    a - z_2 * b / right
}

/// (a - b * z ^ 2) * z ^ 2
fn cos_p4_sin_p5_impl<T: PrimInt>(a: T, b: T, z: T, right: T) -> T {
    let z_2 = square(z, right);
    sin_p3_cos_p4_impl(a, b, z_2, right) * z_2
}

/// (k + 1 - k * z ^ 2) * z ^ 2
fn cos_p4_impl<T: PrimInt>(k: T, z: T, right: T) -> T {
    cos_p4_sin_p5_impl(k + right, k, z, right)
}

/// x
fn sin_p1<T>(x: T, right: T) -> T
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

/// (1 + k - k * x ^ 2) * x
fn sin_p3_impl<T>(k: T, x: T, right: T) -> T
where
    T: AsPrimitive<i8> + PrimInt + Signed,
    i8: AsPrimitive<T>,
{
    let z = sin_p1(x, right);
    sin_p3_cos_p4_impl(right + k, k, square(z, right), right) * z
}

/// (k - (2 * k - 2.5 - (k - 1.5) * x ^ 2) * x ^ 2) * x
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

pub(crate) trait Sin<T> {
    const RIGHT: T;
    fn sin(x: T) -> T;
    fn cos(x: T) -> T
    where
        T: 'static + PrimInt,
        i8: AsPrimitive<T>,
    {
        Self::sin(odd_cos_impl(x, Self::RIGHT))
    }
}

pub(crate) trait Cos<T> {
    const RIGHT: T;
    fn cos(x: T) -> T;
    fn sin(x: T) -> T
    where
        T: 'static + PrimInt,
        i8: AsPrimitive<T>,
    {
        Self::cos(even_sin_impl(x, Self::RIGHT))
    }
}

/// 1 - x ^ 2
fn cos_p2<T>(x: T, right: T) -> T
where
    T: AsPrimitive<i8> + PrimInt + Signed,
    i8: AsPrimitive<T>,
{
    even_cos_impl(x, right, |z, _| z.pow(2))
}

pub(crate) struct CosP2I32();

impl Cos<i32> for CosP2I32 {
    const RIGHT: i32 = RIGHT_I32_DEFAULT;
    fn cos(x: i32) -> i32 {
        cos_p2(x, RIGHT_I32_DEFAULT)
    }
}

/// (1.5 - 0.5 * x ^ 2) * x
pub(crate) struct SinP3_16384();

impl Sin<i32> for SinP3_16384 {
    const RIGHT: i32 = RIGHT_I32_DEFAULT;
    fn sin(x: i32) -> i32 {
        const K: i32 = RIGHT_I32_DEFAULT / 2;
        sin_p3_impl(K, x, RIGHT_I32_DEFAULT)
    }
}

/// Approximate the cosine function by the 4th order polynomial derived by Taylor expansion.
///
/// 1 - (a + 1 - a * z ^ 2) * z ^ 2  
/// a = 1 - pi / 4
pub(crate) struct CosP4_7032();

impl Cos<i32> for CosP4_7032 {
    const RIGHT: i32 = RIGHT_I32_DEFAULT;
    fn cos(x: i32) -> i32 {
        even_cos_impl(x, RIGHT_I32_DEFAULT, |z, _| {
            cos_p4_impl(cos_p4_k(RIGHT_I32_DEFAULT), z, RIGHT_I32_DEFAULT)
        })
    }
}

pub(crate) struct CosP4_7384();

impl Cos<i32> for CosP4_7384 {
    const RIGHT: i32 = RIGHT_I32_DEFAULT;
    fn cos(x: i32) -> i32 {
        even_cos_impl(x, RIGHT_I32_DEFAULT, |z, _| {
            cos_p4_impl(cos_p4o_k(RIGHT_I32_DEFAULT), z, RIGHT_I32_DEFAULT)
        })
    }
}

/// Approximate the sine function by the 5th order polynomial derived by Taylor expansion.
///
/// (a - (2 * a - 2.5 - (a - 1.5) * x ^ 2) * x ^ 2) * x  
/// a = pi / 2
pub fn sin_p5_51472(x: i32) -> i32 {
    sin_p5_impl(sin_p5_k(RIGHT_I32_DEFAULT), x, RIGHT_I32_DEFAULT)
}

/// Approximate the cosine function by the 5th order polynomial derived by Taylor expansion.
pub fn cos_p5_51472(x: i32) -> i32 {
    sin_p5_51472(odd_cos_impl(x, RIGHT_I32_DEFAULT))
}

/// Approximate the sine function by the 5th order polynomial derived by Taylor expansion with
/// coefficients which is adjusted so that the average of the errors is 0.
///
/// (a - (2 * a - 2.5 - (a - 1.5) * x ^ 2) * x ^ 2) * x  
/// a = 4 * (3 / pi - 9 / 16)
pub fn sin_p5_51437(x: i32) -> i32 {
    sin_p5_impl(sin_p5o_k(RIGHT_I32_DEFAULT), x, RIGHT_I32_DEFAULT)
}

/// Approximate the cosine function by the 5th order polynomial derived by Taylor expansion with
/// coefficients which is adjusted so that the average of the errors is 0.
pub fn cos_p5_51437(x: i32) -> i32 {
    sin_p5_51437(odd_cos_impl(x, RIGHT_I32_DEFAULT))
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use crate::{
        cos_p2_i32, cos_p3_16384, cos_p4_7032, cos_p4_7384, sin_p2_i32, sin_p3_16384, sin_p4_7032,
        sin_p4_7384, tests::read_data,
    };

    use super::*;

    #[test]
    fn test_repeat() {
        const LENGTH: i32 = 10;
        for i in -9..=9 {
            for (&expected, offset) in [9, 0, 1].iter().zip([-1, 0, 1]) {
                assert_eq!(expected, repeat(LENGTH * i + offset, LENGTH));
            }
        }
    }

    #[test]
    fn test_calc_quadrant() {
        const RIGHT: i32 = 25;
        for i in -9..=9 {
            let x = 4 * i * RIGHT;
            for expected in 0_i8..4 {
                let x = x + (expected as i32) * RIGHT;
                for offset in [0, 1, RIGHT - 1] {
                    assert_eq!(expected, calc_quadrant(offset + x, RIGHT));
                }
            }
        }
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

    #[test]
    fn test_sin() {
        fn test(f: impl Fn(i32) -> i32, right: i32, one: i32) {
            #[rustfmt::skip] assert_eq!(f(         0),    0);
            #[rustfmt::skip] assert_eq!(f( 2 * right),    0);
            #[rustfmt::skip] assert_eq!(f(-2 * right),    0);
            #[rustfmt::skip] assert_eq!(f(     right),  one);
            #[rustfmt::skip] assert_eq!(f(    -right), -one);
        }

        let right = calc_default_right::<i32>();
        let one = right.pow(2);

        test(sin_p2_i32, right, one);
        test(sin_p3_16384, right, one);
        test(sin_p4_7032, right, one);
        test(sin_p5_51472, right, one);
        test(sin_p4_7384, right, one);
        test(sin_p5_51437, right, one);
    }

    #[test]
    fn test_cos() {
        fn test(f: impl Fn(i32) -> i32, right: i32, one: i32) {
            #[rustfmt::skip] assert_eq!(f(         0),  one);
            #[rustfmt::skip] assert_eq!(f(     right),    0);
            #[rustfmt::skip] assert_eq!(f(    -right),    0);
            #[rustfmt::skip] assert_eq!(f( 2 * right), -one);
            #[rustfmt::skip] assert_eq!(f(-2 * right), -one);
        }

        let right = calc_default_right::<i32>();
        let one = right.pow(2);

        test(cos_p2_i32, right, one);
        test(cos_p3_16384, right, one);
        test(cos_p4_7032, right, one);
        test(cos_p5_51472, right, one);
        test(cos_p4_7384, right, one);
        test(cos_p5_51437, right, one);
    }

    #[test]
    fn test_sin_p1() {
        const RIGHT: i32 = 25;
        for i in -9..=9 {
            #[rustfmt::skip] assert_eq!(        -1, sin_p1((4 * i    ) * RIGHT - 1, RIGHT));
            #[rustfmt::skip] assert_eq!(         0, sin_p1((4 * i    ) * RIGHT,     RIGHT));
            #[rustfmt::skip] assert_eq!(         1, sin_p1((4 * i    ) * RIGHT + 1, RIGHT));
            #[rustfmt::skip] assert_eq!( RIGHT - 1, sin_p1((4 * i + 1) * RIGHT - 1, RIGHT));
            #[rustfmt::skip] assert_eq!( RIGHT,     sin_p1((4 * i + 1) * RIGHT,     RIGHT));
            #[rustfmt::skip] assert_eq!( RIGHT - 1, sin_p1((4 * i + 1) * RIGHT + 1, RIGHT));
            #[rustfmt::skip] assert_eq!(         1, sin_p1((4 * i + 2) * RIGHT - 1, RIGHT));
            #[rustfmt::skip] assert_eq!(         0, sin_p1((4 * i + 2) * RIGHT,     RIGHT));
            #[rustfmt::skip] assert_eq!(        -1, sin_p1((4 * i + 2) * RIGHT + 1, RIGHT));
            #[rustfmt::skip] assert_eq!(-RIGHT + 1, sin_p1((4 * i + 3) * RIGHT - 1, RIGHT));
            #[rustfmt::skip] assert_eq!(-RIGHT,     sin_p1((4 * i + 3) * RIGHT,     RIGHT));
            #[rustfmt::skip] assert_eq!(-RIGHT + 1, sin_p1((4 * i + 3) * RIGHT + 1, RIGHT));
        }
    }

    fn test_sin_cos(
        f: impl Fn(i32) -> i32,
        one: i32,
        data_path: &str,
        to_period: impl Fn(&[i32]) -> Vec<i32>,
        f_std: impl Fn(f64) -> f64,
        acceptable_error: f64,
    ) {
        // 5th mersenne prime
        const STEP: usize = 8191;

        let right = calc_default_right::<i32>();
        let right_as_usize = right as usize;
        let full = calc_full(right);
        let data = read_data(data_path).unwrap();

        assert_eq!(data.len(), right_as_usize + 1);
        assert_eq!(full % right, 0);
        assert_eq!(i32::MIN % full, 0);
        assert_eq!(i32::MAX % full, full - 1);

        let data = to_period(&data);

        assert_eq!(data.len(), full as usize);

        let x = (-full - 1..=full + 1)
            .chain(i32::MAX - full..=i32::MAX)
            .chain(i32::MIN..=i32::MIN + full + 1)
            .chain((i32::MIN..=i32::MAX).step_by(right_as_usize))
            .chain((i32::MIN..=i32::MAX).skip(1).step_by(right_as_usize))
            .chain(
                (i32::MIN..=i32::MAX)
                    .skip(right_as_usize - 1)
                    .step_by(right_as_usize),
            )
            .chain((i32::MIN..=i32::MAX).step_by(STEP))
            .chain((i32::MIN..=i32::MAX).rev().step_by(STEP));

        let frac_pi_straight = FRAC_PI_2 / right as f64;
        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;

        for x in x {
            let expected = data[repeat(x, full) as usize];

            // The value can be greater than 1 or less than -1
            //assert!(expected.abs() <= one, "expected: {expected}, one: {one}");

            assert_eq!(expected, f(x));

            let actual = f_std(x as f64 * frac_pi_straight);

            if (x % right) != 0 || expected != 0 {
                assert_eq!(expected.is_negative(), actual.is_sign_negative());
                assert_eq!(expected.is_positive(), actual.is_sign_positive());
            }

            let expected = expected as f64 / one as f64;

            assert_abs_diff_eq!(expected, actual, epsilon = acceptable_error);

            let diff = actual - expected;

            min = min.min(diff);
            max = max.max(diff);
        }

        println!("min: {min}, max: {max}");
    }

    fn to_sin_period_odd(data: &[i32]) -> Vec<i32> {
        let n = data.len() - 1;
        let iter = data.iter().cloned();
        let iter = iter.clone().take(n).chain(iter.rev().take(n));
        iter.clone().chain(iter.map(|x| -x)).collect()
    }

    fn to_sin_period_even(data: &[i32]) -> Vec<i32> {
        let n = data.len() - 1;
        let iter = data.iter().cloned();
        let iter = iter.clone().rev().take(n).chain(iter.take(n));
        iter.clone().chain(iter.map(|x| -x)).collect()
    }

    fn to_cos_period_even(data: &[i32]) -> Vec<i32> {
        let n = data.len() - 1;
        let f = |x: i32| -x;
        let iter = data.iter().cloned();
        let iter = iter.clone().take(n).chain(iter.rev().take(n).map(f));
        iter.clone().chain(iter.map(f)).collect()
    }

    fn to_cos_period_odd(data: &[i32]) -> Vec<i32> {
        let n = data.len() - 1;
        let f = |x: i32| -x;
        let iter = data.iter().cloned();
        let iter = iter.clone().rev().take(n).chain(iter.take(n).map(f));
        iter.clone().chain(iter.map(f)).collect()
    }

    #[rustfmt::skip] #[test] fn test_sin_p2()  { test_sin_cos(sin_p2_i32,   calc_default_right::<i32>().pow(2), "data/cos_p2.json",  to_sin_period_even, f64::sin, 0.056010); }
    #[rustfmt::skip] #[test] fn test_sin_p3()  { test_sin_cos(sin_p3_16384, calc_default_right::<i32>().pow(2), "data/sin_p3.json",  to_sin_period_odd,  f64::sin, 0.020017); }
    #[rustfmt::skip] #[test] fn test_sin_p4()  { test_sin_cos(sin_p4_7032,  calc_default_right::<i32>().pow(2), "data/cos_p4.json",  to_sin_period_even, f64::sin, 0.002819); }
    #[rustfmt::skip] #[test] fn test_sin_p5()  { test_sin_cos(sin_p5_51472, calc_default_right::<i32>().pow(2), "data/sin_p5.json",  to_sin_period_odd,  f64::sin, 0.000425); }
    #[rustfmt::skip] #[test] fn test_sin_p4o() { test_sin_cos(sin_p4_7384,  calc_default_right::<i32>().pow(2), "data/cos_p4o.json", to_sin_period_even, f64::sin, 0.001174); }
    #[rustfmt::skip] #[test] fn test_sin_p5o() { test_sin_cos(sin_p5_51437, calc_default_right::<i32>().pow(2), "data/sin_p5o.json", to_sin_period_odd,  f64::sin, 0.000226); }
    #[rustfmt::skip] #[test] fn test_cos_p2()  { test_sin_cos(cos_p2_i32,   calc_default_right::<i32>().pow(2), "data/cos_p2.json",  to_cos_period_even, f64::cos, 0.056010); }
    #[rustfmt::skip] #[test] fn test_cos_p3()  { test_sin_cos(cos_p3_16384, calc_default_right::<i32>().pow(2), "data/sin_p3.json",  to_cos_period_odd,  f64::cos, 0.020017); }
    #[rustfmt::skip] #[test] fn test_cos_p4()  { test_sin_cos(cos_p4_7032,  calc_default_right::<i32>().pow(2), "data/cos_p4.json",  to_cos_period_even, f64::cos, 0.002819); }
    #[rustfmt::skip] #[test] fn test_cos_p5()  { test_sin_cos(cos_p5_51472, calc_default_right::<i32>().pow(2), "data/sin_p5.json",  to_cos_period_odd,  f64::cos, 0.000425); }
    #[rustfmt::skip] #[test] fn test_cos_p4o() { test_sin_cos(cos_p4_7384,  calc_default_right::<i32>().pow(2), "data/cos_p4o.json", to_cos_period_even, f64::cos, 0.001174); }
    #[rustfmt::skip] #[test] fn test_cos_p5o() { test_sin_cos(cos_p5_51437, calc_default_right::<i32>().pow(2), "data/sin_p5o.json", to_cos_period_odd,  f64::cos, 0.000226); }
}
