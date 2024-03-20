use std::{
    f64::consts::{FRAC_2_PI, FRAC_PI_2, FRAC_PI_4},
    ops::Mul,
};

use num_traits::{AsPrimitive, PrimInt, Signed};

use crate::bits::Bits;

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
fn sin_p3_cos_p4_impl<T>(a: T, b: T, z_2: T, right: T) -> T
where
    T: PrimInt,
{
    a - z_2 * b / right
}

/// (a - b * z ^ 2) * z ^ 2
fn cos_p4_sin_p5_impl<T>(a: T, b: T, z: T, right: T) -> T
where
    T: PrimInt,
{
    let z_2 = square(z, right);
    sin_p3_cos_p4_impl(a, b, z_2, right) * z_2
}

/// (k + 1 - k * z ^ 2) * z ^ 2
fn cos_p4_impl<T>(k: T, z: T, right: T) -> T
where
    T: PrimInt,
{
    cos_p4_sin_p5_impl(k + right, k, z, right)
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

/// 1 - x ^ 2
pub fn cos_p2<T>(x: T, right: T) -> T
where
    T: AsPrimitive<i8> + PrimInt + Signed,
    i8: AsPrimitive<T>,
{
    even_cos_impl(x, right, |z, _| z.pow(2))
}

/// (1.5 - 0.5 * x ^ 2) * x
pub fn sin_p3<T>(x: T, right: T) -> T
where
    T: AsPrimitive<i8> + PrimInt + Signed,
    i8: AsPrimitive<T>,
{
    // 1.5 * x - 0.5 * x ^ 3
    // = (1.5 - 0.5 * x ^ 2) * x
    sin_p3_impl(right / 2.as_(), x, right)
}

/// 1 - (a + 1 - a * z ^ 2) * z ^ 2  
/// a = 1 - pi / 4
pub fn cos_p4<T>(x: T, right: T) -> T
where
    T: AsPrimitive<f64> + AsPrimitive<i8> + PrimInt + Signed,
    f64: AsPrimitive<T>,
    i8: AsPrimitive<T>,
{
    even_cos_impl(x, right, |z, right| {
        cos_p4_impl(cos_p4_k::<T>(right), z, right)
    })
}

/// (a - (2 * a - 2.5 - (a - 1.5) * x ^ 2) * x ^ 2) * x  
/// a = pi / 2
pub fn sin_p5<T>(x: T, right: T) -> T
where
    T: AsPrimitive<f64> + AsPrimitive<i8> + PrimInt + Signed,
    f64: AsPrimitive<T>,
    i8: AsPrimitive<T>,
{
    sin_p5_impl(sin_p5_k::<T>(right), x, right)
}

/// 1 - (a + 1 - a * z ^ 2) * z ^ 2  
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

/// (a - (2 * a - 2.5 - (a - 1.5) * x ^ 2) * x ^ 2) * x  
/// a = 4 * (3 / pi - 9 / 16)
pub fn sin_p5o<T>(x: T, right: T) -> T
where
    T: AsPrimitive<f64> + AsPrimitive<i8> + PrimInt + Signed,
    f64: AsPrimitive<T>,
    i8: AsPrimitive<T>,
{
    sin_p5_impl(sin_p5o_k::<T>(right), x, right)
}

pub fn sin_p2<T>(x: T, right: T) -> T
where
    T: AsPrimitive<i8> + PrimInt + Signed,
    i8: AsPrimitive<T>,
{
    cos_p2(even_sin_impl(x, right), right)
}

pub fn sin_p4<T>(x: T, right: T) -> T
where
    T: AsPrimitive<f64> + AsPrimitive<i8> + PrimInt + Signed,
    f64: AsPrimitive<T>,
    i8: AsPrimitive<T>,
{
    cos_p4(even_sin_impl(x, right), right)
}

pub fn cos_p1<T>(x: T, right: T) -> T
where
    T: AsPrimitive<i8> + PrimInt + Signed,
    i8: AsPrimitive<T>,
{
    sin_p1(odd_cos_impl(x, right), right)
}

pub fn cos_p3<T>(x: T, right: T) -> T
where
    T: AsPrimitive<i8> + PrimInt + Signed,
    i8: AsPrimitive<T>,
{
    sin_p3(odd_cos_impl(x, right), right)
}

pub fn cos_p5<T>(x: T, right: T) -> T
where
    T: AsPrimitive<f64> + AsPrimitive<i8> + PrimInt + Signed,
    f64: AsPrimitive<T>,
    i8: AsPrimitive<T>,
{
    sin_p5(odd_cos_impl(x, right), right)
}

pub fn sin_p4o<T>(x: T, right: T) -> T
where
    T: AsPrimitive<f64> + AsPrimitive<i8> + PrimInt + Signed,
    f64: AsPrimitive<T>,
    i8: AsPrimitive<T>,
{
    cos_p4o(even_sin_impl(x, right), right)
}

pub fn cos_p5o<T>(x: T, right: T) -> T
where
    T: AsPrimitive<f64> + AsPrimitive<i8> + PrimInt + Signed,
    f64: AsPrimitive<T>,
    i8: AsPrimitive<T>,
{
    sin_p5o(odd_cos_impl(x, right), right)
}

pub fn sin_p1_default<T>(x: T) -> T
where
    T: AsPrimitive<i8> + Bits + PrimInt + Signed,
    i8: AsPrimitive<T>,
{
    sin_p1(x, calc_default_right::<T>())
}

pub fn sin_p2_default<T>(x: T) -> T
where
    T: AsPrimitive<i8> + Bits + PrimInt + Signed,
    i8: AsPrimitive<T>,
{
    sin_p2(x, calc_default_right::<T>())
}

pub fn sin_p3_default<T>(x: T) -> T
where
    T: AsPrimitive<i8> + Bits + PrimInt + Signed,
    i8: AsPrimitive<T>,
{
    sin_p3(x, calc_default_right::<T>())
}

pub fn sin_p4_default<T>(x: T) -> T
where
    T: AsPrimitive<f64> + AsPrimitive<i8> + Bits + PrimInt + Signed,
    f64: AsPrimitive<T>,
    i8: AsPrimitive<T>,
{
    sin_p4(x, calc_default_right::<T>())
}

pub fn sin_p5_default<T>(x: T) -> T
where
    T: AsPrimitive<f64> + AsPrimitive<i8> + Bits + PrimInt + Signed,
    f64: AsPrimitive<T>,
    i8: AsPrimitive<T>,
{
    sin_p5(x, calc_default_right::<T>())
}

pub fn cos_p1_default<T>(x: T) -> T
where
    T: AsPrimitive<i8> + Bits + PrimInt + Signed,
    i8: AsPrimitive<T>,
{
    cos_p1(x, calc_default_right::<T>())
}

pub fn cos_p2_default<T>(x: T) -> T
where
    T: AsPrimitive<i8> + Bits + PrimInt + Signed,
    i8: AsPrimitive<T>,
{
    cos_p2(x, calc_default_right::<T>())
}

pub fn cos_p3_default<T>(x: T) -> T
where
    T: AsPrimitive<i8> + Bits + PrimInt + Signed,
    i8: AsPrimitive<T>,
{
    cos_p3(x, calc_default_right::<T>())
}

pub fn cos_p4_default<T>(x: T) -> T
where
    T: AsPrimitive<f64> + AsPrimitive<i8> + Bits + PrimInt + Signed,
    f64: AsPrimitive<T>,
    i8: AsPrimitive<T>,
{
    cos_p4(x, calc_default_right::<T>())
}

pub fn cos_p5_default<T>(x: T) -> T
where
    T: AsPrimitive<f64> + AsPrimitive<i8> + Bits + PrimInt + Signed,
    f64: AsPrimitive<T>,
    i8: AsPrimitive<T>,
{
    cos_p5(x, calc_default_right::<T>())
}

pub fn sin_p4o_default<T>(x: T) -> T
where
    T: AsPrimitive<f64> + AsPrimitive<i8> + Bits + PrimInt + Signed,
    f64: AsPrimitive<T>,
    i8: AsPrimitive<T>,
{
    sin_p4o(x, calc_default_right::<T>())
}

pub fn sin_p5o_default<T>(x: T) -> T
where
    T: AsPrimitive<f64> + AsPrimitive<i8> + Bits + PrimInt + Signed,
    f64: AsPrimitive<T>,
    i8: AsPrimitive<T>,
{
    sin_p5o(x, calc_default_right::<T>())
}

pub fn cos_p4o_default<T>(x: T) -> T
where
    T: AsPrimitive<f64> + AsPrimitive<i8> + Bits + PrimInt + Signed,
    f64: AsPrimitive<T>,
    i8: AsPrimitive<T>,
{
    cos_p4o(x, calc_default_right::<T>())
}

pub fn cos_p5o_default<T>(x: T) -> T
where
    T: AsPrimitive<f64> + AsPrimitive<i8> + Bits + PrimInt + Signed,
    f64: AsPrimitive<T>,
    i8: AsPrimitive<T>,
{
    cos_p5o(x, calc_default_right::<T>())
}

#[cfg(test)]
mod tests {
    use std::{fmt::Display, ops::Range};

    use anyhow::Result;
    use approx::assert_abs_diff_eq;

    use crate::tests::read_data;

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
    fn test_default_right() {
        assert_eq!(calc_default_right::<i8>(), 8);
        assert_eq!(calc_default_right::<i16>(), 128);
        assert_eq!(calc_default_right::<i32>(), 32768);
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
        fn test<F>(f: F, one: i32)
        where
            F: Fn(i32) -> i32,
        {
            let right = calc_default_right::<i32>();
            #[rustfmt::skip] assert_eq!(f(         0),    0);
            #[rustfmt::skip] assert_eq!(f( 2 * right),    0);
            #[rustfmt::skip] assert_eq!(f(-2 * right),    0);
            #[rustfmt::skip] assert_eq!(f(     right),  one);
            #[rustfmt::skip] assert_eq!(f(    -right), -one);
        }
        let right = calc_default_right::<i32>();
        let one = right.pow(2);
        test(sin_p1_default, right);
        test(sin_p2_default, one);
        test(sin_p3_default, one);
        test(sin_p4_default, one);
        test(sin_p5_default, one);
        test(sin_p4o_default, one);
        test(sin_p5o_default, one);
    }

    #[test]
    fn test_cos() {
        fn test<F>(f: F, one: i32)
        where
            F: Fn(i32) -> i32,
        {
            let right = calc_default_right::<i32>();
            #[rustfmt::skip] assert_eq!(f(         0),  one);
            #[rustfmt::skip] assert_eq!(f(     right),    0);
            #[rustfmt::skip] assert_eq!(f(    -right),    0);
            #[rustfmt::skip] assert_eq!(f( 2 * right), -one);
            #[rustfmt::skip] assert_eq!(f(-2 * right), -one);
        }
        let right = calc_default_right::<i32>();
        let one = right.pow(2);
        test(cos_p1_default, right);
        test(cos_p2_default, one);
        test(cos_p3_default, one);
        test(cos_p4_default, one);
        test(cos_p5_default, one);
        test(cos_p4o_default, one);
        test(cos_p5o_default, one);
    }

    #[test]
    fn test_default() {
        use std::i32::{MAX, MIN};

        // 17th mersenne prime
        const STEP: usize = 131071;

        let right = calc_default_right::<i32>();
        let full = calc_full(right);

        assert_eq!(MIN % full, 0);
        assert_eq!(MAX % full, full - 1);

        (-full - 1..=full + 1)
            .chain(MAX - full..=MAX)
            .chain(MIN..=MIN + full + 1)
            .chain((MIN..=MAX).step_by(right as usize))
            .chain((MIN..=MAX).step_by(right as usize).map(|x| x + 1))
            .chain((MIN..=MAX).rev().step_by(right as usize))
            .chain((MIN..=MAX).step_by(STEP))
            .chain((MIN..=MAX).rev().step_by(STEP))
            .for_each(|x| {
                assert_eq!(sin_p1_default(x), sin_p1(x, right));
                assert_eq!(sin_p2_default(x), sin_p2(x, right));
                assert_eq!(sin_p3_default(x), sin_p3(x, right));
                assert_eq!(sin_p4_default(x), sin_p4(x, right));
                assert_eq!(sin_p5_default(x), sin_p5(x, right));
                assert_eq!(cos_p1_default(x), cos_p1(x, right));
                assert_eq!(cos_p2_default(x), cos_p2(x, right));
                assert_eq!(cos_p3_default(x), cos_p3(x, right));
                assert_eq!(cos_p4_default(x), cos_p4(x, right));
                assert_eq!(cos_p5_default(x), cos_p5(x, right));
                assert_eq!(sin_p4o_default(x), sin_p4o(x, right));
                assert_eq!(sin_p5o_default(x), sin_p5o(x, right));
                assert_eq!(cos_p4o_default(x), cos_p4o(x, right));
                assert_eq!(cos_p5o_default(x), cos_p5o(x, right));
            });
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

    #[test]
    fn test_cos_p1() {
        const RIGHT: i32 = 25;
        for i in -9..=9 {
            #[rustfmt::skip] assert_eq!( RIGHT - 1, cos_p1((4 * i    ) * RIGHT - 1, RIGHT));
            #[rustfmt::skip] assert_eq!( RIGHT,     cos_p1((4 * i    ) * RIGHT,     RIGHT));
            #[rustfmt::skip] assert_eq!( RIGHT - 1, cos_p1((4 * i    ) * RIGHT + 1, RIGHT));
            #[rustfmt::skip] assert_eq!(         1, cos_p1((4 * i + 1) * RIGHT - 1, RIGHT));
            #[rustfmt::skip] assert_eq!(         0, cos_p1((4 * i + 1) * RIGHT,     RIGHT));
            #[rustfmt::skip] assert_eq!(        -1, cos_p1((4 * i + 1) * RIGHT + 1, RIGHT));
            #[rustfmt::skip] assert_eq!(-RIGHT + 1, cos_p1((4 * i + 2) * RIGHT - 1, RIGHT));
            #[rustfmt::skip] assert_eq!(-RIGHT,     cos_p1((4 * i + 2) * RIGHT,     RIGHT));
            #[rustfmt::skip] assert_eq!(-RIGHT + 1, cos_p1((4 * i + 2) * RIGHT + 1, RIGHT));
            #[rustfmt::skip] assert_eq!(        -1, cos_p1((4 * i + 3) * RIGHT - 1, RIGHT));
            #[rustfmt::skip] assert_eq!(         0, cos_p1((4 * i + 3) * RIGHT,     RIGHT));
            #[rustfmt::skip] assert_eq!(         1, cos_p1((4 * i + 3) * RIGHT + 1, RIGHT));
        }
    }

    fn test_sin_cos<F, G, H, I>(
        data_path: &str,
        f: F,
        f_default: G,
        one: i32,
        to_period: H,
        f_std: I,
        acceptable_error: f64,
    ) -> Result<()>
    where
        F: Fn(i32, i32) -> i32,
        G: Fn(i32) -> i32,
        H: Fn(&[i32]) -> Vec<i32>,
        I: Fn(f64) -> f64,
    {
        use std::i32::{MAX, MIN};

        // 17th mersenne prime
        const STEP: usize = 131071;

        let right = calc_default_right::<i32>();
        let right_as_usize = right as usize;
        let full = calc_full(right);

        assert_eq!(full % right, 0);
        assert_eq!(MIN % full, 0);
        assert_eq!(MAX % full, full - 1);

        let data = to_period(&read_data(data_path)?);

        assert_eq!(data.len(), full as usize);

        let x = (-full - 1..=full + 1)
            .chain(MAX - full..=MAX)
            .chain(MIN..=MIN + full + 1)
            .chain((MIN..=MAX).step_by(right_as_usize))
            .chain((MIN..=MAX).skip(1).step_by(right_as_usize))
            .chain((MIN..=MAX).skip(right_as_usize - 1).step_by(right_as_usize))
            .chain((MIN..=MAX).step_by(STEP))
            .chain((MIN..=MAX).rev().step_by(STEP));

        let frac_pi_straight = FRAC_PI_2 / right as f64;
        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;

        for x in x {
            let expected = data[repeat(x, full) as usize];

            // The value can be greater than 1 or less than -1
            //assert!(expected.abs() <= one, "expected: {expected}, one: {one}");

            assert_eq!(expected, f(x, right));
            assert_eq!(expected, f_default(x));

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

        Ok(())
    }

    fn test_partial<F, G>(f: F, data_path: &str, to_period: G) -> Result<()>
    where
        F: Fn(i32) -> i32,
        G: Fn(&[i32]) -> Vec<i32>,
    {
        use std::i32::{MAX, MIN};

        let data = read_data(data_path)?;
        let right = calc_default_right::<i32>();
        assert_eq!(data.len(), (right + 1) as usize);
        let data = to_period(&data);
        let full = calc_full(right);
        for (x, &expected) in (-full - 1..=full + 1).zip(data.iter().skip(full as usize - 1)) {
            let actual = f(x);
            assert_eq!(expected, actual);
        }
        for (x, &expected) in (MAX - full..=MAX)
            .chain(MIN..=MIN + full + 1)
            .zip(data.iter().skip(full as usize - 1))
        {
            let actual = f(x);
            assert_eq!(expected, actual);
        }
        Ok(())
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

    #[rustfmt::skip] #[test] fn test_sin_p2_partial()  { test_partial(sin_p2_default,  "data/cos_p2.json",  to_sin_period_even).unwrap(); }
    #[rustfmt::skip] #[test] fn test_sin_p3_partial()  { test_partial(sin_p3_default,  "data/sin_p3.json",  to_sin_period_odd).unwrap();  }
    #[rustfmt::skip] #[test] fn test_sin_p4_partial()  { test_partial(sin_p4_default,  "data/cos_p4.json",  to_sin_period_even).unwrap(); }
    #[rustfmt::skip] #[test] fn test_sin_p5_partial()  { test_partial(sin_p5_default,  "data/sin_p5.json",  to_sin_period_odd).unwrap();  }
    #[rustfmt::skip] #[test] fn test_cos_p2_partial()  { test_partial(cos_p2_default,  "data/cos_p2.json",  to_cos_period_even).unwrap(); }
    #[rustfmt::skip] #[test] fn test_cos_p3_partial()  { test_partial(cos_p3_default,  "data/sin_p3.json",  to_cos_period_odd).unwrap();  }
    #[rustfmt::skip] #[test] fn test_cos_p4_partial()  { test_partial(cos_p4_default,  "data/cos_p4.json",  to_cos_period_even).unwrap(); }
    #[rustfmt::skip] #[test] fn test_cos_p5_partial()  { test_partial(cos_p5_default,  "data/sin_p5.json",  to_cos_period_odd).unwrap();  }
    #[rustfmt::skip] #[test] fn test_sin_p4o_partial() { test_partial(sin_p4o_default, "data/cos_p4o.json", to_sin_period_even).unwrap(); }
    #[rustfmt::skip] #[test] fn test_sin_p5o_partial() { test_partial(sin_p5o_default, "data/sin_p5o.json", to_sin_period_odd).unwrap();  }
    #[rustfmt::skip] #[test] fn test_cos_p4o_partial() { test_partial(cos_p4o_default, "data/cos_p4o.json", to_cos_period_even).unwrap(); }
    #[rustfmt::skip] #[test] fn test_cos_p5o_partial() { test_partial(cos_p5o_default, "data/sin_p5o.json", to_cos_period_odd).unwrap();  }

    fn test_periodicity<F>(f: F)
    where
        F: Fn(i32) -> i32,
    {
        let right = calc_default_right::<i32>();
        let full = calc_full(right);
        let x = (0..4)
            .flat_map(|i| {
                let i_right = i * right;
                [i_right, i_right + 1, i_right + right - 1]
            })
            .collect::<Vec<_>>();
        for x in x {
            let expected = f(x);
            for x in (x + std::i32::MIN..=std::i32::MAX).step_by(full as usize) {
                assert_eq!(expected, f(x));
            }
        }
    }

    #[rustfmt::skip] #[test] fn test_sin_p1_periodicity()  { test_periodicity(sin_p1_default);  }
    #[rustfmt::skip] #[test] fn test_sin_p2_periodicity()  { test_periodicity(sin_p2_default);  }
    #[rustfmt::skip] #[test] fn test_sin_p3_periodicity()  { test_periodicity(sin_p3_default);  }
    #[rustfmt::skip] #[test] fn test_sin_p4_periodicity()  { test_periodicity(sin_p4_default);  }
    #[rustfmt::skip] #[test] fn test_sin_p5_periodicity()  { test_periodicity(sin_p5_default);  }
    #[rustfmt::skip] #[test] fn test_cos_p1_periodicity()  { test_periodicity(cos_p1_default);  }
    #[rustfmt::skip] #[test] fn test_cos_p2_periodicity()  { test_periodicity(cos_p2_default);  }
    #[rustfmt::skip] #[test] fn test_cos_p3_periodicity()  { test_periodicity(cos_p3_default);  }
    #[rustfmt::skip] #[test] fn test_cos_p4_periodicity()  { test_periodicity(cos_p4_default);  }
    #[rustfmt::skip] #[test] fn test_cos_p5_periodicity()  { test_periodicity(cos_p5_default);  }
    #[rustfmt::skip] #[test] fn test_sin_p4o_periodicity() { test_periodicity(sin_p4o_default); }
    #[rustfmt::skip] #[test] fn test_sin_p5o_periodicity() { test_periodicity(sin_p5o_default); }
    #[rustfmt::skip] #[test] fn test_cos_p4o_periodicity() { test_periodicity(sin_p4o_default); }
    #[rustfmt::skip] #[test] fn test_cos_p5o_periodicity() { test_periodicity(sin_p5o_default); }

    fn test_with_steps<F>(f: F)
    where
        F: Fn(i32) -> i32,
    {
        use std::i32::{MAX, MIN};

        // 17th mersenne prime
        const STEP: usize = 131071;

        let full = calc_full(calc_default_right::<i32>());
        let assert = |x| assert_eq!(f(x), f(x % full));
        (MIN..=MAX).step_by(STEP).for_each(assert);
        (MIN..=MAX).rev().step_by(STEP).for_each(assert);
    }

    #[rustfmt::skip] #[test] fn test_sin_p1_with_steps()  { test_with_steps(sin_p1_default);  }
    #[rustfmt::skip] #[test] fn test_sin_p2_with_steps()  { test_with_steps(sin_p2_default);  }
    #[rustfmt::skip] #[test] fn test_sin_p3_with_steps()  { test_with_steps(sin_p3_default);  }
    #[rustfmt::skip] #[test] fn test_sin_p4_with_steps()  { test_with_steps(sin_p4_default);  }
    #[rustfmt::skip] #[test] fn test_sin_p5_with_steps()  { test_with_steps(sin_p5_default);  }
    #[rustfmt::skip] #[test] fn test_cos_p1_with_steps()  { test_with_steps(cos_p1_default);  }
    #[rustfmt::skip] #[test] fn test_cos_p2_with_steps()  { test_with_steps(cos_p2_default);  }
    #[rustfmt::skip] #[test] fn test_cos_p3_with_steps()  { test_with_steps(cos_p3_default);  }
    #[rustfmt::skip] #[test] fn test_cos_p4_with_steps()  { test_with_steps(cos_p4_default);  }
    #[rustfmt::skip] #[test] fn test_cos_p5_with_steps()  { test_with_steps(cos_p5_default);  }
    #[rustfmt::skip] #[test] fn test_sin_p4o_with_steps() { test_with_steps(sin_p4o_default); }
    #[rustfmt::skip] #[test] fn test_sin_p5o_with_steps() { test_with_steps(sin_p5o_default); }
    #[rustfmt::skip] #[test] fn test_cos_p4o_with_steps() { test_with_steps(cos_p4o_default); }
    #[rustfmt::skip] #[test] fn test_cos_p5o_with_steps() { test_with_steps(cos_p5o_default); }

    fn compare_sin_cos_f64<Actual, T>(actual: Actual, expected: fn(f64) -> f64, margin: f64, one: T)
    where
        Actual: Fn(T, T) -> T,
        T: Display + AsPrimitive<f64> + Bits + PrimInt + Signed,
        Range<T>: Iterator<Item = T>,
        i8: AsPrimitive<T>,
    {
        const SCALE: f64 = 2_i32.pow(12) as f64;

        let right = calc_default_right::<T>();
        let straight = right * 2.as_();
        let frac_pi_straight = FRAC_PI_2 / right.as_();
        let frac_scale_one = SCALE / one.as_();

        for x in -straight..straight {
            let actual = actual(x, right);
            let expected = expected(frac_pi_straight * x.as_());

            if actual != 0.as_() {
                assert_eq!(expected.is_sign_negative(), actual.is_negative());
                assert_eq!(expected.is_sign_positive(), actual.is_positive());
            }

            // Check that the value is close to the expected value.
            {
                let actual = frac_scale_one * actual.as_();
                let expected = SCALE * expected;
                assert_abs_diff_eq!(expected, actual, epsilon = margin);
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
