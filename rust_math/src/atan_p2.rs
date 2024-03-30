use std::f64::consts::PI;

use fixed::{
    traits::Fixed,
    types::{
        I10F22, I10F6, I11F21, I11F5, I12F20, I12F4, I13F19, I13F3, I14F18, I15F17, I16F16, I17F15,
        I18F14, I19F13, I20F12, I21F11, I22F10, I23F9, I24F8, I25F7, I26F6, I2F6, I3F5, I6F10,
        I6F26, I7F25, I7F9, I8F24, I8F8, I9F23, I9F7,
    },
};
use num_traits::{AsPrimitive, ConstZero, NumOps, PrimInt, Signed};
use primitive_promotion::PrimitivePromotionExt;

use crate::{
    atan::{atan2_impl, atan_impl},
    bits::Bits,
};

pub trait AtanP2Default {
    type Bits;
    const A: Self::Bits;
}

macro_rules! impl_atan_p2_default_fixed {
    ($($t:ty, $a:expr),*) => {
        $(
            impl AtanP2Default for $t {
                type Bits = <Self as Fixed>::Bits;
                const A: Self::Bits = $a;
            }
        )*
    };
}

impl_atan_p2_default_fixed!(
    I3F5, 0, I2F6, 0, I13F3, 179, I12F4, 91, I11F5, 47, I10F6, 24, I9F7, 13, I8F8, 8, I7F9, 5,
    I6F10, 3, I26F6, 1458371, I25F7, 729188, I24F8, 364590, I23F9, 182295, I22F10, 91148, I21F11,
    45575, I20F12, 22789, I19F13, 11395, I18F14, 5699, I17F15, 2850, I16F16, 1426, I15F17, 714,
    I14F18, 358, I13F19, 180, I12F20, 91, I11F21, 47, I10F22, 25, I9F23, 14, I8F24, 8, I7F25, 5,
    I6F26, 3
);

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

fn atan_p2_impl<T>(x: T, x_abs: T, x_k: T, a: T, k: T) -> T
where
    T: 'static + Copy + NumOps,
    i8: AsPrimitive<T>,
{
    x * (k / 4.as_() + a * (x_k - x_abs) / x_k)
}

/// ```rust
/// use std::f64::consts::PI;
/// use approx::assert_abs_diff_eq;
/// use fixed::types::I17F15;
/// use rust_math::atan_p2::*;
/// const EXP: u32 = i32::BITS / 2 - 1;
/// const K: i32 = 2_i32.pow(EXP);
/// let result = atan_p2(1000 * K / 1732, K, I17F15::A, K);
/// assert_abs_diff_eq!(
///     PI / 6.0,
///     result as f64 * PI / K.pow(2) as f64,
///     epsilon = 0.0039,
/// );
/// ```
pub fn atan_p2<T>(x: T, x_k: T, a: T, k: T) -> T
where
    <T as PrimitivePromotionExt>::PrimitivePromotion: PartialOrd + AsPrimitive<T> + Signed,
    T: AsPrimitive<<T as PrimitivePromotionExt>::PrimitivePromotion>
        + PrimitivePromotionExt
        + Signed,
    i8: AsPrimitive<T>,
{
    atan_impl(x, x_k, |x, x_abs| atan_p2_impl(x, x_abs, x_k, a, k))
}

/// ```rust
/// use std::f64::consts::PI;
/// use approx::assert_abs_diff_eq;
/// use rust_math::atan_p2::*;
/// const EXP: u32 = i32::BITS / 2 - 1;
/// const K: i32 = 2_i32.pow(EXP);
/// let result = atan_p2_default(1000 * K / 1732);
/// assert_abs_diff_eq!(
///     PI / 6.0,
///     result as f64 * PI / K.pow(2) as f64,
///     epsilon = 0.0039,
/// );
/// ```
pub fn atan_p2_default<T>(x: T) -> T
where
    <T as PrimitivePromotionExt>::PrimitivePromotion: PartialOrd + AsPrimitive<T> + Signed,
    T: AsPrimitive<<T as PrimitivePromotionExt>::PrimitivePromotion>
        + Bits
        + PrimInt
        + PrimitivePromotionExt
        + Signed,
    f64: AsPrimitive<T>,
    i8: AsPrimitive<T>,
{
    let exp = T::BITS / 2 - 1;
    let k = 2.as_().pow(exp);
    let a = calc_default_p2_k(exp);
    atan_p2(x, k, a, k)
}

/// ```rust
/// use std::f64::consts::PI;
/// use approx::assert_abs_diff_eq;
/// use fixed::types::I17F15;
/// use rust_math::atan_p2::*;
/// const EXP: u32 = i32::BITS / 2 - 1;
/// const K: i32 = 2_i32.pow(EXP);
/// let result = atan2_p2(1000, 1732, K, I17F15::A, K);
/// assert_abs_diff_eq!(
///     PI / 6.0,
///     result as f64 * PI / K.pow(2) as f64,
///     epsilon = 0.0039,
/// );
/// ```
pub fn atan2_p2<T>(y: T, x: T, x_k: T, a: T, k: T) -> T
where
    <T as PrimitivePromotionExt>::PrimitivePromotion: AsPrimitive<T> + PartialOrd + Signed,
    T: AsPrimitive<<T as PrimitivePromotionExt>::PrimitivePromotion>
        + ConstZero
        + PrimitivePromotionExt
        + Signed,
    i8: AsPrimitive<T>,
{
    atan2_impl(y, x, x_k, |x| atan_p2_impl(x, x, x_k, a, k))
}

/// ```rust
/// use std::f64::consts::PI;
/// use approx::assert_abs_diff_eq;
/// use rust_math::atan_p2::*;
/// const EXP: u32 = i32::BITS / 2 - 1;
/// let result = atan2_p2_default(1000, 1732);
/// assert_abs_diff_eq!(
///     PI / 6.0,
///     result as f64 * PI / 2_i32.pow(2 * EXP) as f64,
///     epsilon = 0.0039,
/// );
/// ```
pub fn atan2_p2_default<T>(y: T, x: T) -> T
where
    <T as PrimitivePromotionExt>::PrimitivePromotion: AsPrimitive<T> + PartialOrd + Signed,
    T: AsPrimitive<<T as PrimitivePromotionExt>::PrimitivePromotion>
        + Bits
        + ConstZero
        + PrimInt
        + PrimitivePromotionExt
        + Signed,
    f64: AsPrimitive<T>,
    i8: AsPrimitive<T>,
{
    let exp = T::BITS / 2 - 1;
    let k = 2.as_().pow(exp);
    let a = calc_default_p2_k(exp);
    atan2_p2(y, x, k, a, k)
}

#[cfg(test)]
mod tests {
    use std::{
        cmp::Ordering,
        f64::NEG_INFINITY,
        fmt::{Debug, Display},
        ops::RangeInclusive,
    };

    use chrono::Utc;
    use num_traits::ConstOne;
    use rand::prelude::*;
    use rstest::rstest;

    use super::*;

    #[test]
    fn test_atan2_p2_default() {
        use std::i32::{MAX, MIN};

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

    fn find_optimal_constants<T>(exp: u32) -> Vec<T>
    where
        <T as PrimitivePromotionExt>::PrimitivePromotion: PartialOrd + AsPrimitive<T> + Signed,
        RangeInclusive<T>: Iterator<Item = T>,
        T: Debug
            + AsPrimitive<<T as PrimitivePromotionExt>::PrimitivePromotion>
            + AsPrimitive<f64>
            + AsPrimitive<usize>
            + Bits
            + ConstOne
            + ConstZero
            + PrimitivePromotionExt
            + PrimInt
            + Signed,
        i8: AsPrimitive<T>,
    {
        let (x_k, k, to_rad) = {
            let base: T = 2.as_();
            let to_rad = {
                let pi: f64 = base.pow(T::BITS - 2).as_();
                PI / pi
            };
            (base.pow(exp), base.pow(T::BITS - 2 - exp), to_rad)
        };

        let expected = (T::ZERO..=x_k)
            .map(|x| {
                let x: f64 = x.as_();
                (x).atan2(x_k.as_())
            })
            .collect::<Vec<_>>();

        let time = Utc::now();
        let mut elapsed = 0;
        let mut min_error = f64::INFINITY;
        let mut min_sum_error = f64::INFINITY;
        let mut optimal_constants = Vec::new();
        let mut rng = rand::thread_rng();
        let mut a = (T::ZERO..=k / 4.as_()).collect::<Vec<_>>();

        a.shuffle(&mut rng);

        for (ai, &a) in a.iter().enumerate() {
            if (ai % 1000) == 0 {
                let new_elapsed = Utc::now().signed_duration_since(time).num_seconds();
                if elapsed / 30 < new_elapsed / 30 {
                    elapsed = new_elapsed;
                    println!("exp: {exp}, ai: {ai}");
                }
            }

            let mut max_error = NEG_INFINITY;
            let mut error_sum = 0.0;

            for x in T::ZERO..=x_k {
                let i: usize = x.as_();
                let expected = expected[i];
                let actual: f64 = atan_p2(x, x_k, a, k).as_();
                let actual = to_rad * actual;
                let error = actual - expected;
                error_sum += error;
                max_error = max_error.max(error.abs());
                if max_error > min_error {
                    break;
                }
            }

            let sum_error = error_sum.abs();

            match max_error.total_cmp(&min_error) {
                Ordering::Equal => match sum_error.total_cmp(&min_sum_error) {
                    Ordering::Less => {
                        min_sum_error = sum_error;
                        optimal_constants = vec![a];
                    }
                    Ordering::Equal => {
                        optimal_constants.push(a);
                    }
                    Ordering::Greater => {}
                },
                Ordering::Less => {
                    min_error = max_error;
                    min_sum_error = sum_error;
                    optimal_constants = vec![a];
                }
                Ordering::Greater => {}
            }
        }

        println!(
            "exp: {exp}, a: {:?} ({:?}), max error: {min_error}, error average: {}",
            optimal_constants,
            &optimal_constants
                .iter()
                .map(|&a| {
                    let a: f64 = a.as_();
                    let k: f64 = k.as_();
                    PI * a / k
                })
                .collect::<Vec<_>>(),
            {
                let len: f64 = (x_k + T::ONE).as_();
                min_sum_error / len
            }
        );

        optimal_constants
    }

    fn test_optimal_constants<T>(exp: u32, expected: Vec<T>)
    where
        <T as PrimitivePromotionExt>::PrimitivePromotion: PartialOrd + AsPrimitive<T> + Signed,
        RangeInclusive<T>: Iterator<Item = T>,
        T: Debug
            + Display
            + AsPrimitive<<T as PrimitivePromotionExt>::PrimitivePromotion>
            + AsPrimitive<f64>
            + AsPrimitive<usize>
            + Bits
            + ConstOne
            + ConstZero
            + PrimitivePromotionExt
            + PrimInt
            + Signed,
        i8: AsPrimitive<T>,
    {
        let mut actuals = find_optimal_constants(exp);
        actuals.sort_unstable();
        assert_eq!(actuals, expected);
        //assert!(actuals.contains(&expected));
    }

    #[rstest]
    #[case(1, vec![2, 3])]
    #[case(2, vec![3])]
    #[case(3, vec![0, 1])]
    #[case(4, vec![0, 1])]
    #[case(5, vec![I3F5::A])]
    #[case(6, vec![I2F6::A])]
    fn test_optimal_constants_i8(#[case] exp: u32, #[case] expected: Vec<i8>) {
        test_optimal_constants(exp, expected);
    }

    #[rstest]
    #[case(1, vec![740, 741])]
    #[case(2, vec![360, 361])]
    #[case(3, vec![I13F3::A])]
    #[case(4, vec![I12F4::A])]
    #[case(5, vec![I11F5::A])]
    #[case(6, vec![I10F6::A])]
    #[case(7, vec![I9F7::A])]
    #[case(8, vec![I8F8::A])]
    #[case(9, vec![I7F9::A])]
    #[case(10, vec![I6F10::A])]
    #[case(11, vec! [0, 1] )]
    #[case(12, vec! [0, 1] )]
    #[case(13, vec![0])]
    #[case(14, vec![0])]
    fn test_optimal_constants_i16(#[case] exp: u32, #[case] expected: Vec<i16>) {
        test_optimal_constants(exp, expected);
    }

    #[rstest]
    #[case(6, vec![I26F6::A])]
    #[case(7, vec![I25F7::A])]
    #[case(8, vec![I24F8::A])]
    #[case(9, vec![I23F9::A])]
    #[case(10, vec![I22F10::A])]
    #[case(11, vec![I21F11::A])]
    #[case(12, vec![I20F12::A])]
    #[case(13, vec![I19F13::A])]
    #[case(14, vec![I18F14::A])]
    #[case(15, vec![I17F15::A])]
    #[case(16, vec![I16F16::A])]
    #[case(17, vec![I15F17::A])]
    #[case(18, vec![I14F18::A])]
    #[case(19, vec![I13F19::A])]
    #[case(20, vec![I12F20::A])]
    #[case(21, vec![I11F21::A])]
    #[case(22, vec![I10F22::A])]
    fn test_optimal_constants_i32(#[case] exp: u32, #[case] expected: Vec<i32>) {
        test_optimal_constants(exp, expected);
    }

    mod test_optiomal_constants {
        use super::*;
        macro_rules! define_test {
            ($name:tt, $exp:expr, $expected:expr) => {
                #[test]
                #[ignore]
                fn $name() {
                    test_optimal_constants_i32($exp, $expected);
                }
            };
        }
        define_test!(case_1, 1, vec![48497950, 48497951]);
        define_test!(case_2, 2, vec![23487671]);
        define_test!(case_3, 3, vec![11671032, 11671033]);
        define_test!(case_4, 4, vec![5835516]);
        define_test!(case_5, 5, vec![2917056, 2917057]);
        define_test!(case_23, 23, vec![I9F23::A]);
        define_test!(case_24, 24, vec![I8F24::A]);
        define_test!(case_25, 25, vec![I7F25::A]);
        define_test!(case_26, 26, vec![I6F26::A]);
        define_test!(case_27, 27, vec![0, 1]);
        define_test!(case_28, 28, vec![0, 1]);
        define_test!(case_29, 29, vec![0]);
        define_test!(case_30, 30, vec![0]);
    }
}
