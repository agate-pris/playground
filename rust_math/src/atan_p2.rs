use fixed::{
    traits::Fixed,
    types::{
        I10F22, I10F6, I11F21, I11F5, I12F20, I12F4, I13F19, I13F3, I14F18, I15F17, I16F16, I17F15,
        I18F14, I19F13, I20F12, I21F11, I22F10, I23F9, I24F8, I25F7, I26F6, I2F6, I3F5, I6F10,
        I6F26, I7F25, I7F9, I8F24, I8F8, I9F23, I9F7,
    },
};
use num_traits::{AsPrimitive, ConstZero, NumOps, Pow, Signed};
use primitive_promotion::PrimitivePromotionExt;

use crate::atan::{atan2_impl, atan_impl};

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
/// use fixed::types::I17F15;
/// use rust_math::atan_p2::*;
/// const EXP: u32 = i32::BITS / 2 - 1;
/// const K: i32 = 2_i32.pow(EXP);
/// let result = atan_p2_default(I17F15::from_bits(1000 * K / 1732));
/// assert_abs_diff_eq!(
///     PI / 6.0,
///     result as f64 * PI / K.pow(2) as f64,
///     epsilon = 0.0039,
/// );
/// ```
pub fn atan_p2_default<T>(x: T) -> <T as Fixed>::Bits
where
    <<T as Fixed>::Bits as PrimitivePromotionExt>::PrimitivePromotion:
        PartialOrd + AsPrimitive<<T as Fixed>::Bits> + Signed,
    <T as Fixed>::Bits: AsPrimitive<<<T as Fixed>::Bits as PrimitivePromotionExt>::PrimitivePromotion>
        + Pow<u32, Output = <T as Fixed>::Bits>
        + PrimitivePromotionExt
        + Signed,
    T: AtanP2Default<Bits = <T as Fixed>::Bits> + Fixed,
    i8: AsPrimitive<<T as Fixed>::Bits>,
{
    let base: <T as Fixed>::Bits = 2.as_();
    let x_k = base.pow(T::FRAC_NBITS);
    let k = base.pow(T::INT_NBITS - 2);
    atan_p2(x.to_bits(), x_k, <T as AtanP2Default>::A, k)
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
/// use fixed::types::I17F15;
/// use rust_math::atan_p2::*;
/// const EXP: u32 = i32::BITS / 2 - 1;
/// let result = atan2_p2_default(I17F15::from_bits(1000), I17F15::from_bits(1732));
/// assert_abs_diff_eq!(
///     PI / 6.0,
///     result as f64 * PI / 2_i32.pow(2 * EXP) as f64,
///     epsilon = 0.0039,
/// );
/// ```
pub fn atan2_p2_default<T>(y: T, x: T) -> <T as Fixed>::Bits
where
    <<T as Fixed>::Bits as PrimitivePromotionExt>::PrimitivePromotion:
        PartialOrd + AsPrimitive<<T as Fixed>::Bits> + Signed,
    <T as Fixed>::Bits: AsPrimitive<<<T as Fixed>::Bits as PrimitivePromotionExt>::PrimitivePromotion>
        + ConstZero
        + Pow<u32, Output = <T as Fixed>::Bits>
        + PrimitivePromotionExt
        + Signed,
    T: AtanP2Default<Bits = <T as Fixed>::Bits> + Fixed,
    i8: AsPrimitive<<T as Fixed>::Bits>,
{
    let base: <T as Fixed>::Bits = 2.as_();
    let x_k = base.pow(T::FRAC_NBITS);
    let k = base.pow(T::INT_NBITS - 2);
    atan2_p2(y.to_bits(), x.to_bits(), x_k, <T as AtanP2Default>::A, k)
}

#[cfg(test)]
mod tests {
    use std::{
        cmp::Ordering,
        f64::consts::PI,
        fmt::{Debug, Display},
        ops::RangeInclusive,
    };

    use num_traits::{ConstOne, PrimInt};
    use rand::prelude::*;
    use rayon::prelude::*;
    use rstest::rstest;

    use crate::bits::Bits;

    use super::*;

    #[test]
    fn test_atan2_p2_default() {
        use std::i32::{MAX, MIN};

        fn f(x: i32, y: i32) {
            let expected = (y as f64).atan2(x as f64);
            let actual = {
                let actual = atan2_p2_default(I17F15::from_bits(y), I17F15::from_bits(x));
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

    fn test_optimal_constants<T>(exp: u32, expected: Vec<T>)
    where
        T: Debug
            + Display
            + Send
            + Sync
            + AsPrimitive<<T as PrimitivePromotionExt>::PrimitivePromotion>
            + AsPrimitive<f64>
            + AsPrimitive<usize>
            + Bits
            + ConstOne
            + ConstZero
            + PrimInt
            + PrimitivePromotionExt
            + Signed,
        <T as PrimitivePromotionExt>::PrimitivePromotion: PartialOrd + AsPrimitive<T> + Signed,
        RangeInclusive<T>: Iterator<Item = T>,
        i8: AsPrimitive<T>,
    {
        let num = num_cpus::get();
        let mut rng = rand::thread_rng();
        let base: T = 2.as_();
        let k = base.pow(T::BITS - 2 - exp);
        let mut a = (T::ZERO..=k / 4.as_()).collect::<Vec<_>>();

        a.shuffle(&mut rng);

        let cmp = |(a, b): (f64, f64), (c, d)| a.total_cmp(&c).then_with(|| b.total_cmp(&d));

        let atan_expected = crate::atan::tests::make_atan_data(exp);

        let (mut k, max_error, error_sum) = (0..num)
            .into_par_iter()
            .fold(
                || (vec![], f64::INFINITY, f64::INFINITY),
                |(acc, min_max_error, min_error_sum), n| {
                    let search_range = a
                        .iter()
                        .cloned()
                        .skip(a.len() * n / num)
                        .take(a.len() * (n + 1) / num - a.len() * n / num);

                    let (k, max_error, error_sum) = crate::atan::tests::find_optimal_constants(
                        exp,
                        &atan_expected,
                        search_range,
                        |x, x_k, a, k| atan_p2(x, x_k, a, k),
                    );

                    match cmp((max_error, error_sum), (min_max_error, min_error_sum)) {
                        Ordering::Equal => {
                            (acc.into_iter().chain(k).collect(), max_error, error_sum)
                        }
                        Ordering::Less => (k, max_error, error_sum),
                        Ordering::Greater => (acc, min_max_error, min_error_sum),
                    }
                },
            )
            .reduce(
                || (vec![], f64::INFINITY, f64::INFINITY),
                |(lhs, lmax, lsum), (rhs, rmax, rsum)| match cmp((lmax, lsum), (rmax, rsum)) {
                    Ordering::Equal => (lhs.into_iter().chain(rhs).collect(), lmax, lsum),
                    Ordering::Less => (lhs, lmax, lsum),
                    Ordering::Greater => (rhs, rmax, rsum),
                },
            );

        k.sort_unstable();
        assert_eq!(
            expected, k,
            "exp: {exp}, max_error: {max_error}, error_sum: {error_sum}"
        );
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

    // Test as `cargo test -- atan_p2::tests::test_optimal_constants --ignored --nocapture --test-threads=1`
    mod test_optimal_constants {
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
        define_test!(case_01, 30, vec![0]);
        define_test!(case_02, 29, vec![0]);
        define_test!(case_03, 28, vec![0, 1]);
        define_test!(case_04, 27, vec![0, 1]);
        define_test!(case_05, 26, vec![I6F26::A]);
        define_test!(case_06, 25, vec![I7F25::A]);
        define_test!(case_07, 24, vec![I8F24::A]);
        define_test!(case_08, 23, vec![I9F23::A]);
        define_test!(case_09, 5, vec![2917056, 2917057]);
        define_test!(case_10, 4, vec![5835516]);
        define_test!(case_11, 3, vec![11671032, 11671033]);
        define_test!(case_12, 2, vec![23487671]);
        define_test!(case_13, 1, vec![48497950, 48497951]);
    }
}
