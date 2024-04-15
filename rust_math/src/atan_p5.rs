use std::ops::{Add, Div, Mul, Sub};

use fixed::{
    traits::Fixed,
    types::{
        I10F6, I11F5, I12F4, I13F3, I17F15, I18F14, I19F13, I20F12, I21F11, I22F10, I23F9, I9F7,
    },
};
use num_traits::{AsPrimitive, ConstZero, Pow, Signed};
use primitive_promotion::PrimitivePromotionExt;

use crate::atan::{atan2_impl, atan_impl};

pub trait AtanP5Default {
    type Bits;
    const A: Self::Bits;
    const B: Self::Bits;
    const C: Self::Bits;
}

macro_rules! impl_atan_p5_default_fixed_i16 {
    ($($t:ty, $a:expr, $b:expr),*) => {
        $(
            impl AtanP5Default for $t {
                type Bits = <Self as Fixed>::Bits;
                const A: Self::Bits = $a;
                const B: Self::Bits = $b;
                const C: Self::Bits = 2_i16.pow(i16::BITS - 2 - <Self as Fixed>::FRAC_NBITS) / 4 - Self::A + Self::B;
            }
        )*
    };
}

macro_rules! impl_atan_p5_default_fixed_i32 {
    ($($t:ty, $a:expr, $b:expr),*) => {
        $(
            impl AtanP5Default for $t {
                type Bits = <Self as Fixed>::Bits;
                const A: Self::Bits = $a;
                const B: Self::Bits = $b;
                const C: Self::Bits = 2_i32.pow(i32::BITS - 2 - <Self as Fixed>::FRAC_NBITS) / 4 - Self::A + Self::B;
            }
        )*
    };
}

impl_atan_p5_default_fixed_i16!(
    I13F3, 69, 204, I12F4, 27, 93, I11F5, 13, 46, I10F6, 6, 22, I9F7, 5, 14
);

impl_atan_p5_default_fixed_i32!(
    I17F15, 787, 2968, I18F14, 1582, 5947, I19F13, 3169, 11901, I20F12, 6348, 23813, I21F11, 12707,
    47632, I22F10, 25420, 95234, I23F9, 50981, 190506
);

fn atan_p5_impl<T>(x: T, one: T, a: T, b: T, c: T) -> T
where
    T: Copy + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T>,
{
    let x_2 = x * x / one;
    ((a * x_2 / one - b) * x_2 / one + c) * x
}

/// ```rust
/// use std::f64::consts::PI;
/// use approx::assert_abs_diff_eq;
/// use fixed::types::I17F15;
/// use rust_math::atan_p5::*;
/// const EXP: u32 = i32::BITS / 2 - 1;
/// const K: i32 = 2_i32.pow(EXP);
/// let result = atan_p5(1000 * K / 1732, K, I17F15::A, I17F15::B, I17F15::C);
/// assert_abs_diff_eq!(
///     PI / 6.0,
///     result as f64 * PI / K.pow(2) as f64,
///     epsilon = 0.00085,
/// );
/// ```
pub fn atan_p5<T>(x: T, one: T, a: T, b: T, c: T) -> T
where
    T::PrimitivePromotion: PartialOrd + AsPrimitive<T> + Signed,
    T: PartialOrd + AsPrimitive<T::PrimitivePromotion> + PrimitivePromotionExt + Signed,
    i8: AsPrimitive<T>,
{
    atan_impl(x, one, |x| atan_p5_impl(x, one, a, b, c))
}

/// ```rust
/// use std::f64::consts::PI;
/// use approx::assert_abs_diff_eq;
/// use fixed::types::I17F15;
/// use rust_math::atan_p5::*;
/// const EXP: u32 = i32::BITS / 2 - 1;
/// const K: i32 = 2_i32.pow(EXP);
/// let result = atan_p5_default(I17F15::from_bits(1000 * K / 1732));
/// assert_abs_diff_eq!(
///     PI / 6.0,
///     result as f64 * PI / K.pow(2) as f64,
///     epsilon = 0.00085,
/// );
/// ```
pub fn atan_p5_default<T>(x: T) -> <T as Fixed>::Bits
where
    <<T as Fixed>::Bits as PrimitivePromotionExt>::PrimitivePromotion:
        PartialOrd + AsPrimitive<<T as Fixed>::Bits> + Signed,
    <T as Fixed>::Bits: AsPrimitive<<<T as Fixed>::Bits as PrimitivePromotionExt>::PrimitivePromotion>
        + Pow<u32, Output = <T as Fixed>::Bits>
        + PrimitivePromotionExt
        + Signed,
    T: AtanP5Default<Bits = <T as Fixed>::Bits> + Fixed,
    i8: AsPrimitive<<T as Fixed>::Bits>,
{
    let base: <T as Fixed>::Bits = 2.as_();
    let k = base.pow(T::FRAC_NBITS);
    atan_p5(
        x.to_bits(),
        k,
        <T as AtanP5Default>::A,
        <T as AtanP5Default>::B,
        <T as AtanP5Default>::C,
    )
}

/// ```rust
/// use std::f64::consts::PI;
/// use approx::assert_abs_diff_eq;
/// use fixed::types::I17F15;
/// use rust_math::atan_p5::*;
/// const EXP: u32 = i32::BITS / 2 - 1;
/// const K: i32 = 2_i32.pow(EXP);
/// let result = atan2_p5(1000, 1732, K, I17F15::A, I17F15::B, I17F15::C);
/// assert_abs_diff_eq!(
///     PI / 6.0,
///     result as f64 * PI / K.pow(2) as f64,
///     epsilon = 0.00085,
/// );
/// ```
pub fn atan2_p5<T>(y: T, x: T, one: T, a: T, b: T, c: T) -> T
where
    T::PrimitivePromotion: AsPrimitive<T> + PartialOrd + Signed,
    T: AsPrimitive<T::PrimitivePromotion> + ConstZero + PrimitivePromotionExt + Signed,
    i8: AsPrimitive<T>,
{
    atan2_impl(y, x, one, |x| atan_p5_impl(x, one, a, b, c))
}

/// ```rust
/// use std::f64::consts::PI;
/// use approx::assert_abs_diff_eq;
/// use fixed::types::I17F15;
/// use rust_math::atan_p5::*;
/// const EXP: u32 = i32::BITS / 2 - 1;
/// let result = atan2_p5_default(I17F15::from_bits(1000), I17F15::from_bits(1732));
/// assert_abs_diff_eq!(
///     PI / 6.0,
///     result as f64 * PI / 2_i32.pow(2 * EXP) as f64,
///     epsilon = 0.00085,
/// );
/// ```
pub fn atan2_p5_default<T>(y: T, x: T) -> <T as Fixed>::Bits
where
    <<T as Fixed>::Bits as PrimitivePromotionExt>::PrimitivePromotion:
        PartialOrd + AsPrimitive<<T as Fixed>::Bits> + Signed,
    <T as Fixed>::Bits: AsPrimitive<<<T as Fixed>::Bits as PrimitivePromotionExt>::PrimitivePromotion>
        + ConstZero
        + Pow<u32, Output = <T as Fixed>::Bits>
        + PrimitivePromotionExt
        + Signed,
    T: AtanP5Default<Bits = <T as Fixed>::Bits> + Fixed,
    i8: AsPrimitive<<T as Fixed>::Bits>,
{
    let base: <T as Fixed>::Bits = 2.as_();
    let k = base.pow(T::FRAC_NBITS);
    atan2_p5(
        y.to_bits(),
        x.to_bits(),
        k,
        <T as AtanP5Default>::A,
        <T as AtanP5Default>::B,
        <T as AtanP5Default>::C,
    )
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
    use rand::prelude::SliceRandom;
    use rayon::iter::{IntoParallelIterator, ParallelIterator};
    use rstest::rstest;

    use crate::bits::Bits;

    use super::*;

    fn test_optimal_constants<T>(exp: u32, expected: Vec<(T, T)>)
    where
        T::PrimitivePromotion: PartialOrd + AsPrimitive<T> + Signed,
        RangeInclusive<T>: Iterator<Item = T>,
        T: Debug
            + Display
            + Send
            + Sync
            + AsPrimitive<T::PrimitivePromotion>
            + AsPrimitive<f64>
            + AsPrimitive<usize>
            + Bits
            + ConstOne
            + ConstZero
            + PrimitivePromotionExt
            + PrimInt
            + Signed,
        f64: AsPrimitive<T>,
        i8: AsPrimitive<T>,
    {
        let num = num_cpus::get();
        let (a, b) = {
            let mut rng = rand::thread_rng();
            let base: T = 2.as_();
            let k = base.pow(T::BITS - 2 - exp);
            let mut calc = |scale: f64| -> Vec<T> {
                let k_as_f64: f64 = k.as_();
                let v = scale * k_as_f64;
                let first: T = ((scale - 0.05) * k_as_f64).min(v - 1000.0).max(0.0).as_();
                let last: T = ((scale + 0.05) * k_as_f64)
                    .max(v + 1000.0)
                    .min(k_as_f64 / 4.0)
                    .as_();
                println!("first: {}, last: {}", first, last);
                let mut vec = (first..=last).collect::<Vec<_>>();
                vec.shuffle(&mut rng);
                vec
            };
            (calc(0.0776509570923569 / PI), calc(0.287434475393028 / PI))
        };

        let cmp = |(a, b): (f64, f64), (c, d)| a.total_cmp(&c).then_with(|| b.total_cmp(&d));

        let atan_expected = crate::atan::tests::make_atan_data(exp);

        let (mut k, max_error, error_sum) = (0..num)
            .into_par_iter()
            .fold(
                || (vec![], f64::INFINITY, f64::INFINITY),
                |(acc, min_max_error, min_error_sum), n| {
                    let search_range = a
                        .iter()
                        .skip(a.len() * n / num)
                        .take(a.len() * (n + 1) / num - a.len() * n / num)
                        .flat_map(|&a| b.iter().map(move |&b| (a, b)));

                    let (k, max_error, error_sum) = crate::atan::tests::find_optimal_constants(
                        exp,
                        &atan_expected,
                        search_range,
                        |x, one, k, ab| atan_p5(x, one, ab.0, ab.1, k / 4.as_() - ab.0 + ab.1),
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

        k.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
        println!(
            "exp: {exp}, k: {:?}, max_error: {max_error}, error_sum: {error_sum}",
            k
        );
        assert_eq!(expected, k);
    }

    #[rstest]
    #[case(1, vec![(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8)])]
    #[case(2, vec![(1, 2), (2, 3)])]
    #[case(3, vec![(0, 0), (1, 1), (2, 2)])]
    fn test_optimal_constants_i8(#[case] exp: u32, #[case] expected: Vec<(i8, i8)>) {
        test_optimal_constants(exp, expected);
    }

    #[rstest]
    #[case(2, vec![(139, 409), (140, 410)])]
    #[case(3, vec![(I13F3::A, I13F3::B)])]
    #[case(4, vec![(I12F4::A, I12F4::B)])]
    #[case(5, vec![(I11F5::A, I11F5::B)])]
    #[case(6, vec![(I10F6::A, I10F6::B)])]
    #[case(7, vec![(I9F7::A, I9F7::B)])]
    fn test_optimal_constants_i16(#[case] exp: u32, #[case] expected: Vec<(i16, i16)>) {
        test_optimal_constants(exp, expected);
    }

    mod test_optimal_constants_i32 {
        use super::*;
        macro_rules! define_test {
            ($name:ident, $exp:expr, $expected:expr) => {
                #[test]
                #[ignore]
                fn $name() {
                    test_optimal_constants($exp, $expected);
                }
            };
        }
        define_test!(case_01, 15, vec![(I17F15::A, I17F15::B)]);
        define_test!(case_02, 14, vec![(I18F14::A, I18F14::B)]);
        define_test!(case_03, 13, vec![(I19F13::A, I19F13::B)]);
        define_test!(case_04, 12, vec![(I20F12::A, I20F12::B)]);
        define_test!(case_05, 11, vec![(I21F11::A, I21F11::B)]);
        define_test!(case_06, 10, vec![(I22F10::A, I22F10::B)]);
        define_test!(case_07, 9, vec![(I23F9::A, I23F9::B)]);
    }

    #[test]
    fn test_atan_p3_default_trait() {
        assert_eq!(I13F3::A, 69);
        assert_eq!(I13F3::B, 204);
        assert_eq!(I13F3::C, 647);
        assert_eq!(I12F4::A, 27);
        assert_eq!(I12F4::B, 93);
        assert_eq!(I12F4::C, 322);
        assert_eq!(I11F5::A, 13);
        assert_eq!(I11F5::B, 46);
        assert_eq!(I11F5::C, 161);
        assert_eq!(I10F6::A, 6);
        assert_eq!(I10F6::B, 22);
        assert_eq!(I10F6::C, 80);
        assert_eq!(I9F7::A, 5);
        assert_eq!(I9F7::B, 14);
        assert_eq!(I9F7::C, 41);
        assert_eq!(I17F15::A, 787);
        assert_eq!(I17F15::B, 2968);
        assert_eq!(I17F15::C, 10373);
    }
}
