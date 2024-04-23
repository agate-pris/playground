use num_traits::Signed;

use crate::atan::{div_i32_f15, inv_i32_f15, AtanUtil};

fn atan_p2_impl<T>(x: T, one: T, frac_k_4: T, a: T) -> T
where
    T: Copy + Signed,
{
    x * (frac_k_4 + a * (one - x.abs()) / one)
}

pub trait AtanP2Consts<T> {
    const ONE: T;
    const FRAC_K_4: T;
    const A: T;
    fn calc(x: T) -> T
    where
        T: Copy + Signed,
    {
        atan_p2_impl(x, Self::ONE, Self::FRAC_K_4, Self::A)
    }
}

struct AtanP2ConstsI32();

impl AtanP2Consts<i32> for AtanP2ConstsI32 {
    const ONE: i32 = 2_i32.pow(i32::BITS / 2 - 1);
    const FRAC_K_4: i32 = 2_i32.pow(i32::BITS / 2 - 3);
    const A: i32 = 2850;
}

struct AtanP2I32Util();

impl AtanUtil<i32> for AtanP2I32Util {
    type Output = i32;
    const ONE: i32 = 2_i32.pow(i32::BITS / 2 - 1);
    const NEG_ONE: i32 = -Self::ONE;
    const RIGHT: i32 = Self::STRAIGHT / 2;
    const NEG_RIGHT: i32 = -Self::RIGHT;
    const STRAIGHT: i32 = 2_i32.pow(i32::BITS - 2);
    const NEG_STRAIGHT: i32 = -Self::STRAIGHT;
    fn inv(x: i32) -> i32 {
        inv_i32_f15(x)
    }
    fn div(x: i32, y: i32) -> i32 {
        div_i32_f15(x, y)
    }
    fn calc(x: i32) -> i32 {
        AtanP2ConstsI32::calc(x)
    }
}

pub trait AtanP2 {
    type Output;
    fn atan_p2(self) -> Self::Output;
    fn atan2_p2(self, other: Self) -> Self::Output;
}

impl AtanP2 for i32 {
    type Output = i32;

    fn atan_p2(self) -> Self::Output {
        AtanP2I32Util::atan(self)
    }
    fn atan2_p2(self, other: i32) -> Self::Output {
        AtanP2I32Util::atan2(self, other)
    }
}

#[cfg(feature = "fixed")]
use fixed::types::{I17F15, I2F30};

#[cfg(feature = "fixed")]
struct AtanP2I17F15Util();

#[cfg(feature = "fixed")]
impl AtanUtil<I17F15> for AtanP2I17F15Util {
    type Output = I2F30;
    const ONE: I17F15 = I17F15::ONE;
    const NEG_ONE: I17F15 = I17F15::NEG_ONE;
    const RIGHT: Self::Output = I2F30::from_bits(Self::STRAIGHT.to_bits() / 2);
    const NEG_RIGHT: Self::Output = I2F30::from_bits(-Self::RIGHT.to_bits());
    const STRAIGHT: Self::Output = I2F30::from_bits(1 << 30);
    const NEG_STRAIGHT: Self::Output = I2F30::from_bits(-Self::STRAIGHT.to_bits());
    fn inv(x: I17F15) -> I17F15 {
        x.recip()
    }
    fn div(a: I17F15, b: I17F15) -> I17F15 {
        a / b
    }
    fn calc(x: I17F15) -> Self::Output {
        I2F30::from_bits(AtanP2ConstsI32::calc(x.to_bits()))
    }
}

#[cfg(feature = "fixed")]
impl AtanP2 for I17F15 {
    type Output = I2F30;

    fn atan_p2(self) -> Self::Output {
        AtanP2I17F15Util::atan(self)
    }
    fn atan2_p2(self, other: I17F15) -> Self::Output {
        AtanP2I17F15Util::atan2(self, other)
    }
}

/*
impl_atan_p2_default_fixed!(
    I3F5, 0, I2F6, 0, I13F3, 179, I12F4, 91, I11F5, 47, I10F6, 24, I9F7, 13, I8F8, 8, I7F9, 5,
    I6F10, 3, I26F6, 1458371, I25F7, 729188, I24F8, 364590, I23F9, 182295, I22F10, 91148, I21F11,
    45575, I20F12, 22789, I19F13, 11395, I18F14, 5699, I17F15, 2850, I16F16, 1426, I15F17, 714,
    I14F18, 358, I13F19, 180, I12F20, 91, I11F21, 47, I10F22, 25, I9F23, 14, I8F24, 8, I7F25, 5,
    I6F26, 3
);
*/

#[cfg(test)]
mod tests {
    use std::{
        cmp::Ordering,
        f64::consts::PI,
        fmt::{Debug, Display},
        ops::RangeInclusive,
    };

    use num_traits::{AsPrimitive, ConstOne, ConstZero, PrimInt};
    use primitive_promotion::PrimitivePromotionExt;
    use rand::prelude::*;
    use rayon::prelude::*;
    use rstest::rstest;

    use crate::bits::Bits;

    use super::*;

    fn test_optimal_constants<T>(exp: u32, expected: Vec<T>)
    where
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
            + PrimInt
            + PrimitivePromotionExt
            + Signed,
        T::PrimitivePromotion: PartialOrd + AsPrimitive<T> + Signed,
        RangeInclusive<T>: Iterator<Item = T>,
        f64: AsPrimitive<T>,
        i8: AsPrimitive<T>,
    {
        use Ordering::*;

        let num = num_cpus::get();
        let a = {
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
            calc(0.273 / PI)
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
                        .cloned()
                        .skip(a.len() * n / num)
                        .take(a.len() * (n + 1) / num - a.len() * n / num);

                    let (k, max_error, error_sum) = crate::atan::tests::find_optimal_constants(
                        exp,
                        &atan_expected,
                        search_range,
                        |x, one, k, a| atan_p2_impl(x, one, k / 4.as_(), a),
                    );

                    match cmp((max_error, error_sum), (min_max_error, min_error_sum)) {
                        Equal => (acc.into_iter().chain(k).collect(), max_error, error_sum),
                        Less => (k, max_error, error_sum),
                        Greater => (acc, min_max_error, min_error_sum),
                    }
                },
            )
            .reduce(
                || (vec![], f64::INFINITY, f64::INFINITY),
                |(lhs, lmax, lsum), (rhs, rmax, rsum)| match cmp((lmax, lsum), (rmax, rsum)) {
                    Equal => (lhs.into_iter().chain(rhs).collect(), lmax, lsum),
                    Less => (lhs, lmax, lsum),
                    Greater => (rhs, rmax, rsum),
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
    //#[case(5, vec![I3F5::A])]
    //#[case(6, vec![I2F6::A])]
    fn test_optimal_constants_i8(#[case] exp: u32, #[case] expected: Vec<i8>) {
        test_optimal_constants(exp, expected);
    }

    #[rstest]
    #[case(1, vec![740, 741])]
    #[case(2, vec![360, 361])]
    //#[case(3, vec![I13F3::A])]
    //#[case(4, vec![I12F4::A])]
    //#[case(5, vec![I11F5::A])]
    //#[case(6, vec![I10F6::A])]
    //#[case(7, vec![I9F7::A])]
    //#[case(8, vec![I8F8::A])]
    //#[case(9, vec![I7F9::A])]
    //#[case(10, vec![I6F10::A])]
    #[case(11, vec![0, 1] )]
    #[case(12, vec![0, 1] )]
    #[case(13, vec![0])]
    #[case(14, vec![0])]
    fn test_optimal_constants_i16(#[case] exp: u32, #[case] expected: Vec<i16>) {
        test_optimal_constants(exp, expected);
    }

    #[rstest]
    //#[case(6, vec![I26F6::A])]
    //#[case(7, vec![I25F7::A])]
    //#[case(8, vec![I24F8::A])]
    //#[case(9, vec![I23F9::A])]
    //#[case(10, vec![I22F10::A])]
    //#[case(11, vec![I21F11::A])]
    //#[case(12, vec![I20F12::A])]
    //#[case(13, vec![I19F13::A])]
    //#[case(14, vec![I18F14::A])]
    #[case(15, vec![AtanP2ConstsI32::A])]
    //#[case(16, vec![I16F16::A])]
    //#[case(17, vec![I15F17::A])]
    //#[case(18, vec![I14F18::A])]
    //#[case(19, vec![I13F19::A])]
    //#[case(20, vec![I12F20::A])]
    //#[case(21, vec![I11F21::A])]
    //#[case(22, vec![I10F22::A])]
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
        //define_test!(case_05, 26, vec![I6F26::A]);
        //define_test!(case_06, 25, vec![I7F25::A]);
        //define_test!(case_07, 24, vec![I8F24::A]);
        //define_test!(case_08, 23, vec![I9F23::A]);
        define_test!(case_09, 5, vec![2917056, 2917057]);
        define_test!(case_10, 4, vec![5835516]);
        define_test!(case_11, 3, vec![11671032, 11671033]);
        define_test!(case_12, 2, vec![23487671]);
        define_test!(case_13, 1, vec![48497950, 48497951]);
    }
}
