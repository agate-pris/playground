use std::cmp::Ordering;

use num_traits::{ConstZero, Signed};

use crate::atan::inv_i32_f15;

fn atan_p3_impl<T>(x: T, one: T, frac_k_4: T, a: T, b: T) -> T
where
    T: Copy + Signed,
{
    let x_abs = x.abs();
    x * (frac_k_4 - (x_abs - one) * (a + x_abs * b / one) / one)
}

pub trait AtanP3Consts<T> {
    const ONE: T;
    const FRAC_K_4: T;
    const A: T;
    const B: T;
    fn calc(x: T) -> T
    where
        T: Copy + Signed,
    {
        atan_p3_impl(x, Self::ONE, Self::FRAC_K_4, Self::A, Self::B)
    }
}

impl AtanP3Consts<i32> for i32 {
    const ONE: i32 = 2_i32.pow(i32::BITS / 2 - 1);
    const FRAC_K_4: i32 = 2_i32.pow(i32::BITS / 2 - 3);
    const A: i32 = 2555;
    const B: i32 = 691;
}

pub trait AtanP3 {
    type Output;
    fn atan_p3(self) -> Self::Output;
    fn atan2_p3(self, other: Self) -> Self::Output;
}

impl AtanP3 for i32 {
    type Output = i32;

    fn atan_p3(self) -> Self::Output {
        const RIGHT: i32 = 2_i32.pow(i32::BITS - 3);
        const NEG_ONE: i32 = -<i32 as AtanP3Consts<i32>>::ONE;

        if self < NEG_ONE {
            const NEG_RIGHT: i32 = -RIGHT;
            NEG_RIGHT - i32::calc(inv_i32_f15(self))
        } else if self > <i32 as AtanP3Consts<i32>>::ONE {
            RIGHT - i32::calc(inv_i32_f15(self))
        } else {
            i32::calc(self)
        }
    }
    fn atan2_p3(self, other: i32) -> Self::Output {
        fn div(a: i32, b: i32) -> i32 {
            (a as i64 * <i32 as AtanP3Consts<i32>>::ONE as i64 / b as i64) as i32
        }

        use Ordering::*;

        const STRAIGHT: i32 = 2_i32.pow(i32::BITS - 2);
        const NEG_STRAIGHT: i32 = -STRAIGHT;
        const RIGHT: i32 = STRAIGHT / 2;
        const NEG_RIGHT: i32 = -RIGHT;

        match (self.cmp(&Self::ZERO), other.cmp(&Self::ZERO)) {
            (Less, Less) => {
                if self < other {
                    let x = div(other, self);
                    NEG_RIGHT - i32::calc(x)
                } else {
                    let x = div(self, other);
                    NEG_STRAIGHT + i32::calc(x)
                }
            }
            (Less, Equal) => NEG_RIGHT,
            (Less, Greater) => {
                if self < -other {
                    let x = div(other, self);
                    NEG_RIGHT - i32::calc(x)
                } else {
                    let x = div(self, other);
                    i32::calc(x)
                }
            }
            (Equal, Less) => STRAIGHT,
            (Greater, Less) => {
                if -self < other {
                    let x = div(other, self);
                    RIGHT - i32::calc(x)
                } else {
                    let x = div(self, other);
                    STRAIGHT + i32::calc(x)
                }
            }
            (Greater, Equal) => RIGHT,
            (Greater, Greater) => {
                if self < other {
                    let x = div(self, other);
                    i32::calc(x)
                } else {
                    let x = div(other, self);
                    RIGHT - i32::calc(x)
                }
            }
            _ => Self::ZERO,
        }
    }
}

/*
impl_atan_p3_default_fixed!(
    I3F5, 0, 0, I2F6, 0, 0, I13F3, 159, 46, I12F4, 80, 26, I11F5, 41, 12, I10F6, 20, 9, I9F7, 9, 8,
    I8F8, 4, 7, I7F9, 2, 5, I6F10, 2, 2, I23F9, 163355, 44265, I22F10, 81678, 22133, I21F11, 40841,
    11064, I20F12, 20421, 5534, I19F13, 10212, 2766, I18F14, 5107, 1383, I17F15, 2555, 691, I16F16,
    1279, 344, I15F17, 640, 173, I14F18, 322, 85, I13F19, 162, 42, I12F20, 79, 26, I11F21, 38, 18,
    I10F22, 18, 13, I9F23, 6, 13, I8F24, 4, 7, I7F25, 2, 5, I6F26, 2, 2
);
*/

#[cfg(test)]
mod tests {
    use std::{
        f64::consts::PI,
        fmt::{Debug, Display},
        ops::RangeInclusive,
    };

    use num_traits::{AsPrimitive, ConstOne, PrimInt};
    use primitive_promotion::PrimitivePromotionExt;
    use rand::prelude::*;
    use rayon::prelude::*;
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
        use Ordering::*;

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
            (calc(0.2447 / PI), calc(0.0663 / PI))
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
                        |x, one, k, ab| atan_p3_impl(x, one, k / 4.as_(), ab.0, ab.1),
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

        k.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
        assert_eq!(
            expected, k,
            "exp: {exp}, max_error: {max_error}, error_sum: {error_sum}"
        );
    }

    #[rstest]
    #[case(2, vec![(3, 0), (3, 1)])]
    #[case(3, vec![(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)])]
    #[case(4, vec![(0, 0), (0, 1), (1, 0), (1, 1)])]
    //#[case(5, vec![(I3F5::A, I3F5::B)])]
    //#[case(6, vec![(I2F6::A, I2F6::B)])]
    fn test_optimal_constants_i8(#[case] exp: u32, #[case] expected: Vec<(i8, i8)>) {
        test_optimal_constants(exp, expected);
    }

    #[rstest]
    #[case(2, vec![(323, 86), (324, 84), (324, 85), (325, 82), (325, 83)])]
    //#[case(3, vec![(I13F3::A, I13F3::B)])]
    //#[case(4, vec![(I12F4::A, I12F4::B)])]
    //#[case(5, vec![(I11F5::A, I11F5::B)])]
    //#[case(6, vec![(I10F6::A, I10F6::B)])]
    //#[case(7, vec![(I9F7::A, I9F7::B)])]
    //#[case(8, vec![(I8F8::A, I8F8::B)])]
    //#[case(9, vec![(I7F9::A, I7F9::B)])]
    //#[case(10, vec![(I6F10::A, I6F10::B)])]
    #[case(11, vec! [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)])]
    #[case(12, vec! [(0, 0), (0, 1), (1, 0), (1, 1)])]
    #[case(13, vec![(0, 0)])]
    #[case(14, vec![(0, 0)])]
    fn test_optimal_constants_i16(#[case] exp: u32, #[case] expected: Vec<(i16, i16)>) {
        test_optimal_constants(exp, expected);
    }

    // Test as `cargo test -- atan_p3::tests::test_optimal_constants --ignored --nocapture --test-threads=1`
    mod test_optimal_constants {
        use super::*;
        macro_rules! define_test {
            ($name:tt, $exp:expr, $expected:expr) => {
                #[test]
                #[ignore]
                fn $name() {
                    test_optimal_constants($exp, $expected);
                }
            };
        }
        define_test!(case_01, 30, vec![(0, 0)]);
        define_test!(case_02, 29, vec![(0, 0)]);
        define_test!(case_03, 28, vec![(0, 0), (0, 1), (1, 0), (1, 1)]);
        define_test!(case_04, 27, vec![(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]);
        //define_test!(case_05, 26, vec![(I6F26::A, I6F26::B)]);
        //define_test!(case_06, 25, vec![(I7F25::A, I7F25::B)]);
        //define_test!(case_07, 24, vec![(I8F24::A, I8F24::B)]);
        //define_test!(case_08, 23, vec![(I9F23::A, I9F23::B)]);
        //define_test!(case_09, 22, vec![(I10F22::A, I10F22::B)]);
        //define_test!(case_10, 21, vec![(I11F21::A, I11F21::B)]);
        //define_test!(case_11, 20, vec![(I12F20::A, I12F20::B)]);
        //define_test!(case_12, 19, vec![(I13F19::A, I13F19::B)]);
        //define_test!(case_13, 18, vec![(I14F18::A, I14F18::B)]);
        //define_test!(case_14, 17, vec![(I15F17::A, I15F17::B)]);
        //define_test!(case_15, 16, vec![(I16F16::A, I16F16::B)]);
        define_test!(case_16, 15, vec![(i32::A, i32::B)]);
        //define_test!(case_17, 14, vec![(I18F14::A, I18F14::B)]);
        //define_test!(case_18, 13, vec![(I19F13::A, I19F13::B)]);
        //define_test!(case_19, 12, vec![(I20F12::A, I20F12::B)]);
        //define_test!(case_20, 11, vec![(I21F11::A, I21F11::B)]);
        //define_test!(case_21, 10, vec![(I22F10::A, I22F10::B)]);
        //define_test!(case_22, 9, vec![(I23F9::A, I23F9::B)]);
    }
}
