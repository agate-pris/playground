use std::{
    cmp::Ordering,
    ops::{Add, Div, Mul, Sub},
};

use num_traits::ConstZero;

use crate::atan::inv_i32_f15;

fn atan_p5_impl<T>(x: T, one: T, a: T, b: T, c: T) -> T
where
    T: Copy + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T>,
{
    let x_2 = x * x / one;
    ((a * x_2 / one - b) * x_2 / one + c) * x
}

pub trait AtanP5Consts<T> {
    const ONE: T;
    const A: T;
    const B: T;
    const C: T;
    fn calc(x: T) -> T
    where
        T: Copy + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T>,
    {
        atan_p5_impl(x, Self::ONE, Self::A, Self::B, Self::C)
    }
}

impl AtanP5Consts<i32> for i32 {
    const ONE: i32 = 2_i32.pow(i32::BITS / 2 - 1);
    const A: i32 = 787;
    const B: i32 = 2968;
    const C: i32 = 2_i32.pow(i32::BITS / 2 - 3) + Self::B - Self::A;
}

pub trait AtanP5 {
    type Output;
    fn atan_p5(self) -> Self::Output;
    fn atan2_p5(self, other: Self) -> Self::Output;
}

impl AtanP5 for i32 {
    type Output = i32;

    fn atan_p5(self) -> Self::Output {
        const RIGHT: i32 = 2_i32.pow(i32::BITS - 3);
        const NEG_ONE: i32 = -<i32 as AtanP5Consts<i32>>::ONE;

        if self < NEG_ONE {
            const NEG_RIGHT: i32 = -RIGHT;
            NEG_RIGHT - i32::calc(inv_i32_f15(self))
        } else if self > <i32 as AtanP5Consts<i32>>::ONE {
            RIGHT - i32::calc(inv_i32_f15(self))
        } else {
            i32::calc(self)
        }
    }
    fn atan2_p5(self, other: i32) -> Self::Output {
        fn div(a: i32, b: i32) -> i32 {
            (a as i64 * <i32 as AtanP5Consts<i32>>::ONE as i64 / b as i64) as i32
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
impl_atan_p5_default_fixed_i16!(
    I13F3, 69, 204, I12F4, 27, 93, I11F5, 13, 46, I10F6, 6, 22, I9F7, 5, 14
);

impl_atan_p5_default_fixed_i32!(
    I17F15, 787, 2968, I18F14, 1582, 5947, I19F13, 3169, 11901, I20F12, 6348, 23813, I21F11, 12707,
    47632, I22F10, 25420, 95234, I23F9, 50981, 190506
);
*/

#[cfg(test)]
mod tests {
    use std::{
        f64::consts::PI,
        fmt::{Debug, Display},
        ops::RangeInclusive,
    };

    use num_traits::{AsPrimitive, ConstOne, PrimInt, Signed};
    use primitive_promotion::PrimitivePromotionExt;
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
                        |x, one, k, ab| atan_p5_impl(x, one, ab.0, ab.1, k / 4.as_() - ab.0 + ab.1),
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
    //#[case(3, vec![(I13F3::A, I13F3::B)])]
    //#[case(4, vec![(I12F4::A, I12F4::B)])]
    //#[case(5, vec![(I11F5::A, I11F5::B)])]
    //#[case(6, vec![(I10F6::A, I10F6::B)])]
    //#[case(7, vec![(I9F7::A, I9F7::B)])]
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
        define_test!(case_01, 15, vec![(i32::A, i32::B)]);
        //define_test!(case_02, 14, vec![(I18F14::A, I18F14::B)]);
        //define_test!(case_03, 13, vec![(I19F13::A, I19F13::B)]);
        //define_test!(case_04, 12, vec![(I20F12::A, I20F12::B)]);
        //define_test!(case_05, 11, vec![(I21F11::A, I21F11::B)]);
        //define_test!(case_06, 10, vec![(I22F10::A, I22F10::B)]);
        //define_test!(case_07, 9, vec![(I23F9::A, I23F9::B)]);
    }
}
