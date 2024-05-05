use std::ops::{Add, Div, Mul, Sub};

use crate::atan::{div_i32_f15, inv_i32_f15, AtanUtil};

macro_rules! atan_p5_impl {
    ($x:ident,$one:expr,$a:expr,$b:expr,$c:expr) => {{
        let x_2 = $x * $x / ($one);
        ((($a) * x_2 / ($one) - ($b)) * x_2 / ($one) + ($c)) * $x
    }};
}

pub const A_B_I16: [(i16, i16); 5] = [(5, 14), (6, 22), (13, 46), (27, 93), (69, 204)];

pub const A_B_I32: [(i32, i32); 7] = [
    (787, 2968),
    (1582, 5947),
    (3169, 11901),
    (6348, 23813),
    (12707, 47632),
    (25420, 95234),
    (50981, 190506),
];

pub trait AtanP5Consts<T> {
    const ONE: T;
    const A: T;
    const B: T;
    const C: T;
    fn calc(x: T) -> T
    where
        T: Copy + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T>,
    {
        atan_p5_impl!(x, Self::ONE, Self::A, Self::B, Self::C)
    }
}

struct AtanP5ConstsI32();

impl AtanP5Consts<i32> for AtanP5ConstsI32 {
    const ONE: i32 = 2_i32.pow(i32::BITS / 2 - 1);
    const A: i32 = A_B_I32[0].0;
    const B: i32 = A_B_I32[0].1;
    const C: i32 = 2_i32.pow(i32::BITS / 2 - 3) + Self::B - Self::A;
}

pub(crate) struct AtanP5I32Util();

impl AtanUtil<i32> for AtanP5I32Util {
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
        AtanP5ConstsI32::calc(x)
    }
}

pub trait AtanP5 {
    type Output;
    fn atan_p5(self) -> Self::Output;
    fn atan2_p5(self, other: Self) -> Self::Output;
}

impl AtanP5 for i32 {
    type Output = i32;

    fn atan_p5(self) -> Self::Output {
        AtanP5I32Util::atan(self)
    }
    fn atan2_p5(self, other: i32) -> Self::Output {
        AtanP5I32Util::atan2(self, other)
    }
}

#[cfg(test)]
mod tests {
    use std::{
        cmp::Ordering,
        fmt::{Debug, Display},
        ops::RangeInclusive,
    };

    use num_traits::{AsPrimitive, ConstOne, ConstZero, PrimInt, Signed};
    use primitive_promotion::PrimitivePromotionExt;
    use rand::prelude::SliceRandom;
    use rayon::iter::{IntoParallelIterator, ParallelIterator};
    use rstest::rstest;

    use crate::{
        atan::tests::{compare_error, find_optimal_constants},
        bits::Bits,
    };

    use super::*;

    fn test_optimal_constants<T>(exp: u32, expected: Vec<(T, T)>)
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

        let (a, b) = {
            let mut rng = rand::thread_rng();
            let base: T = 2.as_();
            let quarter = base.pow(T::BITS - 2 - exp) / 4.as_();
            let mut a = (0.as_()..=quarter).collect::<Vec<_>>();
            let mut b = a.clone();
            a.shuffle(&mut rng);
            b.shuffle(&mut rng);
            (a, b)
        };

        let num = num_cpus::get();
        let atan_expected = crate::atan::tests::make_atan_data(exp);

        let (mut k, max_error, error_sum) = (0..num)
            .into_par_iter()
            .map(|n| {
                let search_range = a
                    .iter()
                    .skip(a.len() * n / num)
                    .take(a.len() * (n + 1) / num - a.len() * n / num)
                    .flat_map(|&a| b.iter().map(move |&b| (a, b)));

                find_optimal_constants(exp, &atan_expected, search_range, |x, one, k, ab| {
                    atan_p5_impl!(x, one, ab.0, ab.1, k / 4.as_() - ab.0 + ab.1)
                })
            })
            .reduce(
                || (vec![], f64::INFINITY, f64::INFINITY),
                |(lhs, lmax, lsum), (rhs, rmax, rsum)| match compare_error(lmax, lsum, rmax, rsum) {
                    Equal => (lhs.into_iter().chain(rhs).collect(), lmax, lsum),
                    Less => (lhs, lmax, lsum),
                    Greater => (rhs, rmax, rsum),
                },
            );

        println!("exp: {exp}, max_error: {max_error}, error_sum: {error_sum}",);
        k.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
        assert_eq!(expected, k);
    }

    #[rstest]
    #[case(1, vec![(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8)])]
    #[case(2, vec![(1, 2), (2, 3)])]
    #[case(3, vec![(0, 0), (1, 1), (2, 2)])]
    fn test_optimal_constants_i8(#[case] exp: u32, #[case] expected: Vec<(i8, i8)>) {
        test_optimal_constants(exp, expected);
    }

    #[test]
    fn test_optimal_constants_i16() {
        for (i, &(a, b)) in A_B_I16.iter().enumerate() {
            test_optimal_constants(7 - i as u32, vec![(a, b)]);
        }
        let expected: Vec<(i16, i16)> = vec![(139, 409), (140, 410)];
        test_optimal_constants(2, expected);
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
        define_test!(case_15, 15, vec![(A_B_I32[0])]);
        define_test!(case_14, 14, vec![(A_B_I32[1])]);
        define_test!(case_13, 13, vec![(A_B_I32[2])]);
        define_test!(case_12, 12, vec![(A_B_I32[3])]);
        define_test!(case_11, 11, vec![(A_B_I32[4])]);
        define_test!(case_10, 10, vec![(A_B_I32[5])]);
        define_test!(case_09, 9, vec![(A_B_I32[6])]);
    }
}
