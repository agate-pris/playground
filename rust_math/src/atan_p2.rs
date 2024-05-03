use num_traits::Signed;

use crate::atan::{div_i32_f15, inv_i32_f15, AtanUtil};

macro_rules! atan_p2_impl {
    ($x:ident,$one:expr,$frac_k_4:expr,$a:expr) => {
        $x * (($frac_k_4) + ($a) * (($one) - $x.abs()) / ($one))
    };
}

pub const A_I6F2: i8 = 3;
pub const A_I16: [i16; 8] = [3, 5, 8, 13, 24, 47, 91, 179];
pub const A_I32: [i32; 21] = [
    3, 5, 8, 14, 25, 47, 91, 180, 358, 714, 1426, 2850, 5699, 11395, 22789, 45575, 91148, 182295,
    364590, 729188, 1458371,
];

pub trait AtanP2Consts<T> {
    const ONE: T;
    const FRAC_K_4: T;
    const A: T;
    fn calc(x: T) -> T
    where
        T: Copy + Signed,
    {
        atan_p2_impl!(x, Self::ONE, Self::FRAC_K_4, Self::A)
    }
}

struct AtanP2ConstsI32();

impl AtanP2Consts<i32> for AtanP2ConstsI32 {
    const ONE: i32 = 2_i32.pow(i32::BITS / 2 - 1);
    const FRAC_K_4: i32 = 2_i32.pow(i32::BITS / 2 - 3);
    const A: i32 = A_I32[11];
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

#[cfg(test)]
mod tests {
    use std::{
        cmp::Ordering,
        fmt::{Debug, Display},
        ops::RangeInclusive,
    };

    use num_traits::{AsPrimitive, ConstOne, ConstZero, PrimInt};
    use primitive_promotion::PrimitivePromotionExt;
    use rand::prelude::*;
    use rayon::prelude::*;
    use rstest::rstest;

    use crate::{
        atan::tests::{compare_error, find_optimal_constants},
        bits::Bits,
    };

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

        let a = {
            let base: T = 2.as_();
            let quarter: T = base.pow(T::BITS - 2 - exp) / 4.as_();
            let mut v = (0.as_()..=quarter).collect::<Vec<_>>();
            v.shuffle(&mut rand::thread_rng());
            v
        };

        let num = num_cpus::get();
        let atan_expected = crate::atan::tests::make_atan_data(exp);

        let (mut k, max_error, error_sum) = (0..num)
            .into_par_iter()
            .map(|n| {
                let search_range = a
                    .iter()
                    .cloned()
                    .skip(a.len() * n / num)
                    .take(a.len() * (n + 1) / num - a.len() * n / num);

                find_optimal_constants(exp, &atan_expected, search_range, |x, one, k, a| {
                    atan_p2_impl!(x, one, k / 4.as_(), a)
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

        println!("exp: {exp}, max_error: {max_error}, error_sum: {error_sum}");
        k.sort_unstable();
        assert_eq!(expected, k);
    }

    #[test]
    fn test_optimal_constants_i8() {
        test_optimal_constants(3, Vec::<i8>::from([0, 1]));
        test_optimal_constants(2, vec![A_I6F2]);
        test_optimal_constants(1, Vec::<i8>::from([2, 3]));
    }

    #[test]
    fn test_optimal_constants_i16() {
        test_optimal_constants(11, Vec::<i16>::from([0, 1]));
        for (i, &a) in A_I16.iter().enumerate() {
            test_optimal_constants(10 - i as u32, vec![a]);
        }
        test_optimal_constants(2, Vec::<i16>::from([360, 361]));
    }

    #[rstest]
    #[case(27, vec![0, 1])]
    #[case(26, vec![A_I32[0]])]
    #[case(25, vec![A_I32[1]])]
    #[case(24, vec![A_I32[2]])]
    #[case(23, vec![A_I32[3]])]
    #[case(22, vec![A_I32[4]])]
    #[case(21, vec![A_I32[5]])]
    #[case(20, vec![A_I32[6]])]
    #[case(19, vec![A_I32[7]])]
    #[case(18, vec![A_I32[8]])]
    #[case(17, vec![A_I32[9]])]
    #[case(16, vec![A_I32[10]])]
    #[case(15, vec![A_I32[11]])]
    #[case(14, vec![A_I32[12]])]
    #[case(13, vec![A_I32[13]])]
    #[case(12, vec![A_I32[14]])]
    #[case(11, vec![A_I32[15]])]
    #[case(10, vec![A_I32[16]])]
    #[case(9, vec![A_I32[17]])]
    #[case(8, vec![A_I32[18]])]
    #[case(7, vec![A_I32[19]])]
    #[case(6, vec![A_I32[20]])]
    #[case(5, vec![2917056, 2917057])]
    fn test_optimal_constants_i32(#[case] exp: u32, #[case] expected: Vec<i32>) {
        test_optimal_constants(exp, expected);
    }
}
