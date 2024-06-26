use crate::atan::{div_i32_f15, inv_i32_f15, AtanUtil};

/// x * (0.25 + (A * (1 - x.abs)))
macro_rules! atan_p2_impl {
    ($x:ident,$one:expr,$one_exp:expr,$frac_k_4:expr,$a:expr) => {
        $x * (($frac_k_4) + (($a) * (($one) - $x.abs()) >> ($one_exp)))
    };
}

#[cfg(test)]
const A_I6F2: i8 = 3;

#[cfg(test)]
const A_I16: [i16; 8] = [3, 5, 8, 13, 24, 47, 91, 179];

const A_I32: [i32; 21] = [
    3, 5, 8, 14, 25, 47, 91, 180, 358, 714, 1426, 2850, 5699, 11395, 22789, 45575, 91148, 182295,
    364590, 729188, 1458371,
];

struct AtanP2I32();

impl AtanP2I32 {
    const ONE_EXP: u32 = i32::BITS / 2 - 1;
}

impl AtanUtil<i32> for AtanP2I32 {
    type Output = i32;
    const ONE: i32 = 1 << Self::ONE_EXP;
    const STRAIGHT: i32 = 2_i32.pow(i32::BITS - 2);
    const RIGHT: i32 = Self::STRAIGHT / 2;
    const NEG_ONE: i32 = -Self::ONE;
    const NEG_RIGHT: i32 = -Self::RIGHT;
    fn inv(x: i32) -> i32 {
        inv_i32_f15(x)
    }
    fn div(x: i32, y: i32) -> i32 {
        div_i32_f15(x, y)
    }
    fn calc(x: i32) -> i32 {
        const FRAC_K_4: i32 = 1 << (i32::BITS / 2 - 3);
        const A: i32 = A_I32[11];
        atan_p2_impl!(x, Self::ONE, Self::ONE_EXP, FRAC_K_4, A)
    }
}

pub fn atan_p2_2850(x: i32) -> i32 {
    AtanP2I32::atan(x)
}

pub fn atan2_p2_2850(y: i32, x: i32) -> i32 {
    AtanP2I32::atan2(y, x)
}

#[cfg(test)]
mod tests {
    use std::{
        cmp::Ordering,
        fmt::{Debug, Display},
        ops::{RangeInclusive, Shr},
    };

    use num_traits::{AsPrimitive, ConstOne, ConstZero, PrimInt, Signed};
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
            + Shr<u32, Output = T>
            + Sync
            + AsPrimitive<f64>
            + AsPrimitive<usize>
            + Bits
            + ConstOne
            + ConstZero
            + PrimInt
            + PrimitivePromotionExt
            + Signed,
        RangeInclusive<T>: Iterator<Item = T>,
        f64: AsPrimitive<T>,
        i8: AsPrimitive<T>,
    {
        use Ordering::*;

        let (one, frac_k_4, a) = {
            let base: T = 2.as_();
            let frac_k_4: T = base.pow(T::BITS - 2 - exp) / 4.as_();
            let mut v = (0.as_()..=frac_k_4).collect::<Vec<_>>();
            v.shuffle(&mut rand::thread_rng());
            (base.pow(exp), frac_k_4, v)
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

                find_optimal_constants(exp, &atan_expected, search_range, |x, a| {
                    atan_p2_impl!(x, one, exp, frac_k_4, a)
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

    #[allow(dead_code)]
    fn test_optimal_constants_i8() {
        [
            (3, Vec::<i8>::from([0, 1])),
            (2, vec![A_I6F2]),
            (1, Vec::<i8>::from([2, 3])),
        ]
        .into_iter()
        .for_each(|(exp, expected)| test_optimal_constants(exp, expected));
    }

    #[allow(dead_code)]
    fn test_optimal_constants_i16() {
        test_optimal_constants(11, Vec::<i16>::from([0, 1]));
        for (i, &a) in A_I16.iter().enumerate() {
            test_optimal_constants(10 - i as u32, vec![a]);
        }
        test_optimal_constants(2, Vec::<i16>::from([360, 361]));
    }

    #[rstest]
    /*
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
    */
    #[case(15, vec![A_I32[11]])]
    /*
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
    */
    fn test_optimal_constants_i32(#[case] exp: u32, #[case] expected: Vec<i32>) {
        test_optimal_constants(exp, expected);
    }
}
