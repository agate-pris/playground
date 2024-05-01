use num_traits::Signed;

use crate::atan::{div_i32_f15, inv_i32_f15, AtanUtil};

macro_rules! atan_p2_impl {
    ($x:ident,$one:expr,$frac_k_4:expr,$a:expr) => {
        $x * (($frac_k_4) + ($a) * (($one) - $x.abs()) / ($one))
    };
}

pub const A_I6F2: i8 = 3;
pub const A_I13F3: i16 = 179;
pub const A_I12F4: i16 = 91;
pub const A_I11F5: i16 = 47;
pub const A_I10F6: i16 = 24;
pub const A_I9F7: i16 = 13;
pub const A_I8F8: i16 = 8;
pub const A_I7F9: i16 = 5;
pub const A_I6F10: i16 = 3;
pub const A_I26F6: i32 = 1458371;
pub const A_I25F7: i32 = 729188;
pub const A_I24F8: i32 = 364590;
pub const A_I23F9: i32 = 182295;
pub const A_I22F10: i32 = 91148;
pub const A_I21F11: i32 = 45575;
pub const A_I20F12: i32 = 22789;
pub const A_I19F13: i32 = 11395;
pub const A_I18F14: i32 = 5699;
pub const A_I17F15: i32 = 2850;
pub const A_I16F16: i32 = 1426;
pub const A_I15F17: i32 = 714;
pub const A_I14F18: i32 = 358;
pub const A_I13F19: i32 = 180;
pub const A_I12F20: i32 = 91;
pub const A_I11F21: i32 = 47;
pub const A_I10F22: i32 = 25;
pub const A_I9F23: i32 = 14;
pub const A_I8F24: i32 = 8;
pub const A_I7F25: i32 = 5;
pub const A_I6F26: i32 = 3;

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
    const A: i32 = A_I17F15;
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
        fmt::{Debug, Display},
        ops::RangeInclusive,
    };

    use num_traits::{AsPrimitive, ConstOne, ConstZero, PrimInt};
    use primitive_promotion::PrimitivePromotionExt;
    use rand::prelude::*;
    use rayon::prelude::*;
    use rstest::rstest;

    use crate::{atan::tests::compare_error, bits::Bits};

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
            let base: T = 2.as_();
            let quarter: T = base.pow(T::BITS - 2 - exp - 2);
            let mut v = (0.as_()..=quarter).collect::<Vec<_>>();
            v.shuffle(&mut rand::thread_rng());
            v
        };

        let atan_expected = crate::atan::tests::make_atan_data(exp);

        let (mut k, max_error, error_sum) = (0..num)
            .into_par_iter()
            .map(|n| {
                let search_range = a
                    .iter()
                    .cloned()
                    .skip(a.len() * n / num)
                    .take(a.len() * (n + 1) / num - a.len() * n / num);

                crate::atan::tests::find_optimal_constants(
                    exp,
                    &atan_expected,
                    search_range,
                    |x, one, k, a| atan_p2_impl!(x, one, k / 4.as_(), a),
                )
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

    #[rstest]
    #[case(1, vec![2, 3])]
    #[case(2, vec![A_I6F2])]
    #[case(3, vec![0, 1])]
    fn test_optimal_constants_i8(#[case] exp: u32, #[case] expected: Vec<i8>) {
        test_optimal_constants(exp, expected);
    }

    #[rstest]
    #[case(2, vec![360, 361])]
    #[case(3, vec![A_I13F3])]
    #[case(4, vec![A_I12F4])]
    #[case(5, vec![A_I11F5])]
    #[case(6, vec![A_I10F6])]
    #[case(7, vec![A_I9F7])]
    #[case(8, vec![A_I8F8])]
    #[case(9, vec![A_I7F9])]
    #[case(10, vec![A_I6F10])]
    #[case(11, vec![0, 1])]
    fn test_optimal_constants_i16(#[case] exp: u32, #[case] expected: Vec<i16>) {
        test_optimal_constants(exp, expected);
    }

    #[rstest]
    #[case(5, vec![2917056, 2917057])]
    #[case(6, vec![A_I26F6])]
    #[case(7, vec![A_I25F7])]
    #[case(8, vec![A_I24F8])]
    #[case(9, vec![A_I23F9])]
    #[case(10, vec![A_I22F10])]
    #[case(11, vec![A_I21F11])]
    #[case(12, vec![A_I20F12])]
    #[case(13, vec![A_I19F13])]
    #[case(14, vec![A_I18F14])]
    #[case(15, vec![A_I17F15])]
    #[case(16, vec![A_I16F16])]
    #[case(17, vec![A_I15F17])]
    #[case(18, vec![A_I14F18])]
    #[case(19, vec![A_I13F19])]
    #[case(20, vec![A_I12F20])]
    #[case(21, vec![A_I11F21])]
    #[case(22, vec![A_I10F22])]
    #[case(23, vec![A_I9F23])]
    #[case(24, vec![A_I8F24])]
    #[case(25, vec![A_I7F25])]
    #[case(26, vec![A_I6F26])]
    #[case(27, vec![0, 1])]
    fn test_optimal_constants_i32(#[case] exp: u32, #[case] expected: Vec<i32>) {
        test_optimal_constants(exp, expected);
    }
}
