use num_traits::Signed;

use crate::atan::{div_i32_f15, inv_i32_f15, AtanUtil};

macro_rules! atan_p3_impl {
    ($x:ident,$one:expr,$frac_k_4:expr,$a:expr,$b:expr) => {{
        let x_abs = $x.abs();
        $x * (($frac_k_4) - (x_abs - ($one)) * (($a) + x_abs * ($b) / ($one)) / ($one))
    }};
}

pub const A_B_I8: [(i8, i8); 2] = [(0, 0), (0, 0)];

pub const A_B_I16: [(i16, i16); 8] = [
    (2, 2),
    (2, 5),
    (4, 7),
    (9, 8),
    (20, 9),
    (41, 12),
    (80, 26),
    (159, 46),
];

pub const A_B_I32: [(i32, i32); 18] = [
    (2, 2),
    (2, 5),
    (4, 7),
    (6, 13),
    (18, 13),
    (38, 18),
    (79, 26),
    (162, 42),
    (322, 85),
    (640, 173),
    (1279, 344),
    (2555, 691),
    (5107, 1383),
    (10212, 2766),
    (20421, 5534),
    (40841, 11064),
    (81678, 22133),
    (163355, 44265),
];

pub trait AtanP3Consts<T> {
    const ONE: T;
    const FRAC_K_4: T;
    const A: T;
    const B: T;
    fn calc(x: T) -> T
    where
        T: Copy + Signed,
    {
        atan_p3_impl!(x, Self::ONE, Self::FRAC_K_4, Self::A, Self::B)
    }
}

struct AtanP3ConstsI32();

impl AtanP3Consts<i32> for AtanP3ConstsI32 {
    const ONE: i32 = 2_i32.pow(i32::BITS / 2 - 1);
    const FRAC_K_4: i32 = 2_i32.pow(i32::BITS / 2 - 3);
    const A: i32 = A_B_I32[11].0;
    const B: i32 = A_B_I32[11].1;
}

struct AtanP3I32Util();

impl AtanUtil<i32> for AtanP3I32Util {
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
        AtanP3ConstsI32::calc(x)
    }
}

pub trait AtanP3 {
    type Output;
    fn atan_p3(self) -> Self::Output;
    fn atan2_p3(self, other: Self) -> Self::Output;
}

impl AtanP3 for i32 {
    type Output = i32;

    fn atan_p3(self) -> Self::Output {
        AtanP3I32Util::atan(self)
    }
    fn atan2_p3(self, other: i32) -> Self::Output {
        AtanP3I32Util::atan2(self, other)
    }
}

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

    use crate::{atan::tests::compare_error, bits::Bits};

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

        let atan_expected = crate::atan::tests::make_atan_data(exp);

        let (mut k, max_error, error_sum) = (0..num)
            .into_par_iter()
            .map(|n| {
                let search_range = a
                    .iter()
                    .skip(a.len() * n / num)
                    .take(a.len() * (n + 1) / num - a.len() * n / num)
                    .flat_map(|&a| b.iter().map(move |&b| (a, b)));

                crate::atan::tests::find_optimal_constants(
                    exp,
                    &atan_expected,
                    search_range,
                    |x, one, k, ab| atan_p3_impl!(x, one, k / 4.as_(), ab.0, ab.1),
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
        k.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
        assert_eq!(expected, k);
    }

    #[test]
    fn test_optimal_constants_i8() {
        for (i, &(a, b)) in A_B_I8.iter().enumerate() {
            test_optimal_constants(6 - i as u32, vec![(a, b)]);
        }
        let expected: Vec<(i8, i8)> = vec![(0, 0), (0, 1), (1, 0), (1, 1)];
        test_optimal_constants(4, expected);
    }

    #[test]
    fn test_optimal_constants_i16() {
        let expected: Vec<(i16, i16)> = vec![(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)];
        test_optimal_constants(11, expected);
        for (i, &(a, b)) in A_B_I16.iter().enumerate() {
            test_optimal_constants(10 - i as u32, vec![(a, b)]);
        }
        let expected: Vec<(i16, i16)> = vec![(323, 86), (324, 84), (324, 85), (325, 82), (325, 83)];
        test_optimal_constants(2, expected);
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
        define_test!(case_27, 27, vec![(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]);
        define_test!(case_26, 26, vec![(A_B_I32[0])]);
        define_test!(case_25, 25, vec![(A_B_I32[1])]);
        define_test!(case_24, 24, vec![(A_B_I32[2])]);
        define_test!(case_23, 23, vec![(A_B_I32[3])]);
        define_test!(case_22, 22, vec![(A_B_I32[4])]);
        define_test!(case_21, 21, vec![(A_B_I32[5])]);
        define_test!(case_20, 20, vec![(A_B_I32[6])]);
        define_test!(case_19, 19, vec![(A_B_I32[7])]);
        define_test!(case_18, 18, vec![(A_B_I32[8])]);
        define_test!(case_17, 17, vec![(A_B_I32[9])]);
        define_test!(case_16, 16, vec![(A_B_I32[10])]);
        define_test!(case_15, 15, vec![(A_B_I32[11])]);
        define_test!(case_14, 14, vec![(A_B_I32[12])]);
        define_test!(case_13, 13, vec![(A_B_I32[13])]);
        define_test!(case_12, 12, vec![(A_B_I32[14])]);
        define_test!(case_11, 11, vec![(A_B_I32[15])]);
        define_test!(case_10, 10, vec![(A_B_I32[16])]);
        define_test!(case_09, 9, vec![(A_B_I32[17])]);
    }
}
