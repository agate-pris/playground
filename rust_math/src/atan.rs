use std::{
    cmp::Ordering,
    ops::{Neg, Sub},
};

use num_traits::ConstZero;

pub(crate) const fn inv_i32_f15(x: i32) -> i32 {
    const K: i64 = 2_i64.pow(2 * 15 + 1);
    let x = x as i64;
    ((K + x.abs()) / (2 * x)) as i32
}

pub(crate) const fn div_i32_f15(a: i32, b: i32) -> i32 {
    const K: i64 = 1 << (15 + 1);
    let a = a as i64 * K;
    let b = b as i64;
    ((a + a.signum() * b.abs()) / (2 * b)) as i32
}

pub(crate) trait AtanUtil<T> {
    type Output;
    const ONE: T;
    const STRAIGHT: Self::Output;
    const RIGHT: Self::Output;
    const NEG_ONE: T;
    const NEG_STRAIGHT: Self::Output;
    const NEG_RIGHT: Self::Output;
    fn inv(x: T) -> T;
    fn div(a: T, b: T) -> T;
    fn calc(x: T) -> Self::Output;
    fn atan(x: T) -> Self::Output
    where
        T: PartialOrd,
        Self::Output: Sub<Output = Self::Output>,
    {
        if x < Self::NEG_ONE {
            Self::NEG_RIGHT - Self::calc(Self::inv(x))
        } else if x > Self::ONE {
            Self::RIGHT - Self::calc(Self::inv(x))
        } else {
            Self::calc(x)
        }
    }
    fn atan2(y: T, x: T) -> Self::Output
    where
        T: Copy + Ord + Neg<Output = T> + ConstZero,
        Self::Output: Sub<Output = Self::Output> + ConstZero,
    {
        use Ordering::*;

        match (y.cmp(&T::ZERO), x.cmp(&T::ZERO)) {
            (Less, Less) => {
                if y < x {
                    let x = Self::div(x, y);
                    Self::NEG_RIGHT - Self::calc(x)
                } else {
                    let x = Self::div(y, x);
                    Self::NEG_STRAIGHT + Self::calc(x)
                }
            }
            (Less, Equal) => Self::NEG_RIGHT,
            (Less, Greater) => {
                if y < -x {
                    let x = Self::div(x, y);
                    Self::NEG_RIGHT - Self::calc(x)
                } else {
                    let x = Self::div(y, x);
                    Self::calc(x)
                }
            }
            (Equal, Less) => Self::STRAIGHT,
            (Greater, Less) => {
                if -y < x {
                    let x = Self::div(x, y);
                    Self::RIGHT - Self::calc(x)
                } else {
                    let x = Self::div(y, x);
                    Self::STRAIGHT + Self::calc(x)
                }
            }
            (Greater, Equal) => Self::RIGHT,
            (Greater, Greater) => {
                if y < x {
                    let x = Self::div(y, x);
                    Self::calc(x)
                } else {
                    let x = Self::div(x, y);
                    Self::RIGHT - Self::calc(x)
                }
            }
            _ => Self::Output::ZERO,
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use std::{cmp::Ordering, f64::consts::PI, fmt::Debug, ops::RangeInclusive};

    use anyhow::Result;
    use approx::abs_diff_eq;
    use itertools::Itertools;
    use num_traits::{AsPrimitive, ConstOne, ConstZero, PrimInt, Signed};
    use primitive_promotion::PrimitivePromotionExt;
    use rayon::iter::{IntoParallelIterator, ParallelIterator};

    use crate::{
        atan2_p2_2850, atan2_p3_2555_691, atan2_p5_787_2968, atan_p2_2850, atan_p3_2555_691,
        atan_p5_787_2968, bits::Bits, tests::read_data,
    };

    pub(crate) fn compare_max_error(lhs: &f64, rhs: &f64) -> Ordering {
        lhs.abs().total_cmp(&rhs.abs())
    }

    pub(crate) fn compare_error(
        lerror_max: f64,
        lerror_sum: f64,
        rerror_max: f64,
        rerror_sum: f64,
    ) -> Ordering {
        compare_max_error(&lerror_max, &rerror_max)
            .then_with(|| lerror_sum.abs().total_cmp(&rerror_sum.abs()))
    }

    fn test_atan<F>(f: F, data_path: &str, acceptable_error: f64)
    where
        F: Sync + Fn(i32) -> i32,
    {
        const ONE: i32 = 2_i32.pow(i32::BITS / 2 - 1);
        const RIGHT: i32 = 2_i32.pow(i32::BITS - 3);
        const NEG_RIGHT: i32 = -RIGHT;
        const K: i64 = 2_i64.pow(i32::BITS - 1);
        const NEG_K: i64 = -K;
        const ERRORS_LEN: usize = 6;

        fn fold_errors(
            init: [(f64, f64); ERRORS_LEN],
            errors: [f64; ERRORS_LEN],
        ) -> [(f64, f64); ERRORS_LEN] {
            std::array::from_fn(|i| (init[i].0.min(errors[i]), init[i].1.max(errors[i])))
        }
        fn reduce_errors(
            lhs: [(f64, f64); ERRORS_LEN],
            rhs: [(f64, f64); ERRORS_LEN],
        ) -> [(f64, f64); ERRORS_LEN] {
            std::array::from_fn(|i| (lhs[i].0.min(rhs[i].0), lhs[i].1.max(rhs[i].1)))
        }

        let expected = read_data::<i32>(data_path).unwrap();

        assert_eq!(expected.len(), (ONE + 1) as usize);
        assert_eq!(expected[0], 0);
        assert_eq!(expected[ONE as usize], RIGHT / 2);

        let f = |x, expected| -> Result<_> {
            let actual = f(x);
            anyhow::ensure!(
                actual == expected,
                "x: {x}, actual: {actual}, expected: {expected}"
            );

            const SCALE: f64 = PI / ONE.pow(2) as f64;
            let actual = SCALE * actual as f64;
            let expected = (x as f64 / ONE as f64).atan();
            anyhow::ensure!(
                abs_diff_eq!(actual, expected, epsilon = acceptable_error),
                "x: {x}, actual: {actual}, expected: {expected}"
            );
            Ok(actual - expected)
        };

        let (neg_inv_error_near, neg_inv_error_far) = {
            let expected = NEG_RIGHT + expected[1];
            let neg_inv_error_near = f((NEG_K / 3 - 1) as i32, expected).unwrap();
            let neg_inv_error_far = f(i32::MIN, expected).unwrap();
            (neg_inv_error_near, neg_inv_error_far)
        };
        let (inv_error_near, inv_error_far) = {
            let expected = RIGHT - expected[1];
            let inv_error_near = f((K / 3 + 1) as i32, expected).unwrap();
            let inv_error_far = f(i32::MAX, expected).unwrap();
            (inv_error_near, inv_error_far)
        };

        let neg_error = f(-1, -expected[1]).unwrap();
        let error = f(1, expected[1]).unwrap();
        let zero_error = f(0, expected[0]).unwrap();
        let diff_sum = error + zero_error;
        let num = num_cpus::get();

        let map_op = |n| {
            let begin = 2 + n as i32 * (ONE - 1) / num as i32;
            let end = 2 + (n + 1) as i32 * (ONE - 1) / num as i32;
            (begin..end).try_fold(
                (0.0, [(f64::INFINITY, f64::NEG_INFINITY); ERRORS_LEN]),
                |(error_sum, errors), i| -> Result<_> {
                    let expected = expected[i as usize];
                    let expected = [expected, -expected, RIGHT - expected, NEG_RIGHT + expected];
                    let e = [
                        (expected[0], i),
                        (expected[1], -i),
                        (expected[2], (K / (2 * i as i64 + 1)) as i32 + 1),
                        (expected[2], (K / (2 * i as i64 - 1)) as i32),
                        (expected[3], (NEG_K / (2 * i as i64 + 1)) as i32 - 1),
                        (expected[3], (NEG_K / (2 * i as i64 - 1)) as i32),
                    ]
                    .try_map(|(expected, x)| f(x, expected))?;
                    let error_sum = error_sum + e[0];
                    let errors = fold_errors(errors, e);
                    Ok((error_sum, errors))
                },
            )
        };

        let errors = [
            error,
            neg_error,
            inv_error_near,
            inv_error_far,
            neg_inv_error_near,
            neg_inv_error_far,
        ];
        let (diff_sum, errors) = (0..num)
            .into_par_iter()
            .map(map_op)
            .try_reduce(
                || (diff_sum, errors.map(|e| (e, e))),
                |(ldiff_sum, lerrors), (rdiff_sum, rerrors)| -> Result<_> {
                    Ok((ldiff_sum + rdiff_sum, reduce_errors(lerrors, rerrors)))
                },
            )
            .unwrap();

        println!("error_zero: {:12.9}", zero_error);
        for (i, e) in errors.iter().enumerate() {
            println!("errors[{}]: ({:12.9}, {:12.9})", i, e.0, e.1);
        }
        println!("average: {:15.9}", diff_sum / (ONE + 1) as f64);
    }

    #[rustfmt::skip] #[test] fn test_atan_p2() { test_atan(atan_p2_2850,     "data/atan_p2_i17f15.json", 0.003778); }
    #[rustfmt::skip] #[test] fn test_atan_p3() { test_atan(atan_p3_2555_691, "data/atan_p3_i17f15.json", 0.001543); }
    #[rustfmt::skip] #[test] fn test_atan_p5() { test_atan(atan_p5_787_2968, "data/atan_p5_i17f15.json", 0.000767); }

    fn test_atan2<F>(f: F, data_path: &str, acceptable_error: f64)
    where
        F: Fn(i32, i32) -> i32 + Sync,
    {
        const K: i32 = 2_i32.pow(i32::BITS / 2 - 1);
        const STRAIGHT: i32 = K.pow(2);
        const RIGHT: i32 = STRAIGHT / 2;
        const HALF_RIGHT: i32 = RIGHT / 2;
        const NEG_STRAIGHT: i32 = -STRAIGHT;
        const NEG_RIGHT: i32 = -RIGHT;
        const NEG_HALF_RIGHT: i32 = -HALF_RIGHT;
        const OPPOSITE_HALF_RIGHT: i32 = NEG_STRAIGHT + HALF_RIGHT;
        const OPPOSITE_NEG_HALF_RIGHT: i32 = STRAIGHT + NEG_HALF_RIGHT;
        const ERRORS_LEN: usize = 8;

        fn to_4_points(x: i32, y: i32) -> [[i32; 2]; 4] {
            [[x, y], [y, x], [-y, x], [x, -y]]
        }

        fn to_8_points(x: i32, y: i32, z: i32, w: i32) -> [[i32; 2]; ERRORS_LEN] {
            let p = to_4_points(x, y);
            let n = to_4_points(z, w);
            [p[0], p[1], p[2], n[3], n[0], n[1], n[2], p[3]]
        }

        fn to_8_points_default(x: i32, y: i32) -> [[i32; 2]; ERRORS_LEN] {
            to_8_points(x, y, -x, -y)
        }

        fn compare_steep(a: &[i32; 2], b: &[i32; 2]) -> Ordering {
            let aybx = a[1] as i64 * b[0] as i64;
            let byax = b[1] as i64 * a[0] as i64;
            aybx.cmp(&byax)
        }

        fn collect_most_steep_points(n: i32) -> Vec<[i32; 2]> {
            const WIDTH: i32 = 2_i32.pow(16);
            const X: i32 = i32::MAX - (WIDTH - 1);
            const X_BEGIN: i32 = X + 1;

            let points = (X_BEGIN..=i32::MAX)
                .map(|x| {
                    let y = {
                        const WIDTH_AS_I64: i64 = WIDTH as i64;
                        let mul = x as i64 * (2 * n + 1) as i64;
                        let rem = mul % WIDTH_AS_I64;
                        mul / WIDTH_AS_I64 - if rem == 0 { 1 } else { 0 }
                    } as i32;
                    [x, y]
                })
                .collect::<Vec<_>>();

            points.into_iter().max_set_by(compare_steep)
        }

        fn fold_errors(
            init: [(f64, f64); ERRORS_LEN],
            errors: [f64; ERRORS_LEN],
        ) -> [(f64, f64); ERRORS_LEN] {
            std::array::from_fn(|i| (init[i].0.min(errors[i]), init[i].1.max(errors[i])))
        }
        fn reduce_errors(
            lhs: [(f64, f64); ERRORS_LEN],
            rhs: [(f64, f64); ERRORS_LEN],
        ) -> [(f64, f64); ERRORS_LEN] {
            std::array::from_fn(|i| (lhs[i].0.min(rhs[i].0), lhs[i].1.max(rhs[i].1)))
        }

        for (y, x, expected) in [
            (0, 0, 0),
            (0, 1, 0),
            (1, 1, HALF_RIGHT),
            (1, 0, RIGHT),
            (1, -1, OPPOSITE_NEG_HALF_RIGHT),
            (0, -1, STRAIGHT),
            (-1, 1, NEG_HALF_RIGHT),
            (-1, 0, NEG_RIGHT),
            (-1, -1, OPPOSITE_HALF_RIGHT),
            (0, i32::MAX, 0),
            (0, i32::MIN, STRAIGHT),
            (i32::MAX, 0, RIGHT),
            (i32::MIN, 0, NEG_RIGHT),
            (i32::MAX, i32::MAX, HALF_RIGHT),
            (i32::MIN, i32::MIN, OPPOSITE_HALF_RIGHT),
            (i32::MIN, i32::MAX, NEG_HALF_RIGHT),
            (i32::MAX, i32::MIN, OPPOSITE_NEG_HALF_RIGHT),
            (0, -i32::MAX, STRAIGHT),
            (i32::MIN, -i32::MAX, OPPOSITE_HALF_RIGHT),
            (i32::MAX, -i32::MAX, OPPOSITE_NEG_HALF_RIGHT),
            (-i32::MAX, 0, NEG_RIGHT),
            (-i32::MAX, i32::MIN, OPPOSITE_HALF_RIGHT),
            (-i32::MAX, i32::MAX, NEG_HALF_RIGHT),
            (-i32::MAX, -i32::MAX, OPPOSITE_HALF_RIGHT),
            (1, i32::MAX, 0),
            (1, i32::MIN, STRAIGHT),
            (1, -i32::MAX, STRAIGHT),
            (-1, i32::MAX, 0),
            (-1, i32::MIN, NEG_STRAIGHT),
            (-1, -i32::MAX, NEG_STRAIGHT),
            (i32::MIN, 1, NEG_RIGHT),
            (i32::MAX, 1, RIGHT),
            (i32::MIN, -1, NEG_RIGHT),
            (i32::MAX, -1, RIGHT),
            (-i32::MAX, 1, NEG_RIGHT),
            (-i32::MAX, -1, NEG_RIGHT),
        ] {
            assert_eq!(f(y, x), expected, "p: [{x}, {y}]");
        }

        let f = |p: &[i32; 2]| -> Result<_> {
            let actual = f(p[1], p[0]);
            let error = {
                const SCALE: f64 = PI / STRAIGHT as f64;
                let actual = SCALE * actual as f64;
                let expected = (p[1] as f64).atan2(p[0] as f64);
                anyhow::ensure!(
                    abs_diff_eq!(expected, actual, epsilon = acceptable_error),
                    "p: {:?}, expected: {expected}, actual: {actual}",
                    p
                );
                actual - expected
            };
            Ok((actual, error))
        };

        let data = read_data::<i32>(data_path).unwrap();

        assert_eq!(data.len(), (K + 1) as usize);
        assert_eq!(data[0], 0);
        assert_eq!(data[K as usize], HALF_RIGHT);

        // 0
        let far = {
            const EXPECTED: [i32; ERRORS_LEN] = [
                0,
                RIGHT,
                RIGHT,
                STRAIGHT,
                NEG_STRAIGHT,
                NEG_RIGHT,
                NEG_RIGHT,
                0,
            ];
            collect_most_steep_points(0)
                .into_iter()
                .try_fold(
                    [(f64::INFINITY, f64::NEG_INFINITY); ERRORS_LEN],
                    |errors, p| -> Result<_> {
                        let points = to_8_points_default(p[0], p[1]);
                        anyhow::ensure!(EXPECTED.len() == points.len());
                        let e = std::array::try_from_fn(|i| {
                            let expected = EXPECTED[i];
                            let (actual, error) = f(&points[i])?;
                            anyhow::ensure!(expected == actual);
                            Ok(error)
                        })?;
                        Ok(fold_errors(errors, e))
                    },
                )
                .unwrap()
        };

        // 32768
        let near = {
            const EXPECTED: [i32; ERRORS_LEN] = [
                HALF_RIGHT,
                HALF_RIGHT,
                OPPOSITE_NEG_HALF_RIGHT,
                OPPOSITE_NEG_HALF_RIGHT,
                OPPOSITE_HALF_RIGHT,
                OPPOSITE_HALF_RIGHT,
                NEG_HALF_RIGHT,
                NEG_HALF_RIGHT,
            ];
            let points = to_8_points_default(2_i32.pow(16), 2_i32.pow(16) - 1);
            assert_eq!(EXPECTED.len(), points.len());
            std::array::try_from_fn(|i| {
                let expected = EXPECTED[i];
                let (actual, error) = f(&points[i])?;
                anyhow::ensure!(expected == actual);
                Ok(error)
            })
            .unwrap()
        };

        let f = |p: &[i32; 2], i: usize| -> Result<_> {
            let expected = data[i];
            let expected = [
                expected,
                RIGHT - expected,
                RIGHT + expected,
                STRAIGHT - expected,
                NEG_STRAIGHT + expected,
                NEG_RIGHT - expected,
                NEG_RIGHT + expected,
                -expected,
            ];
            let points = to_8_points_default(p[0], p[1]);
            anyhow::ensure!(expected.len() == points.len());
            let result = points.try_map(|p| f(&p))?;
            for (&expected, &actual) in expected.iter().zip(result.iter().map(|(r, _)| r)) {
                anyhow::ensure!(expected == actual);
            }
            Ok(result.map(|(_, diff)| diff))
        };

        let num = num_cpus::get();

        let map_op = |n| {
            let begin = 1 + (K - 1) * n as i32 / num as i32;
            let end = 1 + (K - 1) * (n + 1) as i32 / num as i32;
            (begin..end).try_fold(
                [[(f64::INFINITY, f64::NEG_INFINITY); ERRORS_LEN]; 2],
                |errors, i| -> Result<_> {
                    let near = {
                        let errors = errors[0];
                        let point = [2 * K, 1 + 2 * (i - 1)];
                        let e = f(&point, i as usize)?;
                        fold_errors(errors, e)
                    };
                    let far = collect_most_steep_points(i)
                        .into_iter()
                        .try_fold(errors[1], |errors, p| -> Result<_> {
                            Ok(fold_errors(errors, f(&p, i as usize)?))
                        })?;
                    Ok([near, far])
                },
            )
        };

        let errors = (0..num)
            .into_par_iter()
            .map(map_op)
            .try_reduce(
                || [near.map(|a| (a, a)), far],
                |l, r| Ok(std::array::from_fn(|i| reduce_errors(l[i], r[i]))),
            )
            .unwrap();

        for (i, e) in errors[0].iter().enumerate() {
            println!("near_errors[{i}]: ({:12.9}, {:12.9})", e.0, e.1);
        }
        for (i, e) in errors[1].iter().enumerate() {
            println!("far_errors[{i}]:  ({:12.9}, {:12.9})", e.0, e.1);
        }
    }

    #[rustfmt::skip] #[test] fn test_atan2_p2() { test_atan2(atan2_p2_2850,     "data/atan_p2_i17f15.json", 0.003778); }
    #[rustfmt::skip] #[test] fn test_atan2_p3() { test_atan2(atan2_p3_2555_691, "data/atan_p3_i17f15.json", 0.001543); }
    #[rustfmt::skip] #[test] fn test_atan2_p5() { test_atan2(atan2_p5_787_2968, "data/atan_p5_i17f15.json", 0.000767); }

    pub fn make_atan_data(exp: u32) -> Vec<f64> {
        let num = num_cpus::get();
        let k = 2_i64.pow(exp);
        (0..num)
            .into_par_iter()
            .flat_map(|n| {
                let begin = (k as i128 * n as i128 / num as i128) as i64;
                let end = (k as i128 * (n as i128 + 1) / num as i128) as i64;
                let f = |x| (x as f64).atan2(k as f64); // (x as f64 / k as f64).atan();
                if n == num - 1 {
                    (begin..=end).map(f).collect::<Vec<_>>()
                } else {
                    (begin..end).map(f).collect::<Vec<_>>()
                }
            })
            .collect()
    }

    // Find the optimal constants for the atan2 function.
    // The expected values are given from the out of this function
    // because it is too large so it has to be shared over threads.
    pub fn find_optimal_constants<T, R, F>(
        exp: u32,
        expected: &[f64],
        search_range: R,
        f: F,
    ) -> (Vec<<R as Iterator>::Item>, f64, f64)
    where
        T: Debug
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
        R: Iterator,
        F: Fn(T, T, T, R::Item) -> T,
        T::PrimitivePromotion: PartialOrd + AsPrimitive<T> + Signed,
        R::Item: Clone,
        RangeInclusive<T>: Iterator<Item = T>,
        i8: AsPrimitive<T>,
    {
        use Ordering::*;

        let (one, k, to_rad) = {
            let base: T = 2.as_();
            let to_rad = {
                let pi: f64 = base.pow(T::BITS - 2).as_();
                PI / pi
            };
            (base.pow(exp), base.pow(T::BITS - 2 - exp), to_rad)
        };

        let time = std::time::Instant::now();
        let mut elapsed = 0;

        search_range.enumerate().fold(
            (vec![], f64::INFINITY, f64::INFINITY),
            |(mut acc, min_max_error, min_error_sum), (i, item)| {
                if i % 10000 == 0 {
                    let e = time.elapsed().as_secs();
                    if e / 30 != elapsed / 30 {
                        elapsed = e;
                        println!("i: {i}, elapsed: {elapsed}");
                    }
                }

                match (T::ZERO..=one).try_fold((0.0, 0.0), |(max_error, error_sum), x| {
                    let i: usize = x.as_();
                    let expected = expected[i];
                    let actual: f64 = f(x, one, k, item.clone()).as_();
                    let error = to_rad * actual - expected;

                    let error_sum = error_sum + error;
                    let max_error = std::cmp::max_by(max_error, error, compare_max_error);

                    match compare_max_error(&max_error, &min_max_error) {
                        Greater => None,
                        _ => Some((max_error, error_sum)),
                    }
                }) {
                    None => (acc, min_max_error, min_error_sum),
                    Some((max_error, error_sum)) => {
                        match compare_error(max_error, error_sum, min_max_error, min_error_sum) {
                            Equal => {
                                acc.push(item);
                                (acc, min_max_error, min_error_sum)
                            }
                            Greater => (acc, min_max_error, min_error_sum),
                            Less => (vec![item], max_error, error_sum),
                        }
                    }
                }
            },
        )
    }
}
