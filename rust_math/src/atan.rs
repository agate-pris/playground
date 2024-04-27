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
    const K: i64 = 1 << 15;
    (a as i64 * K / b as i64) as i32
}

pub(crate) trait AtanUtil<T> {
    type Output;
    const ONE: T;
    const NEG_ONE: T;
    const RIGHT: Self::Output;
    const NEG_RIGHT: Self::Output;
    const STRAIGHT: Self::Output;
    const NEG_STRAIGHT: Self::Output;
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
    use std::{
        cmp::Ordering,
        f64::consts::{FRAC_PI_2, PI},
        fmt::Debug,
        iter::once,
        ops::RangeInclusive,
    };

    use anyhow::Result;
    use approx::abs_diff_eq;
    use num_traits::{AsPrimitive, ConstOne, ConstZero, PrimInt, Signed};
    use primitive_promotion::PrimitivePromotionExt;
    use rayon::iter::{IntoParallelIterator, ParallelIterator};

    use crate::{atan_p2::AtanP2, atan_p3::AtanP3, atan_p5::AtanP5, bits::Bits, tests::read_data};

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

        type Errors = [f64; ERRORS_LEN];

        fn make_min_errors(lhs: Errors, rhs: Errors) -> Errors {
            std::array::from_fn(|i| lhs[i].min(rhs[i]))
        }
        fn make_max_errors(lhs: Errors, rhs: Errors) -> Errors {
            std::array::from_fn(|i| lhs[i].max(rhs[i]))
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
                (
                    0.0,
                    [f64::INFINITY; ERRORS_LEN],
                    [f64::NEG_INFINITY; ERRORS_LEN],
                ),
                |(diff_sum, errors_min, errors_max), i| -> Result<_> {
                    let expected = expected[i as usize];
                    let expected = [expected, -expected, RIGHT - expected, NEG_RIGHT + expected];
                    let errors = [
                        (expected[0], i),
                        (expected[1], -i),
                        (expected[2], (K / (2 * i as i64 + 1)) as i32 + 1),
                        (expected[2], (K / (2 * i as i64 - 1)) as i32),
                        (expected[3], (NEG_K / (2 * i as i64 + 1)) as i32 - 1),
                        (expected[3], (NEG_K / (2 * i as i64 - 1)) as i32),
                    ]
                    .try_map(|(expected, x)| f(x, expected))?;
                    anyhow::ensure!(errors_min.len() == ERRORS_LEN);
                    anyhow::ensure!(errors_max.len() == ERRORS_LEN);
                    anyhow::ensure!(errors.len() == ERRORS_LEN);
                    let errors_min = make_min_errors(errors_min, errors);
                    let errors_max = make_max_errors(errors_max, errors);
                    Ok((diff_sum + errors[0], errors_min, errors_max))
                },
            )
        };

        let try_reduce_op = |(ldiff_sum, lerrors_min, lerrors_max): (_, Errors, Errors),
                             (rdiff_sum, rerrors_min, rerrors_max): (_, Errors, Errors)|
         -> Result<_> {
            let errors_min = make_min_errors(lerrors_min, rerrors_min);
            let errors_max = make_max_errors(lerrors_max, rerrors_max);
            Ok((ldiff_sum + rdiff_sum, errors_min, errors_max))
        };

        let errors = [
            error,
            neg_error,
            inv_error_near,
            inv_error_far,
            neg_inv_error_near,
            neg_inv_error_far,
        ];
        let (diff_sum, errors_min, errors_max) = (0..num)
            .into_par_iter()
            .map(map_op)
            .try_reduce(|| (diff_sum, errors.clone(), errors.clone()), try_reduce_op)
            .unwrap();

        println!("error_zero: {:15.9}", zero_error);
        for (i, e) in errors_min.iter().enumerate() {
            println!("errors_min[{}]: {:12.9}", i, e);
        }
        for (i, e) in errors_max.iter().enumerate() {
            println!("errors_max[{}]: {:12.9}", i, e);
        }
        println!("average: {:18.9}", diff_sum / (ONE + 1) as f64);
    }

    #[rustfmt::skip] #[test] fn test_atan_p2() { test_atan(AtanP2::atan_p2, "data/atan_p2_i17f15.json", 0.003778); }
    #[rustfmt::skip] #[test] fn test_atan_p3() { test_atan(AtanP3::atan_p3, "data/atan_p3_i17f15.json", 0.001543); }
    #[rustfmt::skip] #[test] fn test_atan_p5() { test_atan(AtanP5::atan_p5, "data/atan_p5_i17f15.json", 0.000767); }

    fn test_atan2<F>(f: F, data_path: &str, acceptable_error: f64)
    where
        F: Fn(i32, i32) -> i32,
    {
        assert_eq!(f(0, 0), 0);

        const K: i32 = 2_i32.pow(i32::BITS / 2 - 1);
        const STRAIGHT: i32 = K.pow(2);
        const RIGHT: i32 = STRAIGHT / 2;
        const HALF_RIGHT: i32 = RIGHT / 2;
        const NEG_STRAIGHT: i32 = -STRAIGHT;
        const NEG_RIGHT: i32 = -RIGHT;
        const NEG_HALF_RIGHT: i32 = -HALF_RIGHT;
        const OPPOSITE_HALF_RIGHT: i32 = NEG_STRAIGHT + HALF_RIGHT;
        const OPPOSITE_NEG_HALF_RIGHT: i32 = STRAIGHT + NEG_HALF_RIGHT;

        // Find the largest error for each of the eight regions
        // that are divided by the straight lines y = x, y = -x, y = 0, x = 0.

        let mut max_error = f64::NEG_INFINITY;

        // Calculate the expected and actual value and store value.
        let mut calc = |p: &[i32; 2]| {
            let expected = (p[1] as f64).atan2(p[0] as f64);
            let actual = f(p[1], p[0]);
            {
                const SCALE: f64 = FRAC_PI_2 / RIGHT as f64;
                let actual = actual as f64 * SCALE;
                let cond = abs_diff_eq!(expected, actual, epsilon = acceptable_error);
                assert!(cond, "p: {p:?}, expected: {expected}, actual: {actual}");
                max_error = max_error.max((actual - expected).abs());
            }
            (expected, actual)
        };

        // On the straight lintes y = 0, x = 0.
        {
            let points = [[1, 0], [0, 1], [-1, 0], [0, -1]];
            let angles = points.iter().map(&mut calc).collect::<Vec<_>>();
            let e = [0, RIGHT, STRAIGHT, NEG_RIGHT];
            angles.iter().zip(e).for_each(|(a, e)| assert_eq!(a.1, e));
        }

        // On the straight lines y = x, y = -x.
        {
            let points = [[1, 1], [-1, 1], [-1, -1], [1, -1]];
            let angles = points.iter().map(&mut calc).collect::<Vec<_>>();
            let e = [
                HALF_RIGHT,
                OPPOSITE_NEG_HALF_RIGHT,
                OPPOSITE_HALF_RIGHT,
                NEG_HALF_RIGHT,
            ];
            angles.iter().zip(e).for_each(|(a, e)| assert_eq!(a.1, e));
        }

        fn to_4_points(x: i32, y: i32) -> [[i32; 2]; 4] {
            [[x, y], [y, x], [-y, x], [x, -y]]
        }

        fn to_8_points(x: i32, y: i32, z: i32, w: i32) -> [[i32; 2]; 8] {
            let p = to_4_points(x, y);
            let n = to_4_points(z, w);
            [p[0], p[1], p[2], n[3], n[0], n[1], n[2], p[3]]
        }

        fn to_8_points_default(x: i32, y: i32) -> [[i32; 2]; 8] {
            to_8_points(x, y, -x, -y)
        }

        let data = read_data::<i32>(data_path).unwrap();

        assert_eq!(data.len(), (K + 1) as usize);
        assert_eq!(data[0], 0);
        assert_eq!(data[K as usize], HALF_RIGHT);

        let assert_equality = |angles: &[(f64, i32)], i: usize| {
            assert_eq!(angles.len(), 8);

            #[rustfmt::skip] assert_eq!(angles[0].1,              data[i]);
            #[rustfmt::skip] assert_eq!(angles[1].1,      RIGHT - data[i]);
            #[rustfmt::skip] assert_eq!(angles[2].1,      RIGHT + data[i]);
            #[rustfmt::skip] assert_eq!(angles[3].1,  2 * RIGHT - data[i]);
            #[rustfmt::skip] assert_eq!(angles[4].1, -2 * RIGHT + data[i]);
            #[rustfmt::skip] assert_eq!(angles[5].1,     -RIGHT - data[i]);
            #[rustfmt::skip] assert_eq!(angles[6].1,     -RIGHT + data[i]);
            #[rustfmt::skip] assert_eq!(angles[7].1,             -data[i]);
        };

        // (K, K - 1)
        {
            let points = to_8_points_default(K, K - 1);
            let angles = points.iter().map(&mut calc).collect::<Vec<_>>();
            assert_equality(&angles, K as usize - 1);
        }

        // (MAX, MAX - 1), (MIN, -MAX)
        {
            let points = to_8_points(i32::MAX, i32::MAX - 1, i32::MIN, -i32::MAX);
            let angles = points.iter().map(&mut calc).collect::<Vec<_>>();

            #[rustfmt::skip] assert_ne!(angles[0].1,      HALF_RIGHT);
            #[rustfmt::skip] assert_ne!(angles[1].1,  3 * HALF_RIGHT);
            #[rustfmt::skip] assert_ne!(angles[2].1, -3 * HALF_RIGHT);
            #[rustfmt::skip] assert_ne!(angles[3].1,     -HALF_RIGHT);

            assert_equality(&angles, K as usize - 1);
        }

        fn collect_most_steep_points(n: i32) -> Vec<[i32; 2]> {
            fn compare_steep(a: &[i32; 2], b: &[i32; 2]) -> Ordering {
                let aybx = a[1] as i64 * b[0] as i64;
                let byax = b[1] as i64 * a[0] as i64;
                aybx.cmp(&byax)
            }

            let count = n + 1;
            let begin = {
                let mut copy = count;
                while copy % 2 == 0 && copy > 1 {
                    copy /= 2;
                }
                count - copy
            };

            const OFFSET_X: i32 = i32::MAX - K + 1;
            let offset_y = OFFSET_X / K * count;

            let positions = (begin..=n)
                .map(|m| [OFFSET_X + K * m / count + 1, offset_y + m])
                .collect::<Vec<_>>();

            let max = positions.iter().cloned().max_by(compare_steep).unwrap();

            positions
                .into_iter()
                .filter(|a| compare_steep(a, &max).is_eq())
                .collect()
        }

        for n in 1..K - 1 {
            // (K, n)
            {
                let angles = to_8_points_default(K, n)
                    .iter()
                    .map(&mut calc)
                    .collect::<Vec<_>>();

                assert_equality(&angles, n as usize);
            }

            // most steep points
            {
                let positions = collect_most_steep_points(n);

                for position in positions.into_iter() {
                    let positions = to_8_points_default(position[0], position[1]);
                    let angles = positions.iter().map(&mut calc).collect::<Vec<_>>();

                    assert_equality(&angles, n as usize);
                }
            }
        }

        println!("max error: {max_error}");
    }

    #[rustfmt::skip] #[test] fn test_atan2_p2() { test_atan2(AtanP2::atan2_p2, "data/atan_p2_i17f15.json", 0.003789); }
    #[rustfmt::skip] #[test] fn test_atan2_p3() { test_atan2(AtanP3::atan2_p3, "data/atan_p3_i17f15.json", 0.001603); }
    #[rustfmt::skip] #[test] fn test_atan2_p5() { test_atan2(AtanP5::atan2_p5, "data/atan_p5_i17f15.json", 0.000928); }

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
        F: Fn(T, T, T, <R as Iterator>::Item) -> T,
        T::PrimitivePromotion: PartialOrd + AsPrimitive<T> + Signed,
        <R as Iterator>::Item: Clone,
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

        let cmp = |(lmax, lsum): (f64, f64), (rmax, rsum): (f64, f64)| {
            lmax.total_cmp(&rmax).then_with(|| lsum.total_cmp(&rsum))
        };

        let time = std::time::Instant::now();
        let mut elapsed = 0;

        search_range.enumerate().fold(
            (vec![], f64::INFINITY, f64::INFINITY),
            |(acc, min_max_error, min_error_sum), (i, item)| {
                if i % 10000 == 0 {
                    let e = time.elapsed().as_secs();
                    if e / 30 != elapsed / 30 {
                        elapsed = e;
                        println!("i: {i}, elapsed: {elapsed}");
                    }
                }

                let mut max_error = f64::NEG_INFINITY;
                let mut error_sum = 0.0;

                for x in T::ZERO..=one {
                    let i: usize = x.as_();
                    let expected = expected[i];
                    let actual: f64 = f(x, one, k, item.clone()).as_();
                    let error = to_rad * actual - expected;

                    error_sum += error;
                    max_error = max_error.max(error.abs());

                    if max_error > min_max_error {
                        break;
                    }
                }

                let error_sum = error_sum.abs();

                match cmp((max_error, error_sum), (min_max_error, min_error_sum)) {
                    Equal => (
                        acc.into_iter().chain(once(item.clone())).collect(),
                        max_error,
                        error_sum,
                    ),
                    Less => (vec![item.clone()], max_error, error_sum),
                    Greater => (acc, min_max_error, min_error_sum),
                }
            },
        )
    }
}
