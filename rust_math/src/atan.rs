use num_traits::{AsPrimitive, ConstZero, Signed};
use primitive_promotion::PrimitivePromotionExt;

pub(crate) fn atan_impl<T, F>(x: T, k: T, f: F) -> T
where
    <T as PrimitivePromotionExt>::PrimitivePromotion: PartialOrd + AsPrimitive<T> + Signed,
    T: AsPrimitive<<T as PrimitivePromotionExt>::PrimitivePromotion>
        + PrimitivePromotionExt
        + Signed,
    F: Fn(T, T) -> T,
    i8: AsPrimitive<T>,
{
    let x_abs = x.as_().abs();
    if x_abs > k.as_() {
        let signum = x.signum();
        let k_2 = k * k;
        let x = (k_2.as_() / x_abs).as_();
        signum * (k_2 / 2.as_() - f(x, x))
    } else {
        f(x, x_abs.as_())
    }
}

pub(crate) fn atan2_impl<T, F>(y: T, x: T, k: T, f: F) -> T
where
    <T as PrimitivePromotionExt>::PrimitivePromotion: AsPrimitive<T> + PartialOrd + Signed,
    T: AsPrimitive<<T as PrimitivePromotionExt>::PrimitivePromotion>
        + ConstZero
        + PrimitivePromotionExt
        + Signed,
    F: Fn(T) -> T,
    i8: AsPrimitive<T>,
{
    if y.is_zero() && x.is_zero() {
        return T::ZERO;
    }

    let x_abs = x.as_().abs();
    let y_abs = y.as_().abs();
    let x_is_negative = x.is_negative();
    let y_is_negative = y.is_negative();

    if x_abs > y_abs {
        let x = (y_abs * k.as_() / x_abs).as_();
        let v = f(x);
        match (x_is_negative, y_is_negative) {
            (false, false) => v,
            (true, false) => k * k - v,
            (false, true) => -v,
            (true, true) => v - k * k,
        }
    } else {
        let x = (x_abs * k.as_() / y_abs).as_();
        let v = f(x);
        let right = k * k / 2.as_();
        match (x_is_negative, y_is_negative) {
            (false, false) => right - v,
            (true, false) => right + v,
            (false, true) => -right + v,
            (true, true) => -right - v,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        cmp::Ordering,
        f64::{
            consts::{FRAC_PI_2, PI},
            INFINITY, NEG_INFINITY,
        },
    };

    use approx::{abs_diff_eq, assert_abs_diff_eq};
    use fixed::types::I17F15;

    use crate::{
        atan_p2::{atan2_p2_default, atan_p2_default},
        atan_p3::{atan2_p3_default, atan_p3_default},
        atan_p5::{atan2_p5_default, atan_p5_default},
        tests::read_data,
    };

    fn test_atan<F>(f: F, data_path: &str, acceptable_error: f64)
    where
        F: Fn(i32) -> i32,
    {
        use std::i32::{MAX, MIN};

        const EXP: u32 = i32::BITS / 2 - 1;
        const K: i32 = 2_i32.pow(EXP);
        const K_2: i32 = K.pow(2);
        const RIGHT: i32 = K_2 / 2;
        const HALF_RIGHT: i32 = RIGHT / 2;

        let expected = read_data::<i32>(data_path).unwrap();

        assert_eq!(expected.len(), (K + 1) as usize);
        assert_eq!(expected[0], 0);
        assert_eq!(expected[K as usize], HALF_RIGHT);

        let mut min = INFINITY;
        let mut max = NEG_INFINITY;

        let mut verify = |x: i32, expected: i32, neg: bool| {
            let mut f = |x: i32, expected: i32| {
                assert_eq!(f(x), expected);
                let actual = expected as f64 * PI / K_2 as f64;
                let expected = (x as f64 / K as f64).atan();
                assert_abs_diff_eq!(expected, actual, epsilon = acceptable_error);
                min = min.min(actual - expected);
                max = max.max(actual - expected);
            };
            f(x, expected);
            if neg {
                f(-x, -expected);
            }
        };

        verify(0, 0, false);
        verify(MAX, RIGHT, false);
        verify(MIN, -RIGHT, false);
        verify(K_2 + 1, RIGHT, true);
        verify(K, HALF_RIGHT, true);

        for i in 1..K {
            let expected = expected[i as usize];
            let inv = RIGHT - expected;
            verify(i, expected, true);
            verify(K * K / i, inv, true);
            verify(K * K / (i + 1) + 1, inv, true);
        }

        println!("min: {min}, max: {max}");
    }

    #[rustfmt::skip] #[test] fn test_atan_p2() { test_atan(|a| atan_p2_default(I17F15::from_bits(a)), "data/atan_p2_i17f15.json", 0.003789); }
    #[rustfmt::skip] #[test] fn test_atan_p3() { test_atan(atan_p3_default, "data/atan_p3.json", 0.001601); }
    #[rustfmt::skip] #[test] fn test_atan_p5() { test_atan(atan_p5_default, "data/atan_p5.json", 0.000922); }

    fn test_atan2<F>(f: F, data_path: &str, acceptable_error: f64)
    where
        F: Fn(i32, i32) -> i32,
    {
        use std::i32::{MAX, MIN};

        const K: i32 = 2_i32.pow(i32::BITS / 2 - 1);
        const RIGHT: i32 = K.pow(2) / 2;
        const HALF_RIGHT: i32 = RIGHT / 2;

        // Find the largest error for each of the eight regions
        // that are divided by the straight lines y = x, y = -x, y = 0, x = 0.

        let mut max_error = NEG_INFINITY;

        // Calculate the expected and actual value and store value.
        let mut calc = |p: &[i32; 2]| {
            let expected = (p[1] as f64).atan2(p[0] as f64);
            let actual = f(p[1], p[0]);
            {
                let actual = actual as f64 * FRAC_PI_2 / RIGHT as f64;
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

            #[rustfmt::skip] assert_eq!(angles[0].1, 0        );
            #[rustfmt::skip] assert_eq!(angles[1].1,     RIGHT);
            #[rustfmt::skip] assert_eq!(angles[2].1, 2 * RIGHT);
            #[rustfmt::skip] assert_eq!(angles[3].1,    -RIGHT);
        }

        // On the straight lines y = x, y = -x.
        {
            let points = [[1, 1], [-1, 1], [-1, -1], [1, -1]];
            let angles = points.iter().map(&mut calc).collect::<Vec<_>>();

            #[rustfmt::skip] assert_eq!(angles[0].1,      HALF_RIGHT);
            #[rustfmt::skip] assert_eq!(angles[1].1,  3 * HALF_RIGHT);
            #[rustfmt::skip] assert_eq!(angles[2].1, -3 * HALF_RIGHT);
            #[rustfmt::skip] assert_eq!(angles[3].1,     -HALF_RIGHT);
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
            let points = to_8_points(MAX, MAX - 1, MIN, -MAX);
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

            const OFFSET_X: i32 = MAX - K + 1;
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

    #[rustfmt::skip] #[test] fn test_atan2_p2() { test_atan2(|y, x| atan2_p2_default(I17F15::from_bits(y), I17F15::from_bits(x)), "data/atan_p2_i17f15.json", 0.003789); }
    #[rustfmt::skip] #[test] fn test_atan2_p3() { test_atan2(atan2_p3_default, "data/atan_p3.json", 0.001603); }
    #[rustfmt::skip] #[test] fn test_atan2_p5() { test_atan2(atan2_p5_default, "data/atan_p5.json", 0.000928); }
}
