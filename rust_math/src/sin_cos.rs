use std::f64::consts::{FRAC_2_PI, FRAC_PI_2, FRAC_PI_4};

trait Consts<T> {
    const RIGHT_EXP: u32;
    const RIGHT: T;
    const RIGHT_MASK: T;
    const ONE: T;
}

macro_rules! consts_impl {
    ($u:ty, $t:ty) => {
        impl Consts<$t> for $u {
            const RIGHT_EXP: u32 = <$t>::BITS / 2 - 1;
            const RIGHT: $t = 1 << Self::RIGHT_EXP;
            const RIGHT_MASK: $t = Self::RIGHT - 1;
            const ONE: $t = Self::RIGHT.pow(2);
        }
    };
}

pub(crate) trait Sin<T> {
    fn sin(x: T) -> T;
}

pub(crate) trait Cos<T> {
    fn cos(x: T) -> T;
}

macro_rules! even_sin_cos_impl {
    ($u:ty, $t:ty) => {
        impl Cos<$t> for $u {
            fn cos(x: $t) -> $t {
                let masked = x & Self::RIGHT_MASK;
                let quadrant = (x >> Self::RIGHT_EXP) & 3;
                match quadrant {
                    1 => Self::cos_detail(Self::RIGHT - masked) - Self::ONE,
                    3 => Self::ONE - Self::cos_detail(Self::RIGHT - masked),
                    2 => Self::cos_detail(masked) - Self::ONE,
                    0 => Self::ONE - Self::cos_detail(masked),
                    _ => unreachable!(),
                }
            }
        }
        impl Sin<$t> for $u {
            fn sin(x: $t) -> $t {
                Self::cos(x.wrapping_sub(Self::RIGHT))
            }
        }
    };
}

macro_rules! odd_sin_cos_impl {
    ($u:ty, $t:ty) => {
        impl Sin<$t> for $u {
            fn sin(x: $t) -> $t {
                let masked = x & Self::RIGHT_MASK;
                let quadrant = (x >> Self::RIGHT_EXP) & 3;
                let z = match quadrant {
                    1 => Self::RIGHT - masked,
                    3 => masked - Self::RIGHT,
                    2 => -masked,
                    0 => masked,
                    _ => unreachable!(),
                };
                Self::sin_detail(z)
            }
        }
        impl Cos<$t> for $u {
            fn cos(x: $t) -> $t {
                Self::sin(x.wrapping_add(Self::RIGHT))
            }
        }
    };
}

/// a - b * z ^ 2
macro_rules! sin_p3_cos_p4_impl {
    ($a:ident, $b:expr, $z_2:ident) => {
        ($a - (($z_2 * $b) >> Self::RIGHT_EXP))
    };
}

/// (a - b * z ^ 2) * z ^ 2
macro_rules! cos_p4_sin_p5_impl {
    ($a:ident, $b:expr, $z:ident) => {{
        let z_2 = ($z * $z) >> Self::RIGHT_EXP;
        sin_p3_cos_p4_impl!($a, $b, z_2) * z_2
    }};
}

/// (k - (2 * k - 2.5 - (k - 1.5) * x ^ 2) * x ^ 2) * x
macro_rules! sin_p5_impl {
    ($k:expr, $right:expr, $z: ident) => {{
        const A: i32 = $k * 2 - $right * 5 / 2;
        const B: i32 = $k - $right * 3 / 2;
        ($k - (cos_p4_sin_p5_impl!(A, B, $z) >> Self::RIGHT_EXP)) * $z
    }};
}

/// 1 - pi / 4
macro_rules! cos_p4_k {
    () => {
        ((1.0 - FRAC_PI_4) * Self::RIGHT as f64 + 0.5)
    };
}

/// 5 * (1 - 3 / pi)
macro_rules! cos_p4o_k {
    () => {
        (5.0 * (1.0 - 1.5 * FRAC_2_PI) * Self::RIGHT as f64 + 0.5)
    };
}

/// pi / 2
macro_rules! sin_p5_k {
    () => {
        (FRAC_PI_2 * Self::RIGHT as f64 + 0.5)
    };
}

/// 4 * (3 / pi - 9 / 16)
macro_rules! sin_p5o_k {
    () => {
        (4.0 * (1.5 * FRAC_2_PI - 9.0 / 16.0) * Self::RIGHT as f64 + 0.5)
    };
}

pub(crate) struct SinP3_16384();
pub(crate) struct SinP5_51472();
pub(crate) struct SinP5_51437();
pub(crate) struct CosP2I32();
pub(crate) struct CosP4_7032();
pub(crate) struct CosP4_7384();

consts_impl!(SinP3_16384, i32);
consts_impl!(SinP5_51472, i32);
consts_impl!(SinP5_51437, i32);
consts_impl!(CosP2I32, i32);
consts_impl!(CosP4_7032, i32);
consts_impl!(CosP4_7384, i32);

impl CosP2I32 {
    pub fn cos_detail(z: i32) -> i32 {
        z * z
    }
}

impl SinP3_16384 {
    /// (1.5 - 0.5 * x ^ 2) * x
    pub fn sin_detail(z: i32) -> i32 {
        const B: i32 = SinP3_16384::RIGHT / 2;
        const A: i32 = SinP3_16384::RIGHT + B;
        let z_2 = (z * z) >> Self::RIGHT_EXP;
        sin_p3_cos_p4_impl!(A, B, z_2) * z
    }
}

impl CosP4_7032 {
    const K: i32 = cos_p4_k!() as i32;

    /// Approximate the cosine function by the 4th order polynomial derived by Taylor expansion.
    ///
    /// 1 - (a + 1 - a * z ^ 2) * z ^ 2  
    /// a = 1 - pi / 4
    pub fn cos_detail(z: i32) -> i32 {
        const A: i32 = CosP4_7032::K + CosP4_7032::RIGHT;
        cos_p4_sin_p5_impl!(A, CosP4_7032::K, z)
    }
}

impl CosP4_7384 {
    const K: i32 = cos_p4o_k!() as i32;

    /// (k + 1 - k * z ^ 2) * z ^ 2
    fn cos_detail(z: i32) -> i32 {
        const A: i32 = CosP4_7384::K + CosP4_7384::RIGHT;
        cos_p4_sin_p5_impl!(A, CosP4_7384::K, z)
    }
}

impl SinP5_51472 {
    const K: i32 = sin_p5_k!() as i32;

    /// (k - (2 * k - 2.5 - (k - 1.5) * x ^ 2) * x ^ 2) * x
    fn sin_detail(z: i32) -> i32 {
        sin_p5_impl!(SinP5_51472::K, SinP5_51472::RIGHT, z)
    }
}

impl SinP5_51437 {
    const K: i32 = sin_p5o_k!() as i32;

    /// (k - (2 * k - 2.5 - (k - 1.5) * x ^ 2) * x ^ 2) * x
    fn sin_detail(z: i32) -> i32 {
        sin_p5_impl!(SinP5_51437::K, SinP5_51437::RIGHT, z)
    }
}

even_sin_cos_impl!(CosP2I32, i32);
even_sin_cos_impl!(CosP4_7032, i32);
even_sin_cos_impl!(CosP4_7384, i32);

odd_sin_cos_impl!(SinP3_16384, i32);
odd_sin_cos_impl!(SinP5_51472, i32);
odd_sin_cos_impl!(SinP5_51437, i32);

#[cfg(test)]
mod tests {
    use std::{fmt::Display, ops::Neg};

    use anyhow::Context;
    use anyhow::Result;
    use approx::{abs_diff_eq, assert_abs_diff_eq};
    use rand::Rng;
    use rstest::rstest;

    use crate::{
        cos_p2_i32, cos_p3_16384, cos_p4_7032, cos_p4_7384, cos_p5_51437, cos_p5_51472, sin_p2_i32,
        sin_p3_16384, sin_p4_7032, sin_p4_7384, sin_p5_51437, sin_p5_51472, tests::read_data,
    };

    use super::*;

    #[rstest]
    //#[case(2, CosP4_2::K)]
    //#[case(27, CosP4_27::K)]
    #[case(7032, CosP4_7032::K)]
    //#[case(13, SinP5_13::K)]
    //#[case(201, SinP5_201::K)]
    #[case(51472, SinP5_51472::K)]
    //#[case(2, CosP4_2::K)]
    //#[case(29, CosP4_29::K)]
    #[case(7384, CosP4_7384::K)]
    //#[case(13, SinP5_13::K)]
    //#[case(201, SinP5_201::K)]
    #[case(51437, SinP5_51437::K)]
    fn test_constants(#[case] expected: i32, #[case] actual: i32) {
        assert_eq!(expected, actual);
    }

    const RIGHT: i32 = 1 << (i32::BITS / 2 - 1);
    const RIGHT_AS_USIZE: usize = RIGHT as usize;
    const RIGHT_MASK: i32 = RIGHT - 1;
    const RIGHT_MASK_AS_USIZE: usize = RIGHT_MASK as usize;
    const STRAIGHT: i32 = 2 * RIGHT;
    const FULL: i32 = 2 * STRAIGHT;
    const FULL_MASK: i32 = FULL - 1;
    const ONE: i32 = RIGHT.pow(2);
    const ONE_AS_F64: f64 = ONE as f64;
    const NEG_RIGHT: i32 = -RIGHT;
    const NEG_STRAIGHT: i32 = -STRAIGHT;
    const NEG_FULL: i32 = -FULL;
    const NEG_ONE: i32 = -ONE;
    const FRAC_PI_STRAIGHT: f64 = FRAC_PI_2 / RIGHT as f64;

    fn ensure_all_eq<A, Actuals>(expected: A, actuals: Actuals) -> Result<()>
    where
        A: PartialEq + Display,
        Actuals: IntoIterator<Item = A>,
    {
        for actual in actuals {
            anyhow::ensure!(expected == actual, "expected: {expected}, actual: {actual}");
        }
        Ok(())
    }

    #[rstest]
    #[case(sin_p2_i32, cos_p2_i32, "data/sin_p2.json", 0.056010)]
    #[case(sin_p3_16384, cos_p3_16384, "data/sin_p3.json", 0.020017)]
    #[case(sin_p4_7032, cos_p4_7032, "data/sin_p4_7032.json", 0.002819)]
    #[case(sin_p4_7384, cos_p4_7384, "data/sin_p4_7384.json", 0.001174)]
    #[case(sin_p5_51472, cos_p5_51472, "data/sin_p5_51472.json", 0.000425)]
    #[case(sin_p5_51437, cos_p5_51437, "data/sin_p5_51437.json", 0.000226)]
    fn new_sin_test(
        #[case] sin: impl Fn(i32) -> i32,
        #[case] cos: impl Fn(i32) -> i32,
        #[case] data_path: &str,
        #[case] acceptable_error: f64,
    ) -> Result<()> {
        let sin = |x| -> Result<_> {
            let actual = sin(x);
            let actual_real = actual as f64 / ONE_AS_F64;
            let expected = (x as f64 * FRAC_PI_STRAIGHT).sin();
            anyhow::ensure!(
                abs_diff_eq!(actual_real, expected, epsilon = acceptable_error),
                "x: {x}, expected: {expected}, actual_real: {actual_real}, actual: {actual}"
            );
            Ok(actual)
        };

        let cos = |x| -> Result<_> {
            let actual = cos(x);
            let actual_real = actual as f64 / ONE_AS_F64;
            let expected = (x as f64 * FRAC_PI_STRAIGHT).cos();
            anyhow::ensure!(
                abs_diff_eq!(actual_real, expected, epsilon = acceptable_error),
                "x: {x}, expected: {expected}, actual_real: {actual_real}, actual: {actual}"
            );
            Ok(actual)
        };

        let data: Vec<i32> = read_data(data_path)?;

        {
            let a = data[1];
            let b = data[RIGHT_MASK_AS_USIZE];
            anyhow::ensure!(a.is_positive(), "a: {a}");
            anyhow::ensure!(b.is_positive(), "b: {b}");
            for x in
                (0..=u32::MAX / FULL as u32).map(|i| i32::MIN.strict_add_unsigned(i * FULL as u32))
            {
                ensure_all_eq(ONE, [cos(x)?, sin(x + RIGHT)?])?;
                ensure_all_eq(NEG_ONE, [cos(x + STRAIGHT)?, sin(x + STRAIGHT + RIGHT)?])?;
                ensure_all_eq(
                    0,
                    [
                        sin(x)?,
                        cos(x + RIGHT)?,
                        sin(x + STRAIGHT)?,
                        cos(x + STRAIGHT + RIGHT)?,
                    ],
                )?;
                ensure_all_eq(
                    a,
                    [
                        sin(x + 1)?,
                        cos(x + RIGHT - 1)?,
                        sin(x + STRAIGHT - 1)?,
                        cos(x + STRAIGHT + RIGHT + 1)?,
                    ],
                )?;
                ensure_all_eq(
                    b,
                    [
                        cos(x + 1)?,
                        sin(x + RIGHT - 1)?,
                        sin(x + RIGHT + 1)?,
                        cos(x + FULL_MASK)?,
                    ],
                )?;
                ensure_all_eq(
                    -a,
                    [
                        cos(x + RIGHT + 1)?,
                        sin(x + STRAIGHT + 1)?,
                        cos(x + STRAIGHT + RIGHT - 1)?,
                        sin(x + FULL_MASK)?,
                    ],
                )?;
                ensure_all_eq(
                    -b,
                    [
                        cos(x + STRAIGHT - 1)?,
                        cos(x + STRAIGHT + 1)?,
                        sin(x + STRAIGHT + RIGHT - 1)?,
                        sin(x + STRAIGHT + RIGHT + 1)?,
                    ],
                )?;
            }
        }

        const STARTS: [i32; 4] = [i32::MIN.wrapping_sub(FULL), i32::MIN, NEG_FULL, 0];
        for start in STARTS {
            for x in 2..RIGHT - 1 {
                let a = data[x as usize];
                anyhow::ensure!(a.is_positive(), "x: {x}, a: {a}");
                ensure_all_eq(
                    a,
                    [
                        sin(start + x)?,
                        sin(start + STRAIGHT - x)?,
                        cos(start + RIGHT - x)?,
                        cos(start + RIGHT + x + STRAIGHT)?,
                    ],
                )
                .with_context(|| format!("x: {x}"))?;
                ensure_all_eq(
                    -a,
                    [
                        sin(start + STRAIGHT + x)?,
                        sin(start + FULL_MASK - x + 1)?,
                        cos(start + RIGHT + x)?,
                        cos(start + RIGHT - x + STRAIGHT)?,
                    ],
                )
                .with_context(|| format!("x: {x}"))?;
            }
        }

        let mut rng = rand::thread_rng();
        for _ in 0..999 {
            let x = rng.gen_range(i32::MIN..=i32::MAX);
            let sin = sin(x)?;
            let cos = cos(x)?;
            let i = (x & RIGHT_MASK) as usize;
            let a = data[i];
            let b = data[RIGHT_AS_USIZE - i];
            match (x >> (i32::BITS / 2 - 1)) & 3 {
                0 => {
                    anyhow::ensure!(sin == a, "x: {x}, sin: {sin}, a: {a}");
                    anyhow::ensure!(cos == b, "x: {x}, sin: {sin}, b: {b}");
                }
                1 => {
                    anyhow::ensure!(sin == b, "x: {x}, sin: {sin}, b: {b}");
                    anyhow::ensure!(cos == -a, "x: {x}, sin: {sin}, a: {a}");
                }
                2 => {
                    anyhow::ensure!(sin == -a, "x: {x}, sin: {sin}, a: {a}");
                    anyhow::ensure!(cos == -b, "x: {x}, sin: {sin}, b: {b}");
                }
                3 => {
                    anyhow::ensure!(sin == -b, "x: {x}, sin: {sin}, b: {b}");
                    anyhow::ensure!(cos == a, "x: {x}, sin: {sin}, a: {a}");
                }
                _ => unreachable!(),
            }
        }

        Ok(())
    }

    #[rstest]
    #[case(sin_p2_i32)]
    #[case(sin_p3_16384)]
    #[case(sin_p4_7032)]
    #[case(sin_p5_51472)]
    #[case(sin_p4_7384)]
    #[case(sin_p5_51437)]
    fn test_sin(#[case] f: impl Fn(i32) -> i32) {
        assert_eq!(f(0), 0);
        assert_eq!(f(STRAIGHT), 0);
        assert_eq!(f(NEG_STRAIGHT), 0);
        assert_eq!(f(RIGHT), ONE);
        assert_eq!(f(NEG_RIGHT), NEG_ONE);
    }

    #[rstest]
    #[case(cos_p2_i32)]
    #[case(cos_p3_16384)]
    #[case(cos_p4_7032)]
    #[case(cos_p5_51472)]
    #[case(cos_p4_7384)]
    #[case(cos_p5_51437)]
    fn test_cos(#[case] f: impl Fn(i32) -> i32) {
        assert_eq!(f(0), ONE);
        assert_eq!(f(RIGHT), 0);
        assert_eq!(f(NEG_RIGHT), 0);
        assert_eq!(f(STRAIGHT), NEG_ONE);
        assert_eq!(f(NEG_STRAIGHT), NEG_ONE);
    }

    fn test_sin_cos(
        f: impl Fn(i32) -> i32,
        data_path: &str,
        to_period: impl Fn(&[i32]) -> Vec<i32>,
        f_std: impl Fn(f64) -> f64,
        acceptable_error: f64,
    ) {
        // 5th mersenne prime
        const STEP: usize = 8191;

        let right_as_usize = RIGHT as usize;
        let data = read_data(data_path).unwrap();

        assert_eq!(data.len(), right_as_usize + 1);
        assert_eq!(FULL % RIGHT, 0);
        assert_eq!(i32::MIN % FULL, 0);
        assert_eq!(i32::MAX % FULL, FULL - 1);

        let data = to_period(&data);

        assert_eq!(data.len(), FULL as usize);

        let x = (-FULL - 1..=FULL + 1)
            .chain(i32::MAX - FULL..=i32::MAX)
            .chain(i32::MIN..=i32::MIN + FULL + 1)
            .chain((i32::MIN..=i32::MAX).step_by(right_as_usize))
            .chain((i32::MIN..=i32::MAX).skip(1).step_by(right_as_usize))
            .chain(
                (i32::MIN..=i32::MAX)
                    .skip(right_as_usize - 1)
                    .step_by(right_as_usize),
            )
            .chain((i32::MIN..=i32::MAX).step_by(STEP))
            .chain((i32::MIN..=i32::MAX).rev().step_by(STEP));

        let frac_pi_straight = FRAC_PI_2 / RIGHT as f64;
        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;

        for x in x {
            let expected = data[(x & FULL_MASK) as usize];

            // The value can be greater than 1 or less than -1
            //assert!(expected.abs() <= ONE, "expected: {expected}, ONE: {ONE}");

            assert_eq!(expected, f(x));

            let actual = f_std(x as f64 * frac_pi_straight);

            if (x % RIGHT) != 0 || expected != 0 {
                assert_eq!(expected.is_negative(), actual.is_sign_negative());
                assert_eq!(expected.is_positive(), actual.is_sign_positive());
            }

            let expected = expected as f64 / ONE as f64;

            assert_abs_diff_eq!(expected, actual, epsilon = acceptable_error);

            let diff = actual - expected;

            min = min.min(diff);
            max = max.max(diff);
        }

        println!("min: {min}, max: {max}");
    }

    fn to_sin_period_odd(data: &[i32]) -> Vec<i32> {
        let n = data.len() - 1;
        let iter = data.iter().cloned();
        let iter = iter.clone().take(n).chain(iter.rev().take(n));
        iter.clone().chain(iter.map(Neg::neg)).collect()
    }

    fn to_sin_period_even(data: &[i32]) -> Vec<i32> {
        let n = data.len() - 1;
        let iter = data.iter().cloned();
        let iter = iter.clone().rev().take(n).chain(iter.take(n));
        iter.clone().chain(iter.map(Neg::neg)).collect()
    }

    fn to_cos_period_even(data: &[i32]) -> Vec<i32> {
        let n = data.len() - 1;
        let iter = data.iter().cloned();
        let iter = iter.clone().take(n).chain(iter.rev().take(n).map(Neg::neg));
        iter.clone().chain(iter.map(Neg::neg)).collect()
    }

    fn to_cos_period_odd(data: &[i32]) -> Vec<i32> {
        let n = data.len() - 1;
        let iter = data.iter().cloned();
        let iter = iter.clone().rev().take(n).chain(iter.take(n).map(Neg::neg));
        iter.clone().chain(iter.map(Neg::neg)).collect()
    }

    #[rstest]
    #[case(sin_p2_i32)]
    #[case(sin_p3_16384)]
    #[case(sin_p4_7032)]
    #[case(sin_p5_51472)]
    #[case(sin_p4_7384)]
    #[case(sin_p5_51437)]
    #[case(cos_p2_i32)]
    #[case(cos_p3_16384)]
    #[case(cos_p4_7032)]
    #[case(cos_p5_51472)]
    #[case(cos_p4_7384)]
    #[case(cos_p5_51437)]
    fn test_common(#[case] f: impl Copy + Fn(i32) -> i32) {
        const FULL_AS_USIZE: usize = FULL as usize;
        let expected = (0..FULL).map(f).collect::<Vec<_>>();
        for (i, x) in (NEG_FULL..).take(FULL_AS_USIZE).enumerate() {
            assert_eq!(f(x), expected[i]);
        }
        for (i, x) in (i32::MIN..).take(FULL_AS_USIZE).enumerate() {
            assert_eq!(f(x), expected[i]);
        }
        for (i, x) in (i32::MIN.wrapping_sub(FULL)..=i32::MAX).enumerate() {
            assert_eq!(f(x), expected[i]);
        }
        assert_eq!(expected[0], f(i32::MIN + FULL));
        assert_eq!(expected[1], f(i32::MIN + FULL + 1));
        assert_eq!(*expected.last().unwrap(), f(i32::MAX - FULL));
    }

    #[rustfmt::skip] #[test] fn test_sin_p2()  { test_sin_cos(sin_p2_i32,   "data/cos_p2.json",  to_sin_period_even, f64::sin, 0.056010); }
    #[rustfmt::skip] #[test] fn test_sin_p3()  { test_sin_cos(sin_p3_16384, "data/sin_p3.json",  to_sin_period_odd,  f64::sin, 0.020017); }
    #[rustfmt::skip] #[test] fn test_sin_p4()  { test_sin_cos(sin_p4_7032,  "data/cos_p4.json",  to_sin_period_even, f64::sin, 0.002819); }
    #[rustfmt::skip] #[test] fn test_sin_p5()  { test_sin_cos(sin_p5_51472, "data/sin_p5.json",  to_sin_period_odd,  f64::sin, 0.000425); }
    #[rustfmt::skip] #[test] fn test_sin_p4o() { test_sin_cos(sin_p4_7384,  "data/cos_p4o.json", to_sin_period_even, f64::sin, 0.001174); }
    #[rustfmt::skip] #[test] fn test_sin_p5o() { test_sin_cos(sin_p5_51437, "data/sin_p5o.json", to_sin_period_odd,  f64::sin, 0.000226); }
    #[rustfmt::skip] #[test] fn test_cos_p2()  { test_sin_cos(cos_p2_i32,   "data/cos_p2.json",  to_cos_period_even, f64::cos, 0.056010); }
    #[rustfmt::skip] #[test] fn test_cos_p3()  { test_sin_cos(cos_p3_16384, "data/sin_p3.json",  to_cos_period_odd,  f64::cos, 0.020017); }
    #[rustfmt::skip] #[test] fn test_cos_p4()  { test_sin_cos(cos_p4_7032,  "data/cos_p4.json",  to_cos_period_even, f64::cos, 0.002819); }
    #[rustfmt::skip] #[test] fn test_cos_p5()  { test_sin_cos(cos_p5_51472, "data/sin_p5.json",  to_cos_period_odd,  f64::cos, 0.000425); }
    #[rustfmt::skip] #[test] fn test_cos_p4o() { test_sin_cos(cos_p4_7384,  "data/cos_p4o.json", to_cos_period_even, f64::cos, 0.001174); }
    #[rustfmt::skip] #[test] fn test_cos_p5o() { test_sin_cos(cos_p5_51437, "data/sin_p5o.json", to_cos_period_odd,  f64::cos, 0.000226); }
}
