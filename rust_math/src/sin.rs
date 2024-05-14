use std::f64::consts::{FRAC_2_PI, FRAC_PI_2, FRAC_PI_4};

trait Consts<T> {
    const RIGHT_EXP: u32;
    const RIGHT: T;
    const RIGHT_MASK: T;
    const ONE: T;
}

macro_rules! consts_impl {
    ($(($u:ty, $t:ty)),*) => {$(
        impl Consts<$t> for $u {
            const RIGHT_EXP: u32 = <$t>::BITS / 2 - 1;
            const RIGHT: $t = 1 << Self::RIGHT_EXP;
            const RIGHT_MASK: $t = Self::RIGHT - 1;
            const ONE: $t = Self::RIGHT.pow(2);
        }
    )*};
}

trait Sin<T> {
    fn sin(x: T) -> T;
}

trait Cos<T> {
    fn cos(x: T) -> T;
}

macro_rules! sin_cos_impl_even {
    ($(($u:ty, $t:ty)),*) => {$(
        impl Cos<$t> for $u {
            fn cos(x: $t) -> $t {
                let masked = x & Self::RIGHT_MASK;
                match (x >> Self::RIGHT_EXP) & 3 {
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
    )*};
}

macro_rules! sin_cos_impl_odd {
    ($(($u:ty, $t:ty)),*) => {$(
        impl Sin<$t> for $u {
            fn sin(x: $t) -> $t {
                let masked = x & Self::RIGHT_MASK;
                let z = match (x >> Self::RIGHT_EXP) & 3 {
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
    )*};
}

/// a - b * z ^ 2
///
/// NOTE: only correctly works for b is not odd negative number.
/// Use div ops for b is 0 or positive or negative even.
macro_rules! sin_p3_cos_p4_impl {
    ($a:ident, $b:expr, $z_2:ident) => {
        ($a - (($z_2 * $b) >> Self::RIGHT_EXP))
    };
}

/// (a - b * z ^ 2) * z ^ 2
///
/// NOTE: only correctly works for b is not odd negative number.
/// Use div ops for b is 0 or positive or negative even.
macro_rules! cos_p4_sin_p5_impl {
    ($a:ident, $b:expr, $z:ident) => {{
        let z_2 = ($z * $z) >> Self::RIGHT_EXP;
        sin_p3_cos_p4_impl!($a, $b, z_2) * z_2
    }};
}

macro_rules! cos_p4_impl {
    ($t:ty, $z: ident) => {{
        const A: i32 = <$t>::K + <$t>::RIGHT;
        cos_p4_sin_p5_impl!(A, <$t>::K, $z)
    }};
}

/// (k - (2 * k - 2.5 - (k - 1.5) * x ^ 2) * x ^ 2) * x
///
/// NOTE: Prerequisition 1.5 <= k
macro_rules! sin_p5_impl {
    ($t:ty, $z: ident) => {{
        const A: i32 = <$t>::K * 2 - <$t>::RIGHT * 5 / 2;
        const B: i32 = <$t>::K - <$t>::RIGHT * 3 / 2;
        (<$t>::K - (cos_p4_sin_p5_impl!(A, B, $z) >> Self::RIGHT_EXP)) * $z
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

struct SinP3_16384();
struct SinP5_51472();
struct SinP5_51437();
struct CosP2I32();
struct CosP4_7032();
struct CosP4_7384();

consts_impl![
    (SinP3_16384, i32),
    (SinP5_51472, i32),
    (SinP5_51437, i32),
    (CosP2I32, i32),
    (CosP4_7032, i32),
    (CosP4_7384, i32)
];

impl CosP2I32 {
    fn cos_detail(z: i32) -> i32 {
        z * z
    }
}

impl SinP3_16384 {
    /// (1.5 - 0.5 * z ^ 2) * z
    fn sin_detail(z: i32) -> i32 {
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
    /// (k + 1 - k * z ^ 2) * z ^ 2  
    /// k = 1 - pi / 4
    fn cos_detail(z: i32) -> i32 {
        cos_p4_impl!(CosP4_7032, z)
    }
}

impl CosP4_7384 {
    const K: i32 = cos_p4o_k!() as i32;

    /// (k + 1 - k * z ^ 2) * z ^ 2  
    /// k = 5 * (1 - 3 / pi)
    fn cos_detail(z: i32) -> i32 {
        cos_p4_impl!(CosP4_7384, z)
    }
}

impl SinP5_51472 {
    const K: i32 = sin_p5_k!() as i32;

    /// (k - (2 * k - 2.5 - (k - 1.5) * z ^ 2) * z ^ 2) * z  
    /// k = pi / 2
    fn sin_detail(z: i32) -> i32 {
        sin_p5_impl!(SinP5_51472, z)
    }
}

impl SinP5_51437 {
    const K: i32 = sin_p5o_k!() as i32;

    /// (k - (2 * k - 2.5 - (k - 1.5) * z ^ 2) * z ^ 2) * z  
    /// k = 4 * (3 / pi - 9 / 16)
    fn sin_detail(z: i32) -> i32 {
        sin_p5_impl!(SinP5_51437, z)
    }
}

sin_cos_impl_even![(CosP2I32, i32), (CosP4_7032, i32), (CosP4_7384, i32)];
sin_cos_impl_odd![(SinP3_16384, i32), (SinP5_51472, i32), (SinP5_51437, i32)];

pub fn sin_p2_i32(x: i32) -> i32 {
    CosP2I32::sin(x)
}

pub fn cos_p2_i32(x: i32) -> i32 {
    CosP2I32::cos(x)
}

pub fn sin_p3_16384(x: i32) -> i32 {
    SinP3_16384::sin(x)
}

pub fn cos_p3_16384(x: i32) -> i32 {
    SinP3_16384::cos(x)
}

/// Approximate the sine function by the 4th order polynomial derived by Taylor expansion.
pub fn sin_p4_7032(x: i32) -> i32 {
    CosP4_7032::sin(x)
}

/// Approximate the cosine function by the 4th order polynomial derived by Taylor expansion.
///
/// 1 - (k + 1 - k * z ^ 2) * z ^ 2  
/// k = 1 - pi / 4
pub fn cos_p4_7032(x: i32) -> i32 {
    CosP4_7032::cos(x)
}

/// Approximate the sine function by the 4th order polynomial derived by Taylor expansion  with
/// coefficients which is adjusted so that the average of the errors is 0.
pub fn sin_p4_7384(x: i32) -> i32 {
    CosP4_7384::sin(x)
}

/// Approximate the cosine function by the 4th order polynomial derived by Taylor expansion  with
/// coefficients which is adjusted so that the average of the errors is 0.
///
/// 1 - (k + 1 - k * z ^ 2) * z ^ 2  
/// k = 5 * (1 - 3 / pi)
pub fn cos_p4_7384(x: i32) -> i32 {
    CosP4_7384::cos(x)
}

/// Approximate the sine function by the 5th order polynomial derived by Taylor expansion.
///
/// (k - (2 * k - 2.5 - (k - 1.5) * x ^ 2) * x ^ 2) * x  
/// k = pi / 2
pub fn sin_p5_51472(x: i32) -> i32 {
    SinP5_51472::sin(x)
}

/// Approximate the cosine function by the 5th order polynomial derived by Taylor expansion.
pub fn cos_p5_51472(x: i32) -> i32 {
    SinP5_51472::cos(x)
}

/// Approximate the sine function by the 5th order polynomial derived by Taylor expansion with
/// coefficients which is adjusted so that the average of the errors is 0.
///
/// (k - (2 * k - 2.5 - (k - 1.5) * x ^ 2) * x ^ 2) * x  
/// k = 4 * (3 / pi - 9 / 16)
pub fn sin_p5_51437(x: i32) -> i32 {
    SinP5_51437::sin(x)
}

/// Approximate the cosine function by the 5th order polynomial derived by Taylor expansion with
/// coefficients which is adjusted so that the average of the errors is 0.
pub fn cos_p5_51437(x: i32) -> i32 {
    SinP5_51437::cos(x)
}

#[cfg(test)]
mod tests {
    use std::fmt::Debug;

    use anyhow::{Context, Result};
    use approx::abs_diff_eq;
    use num_traits::Zero;
    use rand::Rng;
    use rstest::rstest;

    use crate::tests::read_data;

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

    const RIGHT_EXP: u32 = i32::BITS / 2 - 1;
    const RIGHT: i32 = 1 << RIGHT_EXP;
    const RIGHT_AS_U32: u32 = RIGHT as u32;
    const RIGHT_AS_USIZE: usize = RIGHT as usize;
    const RIGHT_MASK: i32 = RIGHT - 1;
    const FULL: i32 = 4 * RIGHT;
    const ONE: i32 = 1 << (2 * RIGHT_EXP);
    const ONE_AS_F64: f64 = ONE as f64;
    const NEG_FULL: i32 = -FULL;
    const FRAC_PI_STRAIGHT: f64 = FRAC_PI_2 / RIGHT as f64;

    fn to_real(x: i32) -> f64 {
        x as f64 / ONE_AS_F64
    }
    fn to_rad(x: i32) -> f64 {
        x as f64 * FRAC_PI_STRAIGHT
    }
    fn ensure_eq<T>(l: T, r: T) -> Result<()>
    where
        T: Debug + PartialEq,
    {
        Ok(anyhow::ensure!(l == r, "l: {:?}, r: {:?}", l, r))
    }
    fn ensure_abs_diff_eq(l: f64, r: f64, epsilon: f64) -> Result<()> {
        Ok(anyhow::ensure!(
            abs_diff_eq!(l, r, epsilon = epsilon),
            "l: {l}, r: {r}"
        ))
    }

    #[rstest]
    #[case(sin_p2_i32, cos_p2_i32, "data/sin_p2.json", 0.056010)]
    #[case(sin_p3_16384, cos_p3_16384, "data/sin_p3.json", 0.020017)]
    #[case(sin_p4_7032, cos_p4_7032, "data/sin_p4_7032.json", 0.002819)]
    #[case(sin_p4_7384, cos_p4_7384, "data/sin_p4_7384.json", 0.001174)]
    #[case(sin_p5_51472, cos_p5_51472, "data/sin_p5_51472.json", 0.000425)]
    #[case(sin_p5_51437, cos_p5_51437, "data/sin_p5_51437.json", 0.000226)]
    fn sin_test(
        #[case] sin: impl Fn(i32) -> i32,
        #[case] cos: impl Fn(i32) -> i32,
        #[case] data_path: &str,
        #[case] acceptable_error: f64,
    ) -> Result<()> {
        let data: Vec<i32> = read_data(data_path)?;
        anyhow::ensure!(data[0].is_zero(), "{:?}", data[0]);
        anyhow::ensure!(
            data.iter().skip(1).cloned().all(i32::is_positive),
            "{:?}",
            data
        );
        ensure_eq(data[RIGHT_AS_USIZE], ONE)?;
        ensure_eq(data.len(), RIGHT_AS_USIZE + 1)?;

        let test_sin = |x| -> Result<()> {
            let actual = sin(x);
            {
                let masked = (x & RIGHT_MASK) as usize;
                let expected = match (x >> RIGHT_EXP) & 3 {
                    0 => data[masked],
                    1 => data[RIGHT_AS_USIZE - masked],
                    2 => -data[masked],
                    3 => -data[RIGHT_AS_USIZE - masked],
                    _ => unreachable!(),
                };
                ensure_eq(expected, actual).with_context(|| format!("x: {x}"))?;
            }
            ensure_abs_diff_eq(to_rad(x).sin(), to_real(actual), acceptable_error)
                .with_context(|| format!("x: {x}, actual: {actual}"))
        };

        let test_cos = |x| -> Result<_> {
            let actual = cos(x);
            {
                let masked = (x & RIGHT_MASK) as usize;
                let expected = match (x >> RIGHT_EXP) & 3 {
                    0 => data[RIGHT_AS_USIZE - masked],
                    1 => -data[masked],
                    2 => -data[RIGHT_AS_USIZE - masked],
                    3 => data[masked],
                    _ => unreachable!(),
                };
                ensure_eq(expected, actual).with_context(|| format!("x: {x}"))?;
            }
            ensure_abs_diff_eq(to_rad(x).cos(), to_real(actual), acceptable_error)
                .with_context(|| format!("x: {x}, actual: {actual}"))
        };

        for x in (0..=u32::MAX / RIGHT_AS_U32).map(|i| (i * RIGHT_AS_U32) as i32) {
            test_sin(x)?;
            test_cos(x)?;
            test_sin(x + 1)?;
            test_cos(x + 1)?;
            test_sin(x + RIGHT_MASK)?;
            test_cos(x + RIGHT_MASK)?;
        }

        const STARTS: [i32; 4] = [i32::MIN.wrapping_sub(FULL), i32::MIN, NEG_FULL, 0];
        for q in 0..4 {
            let q = q * RIGHT;
            for start in STARTS {
                for x in 2..RIGHT - 1 {
                    test_sin(start + q + x)?;
                    test_cos(start + q + x)?;
                }
            }
        }

        let mut rng = rand::thread_rng();
        for _ in 0..999 {
            let x = rng.gen_range(i32::MIN..=i32::MAX);
            test_sin(x)?;
            test_cos(x)?;
        }

        Ok(())
    }
}
