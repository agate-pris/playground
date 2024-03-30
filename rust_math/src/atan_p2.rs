use std::f64::consts::PI;

use fixed::{
    traits::Fixed,
    types::{
        I10F22, I10F6, I11F21, I11F5, I12F20, I12F4, I13F19, I13F3, I14F18, I15F17, I16F16, I17F15,
        I18F14, I19F13, I20F12, I21F11, I22F10, I23F9, I24F8, I25F7, I26F6, I2F6, I3F5, I6F10,
        I6F26, I7F25, I7F9, I8F24, I8F8, I9F23, I9F7,
    },
};
use num_traits::{AsPrimitive, ConstZero, NumOps, PrimInt, Signed};
use primitive_promotion::PrimitivePromotionExt;

use crate::{
    atan::{atan2_impl, atan_impl},
    bits::Bits,
};

pub trait AtanP2Default {
    type Bits;
    const A: Self::Bits;
}

macro_rules! impl_atan_p2_default_fixed {
    ($($t:ty, $a:expr),*) => {
        $(
            impl AtanP2Default for $t {
                type Bits = <Self as Fixed>::Bits;
                const A: Self::Bits = $a;
            }
        )*
    };
}

impl_atan_p2_default_fixed!(
    I3F5, 0, I2F6, 0, I13F3, 179, I12F4, 91, I11F5, 47, I10F6, 24, I9F7, 13, I8F8, 8, I7F9, 5,
    I6F10, 3, I26F6, 1458371, I25F7, 729188, I24F8, 364590, I23F9, 182295, I22F10, 91148, I21F11,
    45575, I20F12, 22789, I19F13, 11395, I18F14, 5699, I17F15, 2850, I16F16, 1426, I15F17, 714,
    I14F18, 358, I13F19, 180, I12F20, 91, I11F21, 47, I10F22, 25, I9F23, 14, I8F24, 8, I7F25, 5,
    I6F26, 3
);

/// ```rust
/// use rust_math::atan_p2::*;
/// const EXP: u32 = i32::BITS / 2 - 1;
/// let result = calc_default_p2_k::<i32>(EXP);
/// assert_eq!(result, 2847);
/// ```
pub fn calc_default_p2_k<T>(exp: u32) -> T
where
    T: 'static + Copy,
    f64: AsPrimitive<T>,
{
    let k = 2.0_f64.powi(exp as i32);
    (0.273 / PI * k).round_ties_even().as_()
}

fn atan_p2_impl<T>(x: T, x_abs: T, x_k: T, a: T, k: T) -> T
where
    T: 'static + Copy + NumOps,
    i8: AsPrimitive<T>,
{
    x * (k / 4.as_() + a * (x_k - x_abs) / x_k)
}

/// ```rust
/// use std::f64::consts::PI;
/// use approx::assert_abs_diff_eq;
/// use rust_math::atan_p2::*;
/// const EXP: u32 = i32::BITS / 2 - 1;
/// const K: i32 = 2_i32.pow(EXP);
/// let a = calc_default_p2_k::<i32>(EXP);
/// let result = atan_p2(1000 * K / 1732, K, a, K);
/// assert_abs_diff_eq!(
///     PI / 6.0,
///     result as f64 * PI / K.pow(2) as f64,
///     epsilon = 0.0039,
/// );
/// ```
pub fn atan_p2<T>(x: T, x_k: T, a: T, k: T) -> T
where
    <T as PrimitivePromotionExt>::PrimitivePromotion: PartialOrd + AsPrimitive<T> + Signed,
    T: AsPrimitive<<T as PrimitivePromotionExt>::PrimitivePromotion>
        + PrimitivePromotionExt
        + Signed,
    i8: AsPrimitive<T>,
{
    atan_impl(x, x_k, |x, x_abs| atan_p2_impl(x, x_abs, x_k, a, k))
}

/// ```rust
/// use std::f64::consts::PI;
/// use approx::assert_abs_diff_eq;
/// use rust_math::atan_p2::*;
/// const EXP: u32 = i32::BITS / 2 - 1;
/// const K: i32 = 2_i32.pow(EXP);
/// let result = atan_p2_default(1000 * K / 1732);
/// assert_abs_diff_eq!(
///     PI / 6.0,
///     result as f64 * PI / K.pow(2) as f64,
///     epsilon = 0.0039,
/// );
/// ```
pub fn atan_p2_default<T>(x: T) -> T
where
    <T as PrimitivePromotionExt>::PrimitivePromotion: PartialOrd + AsPrimitive<T> + Signed,
    T: AsPrimitive<<T as PrimitivePromotionExt>::PrimitivePromotion>
        + Bits
        + PrimInt
        + PrimitivePromotionExt
        + Signed,
    f64: AsPrimitive<T>,
    i8: AsPrimitive<T>,
{
    let exp = T::BITS / 2 - 1;
    let k = 2.as_().pow(exp);
    let a = calc_default_p2_k(exp);
    atan_p2(x, k, a, k)
}

/// ```rust
/// use std::f64::consts::PI;
/// use approx::assert_abs_diff_eq;
/// use rust_math::atan_p2::*;
/// const EXP: u32 = i32::BITS / 2 - 1;
/// const K: i32 = 2_i32.pow(EXP);
/// let a = calc_default_p2_k::<i32>(EXP);
/// let result = atan2_p2(1000, 1732, K, a, K);
/// assert_abs_diff_eq!(
///     PI / 6.0,
///     result as f64 * PI / K.pow(2) as f64,
///     epsilon = 0.0039,
/// );
/// ```
pub fn atan2_p2<T>(y: T, x: T, x_k: T, a: T, k: T) -> T
where
    <T as PrimitivePromotionExt>::PrimitivePromotion: AsPrimitive<T> + PartialOrd + Signed,
    T: AsPrimitive<<T as PrimitivePromotionExt>::PrimitivePromotion>
        + ConstZero
        + PrimitivePromotionExt
        + Signed,
    i8: AsPrimitive<T>,
{
    atan2_impl(y, x, x_k, |x| atan_p2_impl(x, x, x_k, a, k))
}

/// ```rust
/// use std::f64::consts::PI;
/// use approx::assert_abs_diff_eq;
/// use rust_math::atan_p2::*;
/// const EXP: u32 = i32::BITS / 2 - 1;
/// let result = atan2_p2_default(1000, 1732);
/// assert_abs_diff_eq!(
///     PI / 6.0,
///     result as f64 * PI / 2_i32.pow(2 * EXP) as f64,
///     epsilon = 0.0039,
/// );
/// ```
pub fn atan2_p2_default<T>(y: T, x: T) -> T
where
    <T as PrimitivePromotionExt>::PrimitivePromotion: AsPrimitive<T> + PartialOrd + Signed,
    T: AsPrimitive<<T as PrimitivePromotionExt>::PrimitivePromotion>
        + Bits
        + ConstZero
        + PrimInt
        + PrimitivePromotionExt
        + Signed,
    f64: AsPrimitive<T>,
    i8: AsPrimitive<T>,
{
    let exp = T::BITS / 2 - 1;
    let k = 2.as_().pow(exp);
    let a = calc_default_p2_k(exp);
    atan2_p2(y, x, k, a, k)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_atan2_p2_default() {
        use std::i32::{MAX, MIN};

        fn f(x: i32, y: i32) {
            let expected = (y as f64).atan2(x as f64);
            let actual = {
                let actual = atan2_p2_default(y, x);
                actual as f64 * PI / 2.0_f64.powi(i32::BITS as i32 / 2 * 2 - 2)
            };
            let error = actual - expected;
            assert!(
                error.abs() < 0.0039,
                "y: {y}, x: {x}, actual: {actual}, expected: {expected}"
            );
        }

        {
            let values = [0, MAX, MIN, MAX];
            for x in values {
                for y in values {
                    f(x, y);
                }
            }
        }

        fn g(degrees: i32) {
            const SCALE: f64 = 1000000.0;
            let radians = (degrees as f64).to_radians();
            let x = SCALE * radians.cos();
            let y = SCALE * radians.sin();
            f(x as i32, y as i32);
        }
        for degrees in -195..-180 {
            g(degrees);
        }
        for degrees in -179..180 {
            g(degrees);
        }
        for degrees in 181..195 {
            g(degrees);
        }
    }
}
