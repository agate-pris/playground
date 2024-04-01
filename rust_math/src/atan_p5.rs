use std::f64::consts::PI;

use num_traits::{AsPrimitive, ConstZero, NumOps, PrimInt, Signed};
use primitive_promotion::PrimitivePromotionExt;

use crate::{
    atan::{atan2_impl, atan_impl},
    bits::Bits,
};

/// ```rust
/// use rust_math::atan_p5::*;
/// const EXP: u32 = i32::BITS / 2 - 1;
/// let (a, b, c) = calc_default_p5_k::<i32>(EXP);
/// assert_eq!(a, 810);
/// assert_eq!(b, 2998);
/// assert_eq!(c, 10380);
/// ```
pub fn calc_default_p5_k<T>(exp: u32) -> (T, T, T)
where
    T: 'static + Copy + NumOps,
    f64: AsPrimitive<T>,
    i8: AsPrimitive<T>,
{
    let k = 2.0_f64.powi(exp as i32);
    let a: T = (0.0776509570923569 / PI * k).round_ties_even().as_();
    let b: T = (0.287434475393028 / PI * k).round_ties_even().as_();
    let k: T = k.as_();
    let c = k / 4.as_() - a + b;
    (a, b, c)
}

fn atan_p5_impl<T>(x: T, k: T, a: T, b: T, c: T) -> T
where
    T: 'static + Copy + NumOps,
{
    let x_2 = x * x / k;
    ((a * x_2 / k - b) * x_2 / k + c) * x
}

/// ```rust
/// use std::f64::consts::PI;
/// use approx::assert_abs_diff_eq;
/// use rust_math::atan_p5::*;
/// const EXP: u32 = i32::BITS / 2 - 1;
/// const K: i32 = 2_i32.pow(EXP);
/// let (a, b, c) = calc_default_p5_k::<i32>(EXP);
/// let result = atan_p5(1000 * K / 1732, K, a, b, c);
/// assert_abs_diff_eq!(
///     PI / 6.0,
///     result as f64 * PI / K.pow(2) as f64,
///     epsilon = 0.00085,
/// );
/// ```
pub fn atan_p5<T>(x: T, k: T, a: T, b: T, c: T) -> T
where
    <T as PrimitivePromotionExt>::PrimitivePromotion: PartialOrd + AsPrimitive<T> + Signed,
    T: AsPrimitive<<T as PrimitivePromotionExt>::PrimitivePromotion>
        + PrimitivePromotionExt
        + Signed,
    i8: AsPrimitive<T>,
{
    atan_impl(x, k, |x, _| atan_p5_impl(x, k, a, b, c))
}

/// ```rust
/// use std::f64::consts::PI;
/// use approx::assert_abs_diff_eq;
/// use rust_math::atan_p5::*;
/// const EXP: u32 = i32::BITS / 2 - 1;
/// const K: i32 = 2_i32.pow(EXP);
/// let result = atan_p5_default(1000 * K / 1732);
/// assert_abs_diff_eq!(
///     PI / 6.0,
///     result as f64 * PI / K.pow(2) as f64,
///     epsilon = 0.00085,
/// );
/// ```
pub fn atan_p5_default<T>(x: T) -> T
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
    let (a, b, c) = calc_default_p5_k(exp);
    atan_p5(x, k, a, b, c)
}

/// ```rust
/// use std::f64::consts::PI;
/// use approx::assert_abs_diff_eq;
/// use rust_math::atan_p5::*;
/// const EXP: u32 = i32::BITS / 2 - 1;
/// const K: i32 = 2_i32.pow(EXP);
/// let (a, b, c) = calc_default_p5_k::<i32>(EXP);
/// let result = atan2_p5(1000, 1732, K, a, b, c);
/// assert_abs_diff_eq!(
///     PI / 6.0,
///     result as f64 * PI / K.pow(2) as f64,
///     epsilon = 0.00085,
/// );
/// ```
pub fn atan2_p5<T>(y: T, x: T, k: T, a: T, b: T, c: T) -> T
where
    <T as PrimitivePromotionExt>::PrimitivePromotion: AsPrimitive<T> + PartialOrd + Signed,
    T: AsPrimitive<<T as PrimitivePromotionExt>::PrimitivePromotion>
        + ConstZero
        + PrimitivePromotionExt
        + Signed,
    i8: AsPrimitive<T>,
{
    atan2_impl(y, x, k, |x| atan_p5_impl(x, k, a, b, c))
}

/// ```rust
/// use std::f64::consts::PI;
/// use approx::assert_abs_diff_eq;
/// use rust_math::atan_p5::*;
/// const EXP: u32 = i32::BITS / 2 - 1;
/// let result = atan2_p5_default(1000, 1732);
/// assert_abs_diff_eq!(
///     PI / 6.0,
///     result as f64 * PI / 2_i32.pow(2 * EXP) as f64,
///     epsilon = 0.00085,
/// );
/// ```
pub fn atan2_p5_default<T>(y: T, x: T) -> T
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
    let (a, b, c) = calc_default_p5_k(exp);
    atan2_p5(y, x, k, a, b, c)
}

#[cfg(test)]
mod tests {
    use super::*;
}
