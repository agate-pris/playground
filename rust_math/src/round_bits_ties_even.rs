use std::cmp::Ordering;

use num_traits::{PrimInt, Signed};

/// Round the specified number of bits
/// and truncate it after rounding.
/// The midpoint value is rounded to the nearest even number.
///
/// # Examples
///
/// ```
/// use rust_math::round_bits_ties_even::*;
/// assert_eq!(round_bits_ties_even(-6, 2), -2);
/// assert_eq!(round_bits_ties_even(-5, 2), -1);
/// assert_eq!(round_bits_ties_even(-3, 2), -1);
/// assert_eq!(round_bits_ties_even(-2, 2), 0);
/// assert_eq!(round_bits_ties_even(2, 2), 0);
/// assert_eq!(round_bits_ties_even(3, 2), 1);
/// assert_eq!(round_bits_ties_even(5, 2), 1);
/// assert_eq!(round_bits_ties_even(6, 2), 2);
/// ```
pub fn round_bits_ties_even<I>(i: I, bits: u32) -> I
where
    I: From<i8> + PrimInt + Signed,
{
    // Returns the original value if bits is 0.
    if bits == 0 {
        return i;
    }

    let pow = <I as From<i8>>::from(2).pow(bits);
    let rem = i % pow;
    let div = i / pow;

    if i.is_negative() {
        // Negative
        match rem.cmp(&(-pow / 2.into())) {
            // Round to even if rem is just -0.5.
            Ordering::Equal => {
                // -0.5 -> 0, -1.5 -> -2, -2.5 -> -2, -3.5 -> -4
                if div % 2.into() == 0.into() {
                    div
                } else {
                    div - 1.into()
                }
            }
            // Round down if rem is less than -0.5.
            Ordering::Less => div - 1.into(),
            // Round towards 0 if rem is greater than -0.5.
            Ordering::Greater => div,
        }
    } else {
        // 0 or Positive
        match rem.cmp(&(pow / 2.into())) {
            // Round to even if rem is just 0.5.
            Ordering::Equal => {
                // 0.5 -> 0, 1.5 -> 2, 2.5 -> 2, 3.5 -> 4
                if div % 2.into() == 0.into() {
                    div
                } else {
                    div + 1.into()
                }
            }
            // Round up if rem is greater than 0.5.
            Ordering::Greater => div + 1.into(),
            // Round towards 0 if rem is less than 0.5.
            Ordering::Less => div,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::round_bits_ties_even;

    #[test]
    fn test_round_bits_ties_even() {
        for i in -999..=999 {
            assert_eq!(round_bits_ties_even(i, 0), i);
        }

        assert_eq!(round_bits_ties_even(-9, 1), -4);
        assert_eq!(round_bits_ties_even(-8, 1), -4);
        assert_eq!(round_bits_ties_even(-7, 1), -4);
        assert_eq!(round_bits_ties_even(-6, 1), -3);
        assert_eq!(round_bits_ties_even(-5, 1), -2);
        assert_eq!(round_bits_ties_even(-4, 1), -2);
        assert_eq!(round_bits_ties_even(-3, 1), -2);
        assert_eq!(round_bits_ties_even(-2, 1), -1);
        assert_eq!(round_bits_ties_even(-1, 1), 0);
        assert_eq!(round_bits_ties_even(0, 1), 0);
        assert_eq!(round_bits_ties_even(1, 1), 0);
        assert_eq!(round_bits_ties_even(2, 1), 1);
        assert_eq!(round_bits_ties_even(3, 1), 2);
        assert_eq!(round_bits_ties_even(4, 1), 2);
        assert_eq!(round_bits_ties_even(5, 1), 2);
        assert_eq!(round_bits_ties_even(6, 1), 3);
        assert_eq!(round_bits_ties_even(7, 1), 4);
        assert_eq!(round_bits_ties_even(8, 1), 4);
        assert_eq!(round_bits_ties_even(9, 1), 4);

        assert_eq!(round_bits_ties_even(-9, 2), -2);
        assert_eq!(round_bits_ties_even(-8, 2), -2);
        assert_eq!(round_bits_ties_even(-7, 2), -2);
        assert_eq!(round_bits_ties_even(-6, 2), -2);
        assert_eq!(round_bits_ties_even(-5, 2), -1);
        assert_eq!(round_bits_ties_even(-4, 2), -1);
        assert_eq!(round_bits_ties_even(-3, 2), -1);
        assert_eq!(round_bits_ties_even(-2, 2), 0);
        assert_eq!(round_bits_ties_even(-1, 2), 0);
        assert_eq!(round_bits_ties_even(0, 2), 0);
        assert_eq!(round_bits_ties_even(1, 2), 0);
        assert_eq!(round_bits_ties_even(2, 2), 0);
        assert_eq!(round_bits_ties_even(3, 2), 1);
        assert_eq!(round_bits_ties_even(4, 2), 1);
        assert_eq!(round_bits_ties_even(5, 2), 1);
        assert_eq!(round_bits_ties_even(6, 2), 2);
        assert_eq!(round_bits_ties_even(7, 2), 2);
        assert_eq!(round_bits_ties_even(8, 2), 2);
        assert_eq!(round_bits_ties_even(9, 2), 2);

        for bits in 0..16 {
            let pow = 2_i32.pow(bits) as f64;
            for i in -999..=999 {
                let div = i as f64 / pow;
                let expected = div.round_ties_even() as i32;
                let actual = round_bits_ties_even(i, bits);
                assert_eq!(
                    expected, actual,
                    "bits: {bits}, pow: {pow}, i: {i}, div: {div}"
                );
            }
        }
    }
}
