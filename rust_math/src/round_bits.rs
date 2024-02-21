use num_traits::{PrimInt, Signed};

/// Round the specified number of bits
/// and truncate it after rounding.
/// The midpoint value is rounded away from 0.
///
/// # Examples
///
/// ```
/// assert_eq!(round_bits(-6, 2), -2);
/// assert_eq!(round_bits(-5, 2), -1);
/// assert_eq!(round_bits(-2, 2), -1);
/// assert_eq!(round_bits(-1, 2), 0);
/// assert_eq!(round_bits(1, 2), 0);
/// assert_eq!(round_bits(2, 2), 1);
/// assert_eq!(round_bits(5, 2), 1);
/// assert_eq!(round_bits(6, 2), 2);
/// ```
pub fn round_bits<I>(i: I, bits: u32) -> I
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
        if rem > -pow / 2.into() {
            // Round towards 0 if rem is greater than -0.5.
            div
        } else {
            // Otherwise, round down.
            div - 1.into()
        }
    } else {
        // 0 or Positive
        if rem < pow / 2.into() {
            // Round towards 0 if rem is less than 0.5.
            div
        } else {
            // Otherwise, round up.
            div + 1.into()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::round_bits;

    #[test]
    fn test_round_bits() {
        for i in -999..=999 {
            assert_eq!(round_bits(i, 0), i);
        }

        assert_eq!(round_bits(-9, 1), -5);
        assert_eq!(round_bits(-8, 1), -4);
        assert_eq!(round_bits(-7, 1), -4);
        assert_eq!(round_bits(-6, 1), -3);
        assert_eq!(round_bits(-5, 1), -3);
        assert_eq!(round_bits(-4, 1), -2);
        assert_eq!(round_bits(-3, 1), -2);
        assert_eq!(round_bits(-2, 1), -1);
        assert_eq!(round_bits(-1, 1), -1);
        assert_eq!(round_bits(0, 1), 0);
        assert_eq!(round_bits(1, 1), 1);
        assert_eq!(round_bits(2, 1), 1);
        assert_eq!(round_bits(3, 1), 2);
        assert_eq!(round_bits(4, 1), 2);
        assert_eq!(round_bits(5, 1), 3);
        assert_eq!(round_bits(6, 1), 3);
        assert_eq!(round_bits(7, 1), 4);
        assert_eq!(round_bits(8, 1), 4);
        assert_eq!(round_bits(9, 1), 5);

        assert_eq!(round_bits(-9, 2), -2);
        assert_eq!(round_bits(-8, 2), -2);
        assert_eq!(round_bits(-7, 2), -2);
        assert_eq!(round_bits(-6, 2), -2);
        assert_eq!(round_bits(-5, 2), -1);
        assert_eq!(round_bits(-4, 2), -1);
        assert_eq!(round_bits(-3, 2), -1);
        assert_eq!(round_bits(-2, 2), -1);
        assert_eq!(round_bits(-1, 2), 0);
        assert_eq!(round_bits(0, 2), 0);
        assert_eq!(round_bits(1, 2), 0);
        assert_eq!(round_bits(2, 2), 1);
        assert_eq!(round_bits(3, 2), 1);
        assert_eq!(round_bits(4, 2), 1);
        assert_eq!(round_bits(5, 2), 1);
        assert_eq!(round_bits(6, 2), 2);
        assert_eq!(round_bits(7, 2), 2);
        assert_eq!(round_bits(8, 2), 2);
        assert_eq!(round_bits(9, 2), 2);

        for bits in 0..16 {
            let pow = 2_i32.pow(bits) as f64;
            for i in -999..=999 {
                let div = i as f64 / pow;
                let expected = div.round() as i32;
                let actual = round_bits(i, bits);
                assert_eq!(
                    expected, actual,
                    "bits: {bits}, pow: {pow}, i: {i}, div: {div}"
                );
            }
        }
    }
}
