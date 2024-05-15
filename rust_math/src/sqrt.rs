use std::ops::{Shl, Shr};

use num_traits::{ConstOne, PrimInt};

use crate::Bits;

trait HalfBits {
    const HALF_BITS: u32;
}

impl<T> HalfBits for T
where
    T: Bits,
{
    const HALF_BITS: u32 = <T as Bits>::BITS / 2;
}

pub fn sqrt<T>(x: T) -> T
where
    T: Shr<u32, Output = T> + Shl<u32, Output = T> + Bits + ConstOne + PrimInt,
{
    if x <= T::ONE {
        return x;
    }
    let k = T::HALF_BITS - ((x - T::ONE).leading_zeros() >> 1);
    let mut s = T::ONE << k;
    let mut t = (s + (x >> k)) >> 1_u32;
    while t < s {
        s = t;
        t = (s + (x / s)) >> 1_u32;
    }
    s
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use rand::prelude::*;
    use rayon::prelude::*;

    use super::*;

    #[test]
    fn test_sqrt() -> Result<()> {
        for x in 0..=u8::MAX {
            anyhow::ensure!((x as f64).sqrt() as u8 == sqrt(x));
        }
        for x in 0..=u16::MAX {
            anyhow::ensure!((x as f64).sqrt() as u16 == sqrt(x));
        }
        let mut rng = rand::thread_rng();
        for _ in 0..99999 {
            let x = rng.gen_range(0..=u32::MAX);
            anyhow::ensure!((x as f64).sqrt() as u32 == sqrt(x));
        }
        Ok(())
    }

    #[test]
    #[ignore]
    fn test_sqrt_u32_full() -> Result<()> {
        let num = num_cpus::get() as u32;
        (0..num)
            .into_par_iter()
            .map(|n| {
                let start = (u32::MAX / num) * n;
                let end = (u32::MAX / num) * (n + 1);
                for x in start..=end {
                    anyhow::ensure!((x as f64).sqrt() as u32 == sqrt(x));
                }
                Ok(())
            })
            .collect::<Result<Vec<()>>>()?;
        Ok(())
    }
}
