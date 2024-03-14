pub trait Bits {
    const BITS: u32;
}

macro_rules! impl_bits {
    ($($t:ty),*) => {
        $(
            impl Bits for $t {
                const BITS: u32 = Self::BITS;
            }
        )*
    };
}

impl_bits!(i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bits() {
        assert_eq!(<i8 as Bits>::BITS, 8);
        assert_eq!(<i16 as Bits>::BITS, 16);
        assert_eq!(<i32 as Bits>::BITS, 32);
        assert_eq!(<i64 as Bits>::BITS, 64);
        assert_eq!(<i128 as Bits>::BITS, 128);
        assert_eq!(
            <isize as Bits>::BITS,
            8 * std::mem::size_of::<isize>() as u32
        );
        assert_eq!(<u8 as Bits>::BITS, 8);
        assert_eq!(<u16 as Bits>::BITS, 16);
        assert_eq!(<u32 as Bits>::BITS, 32);
        assert_eq!(<u64 as Bits>::BITS, 64);
        assert_eq!(<u128 as Bits>::BITS, 128);
        assert_eq!(
            <usize as Bits>::BITS,
            8 * std::mem::size_of::<usize>() as u32
        );
    }
}
