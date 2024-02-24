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
