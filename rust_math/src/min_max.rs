pub trait Min {
    const MIN: Self;
}

pub trait Max {
    const MAX: Self;
}

macro_rules! impl_min {
    ($($t:ty),*) => {
        $(
            impl Min for $t {
                const MIN: Self = Self::MIN;
            }
        )*
    };
}

macro_rules! impl_max {
    ($($t:ty),*) => {
        $(
            impl Max for $t {
                const MAX: Self = Self::MAX;
            }
        )*
    };
}

impl_min!(i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize);
impl_max!(i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize);
