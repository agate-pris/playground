#![feature(array_try_from_fn)]
#![feature(array_try_map)]
#![feature(strict_overflow_ops)]

#[cfg(test)]
extern crate num_cpus;

pub(crate) mod atan;
mod atan_p2;
mod atan_p3;
mod atan_p5;
mod bits;
mod sin_cos;

pub use atan_p2::{atan2_p2_2850, atan_p2_2850};
pub use atan_p3::{atan2_p3_2555_691, atan_p3_2555_691};
pub use atan_p5::{atan2_p5_787_2968, atan_p5_787_2968};
pub use bits::Bits;
pub use sin_cos::{
    cos_p2_i32, cos_p3_16384, cos_p4_7032, cos_p4_7384, cos_p5_51437, cos_p5_51472, sin_p2_i32,
    sin_p3_16384, sin_p4_7032, sin_p4_7384, sin_p5_51437, sin_p5_51472,
};

#[cfg(test)]
mod tests {
    use std::{fs::File, io::BufReader};

    use anyhow::Result;
    use serde::de::DeserializeOwned;

    pub(crate) fn read_data<T>(data_path: &str) -> Result<Vec<T>>
    where
        T: DeserializeOwned,
    {
        let inner = File::open(data_path)?;
        let rdr = BufReader::new(inner);
        Ok(serde_json::from_reader(rdr)?)
    }
}
