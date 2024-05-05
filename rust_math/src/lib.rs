#![feature(array_try_from_fn)]
#![feature(array_try_map)]

#[cfg(test)]
extern crate num_cpus;

pub mod atan;
pub mod atan_p2;
pub mod atan_p3;
pub mod atan_p5;
pub mod bits;
pub mod round_bits;
pub mod round_bits_ties_even;
pub mod sin_cos;

use atan::AtanUtil;
use atan_p2::AtanP2I32Util;
use atan_p3::AtanP3I32Util;

pub fn atan_p2_2850(x: i32) -> i32 {
    AtanP2I32Util::atan(x)
}

pub fn atan2_p2_2850(y: i32, x: i32) -> i32 {
    AtanP2I32Util::atan2(y, x)
}

pub fn atan_p3_2555_691(x: i32) -> i32 {
    AtanP3I32Util::atan(x)
}

pub fn atan2_p3_2555_691(y: i32, x: i32) -> i32 {
    AtanP3I32Util::atan2(y, x)
}

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
