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
