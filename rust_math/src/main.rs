pub mod angle;
pub mod bits;
pub mod min_max;
pub mod round_bits;
pub mod round_bits_ties_even;

use std::{
    f64::consts::FRAC_PI_2,
    fmt::Display,
    fs::File,
    io::{BufWriter, Write},
    iter::once,
    ops::RangeInclusive,
    path::Path,
};

use anyhow::{Error, Result};
use clap::Parser;
use num_traits::AsPrimitive;
use serde::{de::DeserializeOwned, ser::Serialize};
use serde_json::{ser::PrettyFormatter, Serializer};

use crate::angle::{cos_p2, cos_p4, cos_p4o, sin_p3, sin_p5, sin_p5o, Angle};

fn serialize<T>(actual: &Vec<T>) -> Result<Vec<u8>>
where
    T: Serialize,
{
    let formatter = PrettyFormatter::with_indent(&[]);
    let mut writer = Vec::new();
    let mut serializer = Serializer::with_formatter(&mut writer, formatter);
    actual.serialize(&mut serializer)?;
    Ok(writer)
}

fn serialize_and_write<T>(dir: &str, file_name: &str, actual: &Vec<T>) -> Result<()>
where
    T: Serialize,
{
    let buf = serialize(actual)?;
    let path = Path::new(dir).join(file_name);
    let inner = File::create(path)?;
    Ok(BufWriter::new(inner).write_all(&buf)?)
}

#[derive(Parser, Debug)]
struct Args {
    #[arg(short, long)]
    output: Option<String>,

    #[arg(short, long)]
    print: bool,
}

fn calc_and_write<F, T>(args: &Args, f: F, right: T, file_name: &str) -> Result<()>
where
    F: Fn(T, T) -> T,
    T: Angle + AsPrimitive<usize> + DeserializeOwned + Display + Serialize,
    RangeInclusive<T>: Iterator<Item = T>,
{
    if let Some(dir) = &args.output {
        let actual = (0.into()..=right).map(|x| f(x, right)).collect();
        serialize_and_write(dir, file_name, &actual)?;
    }
    Ok(())
}

fn calc_and_write_all(args: &Args) -> Vec<Error> {
    [
        calc_and_write(args, cos_p2::<i16>, i16::DEFAULT_RIGHT, "cos_p2_i16.json"),
        calc_and_write(args, cos_p2::<i32>, i32::DEFAULT_RIGHT, "cos_p2_i32.json"),
        calc_and_write(args, sin_p3::<i32>, i32::DEFAULT_RIGHT, "sin_p3_i32.json"),
        calc_and_write(args, cos_p4::<i32>, i32::DEFAULT_RIGHT, "cos_p4_i32.json"),
        calc_and_write(args, cos_p4o::<i32>, i32::DEFAULT_RIGHT, "cos_p4o_i32.json"),
        calc_and_write(args, sin_p5::<i32>, i32::DEFAULT_RIGHT, "sin_p5_i32.json"),
        calc_and_write(args, sin_p5o::<i32>, i32::DEFAULT_RIGHT, "sin_p5o_i32.json"),
    ]
    .into_iter()
    .filter_map(Result::err)
    .collect()
}

fn print_max<Expected, Actual, T>(expected: Expected, actual: Actual, step: usize, right: T, one: T)
where
    Expected: Iterator<Item = f64>,
    Actual: Fn(T, T) -> T,
    T: Angle + AsPrimitive<f64>,
    RangeInclusive<T>: Iterator<Item = T>,
{
    let one: f64 = one.as_();
    let diffs = (0.into()..=right)
        .step_by(step)
        .map(|x| {
            let actual: f64 = actual(x, right).as_();
            actual / one
        })
        .zip(expected)
        .map(|(actual, expected)| actual - expected)
        .collect::<Vec<f64>>();
    let len = diffs.len();
    let ((min_i, min), (max_i, max), sum) = diffs.into_iter().enumerate().fold(
        (
            (0_usize, f64::INFINITY),
            (0_usize, f64::NEG_INFINITY),
            0_f64,
        ),
        |((min_i, min), (max_i, max), sum), (i, diff)| {
            let min = if diff < min { (i, diff) } else { (min_i, min) };
            let max = if diff > max { (i, diff) } else { (max_i, max) };
            (min, max, sum + diff)
        },
    );
    const SCALE: f64 = 2_i32.pow(12) as f64;
    let right: f64 = right.as_();
    let to_deg = 90.0 / right;
    let min_deg = min_i as f64 * to_deg;
    let max_deg = max_i as f64 * to_deg;
    let average = sum / len as f64;
    println!(
        concat!(
            "min: {{{:6} ({:7.3}), {:8.3} / {scale}}}, ",
            "max: {{{:6} ({:7.3}), {:7.3} / {scale}}}, ",
            "average: {:8.3} / {scale}",
        ),
        min_i,
        min_deg,
        SCALE * min,
        max_i,
        max_deg,
        SCALE * max,
        SCALE * average,
        scale = SCALE
    );
}

pub fn print_max_all() {
    let sin: Vec<_>;
    let cos: Vec<_>;
    {
        const FRAC_PI_STRAIGHT: f64 = FRAC_PI_2 / i32::DEFAULT_RIGHT as f64;
        sin = (0..i32::DEFAULT_RIGHT)
            .map(|x| (FRAC_PI_STRAIGHT * x as f64).sin())
            .chain(once(FRAC_PI_2.sin().round()))
            .collect();
        cos = (0..i32::DEFAULT_RIGHT)
            .map(|x| (FRAC_PI_STRAIGHT * x as f64).cos())
            .chain(once(FRAC_PI_2.cos().round()))
            .collect();
    }

    // cos_p2::<i16>
    {
        type Type = i16;
        let expected = cos
            .iter()
            .cloned()
            .step_by((i32::DEFAULT_RIGHT / Type::DEFAULT_RIGHT as i32) as usize);
        let actual = cos_p2::<Type>;
        print!("cos_p2::<i16>:  ");
        print_max(
            expected,
            actual,
            1,
            Type::DEFAULT_RIGHT,
            Type::DEFAULT_RIGHT.pow(2),
        );
    }

    // i32
    {
        type Type = i32;
        const ONE: Type = Type::DEFAULT_RIGHT.pow(2);
        let sin = sin.iter().cloned();
        let cos = cos.iter().cloned();
        print!("sin_p3::<i32>:  ");
        print_max(sin.clone(), sin_p3::<Type>, 1, Type::DEFAULT_RIGHT, ONE);
        print!("cos_p4::<i32>:  ");
        print_max(cos.clone(), cos_p4::<Type>, 1, Type::DEFAULT_RIGHT, ONE);
        print!("cos_p4o::<i32>: ");
        print_max(cos, cos_p4o::<Type>, 1, Type::DEFAULT_RIGHT, ONE);
        print!("sin_p5::<i32>:  ");
        print_max(sin.clone(), sin_p5::<Type>, 1, Type::DEFAULT_RIGHT, ONE);
        print!("sin_p5o::<i32>: ");
        print_max(sin, sin_p5o::<Type>, 1, Type::DEFAULT_RIGHT, ONE);
    }
}

fn main() {
    let args = Args::parse();
    {
        let errors = calc_and_write_all(&args);
        for e in &errors {
            eprintln!("{e}");
        }
        if !errors.is_empty() {
            panic!();
        }
    }
    if args.print {
        print_max_all();
    }
}
