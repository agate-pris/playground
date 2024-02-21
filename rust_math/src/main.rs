pub mod round_bits;
pub mod round_bits_ties_even;

pub use round_bits::round_bits;
pub use round_bits_ties_even::round_bits_ties_even;

use std::{
    f64::consts::{FRAC_2_PI, FRAC_PI_2, FRAC_PI_4},
    fmt::Display,
    fs::File,
    io::{BufReader, BufWriter, Write},
    iter::once,
    ops::RangeInclusive,
    path::Path,
};

use anyhow::{anyhow, ensure, Error, Result};
use clap::Parser;
use num_traits::{AsPrimitive, PrimInt, Signed};
use serde::{de::DeserializeOwned, ser::Serialize};
use serde_json::{ser::PrettyFormatter, Serializer};

pub trait Angle: AsPrimitive<f64> + AsPrimitive<i8> + From<i8> + PrimInt + Signed {}

// i64 and i128 is not supported
// because the coefficients cannot be calculated
// with sufficient precision.

impl Angle for i8 {}
impl Angle for i16 {}
impl Angle for i32 {}

fn square<T: PrimInt>(b: T, denom: T) -> T {
    b.pow(2) / denom
}

fn repeat<T: PrimInt + Signed>(t: T, length: T) -> T {
    let rem = t % length;
    if rem.is_negative() {
        rem + length
    } else {
        rem
    }
}

fn calc_quadrant<T: Angle>(x: T, right: T) -> i8 {
    (repeat(x, right * 4.into()) / right).as_()
}

fn odd_cos_impl<T: Angle>(x: T, right: T) -> T {
    (x % (right * 4.into())) + right
}

fn even_sin_impl<T: Angle>(x: T, right: T) -> T {
    (x % (right * 4.into())) - right
}

/// x
pub fn sin_p1<T: Angle>(x: T, right: T) -> T {
    let rem = repeat(x, right);
    match calc_quadrant(x, right) {
        1 => -rem + right,
        3 => rem - right,
        2 => -rem,
        _ => rem,
    }
}

pub fn cos_p1<T: Angle>(x: T, right: T) -> T {
    sin_p1(odd_cos_impl(x, right), right)
}

fn even_cos_impl<T, F>(x: T, right: T, f: F) -> T
where
    T: Angle,
    F: Fn(T, T) -> T,
{
    let rem = repeat(x, right);
    let k = right.pow(2);
    match calc_quadrant(x, right) {
        1 => -k + f(right - rem, right),
        3 => k - f(right - rem, right),
        2 => -k + f(rem, right),
        _ => k - f(rem, right),
    }
}

/// 1 - x ^ 2
pub fn cos_p2<T: Angle>(x: T, right: T) -> T {
    even_cos_impl(x, right, |z, _| z.pow(2))
}

pub fn sin_p2<T: Angle>(x: T, right: T) -> T {
    cos_p2(even_sin_impl(x, right), right)
}

fn sin_p3_cos_p4_impl<T: Angle>(a: T, b: T, z_2: T, right: T) -> T {
    a - z_2 * b / right
}

/// 1 + k - k * x ^ 2
fn sin_p3_impl<T: Angle>(k: T, x: T, right: T) -> T {
    let z = sin_p1(x, right);
    sin_p3_cos_p4_impl(right + k, k, square(z, right), right) * z
}

/// 1.5 * x - 0.5 * x ^ 3
pub fn sin_p3<T: Angle>(x: T, right: T) -> T {
    // 1.5 * x - 0.5 * x ^ 3
    // = (1.5 - 0.5 * x ^ 2) * x
    sin_p3_impl(right / 2.into(), x, right)
}

pub fn cos_p3<T: Angle>(x: T, right: T) -> T {
    sin_p3(odd_cos_impl(x, right), right)
}

fn cos_p4_sin_p5_impl<T: Angle>(a: T, b: T, z: T, right: T) -> T {
    let z_2 = square(z, right);
    sin_p3_cos_p4_impl(a, b, z_2, right) * z_2
}

/// (k + 1) * z ^ 2 - k * z ^ 4
fn cos_p4_impl<T: Angle>(k: T, z: T, right: T) -> T {
    cos_p4_sin_p5_impl(k + right, k, z, right)
}

/// 1 - pi / 4
fn cos_p4_k<T: Angle>(right: T) -> T
where
    f64: AsPrimitive<T>,
{
    let right: f64 = right.as_();
    ((1.0 - FRAC_PI_4) * right).round().as_()
}

/// 1 - a * z ^ 2 + (a - a) * z ^ 4
/// a = 1 - pi / 4
pub fn cos_p4<T: Angle>(x: T, right: T) -> T
where
    f64: AsPrimitive<T>,
{
    even_cos_impl(x, right, |z, right| {
        cos_p4_impl(cos_p4_k::<T>(right), z, right)
    })
}

pub fn sin_p4<T: Angle>(x: T, right: T) -> T
where
    f64: AsPrimitive<T>,
{
    cos_p4(even_sin_impl(x, right), right)
}

/// 5 * (1 - 3 / pi)
fn cos_p4o_k<T: Angle>(right: T) -> T
where
    f64: AsPrimitive<T>,
{
    let right: f64 = right.as_();
    (5.0 * (1.0 - 1.5 * FRAC_2_PI) * right).round().as_()
}

/// 1 - a * z ^ 2 + (a - a) * z ^ 4
/// a = 5 * (1 - 3 / pi)
pub fn cos_p4o<T: Angle>(x: T, right: T) -> T
where
    f64: AsPrimitive<T>,
{
    even_cos_impl(x, right, |z, right| {
        cos_p4_impl(cos_p4o_k::<T>(right), z, right)
    })
}

pub fn sin_p4o<T: Angle>(x: T, right: T) -> T
where
    f64: AsPrimitive<T>,
{
    cos_p4o(even_sin_impl(x, right), right)
}

/// k * x - (2 * k - 2.5) * x ^ 3 + (k - 1.5) * x ^ 5
fn sin_p5_impl<T: Angle>(k: T, x: T, right: T) -> T {
    let z = sin_p1(x, right);
    let a = k * 2.into() - right * 5.into() / 2.into();
    let b = k - right * 3.into() / 2.into();
    (k - cos_p4_sin_p5_impl(a, b, z, right) / right) * z
}

/// pi / 2
fn sin_p5_k<T: Angle>(right: T) -> T
where
    f64: AsPrimitive<T>,
{
    let right: f64 = right.as_();
    (FRAC_PI_2 * right).round().as_()
}

/// a * x - c * x ^ 3 + c * x ^ 5
/// a = pi / 2
/// b = pi - 2.5
/// c = pi / 2 - 1.5
pub fn sin_p5<T: Angle>(x: T, right: T) -> T
where
    f64: AsPrimitive<T>,
{
    sin_p5_impl(sin_p5_k::<T>(right), x, right)
}

pub fn cos_p5<T: Angle>(x: T, right: T) -> T
where
    f64: AsPrimitive<T>,
{
    sin_p5(odd_cos_impl(x, right), right)
}

/// 4 * (3 / pi - 9 / 16)
fn sin_p5o_k<T: Angle>(right: T) -> T
where
    f64: AsPrimitive<T>,
{
    let right: f64 = right.as_();
    (4.0 * (1.5 * FRAC_2_PI - 9.0 / 16.0) * right).round().as_()
}

/// a * x - c * x ^ 3 + c * x ^ 5
/// a = 4 * (3 / pi - 9 / 16)
/// b = 2 * a - 2.5
/// c = a - 1.5
pub fn sin_p5o<T: Angle>(x: T, right: T) -> T
where
    f64: AsPrimitive<T>,
{
    sin_p5_impl(sin_p5o_k::<T>(right), x, right)
}

pub fn cos_p5o<T: Angle>(x: T, right: T) -> T
where
    f64: AsPrimitive<T>,
{
    sin_p5o(odd_cos_impl(x, right), right)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fmt::Debug;

    #[test]
    fn test_repeat() {
        assert_eq!(repeat(-10, 10), 0);
        assert_eq!(repeat(-9, 10), 1);
        assert_eq!(repeat(-1, 10), 9);
        assert_eq!(repeat(0, 10), 0);
        assert_eq!(repeat(1, 10), 1);
        assert_eq!(repeat(9, 10), 9);
        assert_eq!(repeat(10, 10), 0);
        assert_eq!(repeat(11, 10), 1);
    }

    fn test_sin_cos<Actual, T>(
        actual: Actual,
        expected: fn(f64) -> f64,
        margin: f64,
        exp: u32,
        step: usize,
        right: T,
    ) where
        Actual: Fn(T, T) -> T,
        T: Angle + Display + From<<RangeInclusive<T> as Iterator>::Item>,
        RangeInclusive<T>: Iterator,
    {
        const SCALE: f64 = 2_i32.pow(12) as f64;

        let zero: T = 0.into();
        let one: T = <T as From<i8>>::from(2).pow(exp);
        let full = right * 4.into();
        let frac_pi_straight = {
            let right: f64 = right.as_();
            FRAC_PI_2 / right
        };
        let frac_scale_one = SCALE / 2.0_f64.powf(exp as f64);

        for x in (-full..=full).step_by(step).map(Into::into) {
            let actual: T = actual(x, right);
            let expected = {
                let x: f64 = x.as_();
                expected(frac_pi_straight * x)
            };

            // Check that it is exactly 1, -1 or 0
            // on the coordinate axis,
            // otherwise that the sign is correct.
            if x % right == zero {
                assert!(
                    actual == 0.into() || actual == one || actual == -one,
                    "actual: {}",
                    actual
                );
            } else {
                if 0.0 < expected {
                    assert!(zero < actual);
                } else {
                    assert!(zero > actual);
                }
            }

            let actual = {
                let actual: f64 = actual.as_();
                frac_scale_one * actual
            };
            let expected = SCALE * expected;
            let diff = expected - actual;
            assert!(
                diff.abs() < margin,
                "x: {x}, expected: {expected}, actual: {actual}"
            );
        }
    }

    fn test_sin<F, T>(f: F, right: T, margin: f64, exp: u32, step: usize)
    where
        F: Copy + Fn(T, T) -> T,
        T: Angle + Debug + Display + From<<RangeInclusive<T> as Iterator>::Item>,
        RangeInclusive<T>: Iterator,
    {
        let straight = right * 2.into();
        let full = right * 4.into();
        test_sin_cos(f, f64::sin, margin, exp, step, right);
        for x in (-full..=full).step_by(step).map(Into::into) {
            let fx = f(x, right);
            assert_eq!(fx, f(straight - x, right));
            assert_eq!(-fx, f(straight + x, right));
            assert_eq!(-fx, f(-x, right));
        }
    }

    fn test_cos<F, T>(f: F, right: T, margin: f64, exp: u32, step: usize)
    where
        F: Copy + Fn(T, T) -> T,
        T: Angle + Debug + Display + From<<RangeInclusive<T> as Iterator>::Item>,
        RangeInclusive<T>: Iterator,
    {
        let straight = right * 2.into();
        let full = right * 4.into();
        test_sin_cos(f, f64::cos, margin, exp, step, right);
        for x in (-full..=full).step_by(step).map(Into::into) {
            let fx = f(x, right);
            assert_eq!(fx, f(-x, right));
            assert_eq!(fx, -f(straight - x, right));
            assert_eq!(-fx, f(straight + x, right));
        }
    }

    #[test]
    fn test_sin_p1() {
        const MARGIN: f64 = 862.264;
        test_sin(sin_p1::<i16>, 2_i16.pow(7), MARGIN, 7, 1);
        test_sin(sin_p1::<i32>, 2_i32.pow(15), MARGIN, 15, 1);
        //test_sin(sin_p1::<i64>, MARGIN, 31, 2_usize.pow(16));
        //test_sin(sin_p1::<i128>, MARGIN, 63, 2_usize.pow(48));
    }

    #[test]
    fn test_cos_p1() {
        const MARGIN: f64 = 862.264;
        test_cos(cos_p1::<i16>, 2_i16.pow(7), MARGIN, 7, 1);
        test_cos(cos_p1::<i32>, 2_i32.pow(15), MARGIN, 15, 1);
        //test_cos(cos_p1::<i64>, MARGIN, 31, 2_usize.pow(16));
        //test_cos(cos_p1::<i128>, MARGIN, 63, 2_usize.pow(48));
    }

    #[test]
    fn test_cos_p2() {
        const MARGIN: f64 = 229.416;
        test_cos(cos_p2::<i16>, 2_i16.pow(7), MARGIN, 14, 1);
        test_cos(cos_p2::<i32>, 2_i32.pow(15), MARGIN, 30, 1);
        //test_cos(cos_p2::<i64>, MARGIN, 62, 2_usize.pow(16));
        //test_cos(cos_p2::<i128>, MARGIN, 126, 2_usize.pow(48));
    }

    #[test]
    fn test_sin_p2() {
        const MARGIN: f64 = 229.416;
        test_sin(sin_p2::<i16>, 2_i16.pow(7), MARGIN, 14, 1);
        test_sin(sin_p2::<i32>, 2_i32.pow(15), MARGIN, 30, 1);
        //test_sin(sin_p2::<i64>, MARGIN, 62, 2_usize.pow(16));
        //test_sin(sin_p2::<i128>, MARGIN, 126, 2_usize.pow(48));
    }

    #[test]
    fn test_sin_p3() {
        const MARGIN: f64 = 82.0;
        test_sin(sin_p3::<i32>, 2_i32.pow(15), MARGIN, 30, 1);
        //test_sin(sin_p3::<i64>, MARGIN, 62, 2_usize.pow(16));
        //test_sin(sin_p3::<i128>, MARGIN, 126, 2_usize.pow(48));
    }

    #[test]
    fn test_cos_p3() {
        const MARGIN: f64 = 82.0;
        test_cos(cos_p3::<i32>, 2_i32.pow(15), MARGIN, 30, 1);
        //test_cos(cos_p3::<i64>, MARGIN, 62, 2_usize.pow(16));
        //test_cos(cos_p3::<i128>, MARGIN, 126, 2_usize.pow(48));
    }

    #[test]
    fn test_cos_p4() {
        assert_eq!(2, cos_p4_k::<i8>(2_i8.pow(i8::BITS / 2 - 1)));
        assert_eq!(27, cos_p4_k::<i16>(2_i16.pow(i16::BITS / 2 - 1)));
        assert_eq!(7032, cos_p4_k::<i32>(2_i32.pow(i32::BITS / 2 - 1)));
        //assert_eq!(460853935, cos_p4_k::<i64>());
        //assert_eq!(1979352578777653248, cos_p4_k::<i128>());
        const MARGIN: f64 = 11.5464;
        test_cos(cos_p4::<i32>, 2_i32.pow(15), MARGIN, 30, 1);
        //test_cos(cos_p4::<i64>, MARGIN, 62, 2_usize.pow(16));
        //test_cos(cos_p4::<i128>, MARGIN, 126, 2_usize.pow(48));
    }

    #[test]
    fn test_sin_p4() {
        const MARGIN: f64 = 11.5464;
        test_sin(sin_p4::<i32>, 2_i32.pow(15), MARGIN, 30, 1);
        //test_sin(sin_p4::<i64>, MARGIN, 62, 2_usize.pow(16));
        //test_sin(sin_p4::<i128>, MARGIN, 126, 2_usize.pow(48));
    }

    #[test]
    fn test_cos_p4o() {
        assert_eq!(7384, cos_p4o_k::<i32>(2_i32.pow(i32::BITS / 2 - 1)));
        //assert_eq!(483939106, cos_p4o_k::<i64>());
        const MARGIN: f64 = 4.80746;
        test_cos(cos_p4o::<i32>, 2_i32.pow(i32::BITS / 2 - 1), MARGIN, 30, 1);
        //test_cos(cos_p4o::<i64>, MARGIN, 62, 2_usize.pow(16));
        //test_cos(cos_p4o::<i128>, MARGIN, 126, 2_usize.pow(48));
    }

    #[test]
    fn test_sin_p4o() {
        const MARGIN: f64 = 4.80746;
        test_sin(sin_p4o::<i32>, 2_i32.pow(15), MARGIN, 30, 1);
        //test_sin(sin_p4o::<i64>, MARGIN, 62, 2_usize.pow(16));
        //test_sin(sin_p4o::<i128>, MARGIN, 126, 2_usize.pow(48));
    }

    #[test]
    fn test_sin_p5() {
        assert_eq!(51472, sin_p5_k::<i32>(2_i32.pow(i32::BITS / 2 - 1)));
        //assert_eq!(3373259426, sin_p5_k::<i64>());
        const MARGIN: f64 = 1.73715;
        test_sin(sin_p5::<i32>, 2_i32.pow(15), MARGIN, 30, 1);
        //test_sin(sin_p5::<i64>, MARGIN, 62, 2_usize.pow(16));
        //test_sin(sin_p5::<i128>, MARGIN, 126, 2_usize.pow(48));
    }

    #[test]
    fn test_cos_p5() {
        const MARGIN: f64 = 1.73715;
        test_cos(cos_p5::<i32>, 2_i32.pow(15), MARGIN, 30, 1);
        //test_cos(cos_p5::<i64>, MARGIN, 62, 2_usize.pow(16));
        //test_cos(cos_p5::<i128>, MARGIN, 126, 2_usize.pow(48));
    }

    #[test]
    fn test_sin_p5o() {
        assert_eq!(51437, sin_p5o_k::<i32>(2_i32.pow(i32::BITS / 2 - 1)));
        //assert_eq!(3370945099, sin_p5o_k::<i64>());
        const MARGIN: f64 = 0.925201;
        test_sin(sin_p5o::<i32>, 2_i32.pow(15), MARGIN, 30, 1);
        //test_sin(sin_p5o::<i64>, MARGIN, 62, 2_usize.pow(16));
        //test_sin(sin_p5o::<i128>, MARGIN, 126, 2_usize.pow(48));
    }

    #[test]
    fn test_cos_p5o() {
        const MARGIN: f64 = 0.925201;
        test_cos(cos_p5o::<i32>, 2_i32.pow(15), MARGIN, 30, 1);
        //test_cos(cos_p5o::<i64>, MARGIN, 62, 2_usize.pow(16));
        //test_cos(cos_p5o::<i128>, MARGIN, 126, 2_usize.pow(48));
    }
}

fn read<T>(dir: &str, file_name: &str) -> Result<Vec<T>>
where
    T: DeserializeOwned,
{
    let path = Path::new(dir).join(file_name);
    let inner = File::open(path)?;
    let rdr = BufReader::new(inner);
    Ok(serde_json::from_reader(rdr)?)
}

fn test<T>(expected: &[T], actual: &[T], right: T) -> Result<Vec<Error>>
where
    T: Angle + AsPrimitive<usize> + Display,
{
    let errors: Vec<_> = {
        let len = {
            let right: usize = right.as_();
            right + 1
        };
        ensure!(expected.len() == len);
        ensure!(actual.len() == len);
        (0..len)
            .filter_map(|i| {
                let expected = &expected[i];
                let actual = &actual[i];
                (expected != actual)
                    .then(|| anyhow!("i: {i}, expected: {expected}, actual: {actual}"))
            })
            .collect()
    };
    Ok(errors)
}

fn read_and_test<T>(dir: &str, file_name: &str, actual: &[T], right: T) -> Result<Vec<Error>>
where
    T: Angle + AsPrimitive<usize> + DeserializeOwned + Display,
{
    let expected = read::<T>(dir, file_name)?;
    test(&expected, actual, right)
}

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
    input: Option<String>,

    #[arg(short, long)]
    output: Option<String>,

    #[arg(short, long)]
    print: bool,
}

fn test_and_write<F, T>(args: &Args, f: F, right: T, file_name: &str) -> Result<()>
where
    F: Fn(<RangeInclusive<T> as Iterator>::Item, T) -> T,
    T: Angle + AsPrimitive<usize> + DeserializeOwned + Display + Serialize,
    RangeInclusive<T>: Iterator,
{
    let actual: Vec<_> = (0.into()..=right).map(|x| f(x, right)).collect();
    if let Some(dir) = &args.input {
        let errors = read_and_test(dir, file_name, &actual, right)?;
        for e in &errors {
            eprintln!("{}", e);
        }
        ensure!(errors.is_empty());
    }
    if let Some(dir) = &args.output {
        serialize_and_write(dir, file_name, &actual)?;
    }
    Ok(())
}

fn test_and_write_all(args: &Args) -> Vec<Error> {
    [
        test_and_write(args, sin_p2::<i16>, 2_i16.pow(7), "sin_p2_i16.json"),
        test_and_write(args, sin_p3::<i32>, 2_i32.pow(15), "sin_p3_i32.json"),
        test_and_write(args, sin_p4::<i32>, 2_i32.pow(15), "sin_p4_i32.json"),
        test_and_write(args, sin_p4o::<i32>, 2_i32.pow(15), "sin_p4o_i32.json"),
        test_and_write(args, sin_p5::<i32>, 2_i32.pow(15), "sin_p5_i32.json"),
        test_and_write(args, sin_p5o::<i32>, 2_i32.pow(15), "sin_p5o_i32.json"),
    ]
    .into_iter()
    .filter_map(Result::err)
    .collect()
}

fn print_max<'a, Expected, Actual, T>(
    expected: Expected,
    actual: Actual,
    exp: u32,
    step: usize,
    right: T,
) where
    Expected: Iterator<Item = &'a f64>,
    Actual: Fn(T, T) -> T,
    T: Angle + AsPrimitive<f64> + From<<RangeInclusive<T> as Iterator>::Item>,
    RangeInclusive<T>: Iterator,
{
    let one = 2_i32.pow(exp) as f64;
    let diffs = (0.into()..=right)
        .step_by(step)
        .map(|x| {
            let actual: f64 = actual(x.into(), right).as_();
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
        let frac_pi_straight = {
            let right = 2_i32.pow(i32::BITS / 2 - 1);
            FRAC_PI_2 / right as f64
        };
        let right = 2_i32.pow(i32::BITS / 2 - 1);
        sin = (0..right)
            .map(|x| (frac_pi_straight * x as f64).sin())
            .chain(once(1.0f64))
            .collect();
        cos = (0..right)
            .map(|x| (frac_pi_straight * x as f64).cos())
            .chain(once(0.0f64))
            .collect();
    }

    // sin_p1::<i8>
    {
        type Type = i8;
        let exp = Type::BITS / 2 - 1;
        let expected = {
            let step = 2_i32.pow(i32::BITS / 2 - 1 - exp) as usize;
            sin.iter().step_by(step)
        };
        let actual = sin_p1::<Type>;
        print!("sin_p1::<i8>:   ");
        print_max(expected, actual, exp, 1, 2_i8.pow(3));
    }

    // cos_p2::<i16>
    {
        type Type = i16;
        let exp = Type::BITS - 2;
        let expected = {
            let step = 2_i32.pow(i32::BITS / 2 - 1 - exp / 2) as usize;
            cos.iter().step_by(step)
        };
        let actual = cos_p2::<Type>;
        print!("cos_p2::<i16>:  ");
        print_max(expected, actual, exp, 1, 2_i16.pow(7));
    }

    // i32
    {
        type Type = i32;
        let exp = Type::BITS - 2;
        let right = 2_i32.pow(15);
        print!("sin_p3::<i32>:  ");
        print_max(sin.iter(), sin_p3::<Type>, exp, 1, right);
        print!("cos_p4::<i32>:  ");
        print_max(cos.iter(), cos_p4::<Type>, exp, 1, right);
        print!("cos_p4o::<i32>: ");
        print_max(cos.iter(), cos_p4o::<Type>, exp, 1, right);
        print!("sin_p5::<i32>:  ");
        print_max(sin.iter(), sin_p5::<Type>, exp, 1, right);
        print!("sin_p5o::<i32>: ");
        print_max(sin.iter(), sin_p5o::<Type>, exp, 1, right);
    }
}

fn main() {
    let args = Args::parse();
    {
        let errors = test_and_write_all(&args);
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
