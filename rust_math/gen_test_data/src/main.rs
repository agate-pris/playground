use std::f64::consts::FRAC_PI_2;

use clap::Parser;
use rust_math::{
    atan_p2_2850, atan_p3_2555_691, atan_p5_787_2968, sin_p2_i32, sin_p3_16384, sin_p4_7032,
    sin_p4_7384, sin_p5_51437, sin_p5_51472,
};

const UNIT_EXP: u32 = i32::BITS / 2 - 1;
const UNIT: i32 = 1 << UNIT_EXP;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    sin_errors: bool,

    #[arg(long)]
    sin_p2: bool,

    #[arg(long)]
    sin_p3: bool,

    #[arg(long)]
    sin_p4_7032: bool,

    #[arg(long)]
    sin_p4_7384: bool,

    #[arg(long)]
    sin_p5_51472: bool,

    #[arg(long)]
    sin_p5_51437: bool,

    #[arg(long)]
    atan_p2: bool,

    #[arg(long)]
    atan_p3: bool,

    #[arg(long)]
    atan_p5: bool,
}

fn print_sin_errors() {
    fn f(f: impl Fn(i32) -> i32) -> f64 {
        (0..=UNIT).fold(0.0, |acc, x| {
            const FRAC_PI_STRAIGHT: f64 = FRAC_PI_2 / UNIT as f64;
            const ONE: f64 = (1 << (2 * UNIT_EXP)) as f64;
            let err = f(x) as f64 / ONE - (x as f64 * FRAC_PI_STRAIGHT).sin();
            std::cmp::max_by(err, acc, |a, b| a.abs().total_cmp(&b.abs()))
        })
    }
    let results = [
        ("sin_p2_i32", f(sin_p2_i32)),
        ("sin_p3_16384", f(sin_p3_16384)),
        ("sin_p4_7032", f(sin_p4_7032)),
        ("sin_p4_7384", f(sin_p4_7384)),
        ("sin_p5_51472", f(sin_p5_51472)),
        ("sin_p5_51437", f(sin_p5_51437)),
    ];
    println!("{:>12} | max error", "func");
    for (name, result) in results.iter() {
        println!("{:>12} | {:10.7}", name, result);
    }
}

fn print(f: impl Fn(i32) -> i32) {
    println!("[");
    for x in 0..UNIT {
        print!("{}", f(x));
        println!(",");
    }
    println!("{}", f(UNIT));
    print!("]");
}

fn main() {
    let args = Args::parse();
    if args.sin_errors {
        print_sin_errors();
    }
    if args.sin_p2 {
        print(sin_p2_i32);
    }
    if args.sin_p3 {
        print(sin_p3_16384);
    }
    if args.sin_p4_7032 {
        print(sin_p4_7032);
    }
    if args.sin_p4_7384 {
        print(sin_p4_7384);
    }
    if args.sin_p5_51472 {
        print(sin_p5_51472);
    }
    if args.sin_p5_51437 {
        print(sin_p5_51437);
    }

    if args.atan_p2 {
        print(atan_p2_2850);
    }
    if args.atan_p3 {
        print(atan_p3_2555_691);
    }
    if args.atan_p5 {
        print(atan_p5_787_2968);
    }
}
