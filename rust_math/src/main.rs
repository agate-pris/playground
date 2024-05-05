use clap::Parser;
use rust_math::{
    atan_p2_2850, atan_p3_2555_691, atan_p5_787_2968, cos_p2_i32,
    sin_cos::{cos_p4_7032, cos_p4_7384, sin_p5_51437, sin_p5_51472},
    sin_p3_16384,
};

#[derive(Parser, Debug)]
struct Args {
    #[rustfmt::skip] #[arg(long)] cos_p2: bool,
    #[rustfmt::skip] #[arg(long)] sin_p3: bool,
    #[rustfmt::skip] #[arg(long)] cos_p4: bool,
    #[rustfmt::skip] #[arg(long)] sin_p5: bool,
    #[rustfmt::skip] #[arg(long)] cos_p4o: bool,
    #[rustfmt::skip] #[arg(long)] sin_p5o: bool,
    #[rustfmt::skip] #[arg(long)] atan_p2: bool,
    #[rustfmt::skip] #[arg(long)] atan_p3: bool,
    #[rustfmt::skip] #[arg(long)] atan_p5: bool,
}

fn main() {
    let args = Args::parse();

    fn print<F>(f: F)
    where
        F: Fn(i32) -> i32,
    {
        const K: i32 = 2_i32.pow(i32::BITS / 2 - 1);
        println!("[");
        for x in 0..K {
            println!("{},", f(x));
        }
        println!("{}", f(K));
        print!("]");
    }

    if args.cos_p2 {
        print(cos_p2_i32);
    }
    if args.sin_p3 {
        print(sin_p3_16384);
    }
    if args.cos_p4 {
        print(cos_p4_7032);
    }
    if args.sin_p5 {
        print(sin_p5_51472);
    }
    if args.cos_p4o {
        print(cos_p4_7384);
    }
    if args.sin_p5o {
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
