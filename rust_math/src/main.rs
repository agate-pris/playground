use clap::Parser;
use rust_math::sin_cos::{
    cos_p2_default, cos_p4_default, cos_p4o_default, sin_p3_default, sin_p5_default,
    sin_p5o_default,
};

#[derive(Parser, Debug)]
struct Args {
    #[rustfmt::skip] #[arg(long)] cos_p2: bool,
    #[rustfmt::skip] #[arg(long)] sin_p3: bool,
    #[rustfmt::skip] #[arg(long)] cos_p4: bool,
    #[rustfmt::skip] #[arg(long)] sin_p5: bool,
    #[rustfmt::skip] #[arg(long)] cos_p4o: bool,
    #[rustfmt::skip] #[arg(long)] sin_p5o: bool,
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
        print(cos_p2_default);
    }
    if args.sin_p3 {
        print(sin_p3_default);
    }
    if args.cos_p4 {
        print(cos_p4_default);
    }
    if args.sin_p5 {
        print(sin_p5_default);
    }
    if args.cos_p4o {
        print(cos_p4o_default);
    }
    if args.sin_p5o {
        print(sin_p5o_default);
    }
}
