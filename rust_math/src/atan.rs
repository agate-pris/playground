pub(crate) fn atan_impl<F>(x: i32, k: i32, f: F) -> i32
where
    F: Fn(i32, i32) -> i32,
{
    let x_abs = (x as i64).abs();
    if x_abs > k as i64 {
        let signum = x.signum();
        let k_2 = k * k;
        let x = (k_2 as i64 / x_abs) as i32;
        signum * (k_2 / 2 - f(x, x))
    } else {
        f(x, x_abs as i32)
    }
}

pub(crate) fn atan2_impl<F>(y: i32, x: i32, k: i32, f: F) -> i32
where
    F: Fn(i32) -> i32,
{
    if y == 0 && x == 0 {
        return 0;
    }

    let x_abs = (x as i64).abs();
    let y_abs = (y as i64).abs();
    let x_is_negative = x.is_negative();
    let y_is_negative = y.is_negative();

    if x_abs > y_abs {
        let x = (y_abs * k as i64 / x_abs) as i32;
        let v = f(x);
        match (x_is_negative, y_is_negative) {
            (false, false) => v,
            (true, false) => k * k - v,
            (false, true) => -v,
            (true, true) => v - k * k,
        }
    } else {
        let x = (x_abs * k as i64 / y_abs) as i32;
        let v = f(x);
        let right = k * k / 2;
        match (x_is_negative, y_is_negative) {
            (false, false) => right - v,
            (true, false) => right + v,
            (false, true) => -right + v,
            (true, true) => -right - v,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::f64::consts::PI;

    use crate::{
        atan_p2::{atan2_p2_default, atan_p2_default},
        atan_p3::{atan2_p3_default, atan_p3_default},
        tests::read_data,
    };

    fn test_atan<F>(f: F, data_path: &str, acceptable_error: f64)
    where
        F: Fn(i32) -> i32,
    {
        use std::i32::{MAX, MIN};

        const EXP: u32 = i32::BITS / 2 - 1;
        const K: i32 = 2_i32.pow(EXP);
        const HALF_RIGHT: i32 = K.pow(2) / 4;

        let expected = read_data::<i32>(data_path).unwrap();

        assert_eq!(expected[0], 0);
        assert_eq!(expected[K as usize], HALF_RIGHT);

        assert_eq!(f(0), 0);
        assert_eq!(f(K), HALF_RIGHT);
        assert_eq!(f(-K), -HALF_RIGHT);

        let f_max = f(MAX);
        let f_min = f(MIN);

        assert_eq!(f_max, 2 * HALF_RIGHT);
        assert_eq!(f_min, -2 * HALF_RIGHT);
        assert_eq!(f_max, f(K * K + 1));
        assert_eq!(f_min, f(-K * K - 1));

        for i in 1..=K {
            const RIGHT: i32 = 2 * HALF_RIGHT;

            let expected = expected[i as usize];
            let inv_i = K.pow(2) / i;

            let fi = f(i);

            assert_eq!(fi, expected);
            assert_eq!(f(-i), -expected);
            assert_eq!(f(inv_i), RIGHT - expected);
            assert_eq!(f(-inv_i), expected - RIGHT);

            approx::assert_abs_diff_eq!(
                fi as f64 * PI / K.pow(2) as f64,
                (i as f64 / K as f64).atan(),
                epsilon = acceptable_error
            );
        }

        // bignum
        {
            // 17th mersenne prime
            const STEP: usize = 131071;

            for x in (MIN..=MAX).step_by(STEP) {
                if (-K..=K).contains(&x) {
                    continue;
                }
                let actual = f(x);
                let expected = x.signum() * (K * K / 2 - f((K * K / x).abs()));
                assert_eq!(expected, actual);
            }
        }
    }

    #[rustfmt::skip] #[test] fn test_atan_p2() { test_atan(atan_p2_default, "data/atan_p2.json", 0.0039); }
    #[rustfmt::skip] #[test] fn test_atan_p3() { test_atan(atan_p3_default, "data/atan_p3.json", 0.0016); }

    fn test_atan2_symmetry<F>(f: F, y_abs: i32, x_abs: i32, acceptable_error: f64)
    where
        F: Fn(i32, i32) -> i32,
    {
        #[rustfmt::skip]
        let args = [
           [ y_abs,  x_abs],
           [ x_abs,  y_abs],
           [ y_abs, -x_abs],
           [ x_abs, -y_abs],
           [-y_abs,  x_abs],
           [-x_abs,  y_abs],
           [-y_abs, -x_abs],
           [-x_abs, -y_abs],
        ];

        const HALF_RIGHT: i32 = 2_i32.pow(i32::BITS / 2 - 1).pow(2) / 4;

        let actuals = args.iter().map(|&[y, x]| f(y, x)).collect::<Vec<_>>();

        if y_abs == 0 && x_abs == 0 {
            assert!(actuals.iter().all(|&actual| actual == 0));
        } else {
            #[rustfmt::skip] assert_eq!( 2 * HALF_RIGHT - actuals[0], actuals[1]);
            #[rustfmt::skip] assert_eq!( 4 * HALF_RIGHT - actuals[0], actuals[2]);
            #[rustfmt::skip] assert_eq!( 2 * HALF_RIGHT + actuals[0], actuals[3]);
            #[rustfmt::skip] assert_eq!(                 -actuals[0], actuals[4]);
            #[rustfmt::skip] assert_eq!(-2 * HALF_RIGHT + actuals[0], actuals[5]);
            #[rustfmt::skip] assert_eq!(-2 * HALF_RIGHT - actuals[0], actuals[7]);

            if actuals[0] == 0 && y_abs == 0 {
                assert_eq!(
                    4 * HALF_RIGHT + actuals[0],
                    actuals[6],
                    "y_abs: {y_abs}, x_abs: {x_abs}"
                );
            } else {
                assert_eq!(
                    -4 * HALF_RIGHT + actuals[0],
                    actuals[6],
                    "y_abs: {y_abs}, x_abs: {x_abs}"
                );
            }
        }

        const TO_RAD: f64 = PI / 2_i32.pow(i32::BITS - 2) as f64;

        let expected = args
            .iter()
            .map(|&[y, x]| (y as f64).atan2(x as f64))
            .collect::<Vec<_>>();

        actuals
            .iter()
            .zip(expected.iter())
            .for_each(|(&actual, &expected)| {
                approx::assert_abs_diff_eq!(
                    actual as f64 * TO_RAD,
                    expected,
                    epsilon = acceptable_error
                )
            });
    }

    fn test_atan2<F>(f: F)
    where
        F: Fn(i32, i32) -> i32,
    {
        use std::i32::{MAX, MIN};

        const EXP: u32 = i32::BITS / 2 - 1;
        const K: i32 = 2_i32.pow(EXP);
        const HALF_RIGHT: i32 = K.pow(2) / 4;

        const ACCEPTABLE_ERROR: f64 = 0.000001;

        test_atan2_symmetry(&f, 0, 0, ACCEPTABLE_ERROR);
        test_atan2_symmetry(&f, 0, 1, ACCEPTABLE_ERROR);
        test_atan2_symmetry(&f, 0, MAX, ACCEPTABLE_ERROR);
        test_atan2_symmetry(&f, 1, 1, ACCEPTABLE_ERROR);
        test_atan2_symmetry(&f, 1, MAX, ACCEPTABLE_ERROR);
        test_atan2_symmetry(&f, MAX, MAX, ACCEPTABLE_ERROR);

        assert_eq!(0, f(0, 1));
        assert_eq!(0, f(0, MAX));
        assert_eq!(HALF_RIGHT, f(1, 1));
        assert_eq!(HALF_RIGHT, f(MAX, MAX));
        assert_eq!(HALF_RIGHT * 4, f(0, MIN));
        assert_eq!(HALF_RIGHT * -2, f(MIN, 0));
        assert_eq!(HALF_RIGHT * -3, f(MIN, MIN));

        let div_check = |a: i32, b: i32, c: i32, d: i32| {
            assert_eq!(f(a, b), f(c, d));
            test_atan2_symmetry(&f, a, b, 0.01);
            test_atan2_symmetry(&f, c, d, 0.01);
        };

        #[rustfmt::skip] div_check( 9997, 32768,  3051, 10000); //  9997.51
        #[rustfmt::skip] div_check(10000, 32768,  3052, 10000); // 10000.79
        #[rustfmt::skip] div_check(10004, 32768,  3053, 10000); // 10004.07
        #[rustfmt::skip] div_check( 9999, 32768, 15258, 50000); //  9999.48
        #[rustfmt::skip] div_check(10000, 32768, 15259, 50000); // 10000.13
        #[rustfmt::skip] div_check(10000, 32768, 15260, 50000); // 10000.79
    }

    #[rustfmt::skip] #[test] fn test_atan2_p2() { test_atan2(atan2_p2_default); }
    #[rustfmt::skip] #[test] fn test_atan2_p3() { test_atan2(atan2_p3_default); }

    fn test_atan2_periodicity<F>(f: F, data_path: &str, acceptable_error: f64)
    where
        F: Fn(i32, i32) -> i32,
    {
        use std::i32::{MAX, MIN};

        let expected = read_data::<i32>(data_path).unwrap();

        const K: i32 = 2_i32.pow(i32::BITS / 2 - 1);
        for v in 1..=K {
            let expected = expected[v as usize];

            approx::assert_abs_diff_eq!(
                expected as f64 * PI / K.pow(2) as f64,
                (v as f64).atan2(K as f64),
                epsilon = acceptable_error
            );

            test_atan2_symmetry(&f, v, K, acceptable_error);
        }

        let verify = |y: i32, x: i32| {
            let actual = f(y, x);

            if y != MIN && x != MIN {
                test_atan2_symmetry(&f, y.abs(), x.abs(), acceptable_error);
            } else {
                approx::assert_abs_diff_eq!(
                    actual as f64 * PI / K.pow(2) as f64,
                    (y as f64).atan2(x as f64),
                    epsilon = acceptable_error
                );
            }

            if x == 0 || y == 0 {
                assert_eq!(actual, 0);
            } else {
                let x_abs = (x as i64).abs();
                let y_abs = (y as i64).abs();
                let x_signum = x.signum();
                let y_signum = y.signum();

                let expected = if x_abs > y_abs {
                    let new_y = y_signum * (y_abs * K as i64 / x_abs) as i32;
                    let new_x = x_signum * K;
                    assert!((-K..=K).contains(&new_x));
                    assert!((-K..=K).contains(&new_y));

                    let v = f(new_y, new_x);
                    if new_y == 0 {
                        if x.is_negative() {
                            assert_eq!(v, K.pow(2));
                        } else {
                            assert_eq!(v, 0);
                        }
                        match (x.is_negative(), y.is_negative()) {
                            (true, true) => -K.pow(2),
                            (true, false) => K.pow(2),
                            _ => 0,
                        }
                    } else {
                        v
                    }
                } else {
                    let new_y = y_signum * K;
                    let new_x = x_signum * (x_abs * K as i64 / y_abs) as i32;
                    assert!((-K..=K).contains(&new_x));
                    assert!((-K..=K).contains(&new_y));
                    f(new_y, new_x)
                };
                assert_eq!(actual, expected);
            }
        };

        const STEPS: [usize; 2] = [16777259, 16777289];
        for x_step in STEPS {
            for y_step in STEPS {
                for x in (MIN..=MAX).step_by(x_step) {
                    for y in (MIN..=MAX).step_by(y_step) {
                        verify(y, x);
                    }
                }
                for x in (MIN..=MAX).rev().step_by(x_step) {
                    for y in (MIN..=MAX).rev().step_by(y_step) {
                        verify(y, x);
                    }
                }
            }
        }
    }

    #[rustfmt::skip] #[test] fn test_atan2_p2_periodicity() { test_atan2_periodicity(atan2_p2_default, "data/atan_p2.json", 0.0039); }
    #[rustfmt::skip] #[test] fn test_atan2_p3_periodicity() { test_atan2_periodicity(atan2_p3_default, "data/atan_p3.json", 0.0017); }
}
