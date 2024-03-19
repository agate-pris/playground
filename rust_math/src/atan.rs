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

    use crate::{atan_p2::atan_p2_default, atan_p3::atan_p3_default, tests::read_data};

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
}
