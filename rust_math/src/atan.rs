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
