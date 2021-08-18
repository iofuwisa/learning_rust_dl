use crate::deep_learning::activation_functions::step;

pub fn and(x: &[f64]) -> f64 {
    let w = vec![0.5, 0.5];
    let b = -0.7;
    let mut tmp = 0.0;
    let mut iw = w.iter();
    for ix in x {
        match iw.next() {
            Some(siw) =>
                tmp += ix*siw,
            None =>
                break
        };
    }
    tmp += b;

    return step(tmp)
}

pub fn nand(x: &[f64]) -> f64 {
    let w = vec![-0.5, -0.5];
    let b = 0.7;
    let mut tmp = 0.0;

    let mut iw = w.iter();
    for ix in x {
        match iw.next() {
            Some(siw) =>
                tmp += ix*siw,
            None =>
                break
        };
    }
    tmp += b;

    return step(tmp)
}

pub fn or(x: &[f64]) -> f64 {
    let w = vec![0.5, 0.5];
    let b = -0.2;
    let mut tmp = 0.0;

    let mut iw = w.iter();
    for ix in x {
        match iw.next() {
            Some(siw) =>
                tmp += ix*siw,
            None =>
                break
        };
    }
    tmp += b;

    return step(tmp)
}