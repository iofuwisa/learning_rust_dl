use std::f64::consts::E;
use ndarray::prelude::{
    Array1,
};


// Sum squared error
pub fn sum_squared_error(y: &Array1<f64>, t: &Array1<f64>) -> f64 {
    if y.len() != t.len() {
        panic!("Different len. y:{} t:{}", y.len(), t.len());
    }

    let mut e = 0.0;

    for i in 0..y.len() {
        e += (y[i] - t[i]).powi(2);
    }
    e = e / 2.0;

    return e;

}

// Cross entropy error
pub fn crosss_entropy_erro(y: &Array1<f64>, t: &Array1<f64>) -> f64 {
    if y.len() != t.len() {
        panic!("Different len. y:{} t:{}", y.len(), t.len());
    }

    let mut e = 0.0;

    for i in 0..y.len() {
        e += t[i] * E.log(y[i]);
    }

    e = -e;

    return e;

}