use std::f64::consts::E;
use ndarray::prelude::{
    Array1,
    arr1,
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
pub fn crosss_entropy_error(y: &Array1<f64>, t: &Array1<f64>) -> f64 {
    if y.len() != t.len() {
        panic!("Different len. y:{} t:{}", y.len(), t.len());
    }

    let mut e = 0.0;

    for i in 0..y.len() {
        if t[i] > 0.0 {
            e -= t[i] * y[i].log(E);
        }
    }

    return e;
}


#[cfg(test)]
mod test_mod {
    use super::*;

    #[test]
    fn test_crosss_entropy_error() {
        let y1 = arr1(&[0.1, 0.6, 0.2, 0.1]);
        let y2 = arr1(&[0.0, 1.0, 0.0, 0.0]);
        let t = arr1(&[0.0, 1.0, 0.0, 0.0]);

        let a = crosss_entropy_error(&y1, &t);
        assert_eq!((a * 1000.0).floor() / 1000.0, 0.51);

        let a = crosss_entropy_error(&y2, &t);
        assert_eq!((a * 1000.0).floor() / 1000.0, 0.0);
    }  
}