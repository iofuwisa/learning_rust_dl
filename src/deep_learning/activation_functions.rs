use std::f64::consts::E;
use ndarray::prelude::{
    Array1,
    arr1,
};
use crate::deep_learning::common::*;


// step function
pub fn step(x: f64) -> f64 {
    return if x > 0.0 {
        1.0
    } else {
        0.0
    };
}
pub fn step_array(x: &Array1<f64>) -> Array1<f64> {
    let mut y = x.clone();
    for n in 0..x.len() {
        y[n] = step(x[n]);
    }
    return y;
}


// sigmoid function
pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + E.powf(-x))
}
pub fn sigmoid_array(x: &Array1<f64>) -> Array1<f64> {
    let mut y = x.clone();
    for n in 0..x.len() {
        y[n] = sigmoid(x[n]);
    }
    return y;
}


// ReLU function
pub fn relu(x: f64) -> f64 {
    if x > 0.0 {
        x
    } else {
        0.0
    }
}
pub fn relu_array(x: &Array1<f64>) -> Array1<f64> {
    let mut y = x.clone();
    for n in 0..x.len() {
        y[n] = relu(x[n]);
    }
    return y;
}


// idenntity function
pub fn identity(x: f64) -> f64 {
    x
}
pub fn identity_array(x: &Array1<f64>) -> Array1<f64> {
    let mut y = x.clone();
    for n in 0..x.len() {
        y[n] = identity(x[n]);
    }
    return y;
}


// softmax function
pub fn softmax(x: f64, x_array: &Array1<f64>) -> f64 {
    let c = max(x_array);
    let mut sum = 0.0;
    for ix in x_array {
        sum += E.powf(*ix + c)
    } 
    return E.powf(x + c) / sum;
}
pub fn softmax_array(x: &Array1<f64>) -> Array1<f64> {
    let mut y = x.clone();
    for n in 0..x.len() {
        y[n] = softmax(x[n], x);
    }
    return y;
}


pub fn max(x: &Array1<f64>) -> f64 {
    let mut max = x[0];
    for ix in x {
        max = 
        if max < *ix  { *ix }
        else { max }
    } 
    return max;
}

#[cfg(test)]
mod test_mod {
    use super::*;

    #[test]
    fn test_max() {
        let arr = arr1(&[0.1, -8.0, 5.0, 10.0]);
        let a = max(&arr);
        assert_eq!(a, 10.0);


        let arr = arr1(&[-0.1, -8.0, -2.0]);
        let a =max(&arr);
        assert_eq!(a, -0.1);
    }

    #[test]
    fn test_softmax_array_max() {
        let arr = arr1(&[0.1, -8.0, 5.0, 10.0, 2.0]);
        let a = softmax_array(&arr);
        assert_eq!(max_index_in_arr1(&a), 3);

        let arr = arr1(&[-0.1, -8.0, -5.0, -10.0, -2.0]);
        let a = softmax_array(&arr);
        assert_eq!(max_index_in_arr1(&a), 0);
    }

    #[test]
    fn test_softmax_array_total() {
        let arr = arr1(&[0.1, -8.0, 5.0, 10.0, 2.0]);
        let a = softmax_array(&arr);
        let mut sub = 0.0;
        for i in 0.. arr.len() {
            sub += a[i];
        }
        assert_eq!(round_digit(sub, -3), 1.0);

        let arr = arr1(&[-0.1, -8.0, -5.0, -10.0, -2.0]);
        let a = softmax_array(&arr);
        let mut sub = 0.0;
        for i in 0.. arr.len() {
            sub += a[i];
        }
        assert_eq!(round_digit(sub, -3), 1.0);
    }

}