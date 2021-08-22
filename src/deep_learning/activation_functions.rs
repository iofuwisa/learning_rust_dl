use std::f64::consts::E;
use ndarray::prelude::{
    Array1,
};


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


fn max(x: &Array1<f64>) -> f64 {
    let mut max = x[0];
    for ix in x {
        max = 
        if max < *ix  { *ix }
        else { max }
    } 
    return max;
}