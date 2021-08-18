use std::f64::consts::E;

// step function
pub fn step(x: f64) -> f64 {
    return if x > 0.0 {
        1.0
    } else {
        0.0
    };
}
pub fn step_array(x: &[f64], y: &mut Vec<f64>) {
    for ix in x {
        y.push(
            step(*ix)
        );
    }
}

// sigmoid function
pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + E.powf(-x))
}
pub fn sigmoid_array(x: &[f64], y: &mut Vec<f64>) {
    for ix in x {
        y.push(
            sigmoid(*ix)
        );
    }
}

// ReLU function
pub fn relu(x: f64) -> f64 {
    if x > 0.0 {
        x
    } else {
        0.0
    }
}
pub fn relu_array(x: &[f64], y: &mut Vec<f64>) {
    for ix in x {
        y.push(
            relu(*ix)
        );
    }
}

// idenntity function
pub fn identity(x: f64) -> f64 {
    x
}
pub fn identity_array(x: &[f64], y: &mut Vec<f64>) {
    for ix in x {
        y.push(
            identity(*ix)
        );
    }
}

// softmax function
pub fn softmax(x: f64, x_array: &[f64]) -> f64 {
    let c = max(x_array);
    let mut sum = 0.0;
    for ix in x_array {
        sum += E.powf(*ix + c)
    } 
    return E.powf(x + c) / sum;
}

pub fn softmax_array(x: &[f64], y: &mut Vec<f64>) {
    for ix in x {
        y.push(
            softmax(*ix, x)
        );
    }
}

fn max(x: &[f64]) -> f64 {
    let mut max = x[0];
    for ix in x {
        max = 
        if max < *ix  { *ix }
        else { max }
    } 
    return max;
}