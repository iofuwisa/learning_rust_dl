use ndarray::prelude::{
    arr1,
    arr2,
    s
};

use crate::deep_learning::activation_functions::{
    sigmoid_array,
    identity_array,
    softmax_array
};

pub fn calc(x1: f64, x2: f64) {
    let x = arr1(&[x1, x2]);
    let w1 = arr2(&[
        [0.1, 0.3, 0.5],
        [0.2, 0.4, 0.6]
    ]);
    let b1 = arr1(&[0.1, 0.2, 0.3]);
    let w2 = arr2(&[
        [0.1, 0.4],
        [0.2, 0.5],
        [0.3, 0.6]
    ]);
    let b2 = arr1(&[0.1, 0.2]);
    let w3 = arr2(&[
        [0.1, 0.3],
        [0.2, 0.4]
    ]);
    let b3 = arr1(&[0.1, 0.2]);

    let xw1 = x.dot(&w1) + b1;
    let mut z1: Vec<f64> = Vec::new();
    if let Some(val) = xw1.slice(s![0..;1]).as_slice() {
        sigmoid_array(val, &mut z1);
    } else {
        println!("error1");
        return;
    }
    
    let x2 = arr1(&[z1[0], z1[1], z1[2]]);
    let x2w2 = x2.dot(&w2) + b2;
    let mut z2: Vec<f64> = Vec::new();
    if let Some(val) = x2w2.slice(s![0..;1]).as_slice() {
        sigmoid_array(val, &mut z2);
    } else {
        println!("error2");
        return;
    }

    let x3 = arr1(&[z2[0], z2[1]]);
    let x3w3 = x3.dot(&w3) + b3;
    let mut z3: Vec<f64> = Vec::new();
    if let Some(val) = x3w3.slice(s![0..;1]).as_slice() {
        // identity_array(val, &mut z3);
        softmax_array(val, &mut z3);
    } else {
        println!("error3");
        return;
    }
    println!("{:?}", z3);
}