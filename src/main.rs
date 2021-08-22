pub mod deep_learning;

use crate::deep_learning::logical_operators::*;
use crate::deep_learning::activation_functions::*;
use crate::deep_learning::network::*;

use ndarray::prelude::*;

fn main(){

    let mut x: Vec<f64> = Vec::new();
    x.push(0.0);
    x.push(1.0);
    println!("and:{}", and(&x));
    println!("nand:{}", nand(&x));
    println!("or:{}", or(&x));


    let x = [-5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    let mut step_y: Vec<f64> = Vec::new();
    let mut sigm_y: Vec<f64> = Vec::new();
    let mut relu_y: Vec<f64> = Vec::new();
    step_array(&x, &mut step_y);
    sigmoid_array(&x, &mut sigm_y);
    relu_array(&x, &mut relu_y);
    println!("x: {:?}", x);
    println!("step_y: {:?}", step_y);
    println!("sigm_y: {:?}", sigm_y);
    println!("relu_y: {:?}", relu_y);

    calc(1.0, 0.5);

}


