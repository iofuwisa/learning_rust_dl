pub mod deep_learning;

// use crate::deep_learning::logical_operators::*;
use crate::deep_learning::activation_functions::*;
use crate::deep_learning::loss_functions::*;
use crate::deep_learning::network::*;
use crate::deep_learning::mnist::*;
use crate::deep_learning::common::*;
use crate::deep_learning::graph_plotter::*;

use ndarray::prelude::{
    Array1,
    arr1,
    arr2,
    s
};
use rulinalg::matrix::{
    Matrix, 
    Row,
    BaseMatrix
};

const TRN_IMG_SIZE: usize = 5000;
const VAL_IMG_SIZE: usize = 0;
const TST_IMG_SIZE: usize = 0;

const MINI_BATCH_SIZE: usize = 100;

fn main(){

    // // Load MNIST
    // let mnist_images = MnistImages::new(5000, 0, 0);
    // let trn_img = mnist_images.get_trn_img();
    // let trn_lbl = mnist_images.get_trn_lbl();
    // let trn_lbl_one_hot = mnist_images.get_trn_lbl_one_hot();

    // // Setup NN
    // let nn = NeuralNetwork::new();

    // // minibatch index
    // let indexes = random_choice(MINI_BATCH_SIZE, TRN_IMG_SIZE);

    // for i in indexes {
    //     let img = trn_img.row(i);
    //     let img = img.into_matrix();
    //     let img = img.data();
    //     let img = arr1(img);

    //     let y = nn.forward(&img);

    //     println!("index:{}\nans:{}\nresult:{:?}", i, trn_lbl[i], y);

    // }

    // let plot_data = Vec::<(&str, f64)>::with_capacity(20);
    // for n in 0..20 {
    //     let y = numeric_diff(Box::new(f), n as f64);
    //     println!("x:{} y:{}", n, y);
    //     plot_data.push((&n.to_string(), y));
    // }
    // // prot(&plot_data);

    for x1 in -10..11 {
        for x2 in -10..11 {
            let grad = numeric_gradient(Box::new(f2), &vec![x1 as f64 / 10.0, x2 as f64 / 10.0]);
            print!("{},{}  ",  format!("{:.*}", 2, grad[0] * 100000000.0), format!("{:.*}", 2, grad[1] * 100000000.0));
            // print!("{},{}  ",  grad[0], grad[1]);
        }
        println!();
    }    
}

fn f2(x: &Vec<f64>) -> f64 {
    x[0].powi(2) + x[1].powi(2)
}