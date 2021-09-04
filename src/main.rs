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
    s,
    Axis
};

const TRN_IMG_SIZE: usize = 5000;
const VAL_IMG_SIZE: usize = 0;
const TST_IMG_SIZE: usize = 0;

const MINI_BATCH_SIZE: usize = 100;

fn main(){

    // Load MNIST
    let mnist_images = MnistImages::new(5000, 0, 0);
    let trn_img = mnist_images.get_trn_img();
    let trn_lbl = mnist_images.get_trn_lbl();
    let trn_lbl_one_hot = mnist_images.get_trn_lbl_one_hot();

    // Setup NN
    let nn = NeuralNetwork::new(
        784,
        vec![
            NeuralNetworkLayorBuilder::new(50, Box::new(&sigmoid_array)),   // hidden1
            NeuralNetworkLayorBuilder::new(100, Box::new(&sigmoid_array)),  // hidden2
            NeuralNetworkLayorBuilder::new(10, Box::new(&softmax_array)),  // output
        ]
    );

    // Minibatch index
    let indexes = random_choice(MINI_BATCH_SIZE, TRN_IMG_SIZE);

    // Forwading
    for i in indexes {
        // Convertt image to Array1 from Matrix
        let img = trn_img.index_axis(Axis(0), i);

        // Forwad
        let y = nn.forward(&img.to_owned());

        let mut max_index = 0;
        for i in 0..10 {
            if y[i] > y[max_index] {
                max_index = i;
            }
        }

        println!("ans:{}", trn_lbl[i]);
        println!("res:{}", max_index);
        println!("loss:{}", crosss_entropy_erro(&y, &trn_lbl_one_hot.index_axis(Axis(0), i).to_owned()));

    }

    // let plot_data = Vec::<(&str, f64)>::with_capacity(20);
    // for n in 0..20 {
    //     let y = numeric_diff(Box::new(f), n as f64);
    //     println!("x:{} y:{}", n, y);
    //     plot_data.push((&n.to_string(), y));
    // }
    // // prot(&plot_data);

    // for x1 in -10..11 {
    //     for x2 in -10..11 {
    //         let grad = numeric_gradient(Box::new(f2), &vec![x1 as f64 / 10.0, x2 as f64 / 10.0]);
    //         print!("{},{}  ",  format!("{:.*}", 2, grad[0] * 100000000.0), format!("{:.*}", 2, grad[1] * 100000000.0));
    //         // print!("{},{}  ",  grad[0], grad[1]);
    //     }
    //     println!();
    // }    
}

fn f2(x: &Vec<f64>) -> f64 {
    x[0].powi(2) + x[1].powi(2)
}