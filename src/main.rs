pub mod deep_learning;

// use crate::deep_learning::logical_operators::*;
use crate::deep_learning::activation_functions::*;
use crate::deep_learning::loss_functions::*;
use crate::deep_learning::network::*;
use crate::deep_learning::mnist::*;
use crate::deep_learning::common::*;
use crate::deep_learning::graph_plotter::*;
use crate::deep_learning::network_learning::*;

use ndarray::prelude::{
    Array1,
    Array2,
    arr1,
    arr2,
    s,
    Axis,
    ArrayView
};

use std::time::{Duration, Instant};

const TRN_IMG_SIZE: usize = 5000;
const VAL_IMG_SIZE: usize = 1;
const TST_IMG_SIZE: usize = 2000;

// Hyper parameter
const ITERS_NUM: u32 = 10000;
const MINI_BATCH_SIZE: usize = 10;
const HIDDEN_LAYOR1_NURON_SIZE: u32 = 5;
// const HIDDEN_LAYOR2_NURON_SIZE: u32 = 20;
const LEARNING_RATE: f64 = 0.1;


fn main(){

    // Load MNIST
    let mnist_images = MnistImages::new(TRN_IMG_SIZE as u32, VAL_IMG_SIZE as u32, TST_IMG_SIZE as u32);
    let trn_img = mnist_images.get_trn_img();
    let trn_lbl = mnist_images.get_trn_lbl();
    let trn_lbl_one_hot = mnist_images.get_trn_lbl_one_hot();

    let tst_img = mnist_images.get_tst_img();
    let tst_lbl = mnist_images.get_tst_lbl();
    let tst_lbl_one_hot = mnist_images.get_tst_lbl_one_hot();

    // Setup NN
    let mut nn = NeuralNetwork::new(
        784,
        vec![
            NeuralNetworkLayorBuilder::new(HIDDEN_LAYOR1_NURON_SIZE, Box::new(&sigmoid_array)),   // hidden1
            // NeuralNetworkLayorBuilder::new(HIDDEN_LAYOR2_NURON_SIZE, Box::new(&sigmoid_array)),  // hidden2
            NeuralNetworkLayorBuilder::new(10, Box::new(&softmax_array)),  // output
        ]
    );

    network_learning(&mut nn, trn_img, trn_lbl_one_hot, tst_img, tst_lbl_one_hot, ITERS_NUM, LEARNING_RATE, MINI_BATCH_SIZE);
}