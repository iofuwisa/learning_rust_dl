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

    // Test
    let test_sampl_size = 1000;
    let indexes = random_choice(test_sampl_size, TST_IMG_SIZE);
    let mut loss: f64 = 0.0;
    let mut correct_count = 0;
    for i in &indexes {
        // Guess
        let img = trn_img.index_axis(Axis(0), *i).to_owned();
        let y = nn.forward(&img);
        // Loss
        loss += crosss_entropy_erro(&y, &tst_lbl_one_hot.index_axis(Axis(0), *i).to_owned());
        // Correct answer rate
        let mut max_index: u8 = 0;
        for j in 0..10 {
                if y[j] > y[max_index as usize] {
                    max_index = j as u8;
                }
        }
        correct_count += if max_index==tst_lbl[*i] {1} else {0};
    }
    loss = loss / test_sampl_size as f64;
    println!("Start statis");
    println!("Loss: {}", loss);
    println!("CorrectRate: {}%", correct_count as f64 / test_sampl_size  as f64 * 100.0);
    println!("");

    for iter in 0..ITERS_NUM {

        // Minibatch index
        let indexes = random_choice(MINI_BATCH_SIZE, TRN_IMG_SIZE);

        // Forwading
        let f = |x: &Array1<f64>| -> f64 {
            // Parse weight and bias
            let nn_size = nn.get_network_size();
            let mut weight_h = Vec::<Array2<f64>>::new();
            let mut bias_h = Vec::<Array1<f64>>::new();
            let mut base_index = 0;
            for ((weight_row_size, weight_col_size), bias_size) in nn_size {
                let w = x.slice(s![base_index..(base_index+weight_row_size*weight_col_size)]);
                base_index += weight_row_size*weight_col_size;
                let w = w.into_shape((weight_row_size, weight_col_size)).unwrap();
                let b = x.slice(s![base_index..base_index+bias_size]);
                base_index += bias_size;
                weight_h.push(w.to_owned());
                bias_h.push(b.to_owned());
            }
            let mut loss = 0.0;
            for i in &indexes {
                let img = trn_img.index_axis(Axis(0), *i);
    
                // Forward
                let y = nn.forward_diff(&img.to_owned(), &weight_h, &bias_h);
    
                // Evaluation
                loss += crosss_entropy_erro(&y, &trn_lbl_one_hot.index_axis(Axis(0), *i).to_owned());
    
                // // Print
                // for i in 0..28 {
                //     for j in 0..28 {
                //         if img[(i*28+j)] > 0.0 {
                //             print!("*");
                //         } else {
                //             print!(" ");
                //         }
                //     }
                //     println!("");
                // }
                // let mut max_index = 0;
                // for i in 0..10 {
                //     if y[i] > y[max_index] {
                //         max_index = i;
                //     }
                // }
                // println!("ans:{}", trn_lbl[i]);
                // println!("res:{}", max_index);
                // println!("loss:{}", crosss_entropy_erro(&y, &trn_lbl_one_hot.index_axis(Axis(0), i).to_owned()));
    
            }
            return loss;
        };

        let nn_size = nn.get_network_size();
        let mut parameter_size = 0;
        for ((weight_row_size, weight_col_size), bias_size) in nn_size {
            parameter_size += weight_row_size * weight_col_size + bias_size;
        }
        let x: Array1<f64> = Array1::<f64>::zeros(parameter_size);
        let grad = numeric_gradient(f, &x);
        
        // Parse weight and bias
        let nn_size = nn.get_network_size();
        let mut weight_grad = Vec::<Array2<f64>>::new();
        let mut bias_grad = Vec::<Array1<f64>>::new();
        let mut update_weight_value = Vec::<Array2<f64>>::new();
        let mut update_bias_value = Vec::<Array1<f64>>::new();
        let mut base_index = 0;
        for ((weight_row_size, weight_col_size), bias_size) in nn_size {
            let w = grad.slice(s![base_index..(base_index+weight_row_size*weight_col_size)]);
            base_index += weight_row_size*weight_col_size;
            let w = w.into_shape((weight_row_size, weight_col_size)).unwrap();
            let b = grad.slice(s![base_index..base_index+bias_size]);
            base_index += bias_size;
            weight_grad.push(w.to_owned());
            bias_grad.push(b.to_owned());
            update_weight_value.push(w.mapv(|w:f64| -> f64 {w*-1.0*LEARNING_RATE}));
            update_bias_value.push(b.mapv(|b:f64| -> f64 {b*-1.0*LEARNING_RATE}));
        }
        nn.update_parameters_add(&update_weight_value, &update_bias_value);

        // Test
        let test_sampl_size = 1000;
        let indexes = random_choice(test_sampl_size, TST_IMG_SIZE);
        let mut loss: f64 = 0.0;
        let mut correct_count = 0;
        for i in &indexes {
            // Guess
            let img = trn_img.index_axis(Axis(0), *i).to_owned();
            let y = nn.forward(&img);
            // Loss
            loss += crosss_entropy_erro(&y, &tst_lbl_one_hot.index_axis(Axis(0), *i).to_owned());
            // Correct answer rate
            let mut max_index: u8 = 0;
            for j in 0..10 {
                    if y[j] > y[max_index as usize] {
                        max_index = j as u8;
                    }
            }
            correct_count += if max_index==tst_lbl[*i] {1} else {0};
        }
        loss = loss / test_sampl_size as f64;
        println!("Complete iter {}", iter + 1);
        println!("Loss: {}", loss);
        println!("CorrectRate: {}%", correct_count as f64 / test_sampl_size  as f64 * 100.0);
        println!("");
    }
}