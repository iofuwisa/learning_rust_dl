pub mod deep_learning;

// use crate::deep_learning::logical_operators::*;
use crate::deep_learning::activation_functions::*;
use crate::deep_learning::loss_functions::*;
// use crate::deep_learning::network::*;
use crate::deep_learning::mnist::*;
use crate::deep_learning::common::*;
use crate::deep_learning::graph_plotter::*;
use crate::deep_learning::network_learning::*;
use crate::deep_learning::softmax_with_loss::*;
use crate::deep_learning::affine_layer::*;
use crate::deep_learning::activation_layers::*;
use crate::deep_learning::neural_network::*;

use ndarray::prelude::{
    Array,
    Array1,
    Array2,
    arr1,
    arr2,
    s,
    Axis,
    ArrayView
};

use rand::Rng;
use std::f64::consts::E;
use std::time::{Duration, Instant};

const TRN_IMG_SIZE: usize = 5000;
const VAL_IMG_SIZE: usize = 1;
const TST_IMG_SIZE: usize = 2000;

// Hyper parameter
const ITERS_NUM: u32 = 1;
const MINI_BATCH_SIZE: usize = 20;
const HIDDEN_LAYER1_NURON_SIZE: u32 = 50;
const HIDDEN_LAYER2_NURON_SIZE: u32 = 20;
const LEARNING_RATE: f64 = 0.01;


fn main(){

    // Load MNIST
    let mnist = MnistImages::new(TRN_IMG_SIZE as u32, VAL_IMG_SIZE as u32, TST_IMG_SIZE as u32);
    let trn_img = mnist.get_trn_img();
    let trn_lbl = mnist.get_trn_lbl();
    let trn_lbl_one_hot = mnist.get_trn_lbl_one_hot();

    let tst_img = mnist.get_tst_img();
    let tst_lbl = mnist.get_tst_lbl();
    let tst_lbl_one_hot = mnist.get_tst_lbl_one_hot();



    let iter_num = ITERS_NUM;
    let minibatch_size = MINI_BATCH_SIZE;
    let trn_data = trn_img;
    let trn_lbl_one_hot = trn_lbl_one_hot;

    let mut rng = rand::thread_rng();

    let layers = NetworkBatchValueLayer::new(Array2::<f64>::zeros((minibatch_size, 28*28)));
    let layers = AffineLayer::new_random(layers, 28*28, 200);
    let layers = ReluLayer::new(layers);
    let layers = AffineLayer::new_random(layers, 200, 100);
    let layers = ReluLayer::new(layers);
    let layers = AffineLayer::new_random(layers, 100, 10);
    let layers = SoftmaxWithLoss::new(layers, Array2::<f64>::zeros((minibatch_size, 10)));
    let mut nn = NeuralNetwork::new(layers);


    for iteration in 0..iter_num {
        // Choise minibatch data
        let (batch_data, batch_lbl_onehot) = make_minibatch_data(minibatch_size, &trn_data, &trn_lbl_one_hot);
 
        // Set batch data
        nn.set_input(&batch_data);
        nn.set_lbl(&batch_lbl_onehot);

        println!("forward: {:?}", nn.forward());

    }

    // // Setup NN
    // let nn = SoftmaxWithLoss::new


    // // Setup NN
    // let mut nn = NeuralNetwork::new(
    //     784,
    //     vec![
    //         NeuralNetworkLayerBuilder::new(HIDDEN_LAYER1_NURON_SIZE, Box::new(&sigmoid_array)),   // hidden1
    //         NeuralNetworkLayerBuilder::new(HIDDEN_LAYER2_NURON_SIZE, Box::new(&sigmoid_array)),  // hidden2
    //         NeuralNetworkLayerBuilder::new(10, Box::new(&softmax_array)),  // output
    //     ]
    // );

    // network_learning(&mut nn, trn_img, trn_lbl_one_hot, tst_img, tst_lbl_one_hot, ITERS_NUM, LEARNING_RATE, MINI_BATCH_SIZE);
}

fn make_minibatch_data(minibatch_size: usize, data: &Array2<f64>, lbl_onehot: &Array2<f64>) -> (Array2<f64>, Array2<f64>) {
    let mut minibatch_data = Array2::<f64>::zeros((minibatch_size, data.shape()[1]));
    let mut minibatch_lbl_onehot = Array2::<f64>::zeros((minibatch_size, lbl_onehot.shape()[1]));

    let indexes = random_choice(minibatch_size, data.shape()[0]);

    for row_i in 0..minibatch_size {
        let batch_i = indexes[row_i];

        let data_row = data.index_axis(Axis(0), batch_i);
        let lbl_onehot_row = lbl_onehot.index_axis(Axis(0), batch_i);

        let mut batch_data_row = minibatch_data.index_axis_mut(Axis(0), row_i);
        let mut batch_lbl_onehot_row = minibatch_lbl_onehot.index_axis_mut(Axis(0), row_i);

        batch_data_row.assign(&data_row);
        batch_lbl_onehot_row.assign(&lbl_onehot_row);
    }

    return (minibatch_data, minibatch_lbl_onehot);
}

#[cfg(test)]
mod test_mod {
    use super::*;

    #[test]
    #[ignore]
    fn measure_time_to_execute_log() {
        let f1 = || {
            for _ in 0..1_000_000 {
                let mut rng = rand::thread_rng();
                rng.gen::<f64>().log(E);
            }
        };
        println!("{}", measure_execute_time(&f1));

        let f2 = || {
            for _ in 0..1_000_000 {
                let mut rng = rand::thread_rng();
                E.log(rng.gen::<f64>());
            }
        };
        println!("{}", measure_execute_time(&f1));
    }

    fn measure_execute_time(f: &dyn Fn()) -> u128 {
        let start = Instant::now();

        f();

        let end = start.elapsed();
        return end.as_millis();
    }

    #[test]
    fn test_make_minibatch_data() {
        // Load MNIST
        let mnist = MnistImages::new(TRN_IMG_SIZE as u32, VAL_IMG_SIZE as u32, TST_IMG_SIZE as u32);
        let trn_img = mnist.get_trn_img();
        let trn_lbl_one_hot = mnist.get_trn_lbl_one_hot();

        let minibach_size = 20;

        let (batch_data, batch_lbl_onehot) = make_minibatch_data(minibach_size, &trn_img, &trn_lbl_one_hot);

        assert_eq!(batch_data.shape(), [minibach_size, 28*28]);
        assert_eq!(batch_lbl_onehot.shape(), [minibach_size, 10]);

        for row_i in 0..minibach_size {
            let row_data = batch_data.index_axis(Axis(0), row_i).to_owned();
            let row_lbl_onehot = batch_lbl_onehot.index_axis(Axis(0), row_i).to_owned();

            // println!("digit: {}", max_index_in_arr1(&row_lbl_onehot));
            // print_img(&row_data);
            // println!("");
        }
    }
}