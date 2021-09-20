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
use crate::deep_learning::neural_network_learning::*;

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
const ITERS_NUM: u32 = 100000;
const MINIBATCH_SIZE: usize = 200;
const LEARNING_RATE: f64 = 0.001;


fn main(){

    // Load MNIST
    let mnist = MnistImages::new(TRN_IMG_SIZE as u32, VAL_IMG_SIZE as u32, TST_IMG_SIZE as u32);
    let trn_img = mnist.get_trn_img();
    let trn_lbl = mnist.get_trn_lbl();
    let trn_lbl_onehot = mnist.get_trn_lbl_one_hot();

    let tst_img = mnist.get_tst_img();
    let tst_lbl = mnist.get_tst_lbl();
    let tst_lbl_onehot = mnist.get_tst_lbl_one_hot();

    // Create NN from layers stack
    // Include loss layer
    let layers = NetworkBatchValueLayer::new(Array2::<f64>::zeros((MINIBATCH_SIZE, 28*28)));
    let layers = AffineLayer::new_random(layers, 28*28, 200);
    let layers = ReluLayer::new(layers);
    let layers = AffineLayer::new_random(layers, 200, 10);
    // let layers = ReluLayer::new(layers);
    // let layers = AffineLayer::new_random(layers, 50, 10);
    let layers = SoftmaxWithLoss::new(layers, Array2::<f64>::zeros((MINIBATCH_SIZE, 10)));
    let mut nn = NeuralNetwork::new(layers);

    nn.learn(
        LearningParameter{
            learning_rate:  LEARNING_RATE,
            batch_size:     MINIBATCH_SIZE,
            iterations_num: ITERS_NUM,
        }, 
        LearningResource {
            trn_data:       trn_img.clone(),
            trn_lbl_onehot: trn_lbl_onehot.clone(),
            tst_data:       tst_img.clone(),
            tst_lbl_onehot: tst_lbl_onehot.clone(),
        }
    );
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
}