pub mod deep_learning;

// use crate::deep_learning::logical_operators::*;
// use crate::deep_learning::activation_functions::*;
// use crate::deep_learning::loss_functions::*;
// use crate::deep_learning::network::*;
use crate::deep_learning::mnist::*;
// use crate::deep_learning::common::*;
// use crate::deep_learning::graph_plotter::*;
// use crate::deep_learning::network_learning::*;
use crate::deep_learning::softmax_with_loss::*;
use crate::deep_learning::affine_layer::*;
use crate::deep_learning::activation_layers::*;
use crate::deep_learning::neural_network::*;
// use crate::deep_learning::neural_network_learning::*;
use crate::deep_learning::optimizer::*;

use ndarray::prelude::{
    Array2,
};

const TRN_IMG_SIZE: usize = 5000;
const VAL_IMG_SIZE: usize = 1;
const TST_IMG_SIZE: usize = 2000;

// Hyper parameter
const ITERS_NUM: u32 = 1000000;
const MINIBATCH_SIZE: usize = 200;
const LEARNING_RATE: f64 = 0.1;
const MOMENTUM_FLICTION: f64 = 0.9;
const RMSPROP_FLICTION: f64 = 0.99;
const ADAM_FLICTION_M: f64 = 0.9;
const ADAM_FLICTION_V: f64 = 0.999;


fn main(){

    // Load MNIST
    let mnist = MnistImages::new(TRN_IMG_SIZE, VAL_IMG_SIZE, TST_IMG_SIZE);
    let trn_img = mnist.get_trn_img();
    let trn_lbl = mnist.get_trn_lbl();
    let trn_lbl_onehot = mnist.get_trn_lbl_one_hot();

    let tst_img = mnist.get_tst_img();
    let tst_lbl = mnist.get_tst_lbl();
    let tst_lbl_onehot = mnist.get_tst_lbl_one_hot();

    // Create NN from layers stack
    // Include loss layer
    let layers = NetworkBatchValueLayer::new(Array2::<f64>::zeros((MINIBATCH_SIZE, 28*28)));
    let layers = AffineLayer::new_random(
        layers,
        28*28,
        200,
        // Sgd::new(LEARNING_RATE),
        // Sgd::new(LEARNING_RATE)
        // Momentum::new(LEARNING_RATE, MOMENTUM_FLICTION),
        // Momentum::new(LEARNING_RATE, MOMENTUM_FLICTION)
        // Rmsprop::new(LEARNING_RATE, RMSPROP_FLICTION),
        // Rmsprop::new(LEARNING_RATE, RMSPROP_FLICTION)
        // AdaGrad::new(LEARNING_RATE),
        // AdaGrad::new(LEARNING_RATE)
        Adam::new(LEARNING_RATE, ADAM_FLICTION_M, ADAM_FLICTION_V),
        Adam::new(LEARNING_RATE, ADAM_FLICTION_M, ADAM_FLICTION_V)
    );
    let layers = ReluLayer::new(layers);
    let layers = AffineLayer::new_random(
        layers,
        200,
        10,
        // Sgd::new(LEARNING_RATE),
        // Sgd::new(LEARNING_RATE)
        // Momentum::new(LEARNING_RATE, MOMENTUM_FLICTION),
        // Momentum::new(LEARNING_RATE, MOMENTUM_FLICTION)
        // Rmsprop::new(LEARNING_RATE, RMSPROP_FLICTION),
        // Rmsprop::new(LEARNING_RATE, RMSPROP_FLICTION)
        // AdaGrad::new(LEARNING_RATE),
        // AdaGrad::new(LEARNING_RATE)
        Adam::new(LEARNING_RATE, ADAM_FLICTION_M, ADAM_FLICTION_V),
        Adam::new(LEARNING_RATE, ADAM_FLICTION_M, ADAM_FLICTION_V)
    );
    // let layers = ReluLayer::new(layers);
    // let layers = AffineLayer::new_random(
    //     layers,
    //     50,
    //     10,
    //     // Sgd::new(LEARNING_RATE),
    //     // Sgd::new(LEARNING_RATE)
    //     // Momentum::new(LEARNING_RATE, MOMENTUM_FLICTION),
    //     // Momentum::new(LEARNING_RATE, MOMENTUM_FLICTION)
    //     // Rmsprop::new(LEARNING_RATE, RMSPROP_FLICTION),
    //     // Rmsprop::new(LEARNING_RATE, RMSPROP_FLICTION)
    //     // AdaGrad::new(LEARNING_RATE),
    //     // AdaGrad::new(LEARNING_RATE)
    //     Adam::new(LEARNING_RATE, ADAM_FLICTION_M, ADAM_FLICTION_V),
    //     Adam::new(LEARNING_RATE, ADAM_FLICTION_M, ADAM_FLICTION_V)
    // );
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

    use std::time::Instant;

    fn measure_execute_time(f: &dyn Fn()) -> u128 {
        let start = Instant::now();

        f();

        let end = start.elapsed();
        return end.as_millis();
    }
}