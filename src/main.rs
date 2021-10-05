extern crate deep_learning;

// use deep_learning::logical_operators::*;
// use deep_learning::activation_functions::*;
// use deep_learning::loss_functions::*;
// use deep_learning::network::*;
use deep_learning::deep_learning::mnist::*;
// use deep_learning::deep_learning::common::*;
// use deep_learning::deep_learning::graph_plotter::*;
// use deep_learning::deep_learning::network_learning::*;
use deep_learning::deep_learning::softmax_with_loss::*;
use deep_learning::deep_learning::affine_layer::*;
use deep_learning::deep_learning::activation_layers::*;
use deep_learning::deep_learning::neural_network::*;
// use crate::deep_learning::neural_network_learning::*;
use deep_learning::deep_learning::optimizer::*;
use deep_learning::deep_learning::batch_norm::*;
use deep_learning::deep_learning::dropout_layer::*;

use deep_learning::cnn::*;

use ndarray::prelude::{
    Array2,
};

const TRN_IMG_SIZE: usize = 5000;
const VAL_IMG_SIZE: usize = 1;
const TST_IMG_SIZE: usize = 2000;

// Hyper parameter
const ITERS_NUM: u32 = 1000;
const MINIBATCH_SIZE: usize = 100;
const DROUPOUT_RATE: f64 = 0.15;
const SGD_LEARNING_RATE: f64 = 0.001;
const MOMENTUM_LEARNING_RATE: f64 = 0.1;
const MOMENTUM_FLICTION: f64 = 0.9;
const RMSPROP_LEARNING_RATE: f64 = 0.1;
const RMSPROP_FLICTION: f64 = 0.99;
const ADAGRAD_LEARNING_RATE: f64 = 0.1;
const ADAM_LEARNING_RATE: f64 = 0.1;
const ADAM_FLICTION_M: f64 = 0.9;
const ADAM_FLICTION_V: f64 = 0.999;


fn main(){
    // switch_main();
    // switch_overfitting();
    println!("{}", public());
}

fn switch_main() {
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
    let layers = AffineLayer::new_random_with_name(
        layers,
        28*28,
        200,
        // Sgd::new(SGD_LEARNING_RATE),
        // Sgd::new(SGD_LEARNING_RATE)
        // Momentum::new(MOMENTUM_LEARNING_RATE, MOMENTUM_FLICTION),
        // Momentum::new(MOMENTUM_LEARNING_RATE, MOMENTUM_FLICTION)
        // Rmsprop::new(RMSPROP_LEARNING_RATE, RMSPROP_FLICTION),
        // Rmsprop::new(RMSPROP_LEARNING_RATE, RMSPROP_FLICTION)
        // AdaGrad::new(ADAGRAD_LEARNING_RATE),
        // AdaGrad::new(ADAGRAD_LEARNING_RATE)
        Adam::new(ADAM_LEARNING_RATE, ADAM_FLICTION_M, ADAM_FLICTION_V),
        Adam::new(ADAM_LEARNING_RATE, ADAM_FLICTION_M, ADAM_FLICTION_V),
        "layer1".to_string(),
    );
    let layers = BatchNorm::new(
        layers,
        NetworkBatchNormValueLayer::new(
            Array2::<f64>::ones((200, 200)),
            Adam::new(ADAM_LEARNING_RATE, ADAM_FLICTION_M, ADAM_FLICTION_V),
        ),
        NetworkBatchNormValueLayer::new(
            Array2::<f64>::zeros((200, 200)),
            Adam::new(ADAM_LEARNING_RATE, ADAM_FLICTION_M, ADAM_FLICTION_V),
        )
    );
    let layers = ReluLayer::new(layers);
    let layers = AffineLayer::new_random_with_name(
        layers,
        200,
        10,
        // Sgd::new(SGD_LEARNING_RATE),
        // Sgd::new(MOMENTUM_LEARNING_RATE)
        // Momentum::new(MOMENTUM_LEARNING_RATE, MOMENTUM_FLICTION),
        // Momentum::new(MOMENTUM_LEARNING_RATE, MOMENTUM_FLICTION)
        // Rmsprop::new(RMSPROP_LEARNING_RATE, RMSPROP_FLICTION),
        // Rmsprop::new(RMSPROP_LEARNING_RATE, RMSPROP_FLICTION)
        // AdaGrad::new(ADAGRAD_LEARNING_RATE),
        // AdaGrad::new(ADAGRAD_LEARNING_RATE)
        Adam::new(ADAM_LEARNING_RATE, ADAM_FLICTION_M, ADAM_FLICTION_V),
        Adam::new(ADAM_LEARNING_RATE, ADAM_FLICTION_M, ADAM_FLICTION_V),
        "layer2".to_string(),
    );
    // let layers = BatchNorm::new(
    //     layers,
    //     NetworkBatchNormValueLayer::new(
    //         Array2::<f64>::ones((200, 10)),
    //         Adam::new(ADAM_LEARNING_RATE, ADAM_FLICTION_M, ADAM_FLICTION_V),
    //     ),
    //     NetworkBatchNormValueLayer::new(
    //         Array2::<f64>::zeros((20, 10)),
    //         Adam::new(ADAM_LEARNING_RATE, ADAM_FLICTION_M, ADAM_FLICTION_V),
    //     )
    // );
    // let layers = ReluLayer::new(layers);
    // let layers = AffineLayer::new_random(
    //     layers,
    //     50,
    //     10,
    //     // Sgd::new(MOMENTUM_LEARNING_RATE),
    //     // Sgd::new(MOMENTUM_LEARNING_RATE)
    //     // Momentum::new(MOMENTUM_LEARNING_RATE, MOMENTUM_FLICTION),
    //     // Momentum::new(MOMENTUM_LEARNING_RATE, MOMENTUM_FLICTION)
    //     // Rmsprop::new(RMSPROP_LEARNING_RATE, RMSPROP_FLICTION),
    //     // Rmsprop::new(RMSPROP_LEARNING_RATE, RMSPROP_FLICTION)
    //     // AdaGrad::new(ADAGRAD_LEARNING_RATE),
    //     // AdaGrad::new(ADAGRAD_LEARNING_RATE)
    //     Adam::new(ADAM_LEARNING_RATE, ADAM_FLICTION_M, ADAM_FLICTION_V),
    //     Adam::new(ADAM_LEARNING_RATE, ADAM_FLICTION_M, ADAM_FLICTION_V),
    //     "layer3".to_string(),
    // );
    let layers = SoftmaxWithLoss::new(layers, Array2::<f64>::zeros((MINIBATCH_SIZE, 10)));
    let mut nn = NeuralNetwork::new(layers);

    nn.learn(
        LearningParameter{
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

fn switch_overfitting() {
    // load MNIST
    let mnist = MnistImages::new(TRN_IMG_SIZE, VAL_IMG_SIZE, TST_IMG_SIZE);
    let trn_img = mnist.get_trn_img();
    let trn_lbl = mnist.get_trn_lbl();
    let trn_lbl_onehot = mnist.get_trn_lbl_one_hot();

    let tst_img = mnist.get_tst_img();
    let tst_lbl = mnist.get_tst_lbl();
    let tst_lbl_onehot = mnist.get_tst_lbl_one_hot();

    let (trn_img, trn_lbl_onehot) = make_minibatch_data(300, &trn_img, &trn_lbl_onehot);

    // Create NN from layers stack
    // Include loss layer
    let layers = NetworkBatchValueLayer::new(Array2::<f64>::zeros((MINIBATCH_SIZE, 28*28)));

    // Layer1
    let layers = AffineLayer::new_random_with_name(
        layers,
        28*28,
        100,
        Adam::new(ADAM_LEARNING_RATE, ADAM_FLICTION_M, ADAM_FLICTION_V),
        Adam::new(ADAM_LEARNING_RATE/100f64, ADAM_FLICTION_M, ADAM_FLICTION_V),
        "layer1".to_string(),
    );
    let layers = BatchNorm::new(
        layers,
        NetworkBatchNormValueLayer::new(
            Array2::<f64>::ones((MINIBATCH_SIZE, 100)),
            Adam::new(ADAM_LEARNING_RATE, ADAM_FLICTION_M, ADAM_FLICTION_V),
        ),
        NetworkBatchNormValueLayer::new(
            Array2::<f64>::zeros((MINIBATCH_SIZE, 100)),
            Adam::new(ADAM_LEARNING_RATE, ADAM_FLICTION_M, ADAM_FLICTION_V),
        )
    );
    let layers = DropoutLayer::new(layers, DROUPOUT_RATE);
    let layers = ReluLayer::new(layers);

    // Layer2
    let layers = AffineLayer::new_random_with_name(
        layers,
        100,
        100,
        Adam::new(ADAM_LEARNING_RATE, ADAM_FLICTION_M, ADAM_FLICTION_V),
        Adam::new(ADAM_LEARNING_RATE/100f64, ADAM_FLICTION_M, ADAM_FLICTION_V),
        "layer2".to_string(),
    );
    let layers = BatchNorm::new(
        layers,
        NetworkBatchNormValueLayer::new(
            Array2::<f64>::ones((MINIBATCH_SIZE, 100)),
            Adam::new(ADAM_LEARNING_RATE, ADAM_FLICTION_M, ADAM_FLICTION_V),
        ),
        NetworkBatchNormValueLayer::new(
            Array2::<f64>::zeros((MINIBATCH_SIZE, 100)),
            Adam::new(ADAM_LEARNING_RATE, ADAM_FLICTION_M, ADAM_FLICTION_V),
        )
    );
    let layers = ReluLayer::new(layers);
    let layers = DropoutLayer::new(layers, DROUPOUT_RATE);

    // Layer3
    let layers = AffineLayer::new_random_with_name(
        layers,
        100,
        100,
        Adam::new(ADAM_LEARNING_RATE, ADAM_FLICTION_M, ADAM_FLICTION_V),
        Adam::new(ADAM_LEARNING_RATE/100f64, ADAM_FLICTION_M, ADAM_FLICTION_V),
        "layer3".to_string(),
    );
    let layers = BatchNorm::new(
        layers,
        NetworkBatchNormValueLayer::new(
            Array2::<f64>::ones((MINIBATCH_SIZE, 100)),
            Adam::new(ADAM_LEARNING_RATE, ADAM_FLICTION_M, ADAM_FLICTION_V),
        ),
        NetworkBatchNormValueLayer::new(
            Array2::<f64>::zeros((MINIBATCH_SIZE, 100)),
            Adam::new(ADAM_LEARNING_RATE/100f64, ADAM_FLICTION_M, ADAM_FLICTION_V),
        )
    );
    let layers = ReluLayer::new(layers);
    let layers = DropoutLayer::new(layers, DROUPOUT_RATE);

    // Layer4
    let layers = AffineLayer::new_random_with_name(
        layers,
        100,
        100,
        Adam::new(ADAM_LEARNING_RATE, ADAM_FLICTION_M, ADAM_FLICTION_V),
        Adam::new(ADAM_LEARNING_RATE/100f64, ADAM_FLICTION_M, ADAM_FLICTION_V),
        "layer4".to_string(),
    );
    let layers = BatchNorm::new(
        layers,
        NetworkBatchNormValueLayer::new(
            Array2::<f64>::ones((MINIBATCH_SIZE, 100)),
            Adam::new(ADAM_LEARNING_RATE, ADAM_FLICTION_M, ADAM_FLICTION_V),
        ),
        NetworkBatchNormValueLayer::new(
            Array2::<f64>::zeros((MINIBATCH_SIZE, 100)),
            Adam::new(ADAM_LEARNING_RATE, ADAM_FLICTION_M, ADAM_FLICTION_V),
        )
    );
    let layers = ReluLayer::new(layers);
    let layers = DropoutLayer::new(layers, DROUPOUT_RATE);

    // // Layer5
    // let layers = AffineLayer::new_random_with_name(
    //     layers,
    //     100,
    //     100,
    //     Adam::new(ADAM_LEARNING_RATE, ADAM_FLICTION_M, ADAM_FLICTION_V),
    //     Adam::new(ADAM_LEARNING_RATE, ADAM_FLICTION_M, ADAM_FLICTION_V),
    //     "layer5".to_string(),
    // );
    // let layers = BatchNorm::new(
    //     layers,
    //     NetworkBatchNormValueLayer::new(
    //         Array2::<f64>::ones((MINIBATCH_SIZE, 100)),
    //         Adam::new(ADAM_LEARNING_RATE, ADAM_FLICTION_M, ADAM_FLICTION_V),
    //     ),
    //     NetworkBatchNormValueLayer::new(
    //         Array2::<f64>::zeros((MINIBATCH_SIZE, 100)),
    //         Adam::new(ADAM_LEARNING_RATE, ADAM_FLICTION_M, ADAM_FLICTION_V),
    //     )
    // );
    // let layers = ReluLayer::new(layers);

    // // Layer6
    // let layers = AffineLayer::new_random_with_name(
    //     layers,
    //     100,
    //     100,
    //     Adam::new(ADAM_LEARNING_RATE, ADAM_FLICTION_M, ADAM_FLICTION_V),
    //     Adam::new(ADAM_LEARNING_RATE, ADAM_FLICTION_M, ADAM_FLICTION_V),
    //     "layer6".to_string(),
    // );
    // let layers = BatchNorm::new(
    //     layers,
    //     NetworkBatchNormValueLayer::new(
    //         Array2::<f64>::ones((MINIBATCH_SIZE, 100)),
    //         Adam::new(ADAM_LEARNING_RATE, ADAM_FLICTION_M, ADAM_FLICTION_V),
    //     ),
    //     NetworkBatchNormValueLayer::new(
    //         Array2::<f64>::zeros((200, 100)),
    //         Adam::new(ADAM_LEARNING_RATE, ADAM_FLICTION_M, ADAM_FLICTION_V),
    //     )
    // );
    // let layers = ReluLayer::new(layers);

    // Layer7
    let layers = AffineLayer::new_random_with_name(
        layers,
        100,
        10,
        Adam::new(ADAM_LEARNING_RATE, ADAM_FLICTION_M, ADAM_FLICTION_V),
        Adam::new(ADAM_LEARNING_RATE/100f64, ADAM_FLICTION_M, ADAM_FLICTION_V),
        "layer7".to_string(),
    );
    let layers = ReluLayer::new(layers);
    let layers = DropoutLayer::new(layers, DROUPOUT_RATE);

    let layers = SoftmaxWithLoss::new(layers, Array2::<f64>::zeros((MINIBATCH_SIZE, 10)));

    let mut nn = NeuralNetwork::new(layers);

    nn.learn(
        LearningParameter{
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