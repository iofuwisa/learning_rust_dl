use deep_learning::mnist_images::*;
use deep_learning::deep_learning::neural_network::*;
use deep_learning::deep_learning::layer::*;
use deep_learning::deep_learning::optimizer::*;

use ndarray::{
    Array2,
};

const TRN_IMG_SIZE: usize = 5000;
const VAL_IMG_SIZE: usize = 1;
const TST_IMG_SIZE: usize = 2000;

// Hyper parameter
const ITERS_NUM: u32 = 100;
const MINIBATCH_SIZE: usize = 100;
const CHANNEL_SIZE: usize = 1;
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
    guess_main();
}

fn lern_main() {
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
    let layers = DirectValue::new(Array2::<f64>::zeros((MINIBATCH_SIZE, 28*28)));

    let layers = Convolution::new_random(
        layers, 
        Adam::new(ADAM_LEARNING_RATE, ADAM_FLICTION_M, ADAM_FLICTION_V),
        Adam::new(ADAM_LEARNING_RATE, ADAM_FLICTION_M, ADAM_FLICTION_V),
        // Sgd::new(SGD_LEARNING_RATE),
        // Sgd::new(SGD_LEARNING_RATE),
        MINIBATCH_SIZE,
        CHANNEL_SIZE,
        5,  // filter_num
        4,  // filter_h
        4,  // filter_h
        28, // data_h
        28, // data_w
        2, // stride
        0, // padding
    );  // -> (MINIBATCH_SIZE, 5, 13, 13)
    let layers = Relu::new(layers);

    let layers = Pooling::new(
        layers,
        (MINIBATCH_SIZE, 5, 13, 13),
        3, // filter_h
        3, // filter_w
        3, // stride
        1, // padding
    ); // -> (MINIBATCH_SIZE, 5, 5, 5)

    // let layers = Pooling::new(
    //     layers,
    //     (MINIBATCH_SIZE, 1, 28, 28),
    //     2, // filter_h
    //     2, // filter_w
    //     2, // stride
    //     0, // padding
    // ); // -> (MINIBATCH_SIZE, 1, 14, 14)

    let layers = Affine::new_random_with_name(
        layers,
        // 1*28*28,
        // 1*14*14,
        // 5*13*13,
        5*5*5,
        10,
        Adam::new(ADAM_LEARNING_RATE, ADAM_FLICTION_M, ADAM_FLICTION_V),
        Adam::new(ADAM_LEARNING_RATE, ADAM_FLICTION_M, ADAM_FLICTION_V),
        // Sgd::new(SGD_LEARNING_RATE),
        // Sgd::new(SGD_LEARNING_RATE),
        "layer1".to_string(),
    );
    // let layers = BatchNorm::new(
    //     layers,
    //     NetworkBatchNormValueLayer::new(
    //         Array2::<f64>::ones((200, 200)),
    //         Adam::new(ADAM_LEARNING_RATE, ADAM_FLICTION_M, ADAM_FLICTION_V),
    //     ),
    //     NetworkBatchNormValueLayer::new(
    //         Array2::<f64>::zeros((200, 200)),
    //         Adam::new(ADAM_LEARNING_RATE, ADAM_FLICTION_M, ADAM_FLICTION_V),
    //     )
    // ); // -> (MINIBATCH_SIZE, 20)
    // let layers = Relu::new(layers);

    // let layers = Affine::new_random_with_name(
    //     layers,
    //     20,
    //     10,
    //     // Adam::new(ADAM_LEARNING_RATE, ADAM_FLICTION_M, ADAM_FLICTION_V),
    //     // Adam::new(ADAM_LEARNING_RATE, ADAM_FLICTION_M, ADAM_FLICTION_V),
    //     Sgd::new(SGD_LEARNING_RATE),
    //     Sgd::new(SGD_LEARNING_RATE),
    //     "layer2".to_string(),
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

    let res = nn.export();
    if let Err(e) = res {
        println!("{}", e);
    }
}

fn guess_main() {
    let mut nn =  match NeuralNetwork::import() {
        Ok(v) => v,
        Err(e) => panic!("{}", e.description()),
    };

    // Load MNIST
    let mnist = MnistImages::new(TRN_IMG_SIZE, VAL_IMG_SIZE, TST_IMG_SIZE);
    let tst_img = mnist.get_tst_img();
    let tst_lbl_onehot = mnist.get_tst_lbl_one_hot();

    nn.test(MINIBATCH_SIZE, &tst_img, &tst_lbl_onehot);
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