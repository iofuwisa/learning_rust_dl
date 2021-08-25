pub mod deep_learning;

// use crate::deep_learning::logical_operators::*;
use crate::deep_learning::activation_functions::*;
use crate::deep_learning::loss_functions::*;
use crate::deep_learning::network::*;
use crate::deep_learning::mnist::*;
use crate::deep_learning::common::*;

use ndarray::prelude::{
    Array1,
    arr1,
    arr2,
    s
};

const TRN_IMG_SIZE: u32 = 5000;
const VAL_IMG_SIZE: u32 = 0;
const TST_IMG_SIZE: u32 = 0;

const MINI_BATCH_SIZE: u32 = 100;

fn main(){

    // Load MNIST
    let mnist_images = MnistImages::new(5000, 0, 0);
    let trn_img = mnist_images.get_trn_img();

    // Setup NN
    let nn = NeuralNetwork::new();

    // minibatch index
    let indexes = random_choice(MINI_BATCH_SIZE, TRN_IMG_SIZE);

    // for i in indexes {
    //     let img = trn_img[i];
    //     let img = arr1(&img);

    //     let y = nn.forward(&img);

    //     println!("ans:{}, result:{}", mnist_images.getLabel(n), max(&y));

    // }


    // let y = arr1(&[0.1, 0.2, 0.0, 0.6, 0.1]);
    // let t = arr1(&[0.0, 0.0, 0.0, 1.0, 0.0]);
    // println!("loss: {}", sum_squared_error(&y, &t));
    // println!("loss: {}", crosss_entropy_erro(&y, &t));

}