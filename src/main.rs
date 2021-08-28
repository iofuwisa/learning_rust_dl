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

    // Load MNIST
    let mnist_images = MnistImages::new(5000, 0, 0);
    let trn_img = mnist_images.get_trn_img();
    let trn_lbl = mnist_images.get_trn_lbl();
    let trn_lbl_one_hot = mnist_images.get_trn_lbl_one_hot();

    // Setup NN
    let nn = NeuralNetwork::new();

    // minibatch index
    let indexes = random_choice(MINI_BATCH_SIZE, TRN_IMG_SIZE);

    for i in indexes {
        let img = trn_img.row(i);
        let img = img.into_matrix();
        let img = img.data();
        let img = arr1(img);

        let y = nn.forward(&img);

        println!("index:{}\nans:{}\nresult:{:?}", i, trn_lbl[i], y);

    }


    // let y = arr1(&[0.1, 0.2, 0.0, 0.6, 0.1]);
    // let t = arr1(&[0.0, 0.0, 0.0, 1.0, 0.0]);
    // println!("loss: {}", sum_squared_error(&y, &t));
    // println!("loss: {}", crosss_entropy_erro(&y, &t));

}