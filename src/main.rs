pub mod deep_learning;

// use crate::deep_learning::logical_operators::*;
use crate::deep_learning::activation_functions::*;
use crate::deep_learning::network::*;
use crate::deep_learning::mnist::*;

use ndarray::prelude::{
    Array1,
    arr1,
    arr2,
    s
};

fn main(){

    // Load MNIST
    let mnist_images = MnistImages::new();

    // Setup NN
    let nn = NeuralNetwork::new();

    for n in 0..10 {
        let img = mnist_images.getImgVec(n);
        let img = arr1(&img);

        let y = nn.forward(&img);

        println!("ans:{}, result:{}", mnist_images.getLabel(n), max(&y));

    }
}