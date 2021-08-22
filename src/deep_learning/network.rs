use ndarray::prelude::{
    Array,
    Array1,
    Array2,
    arr1,
    s,
};

use crate::deep_learning::activation_functions::{
    sigmoid_array,
    identity_array,
    softmax_array
};

use crate::deep_learning::mnist::*;

pub struct NeuralNetwork {
    input: Vec<f64>,
    hidden_layors: Vec<NeuralNetworkLayor>,
    output_layor: NeuralNetworkLayor,
}
impl NeuralNetwork {
    pub fn new(input: &Vec<f64>) -> NeuralNetwork {
        let input_num = input.len();
        let hidden1_num = 50;
        let hidden2_num = 100;
        let output_num = 10;

        let hidden_layors = vec![
            NeuralNetworkLayor::new(input_num as u32, hidden1_num, Box::new(sigmoid_array)),
            NeuralNetworkLayor::new(hidden1_num, hidden2_num, Box::new(sigmoid_array)),
        ];

        let output_layor = NeuralNetworkLayor::new(hidden2_num, output_num, Box::new(identity_array));

        return NeuralNetwork {
            input: input.clone(),
            hidden_layors: hidden_layors,
            output_layor: output_layor,
        };

    }
}


pub struct NeuralNetworkLayor {
    weight: Array2<f64>,
    bias: Array2<f64>,
    activation_function: Box<dyn Fn(&Array1<f64>) -> Array1<f64>>,
}
impl NeuralNetworkLayor {
    pub fn new(input_len: u32, neuron_len: u32, activation_function: Box<dyn Fn(&Array1<f64>) -> Array1<f64>>) -> NeuralNetworkLayor {
        let weight: Array2<f64> = Array2::zeros((input_len as usize, neuron_len as usize));
        let bias: Array2<f64> = Array2::zeros((input_len as usize, neuron_len as usize));

        return NeuralNetworkLayor {
            weight: weight,
            bias: bias,
            activation_function: activation_function,
        };
    }

    pub fn forward(&self, x: Array1<f64>) -> Array1<f64> {
        let a = x.dot(&self.weight) + &self.bias;
        let a = arr1(&(a.as_slice().unwrap()));

        let y = &(self.activation_function.as_ref())(&(a));
        return y.clone();
    }
    
}