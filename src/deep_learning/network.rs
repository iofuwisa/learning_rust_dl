use ndarray::prelude::{
    Array,
    Array1,
    Array2,
    arr1,
};

use crate::deep_learning::activation_functions::{
    identity_array,
    sigmoid_array,
    softmax_array
};

use rand::Rng;

use crate::deep_learning::mnist::*;

use rand::prelude::*;

pub struct NeuralNetwork {
    layors: Vec<NeuralNetworkLayor>
}
impl NeuralNetwork {
    pub fn new(input_len: u32, layor_builders: Vec<NeuralNetworkLayorBuilder>) -> NeuralNetwork {

        let mut layors = Vec::<NeuralNetworkLayor>::with_capacity(layor_builders.len());
        let mut before_out_len = input_len;
        for mut builder in layor_builders {
            builder.set_input_len(before_out_len);
            before_out_len = builder.get_neuron_len();
            layors.push(builder.build());
        }

        return NeuralNetwork {
            layors: layors,
        };

    }

    pub fn forward(&self, input: &Array1<f64>) -> Array1<f64>{
        let mut x = input.clone();

        for layor in &self.layors {
            x = layor.forward(&x);
        }

        return x.clone();
    }
}

pub struct NeuralNetworkLayorBuilder {
    input_len: u32,
    neuron_len: u32,
    activation_function: Box<&'static (dyn Fn(&Array1<f64>) -> Array1<f64>)>,
}
impl NeuralNetworkLayorBuilder {
    pub fn new(neuron_len: u32, activation_function: Box<&'static (dyn Fn(&Array1<f64>) -> Array1<f64>)>) -> NeuralNetworkLayorBuilder {
        NeuralNetworkLayorBuilder {
            input_len: 0,
            neuron_len: neuron_len,
            activation_function: activation_function,
        }
    }
    pub fn set_input_len(&mut self, input_len: u32) {
        self.input_len = input_len;
    }
    pub fn set_neuron_len(&mut self, neuron_len: u32) {
        self.neuron_len = neuron_len;
    }
    pub fn get_neuron_len(&self) -> u32 {
        self.neuron_len
    }
    pub fn set_activation_function(&mut self, activation_function: Box<&'static (dyn Fn(&Array1<f64>) -> Array1<f64>)>) {
        self.activation_function = activation_function;
    }
    pub fn build(self) -> NeuralNetworkLayor {
        NeuralNetworkLayor::new(self.input_len, self.neuron_len, Box::new(*self.activation_function))
    }
}

pub struct NeuralNetworkLayor {
    weight: Array2<f64>,
    bias: Array1<f64>,
    activation_function: Box<&'static (dyn Fn(&Array1<f64>) -> Array1<f64>)>,
}
impl NeuralNetworkLayor {
    pub fn new(input_len: u32, neuron_len: u32, activation_function: Box<&'static (dyn Fn(&Array1<f64>) -> Array1<f64>)>) -> NeuralNetworkLayor {
        let mut rng = rand::thread_rng();
        let weight:Array2<f64> = Array::from_shape_fn((input_len as usize, neuron_len as usize), |(_, _)| rng.gen::<f64>());
        let bias:Array1<f64> = Array::from_shape_fn(neuron_len as usize, |_| rng.gen::<f64>());

        return NeuralNetworkLayor {
            weight: weight,
            bias: bias,
            activation_function: activation_function,
        };
    }

    pub fn forward(&self, x: &Array1<f64>) -> Array1<f64> {
        let a = x.dot(&self.weight) + &self.bias;
        let a = arr1(&(a.as_slice().unwrap()));

        let y = &(self.activation_function.as_ref())(&(a));
        return y.clone();
    }
    
}