use ndarray::prelude::{
    Array,
    Array1,
    Array2,
    arr1,
    arr2,
    Dim,
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

    pub fn forward_diff(&self, input: &Array1<f64>, wh: &Vec<Array2<f64>>, bh: &Vec<Array1<f64>>) -> Array1<f64>{
        let mut x = input.clone();

        for i in 0..self.layors.len() {
            x = self.layors[i].forward_diff(&x, &wh[i], &bh[i]);
        }

        return x.clone();
    }

    pub fn get_network_size(&self) -> Vec<((usize, usize), usize)>{
        let mut size = Vec::<((usize, usize), usize)>::with_capacity(self.layors.len());

        for layor in &self.layors {
            size.push(layor.get_layor_size());
        }

        return size;
    }

    pub fn update_parameters_add(&mut self, weight: &Vec<Array2<f64>>, bias: &Vec<Array1<f64>>) {
        for i in 0..self.layors.len() {
            self.layors[i].update_parameters_add(&weight[i], &bias[i]);
        }
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

        let y = &(self.activation_function.as_ref())(&(a));
        return y.clone();
    }

    pub fn forward_diff(&self, x: &Array1<f64>, wh: &Array2<f64>, bh: &Array1<f64>) -> Array1<f64> {
        let a = x.dot(&(&self.weight + wh)) + &self.bias + bh;

        let y = &(self.activation_function.as_ref())(&(a));
        return y.clone();
    }

    pub fn get_layor_size(&self) -> ((usize, usize), usize) {
        (self.weight.dim(), self.bias.len())
    }
    
    pub fn update_parameters_add(&mut self, weight: &Array2<f64>, bias: &Array1<f64>) {
        self.weight = &self.weight + weight;
        self.bias = &self.bias + bias;
    }   

}

#[cfg(test)]
mod NeuralNetwork_test {
    use super::*;

    #[test]
    fn NeuralNetwork_new() {
        let a = NeuralNetwork::new(10, vec![
            NeuralNetworkLayorBuilder::new(80, Box::new(&activation_stub)),
            NeuralNetworkLayorBuilder::new(40, Box::new(&activation_stub)),
            NeuralNetworkLayorBuilder::new(20, Box::new(&activation_stub)),
        ]);

        assert_eq!(a.layors[0].weight.raw_dim(), Dim([10, 80]));
        assert_eq!(a.layors[0].bias.raw_dim(), Dim([80]));
        assert_eq!(&(a.layors[0].activation_function.as_ref())(&arr1(&[1.0;10])), arr1(&[1.0;10]));

        assert_eq!(a.layors[1].weight.raw_dim(), Dim([80, 40]));
        assert_eq!(a.layors[1].bias.raw_dim(), Dim([40]));
        assert_eq!(&(a.layors[1].activation_function.as_ref())(&arr1(&[1.0;10])), arr1(&[1.0;10]));

        assert_eq!(a.layors[2].weight.raw_dim(), Dim([40, 20]));
        assert_eq!(a.layors[2].bias.raw_dim(), Dim([20]));
        assert_eq!(&(a.layors[2].activation_function.as_ref())(&arr1(&[1.0;10])), arr1(&[1.0;10]));
    }

    #[test]
    fn NeuralNetwork_forward() {
        let a = NeuralNetwork {
            layors: vec![
                NeuralNetworkLayor {
                    weight: arr2(&[[1.0, 2.0, 3.0],[4.0, 5.0, 6.0]]),
                    bias: arr1(&[2.0, 2.0, 2.0]),
                    activation_function: Box::new(&activation_stub),
                },
                NeuralNetworkLayor {
                    weight: arr2(&[[1.0, 2.0],[3.0, 4.0], [5.0, 6.0]]),
                    bias: arr1(&[1.0, 1.0]),
                    activation_function: Box::new(&activation_stub),
                },
                NeuralNetworkLayor {
                    weight: arr2(&[[1.0, 3.0, 5.0],[2.0, 4.0, 6.0]]),
                    bias: arr1(&[1.0, 1.0, 1.0]),
                    activation_function: Box::new(&activation_stub),
                }
            ]
        };
        let a = a.forward(&arr1(&[1.0, 2.0]));

        assert_eq!(a, arr1(&[502.0, 1142.0, 1782.0]));
    }

    #[test]
    fn NeuralNetwork_forward_diff() {
        let a = NeuralNetwork {
            layors: vec![
                NeuralNetworkLayor {
                    weight: arr2(&[[0.0, 1.0, 2.0],[3.0, 4.0, 5.0]]),
                    bias: arr1(&[0.0, 0.0, 0.0]),
                    activation_function: Box::new(&activation_stub),
                },
                NeuralNetworkLayor {
                    weight: arr2(&[[0.0, 1.0],[2.0, 3.0], [4.0, 5.0]]),
                    bias: arr1(&[-1.0, -1.0]),
                    activation_function: Box::new(&activation_stub),
                },
                NeuralNetworkLayor {
                    weight: arr2(&[[0.0, 2.0, 4.0],[1.0, 3.0, 5.0]]),
                    bias: arr1(&[-1.0, -1.0, -1.0]),
                    activation_function: Box::new(&activation_stub),
                }
            ]
        };
        let a = a.forward_diff(
                                &arr1(&[1.0, 2.0]),
                                &vec![
                                    Array2::from_elem((2, 3), 1.0),
                                    Array2::from_elem((3, 2), 1.0),
                                    Array2::from_elem((2, 3), 1.0),
                                ],
                                &vec![
                                    Array1::from_elem(3, 2.0),
                                    Array1::from_elem(2, 2.0),
                                    Array1::from_elem(3, 2.0),
                                ],
                            );
        assert_eq!(a, arr1(&[502.0, 1142.0, 1782.0]));
    }

    #[test]
    fn NeuralNetwork_get_layor_size() {
        let a = NeuralNetwork::new(10, vec![
            NeuralNetworkLayorBuilder::new(80, Box::new(&activation_stub)),
            NeuralNetworkLayorBuilder::new(40, Box::new(&activation_stub)),
            NeuralNetworkLayorBuilder::new(20, Box::new(&activation_stub)),
        ]);

        let a = a.get_network_size();

        assert_eq!(
            a,
            vec![
                ((10, 80), 80),
                ((80, 40), 40),
                ((40, 20), 20),
            ]
        );
    }

    fn NeuralNetwork_get_layor_update_parameters_add() {
        let mut a = NeuralNetwork {
            layors: vec![
                NeuralNetworkLayor {
                    weight: arr2(&[[0.0, 1.0, 2.0],[3.0, 4.0, 5.0]]),
                    bias: arr1(&[0.0, 0.0, 0.0]),
                    activation_function: Box::new(&activation_stub),
                },
                NeuralNetworkLayor {
                    weight: arr2(&[[0.0, 1.0],[2.0, 3.0], [4.0, 5.0]]),
                    bias: arr1(&[-1.0, -1.0]),
                    activation_function: Box::new(&activation_stub),
                },
                NeuralNetworkLayor {
                    weight: arr2(&[[0.0, 2.0, 4.0],[1.0, 3.0, 5.0]]),
                    bias: arr1(&[-1.0, -1.0, -1.0]),
                    activation_function: Box::new(&activation_stub),
                }
            ]
        };

        let weight_add = vec![
            arr2(&[[1.0, 4.0, 7.0],[1.0, 4.0, 7.0]]),
            arr2(&[[2.0, 5.0],[8.0, 2.0], [5.0, 8.0]]),
            arr2(&[[3.0, 6.0, 9.0],[3.0, 6.0, 9.0]]),
        ];

        let bias_add = vec![
            arr1(&[3.0, 4.0, 5.0]),
            arr1(&[6.0, 7.0]),
            arr1(&[8.0, 9.0, 3.0]),
        ];

        a.update_parameters_add(&weight_add, &bias_add);

        assert_eq!(a.layors[0].weight, arr2(&[[1.0, 5.0, 9.0],[4.0, 8.0, 12.0]]));
        assert_eq!(a.layors[1].weight, arr2(&[[2.0, 6.0], [10.0, 5.0], [9.0, 13.0]]));
        assert_eq!(a.layors[2].weight, arr2(&[[3.0, 8.0, 13.0],[4.0, 9.0, 14.0]]));

        assert_eq!(a.layors[0].bias, arr1(&[3.0, 4.0, 5.0]));
        assert_eq!(a.layors[1].bias, arr1(&[5.0, 6.0]));
        assert_eq!(a.layors[2].bias, arr1(&[7.0, 8.0, 2.0]));
    }

    fn activation_stub(x: &Array1<f64>) -> Array1<f64> {
        x.clone()
    }
}  

#[cfg(test)]
mod NeuralNetworkLayorBuilder_test {
    use super::*;

    #[test]
    fn NeuralNetworkLayorBuilder_new() {
        let mut a = NeuralNetworkLayorBuilder::new(10, Box::new(&activation_stub));
        a.set_input_len(5);
        let a = a.build();
        assert_eq!(a.weight.raw_dim(), Dim([5, 10]));
        assert_eq!(a.bias.raw_dim(), Dim([10]));
        assert_eq!(&(a.activation_function.as_ref())(&arr1(&[1.0;10])), arr1(&[1.0;10]));
    }

    fn activation_stub(x: &Array1<f64>) -> Array1<f64> {
        x.clone()
    }
}


#[cfg(test)]
mod NeuralNetworkLayor_test {
    use super::*;
    
    #[test]
    fn NeuralNetworkLayor_new() {
        let a = NeuralNetworkLayor::new(10, 20, Box::new(&activation_stub));
        assert_eq!(a.weight.raw_dim(), Dim([10, 20]));
        assert_eq!(a.bias.raw_dim(), Dim([20]));
        assert_eq!(&(a.activation_function.as_ref())(&arr1(&[1.0;10])), arr1(&[1.0;10]));
    }

    #[test]
    fn NeuralNetworkLayor_forward() {
        let nnl = NeuralNetworkLayor {
            weight: arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            bias: arr1(&[7.0, 8.0, 9.0]),
            activation_function: Box::new(&activation_stub),
        };
        let a = nnl.forward(&arr1(&[2.0, 4.0]));
        assert_eq!(a,  arr1(&[25.0, 32.0, 39.0]));
    }

    #[test]
    fn NeuralNetworkLayor_forward_diff() {
        let nnl = NeuralNetworkLayor {
            weight: arr2(&[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]),
            bias: arr1(&[6.0, 7.0, 8.0]),
            activation_function: Box::new(&activation_stub),
        };
        let a = nnl.forward_diff(&arr1(&[2.0, 4.0]), &arr2(&[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]), &arr1(&[1.0, 1.0, 1.0]));
        assert_eq!(a,  arr1(&[25.0, 32.0, 39.0]));
    }

    #[test]
    fn NeuralNetworkLayor_get_layor_size() {
        let nnl = NeuralNetworkLayor {
            weight: arr2(&[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]),
            bias: arr1(&[6.0, 7.0, 8.0]),
            activation_function: Box::new(&activation_stub),
        };
        let a = nnl.get_layor_size();
        assert_eq!(((2, 3), 3), a);
    }

    #[test]
    fn NeuralNetworkLayor_update_parameters_add() {
        let mut nnl = NeuralNetworkLayor {
            weight: arr2(&[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]),
            bias: arr1(&[6.0, 7.0, 8.0]),
            activation_function: Box::new(&activation_stub),
        };
        nnl.update_parameters_add(&arr2(&[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]), &arr1(&[1.0, 1.0, 1.0]));
        assert_eq!(nnl.weight, arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]));
        assert_eq!(nnl.bias, arr1(&[7.0, 8.0, 9.0]));
    }

    fn activation_stub(x: &Array1<f64>) -> Array1<f64> {
        x.clone()
    }
}