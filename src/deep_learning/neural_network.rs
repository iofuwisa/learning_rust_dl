use ndarray::prelude::{
    Array2,
    arr2,
};

use crate::deep_learning::affine_layer::*;
use crate::deep_learning::activation_layers::*;
use crate::deep_learning::softmax_with_loss::*;


pub struct NeuralNetwork<T: NetworkBatchLayer> {
    last_layer: T,
}

impl<T: NetworkBatchLayer> NeuralNetwork<T> {
    pub fn new(last_layer: T) -> Self {
        NeuralNetwork {
            last_layer: last_layer,
        }
    }
    pub fn set_input(&mut self, input: &Array2<f64>) {
        self.last_layer.set_value(input);
    }
    
    pub fn set_lbl(&mut self, lbl_onehot: &Array2<f64>) {
        self.last_layer.set_lbl(lbl_onehot);
    }

    pub fn forward(&mut self) -> Array2<f64> {
        self.last_layer.forward().to_owned()
    }
}

mod test_neuaral_network {
    use super::*;

    #[test]
    fn test_set_input() {
        let batch_size = 2;

        let layers = NetworkBatchValueLayer::new(Array2::<f64>::zeros((batch_size, 3)));
        let layers = AffineLayer::new_random(layers, 28*28, 200);
        let layers = SoftmaxWithLoss::new(layers, Array2::<f64>::zeros((batch_size, 4)));
        let mut nn = NeuralNetwork::new(layers);

        let input = arr2(&
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ]
        );
        nn.set_input(&input);

        assert_eq!(nn.last_layer.get_x().get_x().get_value(), input);
        assert_eq!(nn.last_layer.get_t(), Array2::<f64>::zeros((batch_size, 4)));
    }
    #[test]
    fn test_set_lbl() {
        let batch_size = 2;

        let layers = NetworkBatchValueLayer::new(Array2::<f64>::zeros((batch_size, 3)));
        let layers = AffineLayer::new_random(layers, 28*28, 200);
        let layers = SoftmaxWithLoss::new(layers, Array2::<f64>::zeros((batch_size, 4)));
        let mut nn = NeuralNetwork::new(layers);

        let lbl = arr2(&
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 7.0],
            ]
        );
        nn.set_lbl(&lbl);

        assert_eq!(nn.last_layer.get_x().get_x().get_value(), Array2::<f64>::zeros((batch_size, 3)));
        assert_eq!(nn.last_layer.get_t(), lbl);
    }
}