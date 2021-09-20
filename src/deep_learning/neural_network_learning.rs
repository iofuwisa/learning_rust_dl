// use ndarray::prelude::{
//     Axis,
//     Array2,
// };

// use crate::deep_learning::common::*;
// use crate::deep_learning::affine_layer::*;
// use crate::deep_learning::neural_network::*;
// use crate::deep_learning::softmax_with_loss::*;

// struct NearalNetworkLearning {
//     layers: SoftmaxWithLoss,
//     trn_data: Array2<f64>,
//     trn_lbl_onehot: Array2<f64>,
//     tst_data: Array2<f64>,
//     tst_lbl_onehot: Array2<f64>,
//     learning_rate: f64,
//     batch_size: usize,
//     iterations_num: u32,
// }

// impl<T: NetworkBatchLayer> NearalNetworkLearning {
//     new(neural_network: NeuralNetwork,
//         trn_data: &Array2<f64>,
//         trn_lbl_onehot: &Array2<f64>,
//         tst_data: &Array2>f64>,
//         tst_lbl_onehot: &Array2<f64>,
//         learning_rate: f64
//         batch_size: usize,
//         iterations_num: u32) -> NearalNetworkLearning {
//         // Add loss layer
//         let layers = neural_network::get_layers();
//         let layers = SoftmaxWithLoss::new(layers, Array2::<f64>::zeros((minibatch_size, trn_lbl_onehot.shape()[1])));
//         let neural_network = NeuralNetwork::new(layers);

//         NearalNetworkLearning {
//             neural_network: neural_network
//             trn_data: trn_data,
//             trn_lbl_onehot: trn_lbl_onehot,
//             tst_data: tst_data,
//             tst_lbl_onehot: tst_lbl_onehot,
//             learning_rate: learning_rate,
//             batch_size: batch_size,
//             iterations_num: iterations_num,
//         }
//     }

//     pub fn learn(&mut self) {
//         for iteration in 0..iterations_num {
//             // Choise batch data
//             let (batch_data, batch_lbl_onehot) = make_minibatch_data(minibatch_size, &trn_data, &trn_lbl_one_hot);
     
//             // Set batch data
//             nn.set_input(&batch_data);
//             nn.set_lbl(&batch_lbl_onehot);
    
//             println!("forward: {:?}", nn.forward());
    
//         }
//     }
    
//     fn test(&mut self) -> (f64, f64){
//         // Choise batch data
//         let (batch_data, batch_lbl_onehot) = make_minibatch_data(1000, &tst_data, &trn_lbl_one_hot);

//         // Set batch data
//         nn.set_input(&batch_data);
//         nn.set_lbl(&batch_lbl_onehot);

//         // Forward (skip loss)
//         let test_res = nn.getx().forward();

//         let correct_rate = calc_correct_rate(&test_res, &batch_lbl_onehot);

//         // Forward
//         let batch_loss = nn.forward();

//         // calc loss
//         let mut loss_sum = 0.0;
//         for l in batch_loss {
//             loss_sum += l;
//         }
//         let loss = loss_sum / batch_loss.len();

//         println!("Loss: {}", loss);
//         println!("CorrectRate: {}%", correct_rate * 100.0);
//         println!("");
//     }

//     fn calc_correct_rate(result: &Array2<f64>, lbl_onehot: &Array2<f64>) -> f64 {
//         if result.shape() != lbl_onehot.shape() {
//             panic!("Different shape. result: {:?} lbl_onehot:{:?}", result.shape(), lbl_onehot.shape());
//         }

//         let mut correct_count = 0;
//         for row_i in 0..result.shape()[0] {
//             let max_result_index = max_index_in_arr1(result.index_axis(Axis(0), row_i));
//             let max_lbl_index = max_index_in_arr1(lbl_onehot.index_axis(Axis(0), row_i));

//             if max_result_index == max_lbl_index {
//                 correct_count++;
//             }
//         }

//         return correct_count as f64 / result.shape()[0];
//     }
// }

// fn make_minibatch_data(minibatch_size: usize, data: &Array2<f64>, lbl_onehot: &Array2<f64>) -> (Array2<f64>, Array2<f64>) {
//     let mut minibatch_data = Array2::<f64>::zeros((minibatch_size, data.shape()[1]));
//     let mut minibatch_lbl_onehot = Array2::<f64>::zeros((minibatch_size, lbl_onehot.shape()[1]));

//     let indexes = random_choice(minibatch_size, data.shape()[0]);

//     for row_i in 0..minibatch_size {
//         let batch_i = indexes[row_i];

//         let data_row = data.index_axis(Axis(0), batch_i);
//         let lbl_onehot_row = lbl_onehot.index_axis(Axis(0), batch_i);

//         let mut batch_data_row = minibatch_data.index_axis_mut(Axis(0), row_i);
//         let mut batch_lbl_onehot_row = minibatch_lbl_onehot.index_axis_mut(Axis(0), row_i);

//         batch_data_row.assign(&data_row);
//         batch_lbl_onehot_row.assign(&lbl_onehot_row);
//     }

//     return (minibatch_data, minibatch_lbl_onehot);
// }

// #[cfg(test)]
// mod test_neuaral_network {
//     use super::*;

//     use crate::deep_learning::mnist::*;

//     #[test]
//     fn test_make_minibatch_data() {
//         // Load MNIST
//         let mnist = MnistImages::new(1000, 1, 1);
//         let trn_img = mnist.get_trn_img();
//         let trn_lbl_one_hot = mnist.get_trn_lbl_one_hot();

//         let minibach_size = 20;

//         let (batch_data, batch_lbl_onehot) = make_minibatch_data(minibach_size, &trn_img, &trn_lbl_one_hot);

//         assert_eq!(batch_data.shape(), [minibach_size, 28*28]);
//         assert_eq!(batch_lbl_onehot.shape(), [minibach_size, 10]);

//         for row_i in 0..minibach_size {
//             let row_data = batch_data.index_axis(Axis(0), row_i).to_owned();
//             let row_lbl_onehot = batch_lbl_onehot.index_axis(Axis(0), row_i).to_owned();

//             // println!("digit: {}", max_index_in_arr1(&row_lbl_onehot));
//             // print_img(&row_data);
//             // println!("");
//         }
//     }
// }