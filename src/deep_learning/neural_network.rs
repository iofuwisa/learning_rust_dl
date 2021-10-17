use ndarray::prelude::{
    Axis,
    Array2,
};

use crate::deep_learning::common::*;
use crate::deep_learning::layer::*;
use crate::deep_learning::graph_plotter::*;

pub struct LearningParameter {
    pub batch_size: usize,
    pub iterations_num: u32,
}

pub struct LearningResource {
    pub trn_data: Array2<f64>,
    pub trn_lbl_onehot: Array2<f64>,
    pub tst_data: Array2<f64>,
    pub tst_lbl_onehot: Array2<f64>,
}

pub struct NeuralNetwork {
    last_layer: Box::<dyn NetworkLayer>,
}

impl NeuralNetwork {
    pub fn new<TL>(last_layer: TL) -> Self
    where TL: NetworkLayer + 'static
    {
        NeuralNetwork {
            last_layer: Box::new(last_layer),
        }
    }
    pub fn set_input(&mut self, input: &Array2<f64>) {
        self.last_layer.set_value(input);
    }
    
    pub fn set_lbl(&mut self, lbl_onehot: &Array2<f64>) {
        self.last_layer.set_lbl(lbl_onehot);
    }

    pub fn forward(&mut self, is_learning: bool) -> Array2<f64> {
        self.last_layer.forward(is_learning).to_owned()
    }
    pub fn get_layers(self) -> Box::<dyn NetworkLayer>{
        return self.last_layer;
    }

    pub fn learn(&mut self, parameter: LearningParameter, resource: LearningResource) {
        let mut correct_rates = Vec::<f64>::with_capacity((parameter.iterations_num + 1) as usize);
        let mut losses = Vec::<f64>::with_capacity((parameter.iterations_num + 1) as usize);
        let mut correct_rates_o = Vec::<f64>::with_capacity((parameter.iterations_num + 1) as usize);
        let mut losses_o = Vec::<f64>::with_capacity((parameter.iterations_num + 1) as usize);

        println!("Start learning");
        let (loss, rate) = self.test(parameter.batch_size, &resource.tst_data, &resource.tst_lbl_onehot);
        correct_rates.push(rate);
        losses.push(loss);
        // self.last_layer.plot();
        let (loss, rate) = self.test(parameter.batch_size, &resource.trn_data, &resource.trn_lbl_onehot);
        correct_rates_o.push(rate);
        losses_o.push(loss);

        for iteration in 0..parameter.iterations_num {
            // Choise batch data
            let (batch_data, batch_lbl_onehot) = 
                make_minibatch_data(parameter.batch_size, &resource.trn_data, &resource.trn_lbl_onehot);
     
            // // Set batch data
            self.set_input(&batch_data);
            self.set_lbl(&batch_lbl_onehot);

            // Update value weight and bias
            let init_dout = Array2::<f64>::ones(
                (
                    self.last_layer.forward(true).shape()[0],
                    self.last_layer.forward(true).shape()[1],
                )
            );
            self.last_layer.backward(init_dout);

            println!("Complete iteratioin:{}", iteration);
            let (loss, rate) = self.test(parameter.batch_size, &resource.tst_data, &resource.tst_lbl_onehot);
            correct_rates.push(rate);
            losses.push(loss);
            let (loss, rate) = self.test(parameter.batch_size, &resource.trn_data, &resource.trn_lbl_onehot);
            correct_rates_o.push(rate);
            losses_o.push(loss);
        }
        plot_rate(correct_rates, "correct_rate");
        plot_loss(losses, "loss");
        plot_rate(correct_rates_o, "correct_rate_o");
        plot_loss(losses_o, "loss_o");
        self.last_layer.plot();
    }

    pub fn test(&mut self, batch_size: usize, tst_data: &Array2<f64>, tst_lbl_onehot: &Array2<f64>) -> (f64, f64){
        let mut correct_rate = 0.0;
        let mut loss = 0.0;
        for _ in 0..10 {
            // Choise batch data
            let (batch_data, batch_lbl_onehot) = make_minibatch_data(batch_size, &tst_data, &tst_lbl_onehot);

            // Set batch data
            self.set_input(&batch_data);
            self.set_lbl(&batch_lbl_onehot);

            // Forward (skip loss)
            let test_res = self.last_layer.forward_skip_loss(false);

            correct_rate += calc_correct_rate(&test_res, &batch_lbl_onehot) / 10f64;

            // Forwards
            let batch_loss = self.forward(false);

            // calc loss
            let mut loss_sum = 0f64;
            for l in &batch_loss {
                loss_sum += *l;
            }
            loss += loss_sum / batch_loss.len() as f64 / 10f64;
        }
        
        println!("Test Loss: {}", loss);
        println!("Test CorrectRate: {}%", correct_rate * 100.0);
        println!("");

        return (loss, correct_rate);
    }
} 

fn calc_correct_rate(result: &Array2<f64>, lbl_onehot: &Array2<f64>) -> f64 {
    if result.shape() != lbl_onehot.shape() {
        panic!("Different shape. result: {:?} lbl_onehot:{:?}", result.shape(), lbl_onehot.shape());
    }

    let mut correct_count = 0;
    for row_i in 0..result.shape()[0] {
        let max_result_index = max_index_in_arr1(&result.index_axis(Axis(0), row_i).to_owned());
        let max_lbl_index = max_index_in_arr1(&lbl_onehot.index_axis(Axis(0), row_i).to_owned());

        if max_result_index == max_lbl_index {
            correct_count += 1;
        }
    }

    return correct_count as f64 / result.shape()[0] as f64;
}

pub fn make_minibatch_data(minibatch_size: usize, data: &Array2<f64>, lbl_onehot: &Array2<f64>) -> (Array2<f64>, Array2<f64>) {
    let mut minibatch_data = Array2::<f64>::zeros((minibatch_size, data.shape()[1]));
    let mut minibatch_lbl_onehot = Array2::<f64>::zeros((minibatch_size, lbl_onehot.shape()[1]));

    let indexes = random_choice(minibatch_size, data.shape()[0]);

    for row_i in 0..minibatch_size {
        let batch_i = indexes[row_i];

        let data_row = data.index_axis(Axis(0), batch_i);
        let lbl_onehot_row = lbl_onehot.index_axis(Axis(0), batch_i);

        let mut batch_data_row = minibatch_data.index_axis_mut(Axis(0), row_i);
        let mut batch_lbl_onehot_row = minibatch_lbl_onehot.index_axis_mut(Axis(0), row_i);

        batch_data_row.assign(&data_row);
        batch_lbl_onehot_row.assign(&lbl_onehot_row);
    }

    return (minibatch_data, minibatch_lbl_onehot);
}

#[cfg(test)]
mod test_neuaral_network {
    use super::*;

    use ndarray::prelude::{
        arr2,
    };

    use crate::deep_learning::optimizer::*;

    #[test]
    fn test_set_input() {
        let batch_size = 2;

        let layers = DirectValue::new(Array2::<f64>::zeros((batch_size, 3)));
        let layers = Affine::new_random(
            layers,
            28*28,
            200,
            Sgd::new(0.01),
            Sgd::new(0.01)
        );
        let layers = SoftmaxWithLoss::new(layers, Array2::<f64>::zeros((batch_size, 4)));
        let mut nn = NeuralNetwork::new(layers);

        let input = arr2(&
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ]
        );
        nn.set_input(&input);
    }
    #[test]
    fn test_set_lbl() {
        let batch_size = 2;

        let layers = DirectValue::new(Array2::<f64>::zeros((batch_size, 3)));
        let layers = Affine::new_random(
            layers,
            28*28,
            200,
            Sgd::new(0.01),
            Sgd::new(0.01)
        );
        let layers = SoftmaxWithLoss::new(layers, Array2::<f64>::zeros((batch_size, 4)));
        let mut nn = NeuralNetwork::new(layers);

        let lbl = arr2(&
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 7.0],
            ]
        );
        nn.set_lbl(&lbl);
    }
}