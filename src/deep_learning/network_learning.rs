use crate::deep_learning::network::*;
use crate::deep_learning::loss_functions::*;
use crate::deep_learning::common::*;

use rand::Rng;

use std::time::{Duration, Instant};


use ndarray::prelude::{
    Array1,
    Array2,
    arr1,
    arr2,
    Axis,
    s,
};


pub fn network_learning(network: &mut NeuralNetwork, trn_data: &Array2<f64>, trn_lbl_one_hot: &Array2<f64>, tst_data: &Array2<f64>, tst_lbl_one_hot: &Array2<f64>, iterations_num: u32, learning_rate: f64, minibatch_size: usize) {

    let (loss, rate) = test(network, tst_data, tst_lbl_one_hot);
    println!("Start learning");
    println!("Loss: {}", loss);
    println!("CorrectRate: {}%", rate * 100.0);
    println!("");

    for iteration in 0..iterations_num {

        let (grad_weight, grad_bias) = calc_gradient(network, trn_data, trn_lbl_one_hot, minibatch_size);
        
        // Calc value update
        let mut update_weight = Vec::<Array2<f64>>::new();
        let mut update_bias = Vec::<Array1<f64>>::new();
        for i in 0..grad_weight.len() {
            update_weight.push(grad_weight[i].mapv(|w:f64| -> f64 {w*-1.0*learning_rate}));
            update_bias.push(grad_bias[i].mapv(|b:f64| -> f64 {b*-1.0*learning_rate}));
        }
        network.update_parameters_add(&update_weight, &update_bias);

        let (loss, rate) = test(network, tst_data, tst_lbl_one_hot);
        println!("Complete iter {}", iteration + 1);
        println!("Loss: {}", loss);
        println!("CorrectRate: {}%", rate * 100.0);
        println!("");
    }
}

fn calc_gradient(network: &NeuralNetwork, trn_data: &Array2<f64>, trn_lbl_one_hot: &Array2<f64>, minibatch_size: usize) -> (Vec::<Array2<f64>>, Vec::<Array1<f64>>)  {

    // Choise minibatch indexes
    let indexes = random_choice(minibatch_size, trn_data.shape()[0]);

    // Forwading
    // This closure is called in nuumeric_gradient
    let f = |x: &Array1<f64>| -> f64 {
        // let t1 = Instant::now();
        let (weight_h, bias_h) = parse_weight_bias_from_arr1(x, &network.get_network_size());
        // let du = t1.elapsed();
        // println!("Elapsed parse: {}", du.as_nanos());
        
        let mut loss = 0.0;
        for i in &indexes {
            // let t2 = Instant::now();
            let data = trn_data.index_axis(Axis(0), *i);
            // let du = t2.elapsed();
            // println!("Elapsed idexes: {}", du.as_nanos());

            // let t3 = Instant::now();
            let y = network.forward_diff(&data.to_owned(), &weight_h, &bias_h);
            // let du = t3.elapsed();
            // println!("Elapsed forward: {}", du.as_nanos());

            // let t4 = Instant::now();
            loss += crosss_entropy_erro(&y, &trn_lbl_one_hot.index_axis(Axis(0), *i).to_owned());
            // let du = t4.elapsed();
            // println!("Elapsed loss: {}", du.as_nanos());

        }

        return loss;
    };

    let network_size = network.get_network_size();
    let mut parameter_size = 0;
    for ((weight_row_size, weight_col_size), bias_size) in &network_size {
        parameter_size += weight_row_size * weight_col_size + bias_size;
    }
    let x: Array1<f64> = Array1::<f64>::zeros(parameter_size);
    let grad = numeric_gradient(f, &x);

    return parse_weight_bias_from_arr1(&grad, &network_size);
}

fn parse_weight_bias_from_arr1(value: &Array1<f64>, network_size: &Vec<((usize, usize), usize)>) -> (Vec::<Array2<f64>>, Vec::<Array1<f64>>) {
    let mut weight = Vec::<Array2<f64>>::new();
    let mut bias = Vec::<Array1<f64>>::new();
    let mut base_index = 0;
    for ((weight_row_size, weight_col_size), bias_size) in network_size {
        let w = value.slice(s![base_index..(base_index+weight_row_size*weight_col_size)]);
        base_index += weight_row_size*weight_col_size;
        let w = w.into_shape((*weight_row_size, *weight_col_size)).unwrap();
        let b = value.slice(s![base_index..base_index+bias_size]);
        base_index += bias_size;
        weight.push(w.to_owned());
        bias.push(b.to_owned());
    }

    return (weight, bias);
}

fn random_choice(size: usize, max: usize) -> Vec<usize> {
    let mut rng = rand::thread_rng();

    let mut choice = Vec::<usize>::with_capacity(size as usize);
    for _ in 0..size {
        choice.push((rng.gen::<f32>()*max as f32).floor() as usize);
    }
    
    return choice;
}

fn test(network: &NeuralNetwork, tst_data: &Array2<f64>, tst_lbl_one_hot: &Array2<f64>) -> (f64, f64) {
    // Test
    let test_sampl_size = 1000;
    let indexes = random_choice(test_sampl_size, tst_data.shape()[0]);
    let mut loss: f64 = 0.0;
    let mut correct_count = 0;
    for i in &indexes {
        // Guess
        let data = tst_data.index_axis(Axis(0), *i).to_owned();
        let y = network.forward(&data);
        // Loss
        loss += crosss_entropy_erro(&y, &tst_lbl_one_hot.index_axis(Axis(0), *i).to_owned());
        // Correct answer rate
        let lbl = tst_lbl_one_hot.index_axis(Axis(0), *i).to_owned();
        let mut answer_max_index: u8 = 0;
        let mut lbl_max_index: u8 = 0;
        for j in 0..10 {
                if y[j] > y[answer_max_index as usize] {
                    answer_max_index = j as u8;
                }
                if lbl[j] > lbl[lbl_max_index as usize] {
                    lbl_max_index = j as u8;
                }
        }
        correct_count += if answer_max_index==lbl_max_index {1} else {0};
    }
    loss = loss / test_sampl_size as f64;
    let correct_rate = correct_count as f64 / test_sampl_size  as f64;

    return (loss, correct_rate);
}


#[cfg(test)]
mod test_mod {
    use super::*;

    #[test]
    fn test_parse_weight_bias_from_arr1() {
        let value = arr1(&[
            // weight1
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            // bias1
            2.0, 3.0, 4.0,
            // weight2
            2.0, 5.0, 2.0, 6.0,
            1.0, 2.0, 1.0, 7.0,
            4.0, 5.0, 4.0, 9.0,
            // bias2
            2.0, 5.0, 2.0, 6.0,
        ]);
        let weight_expect = vec![
            arr2(&[   // weight1
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ]),
            arr2(&[   // weight2
                [2.0, 5.0, 2.0, 6.0],
                [1.0, 2.0, 1.0, 7.0],
                [4.0, 5.0, 4.0, 9.0],
            ])
        ];
        let bias_expect = vec![
            // bias1
            arr1(&[2.0, 3.0, 4.0]),
            // bias2
            arr1(&[2.0, 5.0, 2.0, 6.0]),
        ];

        let network_size = vec![
            ((2, 3), 3),
            ((3, 4), 4),
        ];

        let (weight, bias) = parse_weight_bias_from_arr1(&value, &network_size);

        assert_eq!(weight, weight_expect);
        assert_eq!(bias, bias_expect);
        

    }

    #[test]
    fn test_random_choice() {
        let a = random_choice(1_000_000, 50);
        assert_eq!(a.len(), 1_000_000);
        for n in a {
            assert_eq!( n < 50, true);
        }
    }
}