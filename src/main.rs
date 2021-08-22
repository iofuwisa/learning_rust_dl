pub mod deep_learning;

// use crate::deep_learning::logical_operators::*;
use crate::deep_learning::activation_functions::*;
// use crate::deep_learning::network::*;
// use crate::deep_learning::mnist::*;

use ndarray::prelude::{
    Array1,
    arr1,
    arr2,
    s
};

fn main(){
    let x = arr1(&[1.0, 2.0, 0.0, 0.1, -0.1]);
    let y = step_array(&x);   
    println!("{}", y);
}

    // let mut x: Vec<f64> = Vec::new();
    // x.push(0.0);
    // x.push(1.0);
    // println!("and:{}", and(&x));
    // println!("nand:{}", nand(&x));
    // println!("or:{}", or(&x));


    // let x = [-5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    // let mut step_y: Vec<f64> = Vec::new();
    // let mut sigm_y: Vec<f64> = Vec::new();
    // let mut relu_y: Vec<f64> = Vec::new();
    // step_array(&x, &mut step_y);
    // sigmoid_array(&x, &mut sigm_y);
    // relu_array(&x, &mut relu_y);
    // println!("x: {:?}", x);
    // println!("step_y: {:?}", step_y);
    // println!("sigm_y: {:?}", sigm_y);
    // println!("relu_y: {:?}", relu_y);

    // calc(1.0, 0.5);

// }

  
// extern crate mnist;
// extern crate rulinalg;

// use mnist::{Mnist, MnistBuilder};
// use rulinalg::matrix::{Matrix, BaseMatrix, BaseMatrixMut};
// use rulinalg::matrix;


// fn main() {
//     let (trn_size, rows, cols) = (50_000, 28, 28);
//     let index = 200;
//     // Deconstruct the returned Mnist struct.
//     let Mnist { trn_img, trn_lbl, .. } = MnistBuilder::new()
//         .label_format_digit()
//         .training_set_length(trn_size)
//         .validation_set_length(0)
//         .test_set_length(0)
//         .finalize();

//     // Convert the flattened training images vector to a matrix.
//     let trn_img = Matrix::new(trn_size as usize, (rows * cols) as usize, trn_img);

//     // Get the label of the first digit.
//     let first_label = trn_lbl[index];
//     println!("The {} digit is a {}.", index, first_label);

//     // Get the image of the first digit.
//     let index_img = trn_img.select_rows(&[index]);
//     let index_img = Matrix::new(rows, cols, index_img.into_vec().clone());
//     println!("The image looks like... \n{}", index_img);

//     // Normalizeg
//     let trn_img: Matrix<f32> = trn_img.try_into().unwrap() / 255.0;

//     // Get the image of the first digit and round the values to the nearest tenth.
//     let index_img = trn_img.select_rows(&[index]);
//     let index_img = Matrix::new(rows, cols, index_img.into_vec().clone())
//         // floor
//         .apply(&|p| (p * 10.0).round() / 10.0);
//     println!("The image looks like... \n{}", index_img);

//     // let a = matrix![1.0, 2.0, 3.0;
//     //                 4.0, 5.0, 6.0;
//     //                 7.0, 8.0, 9.0];
//     // println!("The image looks like... \n{}", &a.select_rows(&[1,2]));
//     // println!("The image looks like... \n{}", &a.select_cols(&[1,2]));
// }

