extern crate mnist;
extern crate rulinalg;

use mnist::{Mnist, MnistBuilder};
use rulinalg::matrix::{BaseMatrix, BaseMatrixMut, Matrix};

fn main() {
    let (trn_size, rows, cols) = (50_000, 28, 28);
    let index = 200;
    // Deconstruct the returned Mnist struct.
    let Mnist { trn_img, trn_lbl, .. } = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(trn_size)
        .validation_set_length(0)
        .test_set_length(0)
        .finalize();

    // Get the label of the first digit.
    let first_label = trn_lbl[index];
    println!("The {} digit is a {}.", index, first_label);

    // Convert the flattened training images vector to a matrix.
    let trn_img = Matrix::new((trn_size * rows) as usize, cols as usize, trn_img);

    // Get the image of the first digit.
    let row_indexes = ((index*rows as usize)..((index+1)*rows as usize)).collect::<Vec<_>>();
    let first_image = trn_img.select_rows(&row_indexes);
    println!("The image looks like... \n{}", first_image);

    // Normalizeg
    let trn_img: Matrix<f32> = trn_img.try_into().unwrap() / 255.0;

    // Get the image of the first digit and round the values to the nearest tenth.
    let trn_img = trn_img.select_rows(&row_indexes)
        // floor
        .apply(&|p| (p * 10.0).round() / 10.0);
    println!("The image looks like... \n{}", trn_img);
}