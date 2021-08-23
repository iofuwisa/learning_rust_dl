extern crate mnist;
extern crate rulinalg;

use mnist::{Mnist, MnistBuilder};
use rulinalg::matrix::{Matrix, BaseMatrix};


pub struct MnistImages {
    size: u32,
    img_rows: usize,
    img_cols: usize,
    imgs: Matrix<f64>,
    labels: Vec<u8>,
}

impl MnistImages {
    pub fn new() -> MnistImages {
        println!("Start loading mnist.");

        let (size, img_rows, img_cols) = (100, 28, 28);

        // Deconstruct the returned Mnist struct.
        println!("Load mnist resource.");
        let Mnist {
            trn_img: imgs,
            trn_lbl: labels,
            ..
        } = MnistBuilder::new()
            .label_format_digit()
            .training_set_length(size)
            .validation_set_length(0)
            .test_set_length(0)
            .finalize();
        
        // Convert 1D(R:1 C:50000*28*28) to 2D(R:50_000 C:28*28).
        let imgs = Matrix::new(size as usize, (img_rows * img_cols) as usize, imgs);

        // normalyze
        println!("Normalyze.");
        let imgs: Matrix<f64> = imgs.try_into().unwrap() / 255.0;

        println!("Complete loading mnist.");
        return MnistImages{
            size: size,
            img_rows: img_rows,
            img_cols: img_cols,
            imgs: imgs,
            labels: labels
        };
    }

    pub fn getImgMatrix(&self, index: u32) -> Matrix<f64> {
        self.imgs.select_rows(&[index as usize]).clone()
    }

    pub fn getImgVec(&self, index: u32) -> Vec<f64> {
        self.imgs.select_rows(&[index as usize]).into_vec().clone()
    }

    pub fn getLabel(&self, index: u32) -> u8{
        self.labels[index as usize]
    }
}
