extern crate mnist;
extern crate rulinalg;

use mnist::{Mnist, NormalizedMnist, MnistBuilder};
use rulinalg::matrix::{Matrix, BaseMatrix};


pub struct MnistImages {
    trn_img_size: u32,
    trn_lbl: Vec<u8>,
    trn_lbl_one_hot: Matrix<u8>,
    trn_img: Matrix<f64>,

    val_img_size: u32,
    val_lbl: Vec<u8>,
    val_lbl_one_hot: Matrix<u8>,
    val_img: Matrix<f64>,

    tst_img_size: u32,
    tst_lbl: Vec<u8>,
    tst_lbl_one_hot: Matrix<u8>,
    tst_img: Matrix<f64>,
}

impl MnistImages {
    pub fn new(trn_img_size: u32, val_img_size: u32, tst_img_size: u32) -> MnistImages {
        println!("Start loading mnist.");

        let img_rows = 28;
        let img_cols = 28;

        // Deconstruct the returned Mnist struct.
        println!("Load mnist resource.");
        let Mnist {
            trn_img: trn_img,
            trn_lbl: trn_lbl_one_hot,
            val_img: val_img,
            val_lbl: val_lbl_one_hot,
            tst_img: tst_img,
            tst_lbl: tst_lbl_one_hot
        } = MnistBuilder::new()
            .label_format_one_hot()
            .training_set_length(trn_img_size)
            .validation_set_length(val_img_size)
            .test_set_length(tst_img_size)
            .finalize();
        
        
        // Convert 1D(R:1 C:size*28*28) to 2D(R:size C:28*28).
        let trn_img = Matrix::new(trn_img_size as usize, (img_rows * img_cols) as usize, trn_img);

        // Convert label
        let trn_lbl_one_hot = Matrix::new(trn_img_size as usize, 10, trn_lbl_one_hot);
        let mut trn_lbl = Vec::<u8>::with_capacity(trn_img_size as usize);
        for row in trn_lbl_one_hot.row_iter() {
            let mut lbl = 0;
            for n in *row {
                if *n > 0 {
                    break;
                }
                lbl += 1;
            }
            trn_lbl.push(lbl)
        }
        

        // normalyze
        println!("Normalyze trn_img.");
        let trn_img: Matrix<f64> = trn_img.try_into().unwrap() / 255.0;
        

        // Convert 1D(R:1 C:size*28*28) to 2D(R:size C:28*28).
        let val_img = Matrix::new(val_img_size as usize, (img_rows * img_cols) as usize, val_img);

        // Convert label
        let val_lbl_one_hot = Matrix::new(val_img_size as usize, 10, val_lbl_one_hot);
        let mut val_lbl = Vec::<u8>::with_capacity(val_img_size as usize);
        for row in val_lbl_one_hot.row_iter() {
            let mut lbl = 0;
            for n in *row {
                lbl = 
                if *n > 0 {1}
                else {0};
            }
            val_lbl.push(lbl)
        }

        // normalyze
        println!("Normalyze. val_img");
        let val_img: Matrix<f64> = val_img.try_into().unwrap() / 255.0;


        // Convert 1D(R:1 C:size*28*28) to 2D(R:size C:28*28).
        let tst_img = Matrix::new(tst_img_size as usize, (img_rows * img_cols) as usize, tst_img);

        // Convert label
        let tst_lbl_one_hot = Matrix::new(tst_img_size as usize, 10, tst_lbl_one_hot);
        let mut tst_lbl = Vec::<u8>::with_capacity(tst_img_size as usize);
        for row in tst_lbl_one_hot.row_iter() {
            let mut lbl = 0;
            for n in *row {
                lbl = 
                if *n > 0 {1}
                else {0};
            }
            tst_lbl.push(lbl)
        }

        // normalyze
        println!("Normalyze. tst_img");
        let tst_img: Matrix<f64> = tst_img.try_into().unwrap() / 255.0;


        println!("Complete loading mnist.");
        return MnistImages{
            trn_img_size: trn_img_size,
            trn_lbl: trn_lbl,
            trn_lbl_one_hot: trn_lbl_one_hot,
            trn_img: trn_img,
        
            val_img_size: val_img_size,
            val_lbl: val_lbl,
            val_lbl_one_hot: val_lbl_one_hot,
            val_img: val_img,
        
            tst_img_size: tst_img_size,
            tst_lbl: tst_lbl,
            tst_lbl_one_hot: tst_lbl_one_hot,
            tst_img: tst_img,
        };
    }

    pub fn get_trn_img_size(&self) -> u32 { self.trn_img_size }
    pub fn get_trn_lbl(&self) -> &Vec<u8> { &(self.trn_lbl) }
    pub fn get_trn_lbl_one_hot(&self) -> &Matrix<u8> { &(self.trn_lbl_one_hot) }
    pub fn get_trn_img(&self) -> &Matrix<f64> { &(self.trn_img) }
    
    pub fn get_val_img_size(&self) -> u32 { self.val_img_size }
    pub fn get_val_lbl(&self) -> &Vec<u8> { &(self.val_lbl) }
    pub fn get_val_lbl_one_hot(&self) -> &Matrix<u8> { &(self.val_lbl_one_hot) }
    pub fn get_val_img(&self) -> &Matrix<f64> { &(self.val_img) }
    
    pub fn get_tst_img_size(&self) -> u32 { self.tst_img_size }
    pub fn get_tst_lbl(&self) -> &Vec<u8> { &(self.tst_lbl) }
    pub fn get_tst_lbl_one_hot(&self) -> &Matrix<u8> { &(self.tst_lbl_one_hot) }
    pub fn get_tst_img(&self) -> &Matrix<f64> { &(self.tst_img) }
}
