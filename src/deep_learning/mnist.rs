extern crate mnist;
extern crate rulinalg;

use mnist::{Mnist, NormalizedMnist, MnistBuilder};
use ndarray::prelude::{
    ArrayBase,
    Array,
    Array1,
    Array2,
    arr1,
    arr2,
    s
};

const IMG_ROWS: usize = 28;
const IMG_COLS: usize = 28;

pub struct MnistImages {
    trn_img_size: u32,
    trn_lbl: Vec<u8>,
    trn_lbl_one_hot: Array2<f64>,
    trn_img: Array2<f64>,

    val_img_size: u32,
    val_lbl: Vec<u8>,
    val_lbl_one_hot: Array2<f64>,
    val_img: Array2<f64>,

    tst_img_size: u32,
    tst_lbl: Vec<u8>,
    tst_lbl_one_hot: Array2<f64>,
    tst_img: Array2<f64>,
}

impl MnistImages {
    pub fn new(trn_img_size: u32, val_img_size: u32, tst_img_size: u32) -> MnistImages {
        println!("Start loading mnist.");

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

        println!("Setup trn img.");
        let (trn_lbl, trn_lbl_one_hot, trn_img) = Self::setup_img(trn_img_size as usize, trn_lbl_one_hot, trn_img);
        
        println!("Setup val img.");
        let (val_lbl, val_lbl_one_hot, val_img) = Self::setup_img(val_img_size as usize, val_lbl_one_hot, val_img);

        println!("Setup tst img.");
        let (tst_lbl, tst_lbl_one_hot, tst_img) = Self::setup_img(tst_img_size as usize, tst_lbl_one_hot, tst_img);

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

    fn setup_img(img_size: usize, lbl_one_hot: Vec<u8>, img: Vec<u8>) -> (Vec<u8>, Array2<f64>, Array2<f64>) {
        // Normalyze and Convert to f64
        let mut normd_trn_img = Vec::<f64>::with_capacity(img.len());
        for d in img {
            normd_trn_img.push(d as f64 / 255.0);
        }
    
        // Convert 1D(R:1 C:size*28*28) to 2D(R:size C:28*28).
        let img = Array::from_shape_vec((img_size, (IMG_ROWS * IMG_COLS)), normd_trn_img).unwrap();
    
    
        // Convert label one hot to f64
        let mut f64_lbl_one_hot = Vec::<f64>::with_capacity(lbl_one_hot.len());
        for d in lbl_one_hot {
            f64_lbl_one_hot.push(d as f64);
        }
    
        // Convert 1D(R:1 C:size*10) to 2D(R:size C:10).
        let lbl_one_hot = Array::from_shape_vec((img_size, 10), f64_lbl_one_hot).unwrap();
    
        // Generate label digit
        let mut lbl_digit = Vec::<u8>::with_capacity(img_size);
        for row in lbl_one_hot.outer_iter() {
            let mut lbl = 0;
            for n in row {
                if *n > 0.0 {
                    break;
                }
                lbl += 1;
            }
            lbl_digit.push(lbl);
        }
    
        return (
            lbl_digit,
            lbl_one_hot,
            img,
        );
    }

    pub fn get_trn_img_size(&self) -> u32 { self.trn_img_size }
    pub fn get_trn_lbl(&self) -> &Vec<u8> { &(self.trn_lbl) }
    pub fn get_trn_lbl_one_hot(&self) -> &Array2<f64> { &(self.trn_lbl_one_hot) }
    pub fn get_trn_img(&self) -> &Array2<f64> { &(self.trn_img) }
    
    pub fn get_val_img_size(&self) -> u32 { self.val_img_size }
    pub fn get_val_lbl(&self) -> &Vec<u8> { &(self.val_lbl) }
    pub fn get_val_lbl_one_hot(&self) -> &Array2<f64> { &(self.val_lbl_one_hot) }
    pub fn get_val_img(&self) -> &Array2<f64> { &(self.val_img) }
    
    pub fn get_tst_img_size(&self) -> u32 { self.tst_img_size }
    pub fn get_tst_lbl(&self) -> &Vec<u8> { &(self.tst_lbl) }
    pub fn get_tst_lbl_one_hot(&self) -> &Array2<f64> { &(self.tst_lbl_one_hot) }
    pub fn get_tst_img(&self) -> &Array2<f64> { &(self.tst_img) }
}
