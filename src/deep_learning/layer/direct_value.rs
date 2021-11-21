use std::fs::File;
use std::io::{self, Read, Write, BufReader};
use ndarray::prelude::{
    Axis,
    Array2,
};

use crate::deep_learning::layer::*;

const LAYER_LABEL: &str = "direct";
pub struct DirectValue {
    value: Array2<f64>,
}
impl DirectValue {
    pub fn new(value: Array2<f64>) -> DirectValue {
        DirectValue {
            value: value,
        }
    }
    pub fn new_from_len(row_len: usize, col_len: usize) -> DirectValue {
        return DirectValue::new(Array2::<f64>::zeros((row_len, col_len)))
    }
}
impl NetworkLayer for DirectValue {
    fn forward(&mut self, _is_learning: bool) -> Array2<f64> {
        self.value.clone()
    }
    fn backward(&mut self, _dout: Array2<f64>) {
        // Nothinf to do
    }
    fn set_value(&mut self, value: &Array2<f64>) {
        if self.value.shape() != value.shape() {
            panic!("Different shape. self.value: {:?} value:{:?}", self.value.shape(), value.shape());
        }
        self.value.assign(value);
    }
    fn set_lbl(&mut self, _value: &Array2<f64>) {
        // Nothing to do
    }
    fn clean(&mut self) {
        // Nothing to do
    }
    fn plot(&self){
        // Nothing to do
    }
    fn weight_squared_sum(&self) -> f64 {
        return 0f64;
    }
    fn weight_sum(&self) -> f64 {
        return 0f64;
    }
    fn export(&self, file: &mut File) -> Result<(), Box<std::error::Error>> {
        writeln!(file, "{}", LAYER_LABEL)?;

        for row in self.value.axis_iter(Axis(0)) {
            for v in row {
                write!(file, "{},", v)?;
            }
            writeln!(file, "")?;
        }

        file.flush()?;
        Ok(())
    }
}