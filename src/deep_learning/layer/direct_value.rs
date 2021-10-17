use ndarray::prelude::{
    Array2,
};

use crate::deep_learning::layer::*;

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
}