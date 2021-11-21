use std::fs::File;
use std::io::{self, Read, Write, BufReader};
use ndarray::prelude::{
    Array2,
    Axis,
};

use crate::deep_learning::layer::*;
use crate::deep_learning::optimizer::*;
use crate::deep_learning::graph_plotter::*;


// Affine value(weight and bias)
const LAYER_LABEL: &str = "a_direct";
pub struct AffineDirectValue {
    value: Array2<f64>,
    optimizer: Box<dyn Optimizer>,
    name: String,
}
impl AffineDirectValue {
    pub fn new<TO>(value: Array2<f64>, optimizer: TO)
        -> AffineDirectValue
        where TO: Optimizer + 'static
    {
        AffineDirectValue {
            value: value,
            optimizer: Box::new(optimizer),
            name: "".to_string(),
        }
    }
    pub fn new_from_len<TO>(row_len: usize, col_len: usize, optimizer: TO) -> AffineDirectValue
        where TO: Optimizer + 'static
    {
        return AffineDirectValue::new(Array2::<f64>::zeros((row_len, col_len)), optimizer);
    }
    pub fn new_with_name<TO>(value: Array2<f64>, optimizer: TO, name: String)
        -> AffineDirectValue
        where TO: Optimizer + 'static
    {
        AffineDirectValue {
            value: value,
            optimizer: Box::new(optimizer),
            name: name,
        }
    }
    pub fn new_from_len_with_name<TO>(row_len: usize, col_len: usize, optimizer: TO, name: String) -> AffineDirectValue
        where TO: Optimizer + 'static
    {
        return AffineDirectValue::new_with_name(Array2::<f64>::zeros((row_len, col_len)), optimizer, name);
    }
}
impl NetworkLayer for AffineDirectValue {
    fn forward(&mut self, _is_learning: bool) -> Array2<f64> {
        self.value.clone()
    }
    fn backward(&mut self, dout: Array2<f64>) {
        let updated_value = self.optimizer.update(&self.value, &dout);
        self.value.assign(&updated_value);
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
    fn plot(&self) {
        if self.value.shape()[0] == 1 {
            plot_bias_histogram(self.value.clone().into_iter().collect(), &self.name);
        } else {
            plot_histogram(self.value.clone().into_iter().collect(), &self.name);
        };
    }
    fn weight_squared_sum(&self) -> f64 {
        panic!("AffineDirectValue::weight_squared_sum is called");
    }
    fn weight_sum(&self) -> f64 {
        panic!("AffineDirectValue::weight_sum is never weight_sum");
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

#[cfg(test)]
mod test {
    use super::*;
    use ndarray::prelude::{
        arr2,
    };

    #[test]
    fn test_affine_direct_value_backward() {
        let mut mock_optimizer = MockOptimizer::new();
        mock_optimizer.expect_update()
            .returning(|_, value: &_| -> Array2<f64> {
                value.clone()
            })
        ;
        
        let mut affine_direct_value = AffineDirectValue::new_from_len(3, 5, mock_optimizer);

        let expect_value = arr2(&
            [
                [01f64, 02f64, 03f64, 04f64, 05f64],
                [11f64, 12f64, 13f64, 14f64, 15f64],
                [22f64, 22f64, 23f64, 24f64, 25f64],
            ]
        );
        affine_direct_value.backward(expect_value.clone());

        assert_eq!(affine_direct_value.value, expect_value);

    }
}