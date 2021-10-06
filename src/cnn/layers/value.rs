use ndarray::prelude::{
    Array4,
};

use super::cnn_layer::*;


pub struct Value {
    value: Array4<f64>,
}
impl Value {
    pub fn new(value: Array4<f64>) -> Self {
        Self {
            value: value,
        }
    }
}
impl CnnLayer for Value {
    fn forward(&mut self, _is_learning: bool) -> Array4<f64> {
        return self.value.clone();
    }

    fn backward(&mut self, _dout: Array4<f64>) {

    }
}


#[cfg(test)]
mod test {
    use super::*;
    
    use ndarray::prelude::{
        Array,
    };

    #[test]
    fn test_forward() {
        let arr_value = Array::from_shape_vec(
            (2,2,3,3),
            vec![
                1f64, 2f64, 3f64, 4f64, 5f64, 6f64, 7f64, 8f64, 9f64, 0f64,
                1f64, 2f64, 3f64, 4f64, 5f64, 6f64, 7f64, 8f64, 9f64, 0f64,
                1f64, 2f64, 3f64, 4f64, 5f64, 6f64, 7f64, 8f64, 9f64, 0f64,
                1f64, 2f64, 3f64, 4f64, 5f64, 6f64,
            ]
        ).ok().unwrap();
        let mut value = Value::new(arr_value.clone());

        let foreard_res = value.forward(false);

        assert_eq!(foreard_res, arr_value);
    }
}