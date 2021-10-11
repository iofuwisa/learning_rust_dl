use ndarray::prelude::{
    Array2,
};
use rand::Rng;

use crate::deep_learning::affine_layer::*;


// Dropout
pub struct DropoutLayer {
    x: Box<dyn NetworkBatchLayer>,
    y: Option<Array2<f64>>,
    mask: Option<Array2<f64>>,
    dropout_rate: f64,
    is_learning: bool,
}
impl DropoutLayer {
    pub fn new<TX>(x: TX, dropout_rate: f64) -> DropoutLayer
    where TX: NetworkBatchLayer + 'static {
        DropoutLayer {
            x: Box::new(x),
            y: None,
            mask: None,
            dropout_rate: dropout_rate,
            is_learning: false,
        }
    }
    pub fn get_x(&self) -> &Box<dyn NetworkBatchLayer> {&self.x}
}
impl NetworkBatchLayer for DropoutLayer {
    fn forward(&mut self, is_learning: bool) -> Array2<f64> {
        if self.y.is_none() || self.is_learning != is_learning {
            self.is_learning = is_learning;
            let x = self.x.forward(is_learning);
            if is_learning {
                let mut rng = rand::thread_rng();
                let mask = Array2::from_shape_fn(x.dim(),
                    |_| -> f64 {
                        if rng.gen::<f64>() > self.dropout_rate {
                            1f64
                        } else {
                            0f64
                        }
                    }
                );
                let y = &x * &mask;
    
                self.y = Some(y);
                self.mask = Some(mask);
            } else {
                let y = &x * (1f64 - self.dropout_rate);
                self.y = Some(y);
            }
        }
        return self.y.clone().unwrap();
    }
    fn backward(&mut self, dout: Array2<f64>) {
        self.forward(true);
        let mask = self.mask.as_ref().unwrap();
        let dx = dout * mask;
        self.x.backward(dx);
    }
    fn set_value(&mut self, value: &Array2<f64>) {
        self.x.set_value(value);
        self.clean();
    }
    fn set_lbl(&mut self, value: &Array2<f64>) {
        self.x.set_lbl(value);
        self.clean();
    }
    fn clean(&mut self) {
        self.y = None;
        self.mask = None;
    }
    fn plot(&self){
        self.x.plot();
    }
    fn weight_squared_sum(&self) -> f64 {
        return self.x.weight_squared_sum();
    }
    fn weight_sum(&self) -> f64 {
        return self.x.weight_sum();
    }
}