use std::f64::consts::E;
use ndarray::prelude::{
    Array2,
};

use crate::deep_learning::affine_layer::*;

// Relu
// y = x (x > 0)
// y = 0 (x <= 0)
pub struct ReluLayer {
    x: Box<dyn NetworkBatchLayer>,
    y: Option<Array2<f64>>, 
}
impl ReluLayer {
    pub fn new<TX>(x: TX) -> ReluLayer
        where TX: NetworkBatchLayer + 'static {
        ReluLayer {
            x: Box::new(x),
            y: None,
        }
    }
    pub fn get_x(&self) -> &Box<dyn NetworkBatchLayer> {&self.x}
}
impl NetworkBatchLayer for ReluLayer {
    fn forward(&mut self) -> Array2<f64> {
        if self.y.is_none() {
            let x = self.x.forward();
            self.y = Some(x.mapv(|n: f64| -> f64{if n > 0.0 {n} else {0.0}}));
        }
        return self.y.clone().unwrap();
    }
    fn backward(&mut self, dout: Array2<f64>) {
        let x = self.x.forward();
        if dout.shape() != x.shape() {
            panic!("Different shape. dout: {:?} x: {:?}", dout.shape(), x.shape());
        }

        let mut iter_dout = dout.iter();
        let dx = x.mapv(|n: f64| -> f64 {
            let d = iter_dout.next().unwrap();
            if n > 0.0 {    // z = x (x > 0)
                *d
            } else {        // z = 0 (x <= 0)
                0.0
            }
        });

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
    }
}


// Sigmoid
// y = 1 / (1 + exp(-x))
pub struct SigmoidLayer {
    x: Box<dyn NetworkBatchLayer>,
    y: Option<Array2<f64>>,
}
impl SigmoidLayer {
    pub fn new<TX>(x: TX) -> SigmoidLayer
    where TX: NetworkBatchLayer + 'static {
        SigmoidLayer {
            x: Box::new(x),
            y: None,
        }
    }
    pub fn get_x(&self) -> &Box<dyn NetworkBatchLayer> {&self.x}
}
impl NetworkBatchLayer for SigmoidLayer {
    // f(x) =  1 / (1 + exp(-x))
    fn forward(&mut self) -> Array2<f64> {
        if self.y.is_none() {
            // -x
            let x = self.x.forward() * -1.0;
            // exp(-x)
            let x = x.mapv(|n: f64| -> f64 {E.powf(n)});
            // 1 + exp(-x)
            let x = x + 1.0;
            // 1 / (1 + exp(-x))
            let y = 1.0 / x;

            self.y = Some(y);
        }
        return self.y.clone().unwrap();
    }
    // f(x)' = (1 - f(x)) f(x)
    fn backward(&mut self, dout: Array2<f64>) {
        let fx = self.forward();
        if dout.shape() != fx.shape() {
            panic!("Different shape. dout: {:?} fx:{:?}", dout.shape(), fx.shape());
        }

        let mut iter_dout = dout.iter();
        let dx = fx.mapv(|n: f64| -> f64 {
            let d = iter_dout.next().unwrap();
            // (1 - f(x)) f(x)
            return d * (1.0 - n) * n
        });

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
    }
}


#[cfg(test)]
mod test_relu_mod {
    use super::*;

    use ndarray::prelude::{
        arr2,
    };

    #[test]
    fn test_forward() {
        let value = NetworkBatchValueLayer::new(arr2(&
            [
                [2.0, -3.0, 0.0],
                [1.0, 0.1, -0.12345],
            ]
        ));
        let mut relu = ReluLayer::new(value);

        let relu_res = relu.forward();
        
        assert_eq!(relu_res, arr2(&
            [
                [2.0, 0.0, 0.0],
                [1.0, 0.1, 0.0],
            ]
        ));
    }

    #[test]
    fn test_backward() {
        let value = NetworkBatchValueLayer::new(arr2(&
            [
                [2.0, -3.0, 0.0],
                [1.0, 0.1, -0.12345],
            ]
        ));
        let mut relu = ReluLayer::new(value);
        let dout = arr2(&
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ]
        );

        relu.backward(dout);
    }
}

#[cfg(test)]
mod test_sigmoid_mod {
    use super::*;

    use ndarray::prelude::{
        arr2,
    };

    use crate::deep_learning::common::*;
    use crate::deep_learning::optimizer::*;

    #[test]
    fn test_forward() {
        let value = NetworkBatchValueLayer::new(arr2(&
            [
                [2.0, -3.0, 0.0],
                [1.0, 0.1, -0.1],
            ]
        ));
        let mut sigmoid = SigmoidLayer::new(value);

        let sigmoid_res = sigmoid.forward();
        
        assert_eq!(round_digit_arr2(&sigmoid_res, -4), round_digit_arr2(&arr2(&
            [
                [0.88077077, 0.04742587317, 0.5],
                [0.73105857863, 0.52497918747, 0.4750201252],
            ]
        ), -4));
    }

    #[test]
    fn test_backward() {
        let arr2_value = arr2(&
            [
                [2.0, -3.0, 0.0],
                [1.0, 0.1, -0.1],
            ]
        );
        // Use NetworkBatchAffineValueLayer to check side effects
        let value = NetworkBatchAffineValueLayer::new(
            arr2_value.clone(),
            Sgd::new(0.01)
        );
        let mut sigmoid = SigmoidLayer::new(value);
        let dout = arr2(&
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ]
        );

        sigmoid.backward(dout.clone());

        assert_eq!(
            round_digit_arr2(&sigmoid.x.forward(), -4),
            // (1 - f(x)) f(x)
            round_digit_arr2(&(arr2_value.clone()-(((1.0-sigmoid.forward())*sigmoid.forward())*dout*0.01)), -4)
        );
    } 
}