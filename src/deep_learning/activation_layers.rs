use std::f64::consts::E;
use ndarray::prelude::{
    Array2,
};

use crate::deep_learning::affine_layer::*;

// Relu
// y = x (x > 0)
// y = 0 (x <= 0)
pub struct ReluLayer<T: NetworkBatchLayer> {
    x: T,
    y: Option<Array2<f64>>, 
}
impl<T: NetworkBatchLayer> ReluLayer<T> {
    pub fn new(x: T) -> ReluLayer<T> {
        ReluLayer {
            x: x,
            y: None,
        }
    }
}
impl<T: NetworkBatchLayer> NetworkBatchLayer for ReluLayer<T> {
    fn forward(&mut self) -> &Array2<f64> {
        if self.y.is_none() {
            let x = self.x.forward();
            self.y = Some(x.mapv(|n: f64| -> f64{if n > 0.0 {n} else {0.0}}));
        }
        return self.y.as_ref().unwrap();
    }
    fn backward(&mut self, dout: Array2<f64>, diffs: Vec<Array2<f64>>) -> Vec<Array2<f64>> {
        let x = self.x.forward();
        if dout.shape() != x.shape() {
            panic!("Different shape. dout: {:?} x: {:?}", dout.shape(), x.shape());
        }

        let mut diffs = diffs;
        let mut iter_dout = dout.iter();
        let dout = x.mapv(|n: f64| -> f64 {
            let d = iter_dout.next().unwrap();
            if n > 0.0 {    // z = x (x > 0)
                *d
            } else {        // z = 0 (x <= 0)
                0.0
            }
        });

        diffs = self.x.backward(dout, diffs);

        return diffs;
    }
}


// Sigmoid
// y = 1 / (1 + exp(-x))
pub struct SigmoidLayer<T: NetworkBatchLayer> {
    x: T,
    y: Option<Array2<f64>>,
}
impl<T: NetworkBatchLayer> SigmoidLayer<T> {
    pub fn new(x: T) -> SigmoidLayer<T> {
        SigmoidLayer {
            x: x,
            y: None,
        }
    }
}
impl<T: NetworkBatchLayer> NetworkBatchLayer for SigmoidLayer<T> {
    // f(x) =  1 / (1 + exp(-x))
    fn forward(&mut self) -> &Array2<f64> {
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
        return self.y.as_ref().unwrap();
    }
    // f(x)' = (1 - f(x)) f(x)
    fn backward(&mut self, dout: Array2<f64>, diffs: Vec<Array2<f64>>) -> Vec<Array2<f64>> {
        let fx = self.forward();
        if dout.shape() != fx.shape() {
            panic!("Different shape. dout: {:?} fx:{:?}", dout.shape(), fx.shape());
        }

        let mut diffs = diffs;
        let mut iter_dout = dout.iter();
        let dout = fx.mapv(|n: f64| -> f64 {
            let d = iter_dout.next().unwrap();
            // (1 - f(x)) f(x)
            return d * (1.0 - n) * n
        });

        diffs = self.x.backward(dout, diffs);

        return diffs;
    }
}


#[cfg(test)]
mod test_relu_mod {
    use super::*;
    use ndarray::prelude::{
        Array2,
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

        let diffs = relu.backward(dout, Vec::<Array2<f64>>::new());

        assert_eq!(diffs.len(), 1);
        assert_eq!(diffs[0], arr2(&
            [
                [1.0, 0.0, 0.0],
                [4.0, 5.0, 0.0],
            ]
        ));
    }
}

#[cfg(test)]
mod test_sigmoid_mod {
    use super::*;
    use crate::deep_learning::common::*;
    use ndarray::prelude::{
        Array2,
        arr2,
    };

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
        let value = NetworkBatchValueLayer::new(arr2(&
            [
                [2.0, -3.0, 0.0],
                [1.0, 0.1, -0.1],
            ]
        ));
        let mut sigmoid = SigmoidLayer::new(value);
        let dout = arr2(&
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ]
        );

        let diffs = sigmoid.backward(dout.clone(), Vec::<Array2<f64>>::new());

        assert_eq!(diffs.len(), 1);
        let sigmoid_diff = arr2(&
            [
                [0.10501362071, 0.04517665972, 0.25],
                [0.19661193324, 0.24937604019, 0.24937600585],
            ]
        );
        assert_eq!(round_digit_arr2(&diffs[0], -4), round_digit_arr2(&(dout*sigmoid_diff), -4));
    } 
}