use std::f64::consts::E;
use ndarray::prelude::{
    Array1,
    arr1,
};
use ndarray::iter::{
    Iter,
};

use crate::deep_learning::common::*;

use crate::deep_learning::network_layers::*;

// Relu
// z = x (x > 0)
// z = 0 (x <= 0)
pub struct ReluLayer<T: NetworkLayer, U: NetworkLayer> {
    x: T,
    _y: Option<U>,
    z: Option<Array1<f64>>, 
}
impl<T: NetworkLayer> ReluLayer<T, NetworkNoneValue> {
    pub fn new(x: T) -> ReluLayer<T, NetworkNoneValue> {
        ReluLayer {
            x: x,
            _y: None,
            z: None,
        }
    }
}
impl<T: NetworkLayer, U: NetworkLayer> NetworkLayer for ReluLayer<T, U> {
    fn forward(&mut self) -> &Array1<f64> {
        if self.z.is_none() {
            let x = self.x.forward();
            self.z = Some(x.mapv(|n: f64| -> f64{if n > 0.0 {n} else {0.0}}));
        }
        return self.z.as_ref().unwrap();
    }
    fn backward(&mut self, dout: Array1<f64>, diffs: Vec<Array1<f64>>) -> Vec<Array1<f64>> {
        let x = self.x.forward();
        if dout.len() != x.len() {
            panic!("Different len. dout: {} x: {}", dout.len(), x.len());
        }

        let mut diffs = diffs;
        let mut iter_dout = dout.iter();
        diffs.push(x.mapv(|n: f64| -> f64 {
            let d = iter_dout.next().unwrap();
            if n > 0.0 {    // z = x (x > 0)
                *d
            } else {        // z = 0 (x <= 0)
                0.0
            }
        }));
        return diffs;
    }
}


// Sigmoid
// z = 1 / (1 + exp(-x))
pub struct SigmoidLayer<T: NetworkLayer, U: NetworkLayer> {
    x: T,
    _y: Option<U>,
    z: Option<Array1<f64>>, 
}
impl<T: NetworkLayer> SigmoidLayer<T, NetworkNoneValue> {
    pub fn new(x: T) -> SigmoidLayer<T, NetworkNoneValue> {
        SigmoidLayer {
            x: x,
            _y: None,
            z: None,
        }
    }
}
impl<T: NetworkLayer, U: NetworkLayer> NetworkLayer for SigmoidLayer<T, U> {
    fn forward(&mut self) -> &Array1<f64> {
        if self.z.is_none() {
            // -x
            let x = self.x.forward() * -1.0;
            // exp(-x)
            let x = x.mapv(|n: f64| -> f64 {E.powf(n)});
            // 1 + exp(-x)
            let x = x + 1.0;
            // 1 / (1 + exp(-x))
            let z = 1.0 / x;

            self.z = Some(z);
        }
        return self.z.as_ref().unwrap();
    }
    // f(x) =  1 / (1 + exp(-x))
    // f(x)' = (1 - f(x)) f(x)
    fn backward(&mut self, dout: Array1<f64>, diffs: Vec<Array1<f64>>) -> Vec<Array1<f64>> {
        let z = self.forward();
        if dout.len() != z.len() {
            panic!("Different len. dout: {} z:{}", dout.len(), z.len());
        }

        let mut diffs = diffs;
        let mut iter_dout = dout.iter();
        diffs.push(z.mapv(|n: f64| -> f64 {
            let d = iter_dout.next().unwrap();
            // (1 - f(x)) f(x)
            return d * (1.0 - n) * n
        }));
        return diffs;
    }
}


#[cfg(test)]
mod test_relu_mod {
    use super::*;

    #[test]
    fn test_forward() {
        let v1 = NetworkValueLayer::new(arr1(&[2.0, -3.0, 0.0]));
        let mut r1 = ReluLayer::new(v1);
        
        assert_eq!(r1.forward(), arr1(&[2.0, 0.0, 0.0]));
        assert_eq!(r1.z.unwrap(), arr1(&[2.0, 0.0, 0.0]));
    }
    #[test]
    fn test_backward() {
        let mut v1 = NetworkValueLayer::new(arr1(&[2.0, -3.0, 0.0]));
        let mut r1 = ReluLayer::new(v1);
        let dout = arr1(&[-4.0, 0.0, -4.0]);

        let diffs = r1.backward(dout, Vec::<Array1<f64>>::new());

        assert_eq!(diffs.len(), 1);
        assert_eq!(diffs[0], arr1(&[-4.0, 0.0, 0.0]));
    }
}

#[cfg(test)]
mod test_sigmoid_mod {
    use super::*;

    #[test]
    fn test_forward() {
        let v1 = NetworkValueLayer::new(arr1(&[2.0, -3.0, 0.0]));
        let mut s1 = SigmoidLayer::new(v1);
        
        assert_eq!(round_digit_arr1(&s1.forward(), -4), arr1(&[0.8808, 0.047400000000000005, 0.5]));
        assert_eq!(round_digit_arr1(s1.z.as_ref().unwrap(), -4), arr1(&[0.8808, 0.047400000000000005, 0.5]));
    }
    #[test]
    fn test_backward() {
        let mut v1 = NetworkValueLayer::new(arr1(&[2.0, -3.0, 0.0]));
        let mut r1 = SigmoidLayer::new(v1);
        let dout = arr1(&[1.0, 3.0, -4.0]);

        let diffs = r1.backward(dout, Vec::<Array1<f64>>::new());

        // (1 - f(x)) f(x)
        let f = [0.8807970779778824440597, 0.04742587317756678087885, 0.5];
        assert_eq!(diffs.len(), 1);
        assert_eq!(round_digit_arr1(&diffs[0], -4), round_digit_arr1(&arr1(&[(1.0-f[0])*f[0]*1.0, (1.0-f[1])*f[1]*3.0, (1.0-f[2])*f[2]*-4.0]), -4));
    } 
}