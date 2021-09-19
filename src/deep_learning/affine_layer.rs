use ndarray::prelude::{
    Array,
    Array1,
    Array2,
    arr1,
    arr2,
};
use rand::Rng;

use crate::deep_learning::common::*;
use crate::deep_learning::network_layers::*;


pub trait NetworkBatchLayer {
    fn forward(&mut self) -> &Array2<f64>;
    fn backward(&mut self, dout: Array2<f64>, diffs: Vec<Array2<f64>>) -> Vec<Array2<f64>>;
    fn set_value(&mut self, value: &Array2<f64>);
    fn set_lbl(&mut self, value: &Array2<f64>);
    fn clean(&mut self);
}


// Direct value
pub struct NetworkBatchValueLayer {
    value: Array2<f64>,
}
impl NetworkBatchValueLayer {
    pub fn new(value: Array2<f64>) -> NetworkBatchValueLayer {
        NetworkBatchValueLayer {
            value: value,
        }
    }
    pub fn new_from_len(row_len: usize, col_len: usize) -> NetworkBatchValueLayer {
        return NetworkBatchValueLayer::new(Array2::<f64>::zeros((row_len, col_len)))
    }
    pub fn get_value(&self) -> &Array2<f64> {&self.value}
}
impl NetworkBatchLayer for NetworkBatchValueLayer {
    fn forward(&mut self) -> &Array2<f64> {
        &self.value
    }
    fn backward(&mut self, dout: Array2<f64>, diffs: Vec<Array2<f64>>) -> Vec<Array2<f64>> {
        let mut mut_diffs = diffs;
        mut_diffs.push(dout);
        return mut_diffs;
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
}


// Affine
pub struct AffineLayer<TX: NetworkBatchLayer, TW: NetworkBatchLayer, TB: NetworkBatchLayer> {
    x: TX,
    w: TW,
    b: TB,
    z: Option<Array2<f64>>, 
}
impl<TX: NetworkBatchLayer, TW: NetworkBatchLayer, TB: NetworkBatchLayer> AffineLayer<TX, TW, TB> {
    pub fn new(x: TX, w: TW, b: TB) -> AffineLayer<TX, TW, TB> {
        AffineLayer {
            x: x,
            w: w,
            b: b,
            z: None,
        }
    }
    pub fn get_x(&self) -> &TX {&self.x}
    pub fn get_w(&self) -> &TW {&self.w}
    pub fn get_b(&self) -> &TB {&self.b}
}
impl<TX: NetworkBatchLayer> AffineLayer<TX, NetworkBatchValueLayer, NetworkBatchValueLayer> {
    pub fn new_random(x: TX, input_len: usize, neuron_len: usize)
            -> AffineLayer<TX, NetworkBatchValueLayer, NetworkBatchValueLayer> {
        let mut rng = rand::thread_rng();

        // Generate initialize weight and biasn by random.
        // -1.0 <= weight < 1.0
        let affine_weight = NetworkBatchValueLayer::new(Array2::<f64>::from_shape_fn(
            (input_len as usize, neuron_len as usize),
            |(_, _)| rng.gen::<f64>()*2.0-1.0
        ));
        // -0.01 <= bias < 0.01
        let affine_bias = NetworkBatchValueLayer::new(Array2::from_shape_fn(
            (1, neuron_len as usize),
            |_| (rng.gen::<f64>()*2.0-1.0) / 100.0
        ));

       return AffineLayer::new(x, affine_weight, affine_bias);
    }
}
impl<TX: NetworkBatchLayer, TW: NetworkBatchLayer, TB: NetworkBatchLayer>
        NetworkBatchLayer for AffineLayer<TX, TW, TB> {
    fn forward(&mut self) -> &Array2<f64> {
        if self.z.is_none() {
            let x = self.x.forward();
            let w = self.w.forward();
            let b = self.b.forward();
            self.z = Some(x.dot(w) + b);
        }
        self.z.as_ref().unwrap()
    }
    fn backward(&mut self, dout: Array2<f64>, diffs: Vec<Array2<f64>>) -> Vec<Array2<f64>> {
        let mut mut_diffs = diffs;

        let w_t = self.w.forward().t();
        mut_diffs.push(dout.dot(&w_t));

        let x_t = self.x.forward().t();
        mut_diffs.push(x_t.dot(&dout));

        mut_diffs.push(self.b.forward() * dout);

        return mut_diffs;
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
        self.z = None;
    }
}


#[cfg(test)]
mod test_affine_mod {
    use super::*;

    #[test]
    fn test_new_random() {
        let x = NetworkBatchValueLayer::new(arr2(&
            [
                [1.0,  2.0],
                [1.0, -2.0]
            ]
        ));

        let affine = AffineLayer::new_random(x, 2, 10);

        assert_eq!(affine.x.value.shape(), [2, 2]);
        assert_eq!(affine.w.value.shape(), [2, 10]);
        assert_eq!(affine.b.value.shape(), [1, 10]);
        // println!("x:\n{}", affine.x.value);
        // println!("w:\n{}", affine.w.value);
        // println!("b:\n{}", affine.b.value);
    }

    #[test]
    fn test_forward() {
        let x = NetworkBatchValueLayer::new(arr2(&
            [
                [1.0,  2.0],
                [1.0, -2.0]
            ]
        ));
        let w = NetworkBatchValueLayer::new(arr2(&
            [
                [ 0.5,  0.2, 1.5],
                [-1.0, -0.5, 2.0]
            ]
        ));
        let b = NetworkBatchValueLayer::new(arr2(&
            [
                [1.0, 2.0, 1.0]
            ]
        ));

        let mut affine = AffineLayer::new(x, w, b);

        let y = affine.forward();
        assert_eq!(y, arr2(&
            [
                [-0.5, 1.2,  6.5],
                [ 3.5, 3.2, -1.5]
            ]
        ));
    }

    #[test]
    fn test_backward() {
        let x = NetworkBatchValueLayer::new(arr2(&
            [
                [1.0,  2.0],
                [1.0, -2.0]
            ]
        ));
        let w = NetworkBatchValueLayer::new(arr2(&
            [
                [ 0.5,  0.2, 1.5],
                [-1.0, -0.5, 2.0]
            ]
        ));
        let b = NetworkBatchValueLayer::new(arr2(&
            [
                [1.0, 2.0, 1.0]
            ]
        ));

        let mut affine = AffineLayer::new(x, w, b);

        let dout = arr2(&
            [
                [ 1.0,   2.0, -1.0],
                [10.0, -20.0,  5.0],
            ]
        );

        let diffs = vec![
            arr2(&
                [
                    [3.0, 1.0, 4.0],
                    [1.0, 5.0, 9.0],
                ]
            )
        ];

        let diffs = affine.backward(dout, diffs);

        assert_eq!(diffs.len(), 4);
        assert_eq!(diffs[0], arr2(&
            [
                [3.0, 1.0, 4.0],
                [1.0, 5.0, 9.0],
            ]
        ));
        assert_eq!(diffs[1], arr2(&
            [
                [-0.6,  -4.0],
                [ 8.5,  10.0],
            ]
        ));
        assert_eq!(diffs[2], arr2(&
            [
                [11.0, -18.0, 4.0],
                [-18.0, 44.0, -12.0],
            ]
        ));
        assert_eq!(diffs[3], arr2(&
            [
                [1.0, 4.0, -1.0],
                [10.0, -40.0, 5.0]
            ]
        ));

    }
}
