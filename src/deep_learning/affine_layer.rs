use ndarray::prelude::{
    Array,
    Array1,
    Array2,
    ArrayView1,
    Axis,
    arr1,
    arr2,
};
use rand::Rng;

use crate::deep_learning::common::*;
use crate::deep_learning::network_layers::*;


pub trait NetworkBatchLayer {
    fn forward(&mut self) -> Array2<f64>;
    fn forward_skip_loss(&mut self) -> Array2<f64> {self.forward()}
    fn backward(&mut self, dout: Array2<f64>, learning_rate: f64);
    fn set_value(&mut self, value: &Array2<f64>);
    fn set_lbl(&mut self, value: &Array2<f64>);
    fn clean(&mut self);
}

impl dyn NetworkBatchLayer {
    fn is_loss_layer(&self) -> bool {
        false
    }
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
    fn forward(&mut self) -> Array2<f64> {
        self.value.clone()
    }
    fn backward(&mut self, dout: Array2<f64>, learning_rate: f64) {
        // Nothing to do
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

// Affine value(weight and bias)
pub struct NetworkBatchAffineValueLayer {
    value: Array2<f64>,
}
impl NetworkBatchAffineValueLayer {
    pub fn new(value: Array2<f64>) -> NetworkBatchAffineValueLayer {
        // println!("init value:\n{:?}\n\n", value);
        NetworkBatchAffineValueLayer {
            value: value,
        }
    }
    pub fn new_from_len(row_len: usize, col_len: usize) -> NetworkBatchAffineValueLayer {
        return NetworkBatchAffineValueLayer::new(Array2::<f64>::zeros((row_len, col_len)))
    }
    pub fn get_value(&self) -> &Array2<f64> {&self.value}
}
impl NetworkBatchLayer for NetworkBatchAffineValueLayer {
    fn forward(&mut self) -> Array2<f64> {
        self.value.clone()
    }
    fn backward(&mut self, dout: Array2<f64>, learning_rate: f64) {
        // println!("dout:\n{:?}", dout);
        self.value = &self.value - (dout * learning_rate);
        // println!("value:\n{:?}", self.value);
        // println!("\n\n");
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
pub struct AffineLayer {
    x: Box<dyn NetworkBatchLayer>,
    w: Box<dyn NetworkBatchLayer>,
    b: Box<dyn NetworkBatchLayer>,
    z: Option<Array2<f64>>, 
}
impl AffineLayer {
    pub fn new<TX, TW, TB>(x: TX, w: TW, b: TB) -> AffineLayer
        where   TX: NetworkBatchLayer + 'static,
                TW: NetworkBatchLayer + 'static,
                TB: NetworkBatchLayer + 'static
    {
        AffineLayer {
            x: Box::new(x),
            w: Box::new(w),
            b: Box::new(b),
            z: None,
        }
    }
    pub fn new_random<TX>(x: TX, input_len: usize, neuron_len: usize)-> AffineLayer
        where   TX: NetworkBatchLayer + 'static
    {
        let mut rng = rand::thread_rng();

        // Generate initialize weight and biasn by random.
        // -1.0 <= weight < 1.0
        let affine_weight = NetworkBatchAffineValueLayer::new(Array2::<f64>::from_shape_fn(
            (input_len as usize, neuron_len as usize),
            |(_, _)| rng.gen::<f64>()*2.0-1.0
        ));
        // -0.01 <= bias < 0.01
        let affine_bias = NetworkBatchAffineValueLayer::new(Array2::from_shape_fn(
            (1, neuron_len as usize),
            |_| (rng.gen::<f64>()*2.0-1.0) / 100.0
        ));

       return AffineLayer::new(x, affine_weight, affine_bias);
    }
    pub fn get_x(&self) -> &Box<dyn NetworkBatchLayer> {&self.x}
    pub fn get_w(&self) -> &Box<dyn NetworkBatchLayer> {&self.w}
    pub fn get_b(&self) -> &Box<dyn NetworkBatchLayer> {&self.b}
}
impl NetworkBatchLayer for AffineLayer {
    fn forward(&mut self) -> Array2<f64> {
        if self.z.is_none() {
            let x = self.x.forward();
            let w = self.w.forward();
            let b = self.b.forward();
            self.z = Some(x.dot(&w) + b);
        }
        self.z.clone().unwrap()
    }
    fn backward(&mut self, dout: Array2<f64>, learning_rate: f64) {
        let w = self.w.forward();
        let w_t = w.t();
        let dx = dout.dot(&w_t);
        self.x.backward(dx, learning_rate);

        let x = self.x.forward();
        let x_t = x.t();
        let dw = x_t.dot(&dout);
        self.w.backward(dw, learning_rate);

        let mut db = Array2::<f64>::zeros((1, dout.shape()[1]));
        for col_i in 0..dout.shape()[1] {
            let mut sum = 0.0;
            for row_i in 0..dout.shape()[0] {
                sum += dout[(row_i, col_i)];
            }
            db[(0, col_i)] = sum / dout.shape()[1] as f64;
        }
        self.b.backward(db, learning_rate);
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

        let mut affine = AffineLayer::new_random(x, 2, 10);

        assert_eq!(affine.x.forward().shape(), [2, 2]);
        assert_eq!(affine.w.forward().shape(), [2, 10]);
        assert_eq!(affine.b.forward().shape(), [1, 10]);
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

        let diffs = affine.backward(dout, 0.01);
    }
}
