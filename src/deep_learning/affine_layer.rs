use ndarray::prelude::{
    Array2,
};
use rand::Rng;

use crate::deep_learning::optimizer::*;
use crate::deep_learning::common::*;
use crate::deep_learning::graph_plotter::*;

pub trait NetworkBatchLayer {
    fn forward(&mut self) -> Array2<f64>;
    fn forward_skip_loss(&mut self) -> Array2<f64> {self.forward()}
    fn backward(&mut self, _dout: Array2<f64>) {}
    fn set_value(&mut self, value: &Array2<f64>);
    fn set_lbl(&mut self, value: &Array2<f64>);
    fn clean(&mut self);
    fn is_loss_layer(&self) -> bool {false}
    fn prot(&self);
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
}
impl NetworkBatchLayer for NetworkBatchValueLayer {
    fn forward(&mut self) -> Array2<f64> {
        self.value.clone()
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
    fn prot(&self){
        // Nothing to do
    }
}

// Affine value(weight and bias)
pub struct NetworkBatchAffineValueLayer {
    value: Array2<f64>,
    optimizer: Box<dyn Optimizer>,
    name: String,
}
impl NetworkBatchAffineValueLayer {
    pub fn new<TO>(value: Array2<f64>, optimizer: TO)
        -> NetworkBatchAffineValueLayer
        where TO: Optimizer + 'static
    {
        NetworkBatchAffineValueLayer {
            value: value,
            optimizer: Box::new(optimizer),
            name: "".to_string(),
        }
    }
    pub fn new_from_len<TO>(row_len: usize, col_len: usize, optimizer: TO) -> NetworkBatchAffineValueLayer
        where TO: Optimizer + 'static
    {
        return NetworkBatchAffineValueLayer::new(Array2::<f64>::zeros((row_len, col_len)), optimizer);
    }
    pub fn new_with_name<TO>(value: Array2<f64>, optimizer: TO, name: String)
        -> NetworkBatchAffineValueLayer
        where TO: Optimizer + 'static
    {
        NetworkBatchAffineValueLayer {
            value: value,
            optimizer: Box::new(optimizer),
            name: name,
        }
    }
    pub fn new_from_len_with_name<TO>(row_len: usize, col_len: usize, optimizer: TO, name: String) -> NetworkBatchAffineValueLayer
        where TO: Optimizer + 'static
    {
        return NetworkBatchAffineValueLayer::new_with_name(Array2::<f64>::zeros((row_len, col_len)), optimizer, name);
    }
}
impl NetworkBatchLayer for NetworkBatchAffineValueLayer {
    fn forward(&mut self) -> Array2<f64> {
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
    fn prot(&self) {
        prot_histogram(self.value.clone().into_iter().collect(), &self.name);
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
    pub fn new<TX, TW, TB>(x: TX, w: TW, b: TB)
        -> AffineLayer
        where   TX : NetworkBatchLayer + 'static,
                TW : NetworkBatchLayer + 'static,
                TB : NetworkBatchLayer + 'static,
    {
        AffineLayer {
            x: Box::new(x),
            w: Box::new(w),
            b: Box::new(b),
            z: None,
        }
    }
    pub fn new_random<TX, TWO, TBO>(x: TX, input_len: usize, neuron_len: usize, optimizer_w: TWO, optimizer_b: TBO)
        -> AffineLayer
        where   TX : NetworkBatchLayer + 'static,
                TWO: Optimizer + 'static,
                TBO: Optimizer + 'static
    {
        let mut rng = rand::thread_rng();

        // Generate initialize weight and biasn by normal distibution
        let affine_weight = NetworkBatchAffineValueLayer::new(
            Array2::from_shape_vec(
                (input_len as usize, neuron_len as usize),
                norm_random_vec(input_len * neuron_len)
            ).ok().unwrap(),
            optimizer_w    
        );
        let mut norm_bias_iter = norm_random_vec(neuron_len).into_iter();
        let affine_bias = NetworkBatchAffineValueLayer::new(
            Array2::from_shape_vec(
                (1, neuron_len as usize),
                norm_random_vec(neuron_len)
                    .into_iter()
                    .map(|x: f64| {x / 100.0})
                    .collect()
            ).ok().unwrap(),
            optimizer_b
        );

       return AffineLayer::new(x, affine_weight, affine_bias);
    }
    pub fn new_random_with_name<TX, TWO, TBO>(x: TX, input_len: usize, neuron_len: usize, optimizer_w: TWO, optimizer_b: TBO, name: String)
    -> AffineLayer
    where   TX : NetworkBatchLayer + 'static,
            TWO: Optimizer + 'static,
            TBO: Optimizer + 'static
{
    let mut rng = rand::thread_rng();

    // Generate initialize weight and biasn by normal distibution
    let affine_weight = NetworkBatchAffineValueLayer::new_with_name(
        Array2::from_shape_vec(
            (input_len as usize, neuron_len as usize),
            norm_random_vec(input_len * neuron_len)
        ).ok().unwrap(),
        optimizer_w,
        name.clone() + "_weight",
    );
    let mut norm_bias_iter = norm_random_vec(neuron_len).into_iter();
    let affine_bias = NetworkBatchAffineValueLayer::new_with_name(
        Array2::from_shape_vec(
            (1, neuron_len as usize),
            norm_random_vec(neuron_len)
                .into_iter()
                .map(|x: f64| {x / 100.0})
                .collect()
        ).ok().unwrap(),
        optimizer_b,
        name.clone() + "_bias",
    );

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
    fn backward(&mut self, dout: Array2<f64>) {
        let w = self.w.forward();
        let w_t = w.t();
        let dx = dout.dot(&w_t);
        self.x.backward(dx,);

        let x = self.x.forward();
        let x_t = x.t();
        let dw = x_t.dot(&dout);
        self.w.backward(dw);

        let mut db = Array2::<f64>::zeros((1, dout.shape()[1]));
        for col_i in 0..dout.shape()[1] {
            let mut sum = 0.0;
            for row_i in 0..dout.shape()[0] {
                sum += dout[(row_i, col_i)];
            }
            db[(0, col_i)] = sum / dout.shape()[1] as f64;
        }
        self.b.backward(db);
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
    fn prot(&self){
        self.x.prot();
        self.w.prot();
        self.b.prot();
    }
}


#[cfg(test)]
mod test_affine_mod {
    use super::*;

    use ndarray::prelude::{
        arr2,
    };

    #[test]
    fn test_new_random() {
        let x = NetworkBatchValueLayer::new(arr2(&
            [
                [1.0,  2.0],
                [1.0, -2.0]
            ]
        ));

        let mut affine = AffineLayer::new_random(
            x,
            2,
            10,
            Sgd::new(0.01),
            Sgd::new(0.01)
        );

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

        let diffs = affine.backward(dout);
    }
}
