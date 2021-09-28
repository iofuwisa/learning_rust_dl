use ndarray::prelude::{
    Array2,
    Axis,
};

use crate::deep_learning::affine_layer::*;
use crate::deep_learning::optimizer::*;
use crate::deep_learning::common::*;

// Affine value(weight and bias)
pub struct NetworkBatchNormValueLayer {
    value: Array2<f64>,
    optimizer: Box<dyn Optimizer>,
}
impl NetworkBatchNormValueLayer {
    pub fn new<TO>(value: Array2<f64>, optimizer: TO) -> Self
        where TO: Optimizer + 'static
    {
        NetworkBatchNormValueLayer {
            value: value,
            optimizer: Box::new(optimizer),
        }
    }
    pub fn new_from_len<TO>(row_len: usize, col_len: usize, optimizer: TO) -> NetworkBatchAffineValueLayer
        where TO: Optimizer + 'static
    {
        return NetworkBatchAffineValueLayer::new(Array2::<f64>::zeros((row_len, col_len)), optimizer);
    }
}
impl NetworkBatchLayer for NetworkBatchNormValueLayer {
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
        // Nothing to do
    }
}

// Batch normalization
pub struct BatchNorm {
    x: Box<dyn NetworkBatchLayer>,
    y: Option<Array2<f64>>, 
    w: Box<dyn NetworkBatchLayer>,
    b: Box<dyn NetworkBatchLayer>,
    normalized: Option<Array2<f64>>,
    distribute: Option<Array2<f64>>,
    average: Option<Array2<f64>>,
}
impl BatchNorm {
    pub fn new<TX, TW, TB>(x: TX, w: TW, b: TB) -> BatchNorm
        where   TX: NetworkBatchLayer + 'static,
                TW: NetworkBatchLayer + 'static,
                TB: NetworkBatchLayer + 'static,
    {
        BatchNorm {
            x: Box::new(x),
            y: None,
            w: Box::new(w),
            b: Box::new(b),
            normalized: None,
            distribute: None,
            average: None,
        }
    }
    pub fn get_x(&self) -> &Box<dyn NetworkBatchLayer> {&self.x}
}
impl NetworkBatchLayer for BatchNorm {
    fn forward(&mut self) -> Array2<f64> {
        if self.y.is_none() {
            let x = self.x.forward();
            let w = self.w.forward();
            let b = self.b.forward();
           
            // Calc average for each column
            let mut average = Array2::<f64>::zeros(x.dim());
            for col in 0..x.shape()[1] {
                let col_x = x.index_axis(Axis(1), col);
                let mut col_average = average.index_axis_mut(Axis(1), col);

                let mut sum = 0f64;
                for n in &col_x {
                    sum += *n;
                }
                let batch_average = sum / col_x.len() as f64;

                col_average.fill(batch_average);
            }

            // Calc distibute
            let avg_diff_squared = (x.clone() - &average) * (x.clone() - &average);
            let mut distribute = Array2::<f64>::zeros(x.dim());
            for col in 0..x.shape()[1] {
                let col_avg_diff_squared = avg_diff_squared.index_axis(Axis(1), col);
                let mut col_distribute = distribute.index_axis_mut(Axis(1), col);

                let mut sum = 0f64;
                for n in &col_avg_diff_squared {
                    sum += *n;
                }
                let batch_distribute = sum / col_avg_diff_squared.len() as f64;

                col_distribute.fill(batch_distribute);
            }

            // Calc normalize
            let normalized = (&x - &average) / sqrt_arr2(&(&distribute + 10f64.powi(-6)));

            // Apply weight and bias
            let y = normalized.clone() * w + b;

            self.y = Some(y);
            self.normalized = Some(normalized);
            self.distribute = Some(distribute);
            self.average = Some(average);

        }
        return self.y.clone().unwrap();
    }

    // refference: https://qiita.com/t-tkd3a/items/14950dbf55f7a3095600
    fn backward(&mut self, dout: Array2<f64>) {
        let x = self.x.forward();
        let w = self.w.forward();
        let normalized = self.normalized.as_ref().unwrap();
        let distribute = self.distribute.as_ref().unwrap();
        let average = self.average.as_ref().unwrap();

        // d15 +
        let d15 = dout.clone();
        
        // d14 Broadcast
        let mut d14 = Array2::<f64>::zeros((1, d15.shape()[1]));
        for col_i in 0..d15.shape()[1] {
            let col_d15 = d15.index_axis(Axis(1), col_i);
            let mut sum = 0f64;
            for n in &col_d15 {
                sum += *n;
            }
            d14[(0, col_i)] = sum;
        }

        // 13 db
        self.b.backward(d14.clone());

        // d12a *
        let d12a = w * &dout;

        // d12b *
        let d12b = normalized * &dout;

        // d11 Broadcast
        let mut d11 = Array2::<f64>::zeros((1, d12b.shape()[1]));
        for col_i in 0..d12b.shape()[1] {
            let col_d12b = d12b.index_axis(Axis(1), col_i);
            let mut sum = 0f64;
            for n in &col_d12b {
                sum += *n;
            }
            d11[(0, col_i)] = sum;
        }

        // 10
        self.w.backward(d11.clone());

        // d9a *
        let d9a =  &d12a /  sqrt_arr2(&(distribute + 10f64.powi(-6)));

        // d9b *
        let d9b = &d12a * (x.clone() - average);

        // d8 Broadcast
        let mut d8 = Array2::<f64>::zeros((1, d9b.shape()[1]));
        for col_i in 0..d9b.shape()[1] {
            let col_d9b = d9b.index_axis(Axis(1), col_i);
            let mut sum = 0f64;
            for n in &col_d9b {
                sum += *n;
            }
            d8[(0, col_i)] = sum;
        }

        // d7 1/x
        let d7 = d8 * (-1.0 / (distribute + 10f64.powi(-6)));

        // d6 sqrt(x)
        let d6 = d7 / (2.0 * sqrt_arr2(&(distribute + 10f64.powi(-6).sqrt())));

        // d5 avg
        let d5 = d6 / x.shape()[0] as f64;

        // d4 ^2
        let d4 = d5 * (x.clone() - average);
        
        //d3a -
        let d3a = d9a + d4;

        //d3b -
        let d3b = -&d3a;

        // d2 Broadcast
        let mut d2 = Array2::<f64>::zeros((1, d3b.shape()[1]));
        for col_i in 0..d3b.shape()[1] {
            let mut sum = 0f64;
            for row_i in 0..d3b.shape()[0] {
                sum += d3b[(row_i, col_i)];
            }
            d2[(0, col_i)] = sum;
        }

        // d1
        let d1 = d2 /  x.shape()[0] as f64;
        
        let dx = &d3a + d1;

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
    fn prot(&self){
        self.x.prot();
    }
}



#[cfg(test)]
mod batch_norm_test {
    use super::*;

    use crate::deep_learning::mnist::*;
    use crate::deep_learning::neural_network::*;
    use crate::deep_learning::graph_plotter::*;

    #[test]
    fn test_forward() {
        let mut batch_norm = BatchNorm::new(
            NetworkBatchAffineValueLayer::new_from_len(100, 28*28, Sgd::new(0.01)),
            NetworkBatchNormValueLayer::new(
                Array2::<f64>::ones((100, 28*28)),
                Sgd::new(0.01)
            ),
            NetworkBatchNormValueLayer::new(
                Array2::<f64>::zeros((100, 28*28)),
                Sgd::new(0.01)
            ),
        );

        let mnist = MnistImages::new(1000, 1, 1);
        let trn_img = mnist.get_trn_img();
        let trn_lbl_onehot = mnist.get_trn_lbl_one_hot();

        let (batch_data, _) = make_minibatch_data(100, &trn_img, &trn_lbl_onehot);
        batch_norm.set_value(&batch_data);

        prot_histogram(batch_data.clone().into_iter().collect(), "test_batch_norm_forward_before");

        let y = batch_norm.forward();

        prot_histogram(y.clone().into_iter().collect(), "test_batch_norm_forward_after");
    }
}