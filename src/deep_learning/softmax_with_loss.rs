use std::f64::consts::E;
use ndarray::prelude::{
    Array,
    Array1,
    Array2,
    ArrayView1,
    arr1,
    arr2,
    s,
};
use ndarray::Axis;
use ndarray::Dim;

use crate::deep_learning::affine_layer::*;
use crate::deep_learning::common::*;


// Softmax with loss
pub struct SoftmaxWithLoss<TX: NetworkBatchLayer> {
    x: TX,
    t: Array2<f64>,
    z: Option<Array2<f64>>, 
}
impl<TX: NetworkBatchLayer> SoftmaxWithLoss<TX> {
    pub fn new(x: TX, t: Array2<f64>) -> SoftmaxWithLoss<TX> {
        SoftmaxWithLoss {
            x: x,
            t: t,
            z: None,
        }
    }
}
impl<TX: NetworkBatchLayer> NetworkBatchLayer for SoftmaxWithLoss<TX> {
    fn forward(&mut self) -> &Array2<f64> {
        if self.z.is_none() {
            
            let x = self.x.forward();

            let softmax_res = softmax(x);

            let z = crosss_entropy_error(&softmax_res, &self.t);

            self.z = Some(z);
        }
        self.z.as_ref().unwrap()
    }
    fn backward(&mut self, dout: Array2<f64>, diffs: Vec<Array2<f64>>) -> Vec<Array2<f64>> {
        // y - t
        let d = dout * (self.forward().to_owned() - &self.t);

        let diffs = self.x.backward(d, diffs);

        return diffs;
    }
}

fn softmax(x: &Array2<f64>) -> Array2<f64> {
    // Create Array same shape from x
    let mut z = Array2::<f64>::zeros(x.dim());
    // let mut z = x.clone();

    for row_i in 0..x.shape()[0] {
        // Get row
        let r = x.index_axis(Axis(0), row_i).to_owned();

        // Find max
        let max_index = max_index_in_arr1(&r);
        let max = r[max_index];

        // Σ(exp(ai + c)
        let mut sum_exp_a = 0.0;
        for ai in &r {
            sum_exp_a += E.powf(*ai + max);
        }

        // exp(ak + c)/Σ(exp(ai + c))
        for col_i in 0..x.shape()[1] {
            z[(row_i, col_i)] = E.powf(r[col_i] + max) / sum_exp_a;
        }
    }
    return z;
}

fn crosss_entropy_error(x: &Array2<f64>, t: &Array2<f64>) -> Array2<f64> {
    if x.len() != t.len() {
        panic!("Different shape. x:{:?} t:{:?}", x.shape(), t.shape());
    }

    // Create Array same len with row len x has
    let mut z = Array2::<f64>::zeros([x.shape()[0], 1]);

    for row_i in 0..x.shape()[0] {
        // Get row
        let x_row = x.index_axis(Axis(0), row_i).to_owned();
        let t_row = t.index_axis(Axis(0), row_i).to_owned();

        // Find correct label index
        let correct_index = max_index_in_arr1(&t_row);

        z[(row_i, 0)] = x_row[correct_index].log(E) * -1.0;
    }
    return z;
}

#[cfg(test)]
mod test_softmax_with_loss_mod {
    use super::*;

    #[test]
    fn test_softmax() {
        let x = arr2(&
            [
                [2.0,   5.0, 3.0,  3.0],
                [0.0, -10.0, 7.0, 12.0],
            ]
        );

        let softmax_res = softmax(&x);

        // Row0 max index
        assert_eq!(max_index_in_arr1(&softmax_res.index_axis(Axis(0), 0).to_owned()), 1);
        // Row0 sum
        assert_eq!(round_digit(sum_arr1(&softmax_res.index_axis(Axis(0), 0).to_owned()), -4), 1.0);
        // Row1 max index
        assert_eq!(max_index_in_arr1(&softmax_res.index_axis(Axis(0), 1).to_owned()), 3);
        // Row0 sum
        assert_eq!(round_digit(sum_arr1(&softmax_res.index_axis(Axis(0), 1).to_owned()), -4), 1.0);

    }

    #[test]
    fn test_crosss_entropy_error() {
        let x = arr2(&
            [
                [0.3, 0.1, 0.5, 0.1],
                [0.0, 0.0, 0.0, 1.0],
            ]
        );

        let t = arr2(&
            [
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        );

        let cee_res = crosss_entropy_error(&x, &t);

        assert_eq!(cee_res.shape(), [2, 1]);
        assert_eq!(cee_res[(0, 0)], (0.5 as f64).log(E) * -1.0);
        assert_eq!(cee_res[(1, 0)], (1.0 as f64).log(E) * -1.0);
    }
    #[test]
    fn test_backward() {
        let x = NetworkBatchValueLayer::new(arr2(&
            [
                [2.0,   5.0, 3.0,  3.0],
                [0.0, -10.0, 7.0, 12.0],
            ]
        ));
        let t = arr2(&
            [
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        );
        let mut softmaxLoss= SoftmaxWithLoss::new(x, t.clone());

        let dout = arr2(&
            [
                [1.0, -1.0, 2.0, -2.0],
                [0.0,  1.0, 2.0,  0.0],
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
        let diffs = softmaxLoss.backward(dout.clone(), diffs);

        assert_eq!(diffs.len(), 2);
        assert_eq!(diffs[0], arr2(&
            [
                [3.0, 1.0, 4.0],
                [1.0, 5.0, 9.0],
            ]
        ));
        assert_eq!(diffs[1], dout * (softmaxLoss.forward().to_owned() - t));
    }
}