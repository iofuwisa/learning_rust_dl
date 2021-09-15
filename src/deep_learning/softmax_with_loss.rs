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
            
            // exp(ak + c)/Σ(exp(ai + c))
            let z = Array::from_shape_fn(x.dim(), |(i, j)| -> f64 {

                let r = x.index_axis(Axis(0), i).to_owned();

                // Find max
                let max_index = max_index_in_arr1(&r);
                let c = r[max_index];

                // Σ(exp(ai + c)
                let mut sum_exp_a = 0.0;
                for ai in &r {
                    sum_exp_a += E.powf(*ai + c);
                }

                // exp(ak + c)/Σ(exp(ai + c))
                return E.powf(r[j] + c) / sum_exp_a;
            });

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

#[cfg(test)]
mod test_affine_mod {
    use super::*;

    #[test]
    fn test_forward() {
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
        let mut a = SoftmaxWithLoss::new(x, t);

        let y = a.forward();

        assert_eq!(max_index_in_arr1(&y.index_axis(Axis(0), 0).to_owned()), 1);
        assert_eq!(round_digit(sum_arr1(&y.index_axis(Axis(0), 0).to_owned()), -4), 1.0);
        assert_eq!(max_index_in_arr1(&y.index_axis(Axis(0), 1).to_owned()), 3);
        assert_eq!(round_digit(sum_arr1(&y.index_axis(Axis(0), 1).to_owned()), -4), 1.0);
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
        let mut a = SoftmaxWithLoss::new(x, t.clone());

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
        let diffs = a.backward(dout.clone(), diffs);

        assert_eq!(diffs.len(), 2);
        assert_eq!(diffs[0], arr2(&
            [
                [3.0, 1.0, 4.0],
                [1.0, 5.0, 9.0],
            ]
        ));
        assert_eq!(diffs[1], dout * (a.forward().to_owned() - t));
    }

}