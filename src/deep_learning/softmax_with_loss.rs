// use std::f64::consts::E;
// use ndarray::prelude::{
//     Array,
//     Array1,
//     Array2,
//     ArrayView1,
//     arr1,
//     arr2,
//     s,
// };
// use ndarray::Axis;
// use ndarray::Dim;

// use crate::deep_learning::affine_layer::*;
// use crate::deep_learning::common::*;


// // Softmax with loss
// pub struct SoftmaxWithLoss<TX: NetworkBatchLayer> {
//     x: TX,
//     z: Option<Array2<f64>>, 
// }
// impl<TX: NetworkBatchLayer> SoftmaxWithLoss<TX> {
//     pub fn new(x: TX) -> SoftmaxWithLoss<TX> {
//         SoftmaxWithLoss {
//             x: x,
//             z: None,
//         }
//     }
// }
// impl<TX: NetworkBatchLayer> NetworkBatchLayer for SoftmaxWithLoss<TX> {
//     fn forward(&mut self) -> &Array2<f64> {
//         if self.z.is_none() {
            
//             let x = self.x.forward();
            
//             // exp(ak + c)/Σ(exp(ai + c))
//             let z = Array::from_shape_fn(x.shape(), |(i, j)| -> f64 {
//                 0.0
//             });
//             let z = x.map_axis(Axis(0), |a: ArrayView1<f64>| -> Array1<f64> {
//                 // Find max
//                 let max_index = max_index_in_arr1(&a.to_owned());
//                 let c = a[max_index];

//                 // Σ(exp(ai + c)
//                 let sum_exp_a = 0.0;
//                 for ai in a {
//                     sum_exp_a += E.powf(ai + c);
//                 }

//                 Array::from_shape_fn(a.len(), |i| ->f64 {
//                     // exp(ak + c)/Σ(exp(ai + c))
//                     E.powf(a[i]) / sum_exp_a
//                 })
//             });
//             self.z = Some(z);
//         }
//         self.z.as_ref().unwrap()
//     }
//     fn backward(&mut self, dout: Array2<f64>, diffs: Vec<Array2<f64>>) -> Vec<Array2<f64>> {
//         let mut mut_diffs = diffs;

//         let w_t = self.w.forward().t();
//         mut_diffs.push(dout.dot(&w_t));

//         let x_t = self.x.forward().t();
//         mut_diffs.push(x_t.dot(&dout));

//         mut_diffs.push(self.b.forward() * dout);

//         return mut_diffs;
//     }
// }