use ndarray::prelude::{
    Array1,
    Array2,
    arr1,
    arr2,
};
use ndarray::iter::{
    Iter,
};

use crate::deep_learning::common::*;
use crate::deep_learning::network_layers::*;


pub trait NetworkBatchLayer {
    fn forward(&mut self) -> &Array2<f64>;
    fn backward(&mut self, dout: Array2<f64>, diffs: Vec<Array2<f64>>) -> Vec<Array2<f64>>;
}


// Direct value
pub struct NetworkBatchValueLayer {
    z: Array2<f64>,
}
impl NetworkBatchValueLayer {
    pub fn new(z: Array2<f64>) -> NetworkBatchValueLayer {
        NetworkBatchValueLayer {
            z: z,
        }
    }
}
impl NetworkBatchLayer for NetworkBatchValueLayer {
    fn forward(&mut self) -> &Array2<f64> {
        &self.z
    }
    fn backward(&mut self, dout: Array2<f64>, diffs: Vec<Array2<f64>>) -> Vec<Array2<f64>> {
        let mut mut_diffs = diffs;
        mut_diffs.push(dout);
        return mut_diffs;
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
}
impl<TX: NetworkBatchLayer, TW: NetworkBatchLayer, TB: NetworkBatchLayer> NetworkBatchLayer for AffineLayer<TX, TW, TB> {
    fn forward(&mut self) -> &Array2<f64> {
        if self.z.is_none() {
            let x = self.x.forward();
            let w = self.w.forward();
            let b = self.b.forward();
            println!("x: {:?}", x);
            println!("w: {:?}", w);
            println!("b: {:?}", b);
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
}


#[cfg(test)]
mod test_affine_mod {
    use super::*;

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
                [2.4,  0.0],
                [8.5, 10.0],
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
