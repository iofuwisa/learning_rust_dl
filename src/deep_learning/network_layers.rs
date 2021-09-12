use ndarray::prelude::{
    Array1,
    arr1,
};


pub trait NetworkLayer {
    fn forward(&mut self) -> &Array1<f64>;
    fn backward(&mut self, dout: Array1<f64>, diffs: Vec<Array1<f64>>) -> Vec<Array1<f64>>;
}


// No value layer for ValueLayer.
pub struct NetworkNoneValue {}
impl NetworkLayer for NetworkNoneValue {
    fn forward(&mut self) -> &Array1<f64> {
        panic!("This function must not be called.");
    }
    fn backward(&mut self, _dout: Array1<f64>, _diffs: Vec<Array1<f64>>) -> Vec<Array1<f64>> {
        panic!("This function must not be called.");
    }
}


// Direct value
pub struct NetworkValueLayer<T: NetworkLayer, U: NetworkLayer> {
    _x: Option<T>,
    _y: Option<U>,
    z: Array1<f64>,
}
impl NetworkValueLayer<NetworkNoneValue, NetworkNoneValue> {
    pub fn new(z: Array1<f64>) -> NetworkValueLayer<NetworkNoneValue, NetworkNoneValue> {
        NetworkValueLayer {
            _x: None,
            _y: None,
            z: z,
        }
    }
}
impl<T: NetworkLayer, U: NetworkLayer> NetworkLayer for NetworkValueLayer<T, U> {
    fn forward(&mut self) -> &Array1<f64> {
        &self.z
    }
    fn backward(&mut self, dout: Array1<f64>, diffs: Vec<Array1<f64>>) -> Vec<Array1<f64>> {
        let mut mut_diffs = diffs;
        mut_diffs.push(dout);
        return mut_diffs;
    }
}


#[cfg(test)]
mod test_network_value_mod {
    use super::*;

    #[test]
    fn test_forward() {
        let mut v = NetworkValueLayer::new( arr1(&[0.0, 0.2, 5.0, -5.0, 999999.0]));
        assert_eq!(v.forward(),             arr1(&[0.0, 0.2, 5.0, -5.0, 999999.0]));
    }
    #[test]
    fn test_backward() {
        let mut v = NetworkValueLayer::new( arr1(&[0.0, 0.2, 5.0, -5.0, 999999.0]));
        
        let mut diffs = Vec::<Array1<f64>>::new();
        diffs.push(arr1(&[0.1, 0.2, 0.3, 0.4, 0.5]));
        diffs.push(arr1(&[0.5, 0.6, 0.7, 0.8, 0.9]));
        diffs = v.backward(arr1(&[1.0, 2.0, 3.0, 4.0, 5.0]), diffs);

        assert_eq!(diffs[0], arr1(&[0.1, 0.2, 0.3, 0.4, 0.5]));
        assert_eq!(diffs[1], arr1(&[0.5, 0.6, 0.7, 0.8, 0.9]));
        assert_eq!(diffs[2], arr1(&[1.0, 2.0, 3.0, 4.0, 5.0]));
    }
}