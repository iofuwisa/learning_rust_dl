pub trait Layer {
    fn forward(&mut self) -> f64;
    fn backward(&mut self, dout: f64, diff: Vec<f64>) -> Vec<f64>;
}


// No value layer for ValueLayer.
pub struct NoneValue {}
impl Layer for NoneValue {
    fn forward(&mut self) -> f64 {0.0}
    fn backward(&mut self, _dout: f64, diffs: Vec<f64>) -> Vec<f64> {diffs}
}


// Direct value
pub struct ValueLayer<T: Layer, U: Layer> {
    _x: Option<T>,
    _y: Option<U>,
    z: f64,
}
impl ValueLayer<NoneValue, NoneValue> {
    pub fn new(z: f64) -> ValueLayer<NoneValue, NoneValue> {
        ValueLayer {
            _x: None,
            _y: None,
            z: z,
        }
    }
}
impl<T: Layer, U: Layer> Layer for ValueLayer<T, U> {
    fn forward(&mut self) -> f64 {
        self.z
    }
    fn backward(&mut self, dout: f64, diffs: Vec<f64>) -> Vec<f64> {
        let mut mut_diffs = diffs;
        mut_diffs.push(dout);
        return mut_diffs;
    }
}


// Multiplication
pub struct MulLayer<T: Layer, U: Layer> {
    x: T,
    y: U,
    z: Option<f64>,
}
impl<T: Layer, U: Layer> MulLayer<T, U> {
    pub fn new(x: T, y: U) -> MulLayer<T, U> {
        MulLayer {
            x: x,
            y: y,
            z: None,
        }
    }
}
impl<T: Layer, U: Layer> Layer for MulLayer<T, U> {
    fn forward(&mut self) -> f64 {
        match self.z {
            Some(n) => n,   // If caluculated return result.
            None => {
                self.z = Some(self.x.forward() * self.y.forward());
                self.z.unwrap()
            },
        }
    }
    fn backward(&mut self, dout: f64, diffs: Vec<f64>) -> Vec<f64> {
        //dout side of x: dout * y
        let dout_x = dout * self.y.forward();
        let diffs = self.x.backward(dout_x, diffs);

        //dout side of y: dout * x
        let dout_y = dout * self.x.forward();
        let diffs = self.y.backward(dout_y, diffs);

        return diffs;
    }
}


// Add
pub struct AddLayer<T: Layer, U: Layer> {
    x: T,
    y: U,
    z: Option<f64>,
}
impl<T: Layer, U: Layer> AddLayer<T, U> {
    pub fn new(x: T, y: U) -> AddLayer<T, U> {
        AddLayer {
            x: x,
            y: y,
            z: None,
        }
    }
}
impl<T: Layer, U: Layer> Layer for AddLayer<T, U> {
    fn forward(&mut self) -> f64 {
        match self.z {
            Some(n) => n,   // If caluculated return result.
            None => {
                self.z = Some(self.x.forward() + self.y.forward());
                self.z.unwrap()
            },
        }
    }
    fn backward(&mut self, dout: f64, diffs: Vec<f64>) -> Vec<f64> {
        let diffs = self.x.backward(dout, diffs);
        let diffs = self.y.backward(dout, diffs);
        return diffs;
    }
}


#[cfg(test)]
mod test_value_mod {
    use super::*;

    #[test]
    fn test_forward() {
        let mut v = ValueLayer::new(0.6);
        assert_eq!(v.forward(), 0.6);
        assert_eq!(v.z, 0.6);
    }
    #[test]
    fn test_backward() {
        let mut v = ValueLayer::new(0.6);
        let diffs = v.backward(3.0, Vec::<f64>::new());
        assert_eq!(diffs.len(), 1);
        assert_eq!(diffs[0], 3.0);
    }
}

#[cfg(test)]
mod test_mul_mod {
    use super::*;

    #[test]
    fn test_forward() {
        // (2.0 * 5.0) * (3.0 * 4.0)
        let mut v1 = ValueLayer::new(2.0);
        let mut v2 = ValueLayer::new(5.0);
        let mut m1 = MulLayer::new(v1, v2);
        let mut v3 = ValueLayer::new(3.0);
        let mut v4 = ValueLayer::new(4.0);
        let mut m2 = MulLayer::new(v3, v4);
        let mut m3 = MulLayer::new(m1, m2);

        // calc
        assert_eq!(m3.forward(), 120.0);
        // cache
        assert_eq!(m3.z.unwrap(), 120.0);
    }
    #[test]
    fn test_backward() {
        // (2.0 * 5.0) * (3.0 * 4.0)
        let mut v1 = ValueLayer::new(2.0);
        let mut v2 = ValueLayer::new(5.0);
        let mut m1 = MulLayer::new(v1, v2);
        let mut v3 = ValueLayer::new(3.0);
        let mut v4 = ValueLayer::new(4.0);
        let mut m2 = MulLayer::new(v3, v4);
        let mut m3 = MulLayer::new(m1, m2);

        let diffs = m3.backward(3.0, Vec::<f64>::new());
        println!("{:?}", diffs);
        assert_eq!(diffs.len(), 4);
        assert_eq!(diffs[0], 60.0 * 3.0);
        assert_eq!(diffs[1], 24.0 * 3.0);
        assert_eq!(diffs[2], 40.0 * 3.0);
        assert_eq!(diffs[3], 30.0 * 3.0);
    }
}

#[cfg(test)]
mod test_add_mod {
    use super::*;

    #[test]
    fn test_forward() {
        // (2.0 + 5.0) + (3.0 + 4.0)
        let mut v1 = ValueLayer::new(2.0);
        let mut v2 = ValueLayer::new(5.0);
        let mut m1 = AddLayer::new(v1, v2);
        let mut v3 = ValueLayer::new(3.0);
        let mut v4 = ValueLayer::new(4.0);
        let mut m2 = AddLayer::new(v3, v4);
        let mut m3 = AddLayer::new(m1, m2);

        // calc
        assert_eq!(m3.forward(), 14.0);
        // cache
        assert_eq!(m3.z.unwrap(), 14.0);
    }
    #[test]
    fn test_backward() {
        // (2.0 + 5.0) + (3.0 + 4.0)
        let mut v1 = ValueLayer::new(2.0);
        let mut v2 = ValueLayer::new(5.0);
        let mut m1 = AddLayer::new(v1, v2);
        let mut v3 = ValueLayer::new(3.0);
        let mut v4 = ValueLayer::new(4.0);
        let mut m2 = AddLayer::new(v3, v4);
        let mut m3 = AddLayer::new(m1, m2);

        let diffs = m3.backward(3.0, Vec::<f64>::new());
        println!("{:?}", diffs);
        assert_eq!(diffs.len(), 4);
        assert_eq!(diffs[0], 3.0);
        assert_eq!(diffs[1], 3.0);
        assert_eq!(diffs[2], 3.0);
        assert_eq!(diffs[3], 3.0);
    }
}