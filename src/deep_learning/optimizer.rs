use ndarray::prelude::{
    Array2,
};

use crate::deep_learning::common::*;

pub trait Optimizer {
    fn update(&mut self, target: &Array2<f64>, gradient: &Array2<f64>) -> Array2<f64>;
}

// Reference
// https://data-science.gr.jp/theory/tml_optimizer_of_gradient_descent.html

pub struct Sgd {
    learning_rate: f64,
}
impl Sgd {
    pub fn new(learning_rate: f64) -> Self {
        Sgd {
            learning_rate: learning_rate,
        }
    }
}
impl Optimizer for Sgd {
    fn update(&mut self, target: &Array2<f64>, gradient: &Array2<f64>) -> Array2<f64> {
        return target - gradient * self.learning_rate;
    }
}

pub struct Momentum {
    learning_rate: f64,
    velocity: Option<Array2<f64>>,
    friction: f64,
}
impl Momentum {
    pub fn new(learning_rate: f64, friction: f64) -> Self {
        Self {
            learning_rate: learning_rate,
            velocity: None,
            friction: friction,
        }
    }
}
impl Optimizer for Momentum {
    fn update(&mut self, target: &Array2<f64>, gradient: &Array2<f64>) -> Array2<f64> {
        if self.velocity.is_none() {
            self.velocity = Some(Array2::<f64>::zeros(target.dim()));
        }
        let mut velocity = self.velocity.as_mut().unwrap();

        velocity.assign(&(velocity.clone() * self.friction - gradient * self.learning_rate));

        return target + velocity.clone();
    }
}

pub struct Rmsprop {
    learning_rate: f64,
    velocity: Option<Array2<f64>>,
    friction: f64,
}
impl Rmsprop {
    pub fn new(learning_rate: f64, friction: f64) -> Self {
        Self {
            learning_rate: learning_rate,
            velocity: None,
            friction: friction,
        }
    }
}
impl Optimizer for Rmsprop {
    fn update(&mut self, target: &Array2<f64>, gradient: &Array2<f64>) -> Array2<f64> {
        if self.velocity.is_none() {
            self.velocity = Some(Array2::<f64>::zeros(target.dim()));
        }
        let mut velocity = self.velocity.as_mut().unwrap();

        velocity.assign(&(velocity.clone() * self.friction + gradient * gradient * (1.0 - self.friction)));
        return
            target - 
                self.learning_rate /
                sqrt_arr2(&(velocity.clone() + (10.0 as f64).powi(-6)))
                * gradient;
    }
}

pub struct AdaGrad {
    learning_rate: f64,
    grad_squared_sum: Option<Array2<f64>>
}
impl AdaGrad {
    pub fn new(learning_rate: f64) -> Self {
        Self {
            learning_rate: learning_rate,
            grad_squared_sum: None,
        }
    }
}
impl Optimizer for AdaGrad {
    fn update(&mut self, target: &Array2<f64>, gradient: &Array2<f64>) -> Array2<f64> {
        if self.grad_squared_sum.is_none() {
            self.grad_squared_sum = Some(Array2::<f64>::zeros(target.dim()));
        }
        let mut grad_squared_sum = self.grad_squared_sum.as_mut().unwrap();

        grad_squared_sum.assign(&(grad_squared_sum.clone() + gradient * gradient));

        return
            target -
                self.learning_rate /
                sqrt_arr2(&(grad_squared_sum.clone() + (10.0 as f64).powi(-6))) *
                gradient;
    }
}

pub struct Adam {
    learning_rate: f64,
    m: Option<Array2<f64>>,
    v: Option<Array2<f64>>,
    friction_m: f64,
    friction_v: f64,
    update_count: u32,
}
impl Adam {
    pub fn new(learning_rate: f64, friction_m: f64, friction_v: f64) -> Self {
        Self {
            learning_rate: learning_rate,
            m: None,
            v: None,
            friction_m: friction_m,
            friction_v: friction_v,
            update_count: 0,
        }
    }
}
impl Optimizer for Adam {
    fn update(&mut self, target: &Array2<f64>, gradient: &Array2<f64>) -> Array2<f64> {
        self.update_count += 1;

        if self.m.is_none() {
            self.m = Some(Array2::<f64>::zeros(target.dim()));
        }
        let mut m = self.m.as_mut().unwrap();

        if self.v.is_none() {
            self.v = Some(Array2::<f64>::zeros(target.dim()));
        }
        let mut v = self.v.as_mut().unwrap();

        m.assign(&(self.friction_m * m.clone() + (1.0 - self.friction_m) * gradient));

        v.assign(&(self.friction_v * v.clone() + (1.0 - self.friction_v) * gradient * gradient));

        let m_d = m.clone() / (1.0 - self.friction_m.powi(self.update_count as i32));

        let v_d = v.clone() / (1.0 - self.friction_v.powi(self.update_count as i32));

        return 
            target - 
                self.learning_rate *
                m_d / 
                sqrt_arr2(&(v_d + (10.0 as f64).powi(-6)))
        ;
    }
}



#[cfg(test)]
mod test_sgd_mod {
    use super::*;

    use ndarray::prelude::{
        arr2,
    };

    #[test]
    fn update() {
        let mut sgd = Sgd::new(0.1);
        let target = arr2(&
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ]
        );
        let gradient = arr2(&
            [
                [1.0, 3.0, 5.0],
                [2.0, 4.0, 6.0],
            ]
        );

        let updated = sgd.update(&target, &gradient);

        assert_eq!(updated, arr2(&
            [
                [0.9, 1.7, 2.5],
                [3.8, 4.6, 5.4]
            ]
        ));
    }
}

#[cfg(test)]
mod test_momentum_mod {
    use super::*;

    use ndarray::prelude::{
        arr2,
    };

    #[test]
    fn update() {
        let mut momentum = Momentum::new(0.1, 0.9);
        let target = arr2(&
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ]
        );
        let gradient = arr2(&
            [
                [1.0, 3.0, 5.0],
                [2.0, 4.0, 6.0],
            ]
        );

        let updated = momentum.update(&target, &gradient);

        let velocity = gradient * momentum.learning_rate * -1.0;
        let expect_updated = target + velocity;
        assert_eq!(updated, expect_updated);
    }
}

#[cfg(test)]
mod test_rmsprop_mod {
    use super::*;

    use ndarray::prelude::{
        arr2,
    };

    #[test]
    fn update() {
        let mut rmsprop = Rmsprop::new(0.1, 0.9);
        let target = arr2(&
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ]
        );
        let gradient = arr2(&
            [
                [1.0, 3.0, 5.0],
                [2.0, 4.0, 6.0],
            ]
        );

        let updated = rmsprop.update(&target, &gradient);

        let velocity = gradient.clone() * gradient.clone() * 0.1;
        let expect_updated =
            target.clone() -
                0.1 /
                sqrt_arr2(&(velocity.clone() + (10.0 as f64).powi(-6))) *
                gradient.clone();
        assert_eq!(
            round_digit_arr2(&updated, -6),
            round_digit_arr2(&expect_updated, -6)
        );

        let updated = rmsprop.update(&target, &gradient);

        let velocity = velocity.clone() * 0.9 + gradient.clone() * gradient.clone() * 0.1;
        let expect_updated =
        target.clone() -
            0.1 /
            sqrt_arr2(&(velocity.clone() + (10.0 as f64).powi(-6))) *
            gradient.clone();
        assert_eq!(
            round_digit_arr2(&updated, -6),
            round_digit_arr2(&expect_updated, -6)
        );
    }
}

#[cfg(test)]
mod test_adagrad_mod {
    use super::*;

    use ndarray::prelude::{
        arr2,
    };

    #[test]
    fn update() {
        let mut adagrad = AdaGrad::new(0.1);
        let target = arr2(&
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ]
        );
        let gradient = arr2(&
            [
                [1.0, 3.0, 5.0],
                [2.0, 4.0, 6.0],
            ]
        );

        let updated = adagrad.update(&target, &gradient);

        let grad_suared_sum = gradient.clone() * gradient.clone();
        let expect_updated = 
            target.clone() -
                0.1 /
                sqrt_arr2(&(grad_suared_sum.clone() + (10.0 as f64).powi(-6))) *
                gradient.clone();
        assert_eq!(updated, expect_updated);

        let updated = adagrad.update(&target, &gradient);

        let grad_suared_sum = grad_suared_sum.clone() + gradient.clone() * gradient.clone();
        let expect_updated = 
            target.clone() -
                0.1 /
                sqrt_arr2(&(grad_suared_sum.clone() + (10.0 as f64).powi(-6))) *
                gradient.clone();
        assert_eq!(updated, expect_updated);
    }
}

#[cfg(test)]
mod test_adam_mod {
    use super::*;

    use ndarray::prelude::{
        arr2,
    };

    use crate::deep_learning::common::*;

    #[test]
    fn update() {
        let mut adam = Adam::new(0.1, 0.9, 0.99);
        let target = arr2(&
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ]
        );
        let gradient = arr2(&
            [
                [1.0, 3.0, 5.0],
                [2.0, 4.0, 6.0],
            ]
        );

        let updated = adam.update(&target, &gradient);

        let m = 0.1 * gradient.clone();
        let v = 0.01 * gradient.clone() * gradient.clone();
        let m_d = m.clone() / (1.0 - (0.9 as f64).powf(1.0));
        let v_d = v.clone() / (1.0 - (0.99 as f64).powf(1.0));
        
        let expect_updated = 
            target.clone() - 
                0.1 / 
                (v_d + (10.0 as f64).powi(-6)).mapv(|v: f64| -> f64 {v.sqrt()})
                    * m_d;
        assert_eq!(round_digit_arr2(&updated, -6), round_digit_arr2(&expect_updated, -6));


        let updated = adam.update(&target, &gradient);

        let m = m * 0.9 + 0.1 * gradient.clone();
        let v = v * 0.99 + 0.01 * gradient.clone() * gradient.clone();
        let m_d = m.clone() / (1.0 - (0.9 as f64).powf(2.0));
        let v_d = v.clone() / (1.0 - (0.99 as f64).powf(2.0));

        let expect_updated = 
            target.clone() - 
                0.1 / 
                (v_d + (10.0 as f64).powi(-6)).mapv(|v: f64| -> f64 {v.sqrt()})
                    * m_d;
        assert_eq!(round_digit_arr2(&updated, -6), round_digit_arr2(&expect_updated, -6));
    }
}
