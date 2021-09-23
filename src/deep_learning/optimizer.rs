use ndarray::prelude::{
    Array2,
};


trait Optimizer {
    fn update(&mut self, target: &Array2<f64>, gradient: &Array2<f64>) -> Array2<f64>;
}


struct Sgd {
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


struct Momentum {
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


struct AdaGrad {
    learning_rate: f64,
    grad_suared_sum: Option<Array2<f64>>
}
impl AdaGrad {
    pub fn new(learning_rate: f64) -> Self {
        Self {
            learning_rate: learning_rate,
            grad_suared_sum: None,
        }
    }
}
impl Optimizer for AdaGrad {
    fn update(&mut self, target: &Array2<f64>, gradient: &Array2<f64>) -> Array2<f64> {
        if self.grad_suared_sum.is_none() {
            self.grad_suared_sum = Some(Array2::<f64>::zeros(target.dim()));
        }
        let mut grad_suared_sum = self.grad_suared_sum.as_mut().unwrap();

        grad_suared_sum.assign(&(grad_suared_sum.clone() + gradient * gradient));

        return target - grad_suared_sum.clone() * gradient * self.learning_rate;
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
        let expect_updated = target.clone() - grad_suared_sum.clone() * gradient.clone() * 0.1;
        assert_eq!(updated, expect_updated);

        let updated = adagrad.update(&target, &gradient);
        
        let grad_suared_sum = grad_suared_sum.clone() + gradient.clone() * gradient.clone();
        let expect_updated = target.clone() - grad_suared_sum.clone() * gradient.clone() * 0.1;
        assert_eq!(updated, expect_updated);
    }
}