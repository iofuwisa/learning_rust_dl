use ndarray::prelude::{
    Array,
    Array1,
    arr1,
};


// Numerical differentiation
pub fn numeric_diff(func: Box<dyn Fn(f64) -> f64>, x: f64) -> f64 {
    let h = 0.0001;
    return (func(x+h) - func(x-h)) / (h * 2.0);
}

// Numerical gradient
pub fn numeric_gradient<F: Fn(&Array1<f64>) -> f64>(func: F, x: &Array1<f64>) -> Array1<f64> {
    let h = 0.0001;
    let mut grad: Array1<f64> = Array::zeros(x.len());

    
    let mut progress = 0.0;
    let mut argx = x.clone();
    for i in 0..x.len() {
        argx[i] = argx[i] + h;
        let fxh1 = func(&argx);

        argx[i] = argx[i] - h - h;
        let fxh2 = func(&argx);

        argx[i] = argx[i] + h;

        grad[i] = (fxh1 - fxh2) / (2.0 * h);

        // println!("Gradient progress: {}% {}/{}", i*100/x.len(), i, x.len());
        if progress+0.05 < i as f64 / x.len() as f64 {
            progress += 0.05;
            println!("Gradient progress: {}%", progress*100.0);
        }

    }

    return grad;
}

#[cfg(test)]
mod NeuralNetwork_test {
    use super::*;

    #[test]
    fn test_numeric_gradient() {
        let x = arr1(&[0.0, 1.0, 2.0, 3.0, 4.0]);
        let f = |x: &Array1<f64>| -> f64 {
            let mut y = 0.0;
            for i in 0..x.len() {
                y += x[i] * x[i];
            }
            return y;
        };
        let mut grad = numeric_gradient(f, &x);

        for i in 0..grad.len() {
            grad[i] = (grad[i] * 1000.0).round() / 1000.0;
        }

        assert_eq!(grad, arr1(&[0.0, 2.0, 4.0, 6.0, 8.0]));
    }
}