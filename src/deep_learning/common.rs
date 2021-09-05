use rand::Rng;

use ndarray::prelude::{
    Array,
    Array1,

};

pub fn random_choice(size: usize, max: usize) -> Vec<usize> {
    let mut rng = rand::thread_rng();

    let mut choice = Vec::<usize>::with_capacity(size as usize);
    for _i in 0..size {
        choice.push((rng.gen::<f32>()*max as f32).floor() as usize);
    }
    
    return choice;
}


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
    for i in 0..x.len() {
        let mut argx = x.clone();
        argx[i] = argx[i] + h;
        let fxh1 = func(&argx);

        argx[i] = argx[i] - h;
        let fxh2 = func(&argx);

        grad[i] = (fxh1 - fxh2) / 2.0 * h;

        // println!("Gradient progress: {}% {}/{}", i*100/x.len(), i, x.len());
        if progress+0.05 < i as f64 / x.len() as f64 {
            progress += 0.05;
            println!("Gradient progress: {}%", progress*100.0);
        }

    }

    return grad;
}