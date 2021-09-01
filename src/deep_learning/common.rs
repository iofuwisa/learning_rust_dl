use rand::Rng;

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
pub fn numeric_gradient(func: Box<dyn Fn(&Vec<f64>) -> f64>, x: &Vec<f64>) -> Vec<f64> {
    let h = 0.0001;
    let mut grad = Vec::<f64>::with_capacity(x.len());

    for i in 0..x.len() {
        let mut argx = x.clone();
        argx[i] = argx[i] + h;
        let fxh1 = func(&argx);

        argx[i] = argx[i] - h;
        let fxh2 = func(&argx);

        grad.push((fxh1 - fxh2) / 2.0 * h)

    }

    return grad;
}