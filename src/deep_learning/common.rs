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
        if progress+0.01 < i as f64 / x.len() as f64 {
            progress += 0.05;
            println!("Gradient progress: {}%", progress*100.0);
        }

    }

    return grad;
}

pub fn max_index_in_arr1(arr: &Array1<f64>) -> usize {
    let mut max_index: usize = 0;
    for i in 0..arr.len() {
        if arr[i] > arr[max_index as usize] {
            max_index = i;
        }
    }
    return max_index
}

pub fn round_digit(num: f64, digit: i32) -> f64 {
    if digit == 0 {
        num.round()
    } else {
        (num * 10.0_f64.powi(-digit)).round() * 10.0_f64.powi(digit)
    }
}

pub fn print_img(img: &Array1<f64>) {
    for i in 0..28 {
        for j in 0..28 {
            print!("{} ", if img[(i*28+j)] > 0.0 {"*"} else {" "});
        }
        println!("");
    }
}

#[cfg(test)]
mod NeuralNetwork_test {
    use super::*;

    #[test]
    fn test_round_digit() {
        assert_eq!(round_digit(3.14, 0), 3.0);
        assert_eq!(round_digit(123456.789123, 3), 123000.0);
        assert_eq!(round_digit(123456.789123, -3), 123456.789);
    }

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

    
    #[test]
    fn test_numeric_gradient2() {
        
        // Find minimum below function
        // y = x[0]^2 + x[1]^2

        let lerning_rate = 0.05;

        let mut x = arr1(&[3.0, 4.0]);

        // y = x[0]^2 + x[1]^2
        for _ in 0..100 {
            
            let f = |diff: &Array1<f64>| -> f64 {
                let added_x = x.clone() + diff;
                return added_x[0] * added_x[0] + added_x[1] * added_x[1];
            };

            println!("Y {}", f(&x));
            println!("");

            let grad = numeric_gradient(&f, &arr1(&[0.0, 0.0]));

            // Update x
            x[0] -= grad[0] * 0.05;
            x[1] -= grad[1] * 0.05;
            
            println!("Grad {:?}", grad);
            println!("X {:?}", x);

        }

    }  
}