use ndarray::prelude::{
    s,
    Array4
};

// fn im2col(input_data: Array4<f64>, filter_h: usize, filter_w: usize, stride: usize, pad: usize) {
//     let (N, C, H, W) = input_data.dim();
//     let out_h = (H + 2*pad - filter_h);//stride + 1
//     let out_w = (W + 2*pad - filter_w);//stride + 1

//     img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
//     col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))
// }

fn pad_array4(data: &Array4<f64>, pad: [(usize, usize); 4]) -> Array4<f64> {
    let paded_shape = [
        data.shape()[0] + pad[0].0 + pad[0].1,
        data.shape()[1] + pad[1].0 + pad[1].1,
        data.shape()[2] + pad[2].0 + pad[2].1,
        data.shape()[3] + pad[3].0 + pad[3].1,
    ];
    let mut paded = Array4::<f64>::zeros(paded_shape);
    let mut paded_data = paded.slice_mut(
        s![
            pad[0].0..paded.shape()[0]-pad[0].1,
            pad[1].0..paded.shape()[1]-pad[1].1,
            pad[2].0..paded.shape()[2]-pad[2].1,
            pad[3].0..paded.shape()[3]-pad[3].1,
        ]
    );

    paded_data.assign(data);

    return paded;
}

#[cfg(test)]
mod test {
    use super::*;

    use ndarray::prelude::Array;

    #[test]
    fn test_pad_array4() {
        let data = Array::from_shape_vec(
            (2,2,3,3),
            vec![
                1f64, 2f64, 3f64, 4f64, 5f64, 6f64, 7f64, 8f64, 9f64, 0f64,
                1f64, 2f64, 3f64, 4f64, 5f64, 6f64, 7f64, 8f64, 9f64, 0f64,
                1f64, 2f64, 3f64, 4f64, 5f64, 6f64, 7f64, 8f64, 9f64, 0f64,
                1f64, 2f64, 3f64, 4f64, 5f64, 6f64,
            ]
        ).ok().unwrap();

        let paded = pad_array4(&data, [(1, 2), (2, 1), (1, 2), (2, 1)]);

        // assert_eq!(paded, )
        println!("{:?}", paded);
    }
}
