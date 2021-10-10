use ndarray::{
    s,
    Array2,
    Array4,
    Array6,
};

use super::super::super::deep_learning::affine_layer::*;
use super::super::super::deep_learning::optimizer::*;
use super::super::super::deep_learning::common::*;
use super::super::super::deep_learning::statistics::*;


pub struct Convolution {
    x: Box<dyn NetworkBatchLayer>,
    y: Option<Array2<f64>>,
    filter: Box<dyn NetworkBatchLayer>,
    bias: Box<dyn NetworkBatchLayer>,
    x_shape: (usize, usize, usize, usize),
    y_shape: (usize, usize, usize, usize),
    filter_shape: (usize, usize, usize, usize),
    stride: usize,
    pad: usize,
}

impl Convolution {
    pub fn new<TX, TW, TB>(
        x: TX,
        filter: TW,
        bias: TB,
        x_shape: (usize, usize, usize, usize),      // batch_num, channel_size, data_h, data_w
        y_shape: (usize, usize, usize, usize),      // batch_num, channel_size, data_h, data_w
        filter_shape: (usize, usize, usize, usize), // filter_num, channel_size, filter_h, filter_w
        stride: usize,
        pad: usize
    ) -> Self
        where   TX : NetworkBatchLayer + 'static,
                TW : NetworkBatchLayer + 'static,
                TB : NetworkBatchLayer + 'static,
    {
        Self {
            x: Box::new(x),
            y: None,
            filter: Box::new(filter),
            bias: Box::new(bias),
            x_shape: x_shape,
            y_shape: y_shape,
            filter_shape: filter_shape,
            stride: stride,
            pad: pad,
        }
    }

    pub fn new_random<TX, TFO, TBO>(
        x: TX,
        optimizer_f: TFO,
        optimizer_b: TBO,
        batch_num: usize,
        channel_size: usize,
        filter_h: usize,
        filter_w: usize,
        data_h: usize,
        data_w: usize,
        stride: usize,
        pad: usize
    ) -> Convolution
    where   TX : NetworkBatchLayer + 'static,
            TFO: Optimizer + 'static,
            TBO: Optimizer + 'static
    {
        // Calc filter num
        let stride_count_h = (data_h + 2 * pad - filter_h) / stride + 1;
        let stride_count_w = (data_w + 2 * pad - filter_w) / stride + 1;
        let filter_num = stride_count_h * stride_count_w;

        // Generate initialize filter and biasn by normal distibution
        let filter = NetworkBatchAffineValueLayer::new(
            Array2::from_shape_vec(
                (filter_num, channel_size * filter_h * filter_w),
                norm_random_vec(filter_num * channel_size * filter_h * filter_w)
            ).ok().unwrap(),
            optimizer_f    
        );
        let bias = NetworkBatchAffineValueLayer::new(
            Array2::from_shape_vec(
                (filter_num, 1),
                norm_random_vec(filter_num)
                    .into_iter()
                    .map(|x: f64| {x / 100.0})
                    .collect()
            ).ok().unwrap(),
            optimizer_b
        );

        return Convolution::new(
            x,
            filter,
            bias,
            (batch_num, channel_size, data_h, data_w),
            (batch_num, channel_size, stride_count_h, stride_count_w),
            (filter_num, channel_size, filter_h, filter_w),
            stride,
            pad
        );
    }
}

impl NetworkBatchLayer for Convolution {
    fn forward(&mut self, is_learning: bool) -> Array2<f64> {
        if self.y.is_none() {
            let x_2d = self.x.forward(is_learning);
            let filter_2d = self.filter.forward(is_learning);
            let bias_2d = self.bias.forward(is_learning);           

            // Reshape to 4d from 2d
            let x_4d = x_2d.to_shared().reshape(self.x_shape).to_owned();

            
            let (_filter_num, _channel_num, filter_h, filter_w) = self.filter_shape;
            let col_x_2d = im2col(&x_4d.to_owned(), filter_h, filter_w, self.stride, self.pad);

            let col_y = filter_2d.dot(&col_x_2d.t());

            let y = col_y.to_shared().reshape((self.y_shape.0, self.y_shape.1*self.y_shape.2*self.y_shape.3)).to_owned();

            // let y = y + bias_2d;

            self.y = Some(y);
        }

        return self.y.clone().unwrap();
    }
    fn backward(&mut self, _dout: Array2<f64>) {}
    fn set_value(&mut self, value: &Array2<f64>) {
        self.x.set_value(value);
        self.clean();
    }
    fn set_lbl(&mut self, value: &Array2<f64>) {
        self.x.set_lbl(value);
    }
    fn clean(&mut self) {
        self.y = None;
    }
    fn is_loss_layer(&self) -> bool {
        false
    }
    fn plot(&self) {
        self.x.plot();
    }
    fn weight_squared_sum(&self) -> f64 {
        return self.x.weight_squared_sum();
    }
    fn weight_sum(&self) -> f64 {
        return self.weight_sum();
    }
}

fn im2col(input_data: &Array4<f64>, filter_h: usize, filter_w: usize, stride: usize, pad: usize) -> Array2<f64>{
    let img = pad_array4(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)]);

    let (batch_size, channel_size, input_h, input_w) = input_data.dim();
    let stride_count_h = (input_h + 2 * pad - filter_h) / stride + 1;
    let stride_count_w = (input_w + 2 * pad - filter_w) / stride + 1;
    let mut col_6d = Array6::<f64>::zeros((batch_size, channel_size, filter_h, filter_w, stride_count_h, stride_count_w));

    for y in 0..filter_h {
        for x in 0..filter_w {
            let col_index = y * filter_w + x;
            let mut c = col_6d.slice_mut(s![.., .., y, x, .., ..]);

            let i = img.slice(s![.., .., y..=input_h-filter_h+y;stride, x..=input_w-filter_w+x;stride]);

            let shaped_i = i.to_owned().into_shared().reshape(c.shape());

            c.assign(&shaped_i);
        }
    }

    // Swap axes (0, 1, 2, 3, 4, 5) -> (0, 4, 5, 1, 2, 3)
    col_6d.swap_axes(1, 4);
    col_6d.swap_axes(2, 5);
    col_6d.swap_axes(3, 4);
    col_6d.swap_axes(4, 5);

    // Reshape to 2d 
    let col_2d = col_6d.to_shared().reshape((batch_size*stride_count_h*stride_count_w, filter_h*filter_w*channel_size)).to_owned();

    return col_2d;
}

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

    use ndarray::{
        Array,
        Axis,
        arr2,
        arr3,
    };

    #[test]
    fn test_convolution_new_random() {
        let x = NetworkBatchValueLayer::new(arr2(&
            [
                [1f64, 2f64, 3f64],
            ]
        ));
        let opt_f = Sgd::new(0.01);
        let opt_b = Sgd::new(0.01);
        let mut conv = Convolution::new_random(
            x,      // x
            opt_f,  // optimizer_f
            opt_b,  // optimizer_b
            1,      // batch_num
            3,      // channel_size
            3,      // filter_h
            3,      // filter_w
            28,     // img_h
            28,     // img_w 
            3,      // stride
            1       // pad
        );

        let mut filter = conv.filter;
        let mut bias = conv.bias;

        let filter_value = filter.forward(true);
        let bias_value = bias.forward(true);

        assert_eq!(filter_value.shape(), [100, 27]);
        assert_eq!(bias_value.shape(), [100, 1]);

        let (filter_std_dev, _, filter_avg) = standard_devication(&filter_value.to_shared().reshape(filter_value.len()).to_vec());
        let (bias_std_dev, _, bias_avg) = standard_devication(&bias_value.to_shared().reshape(bias_value.len()).to_vec());

        assert_eq!(round_digit(filter_avg, -1), 0f64);
        assert_eq!(round_digit(filter_std_dev, -1), 1f64);
        assert_eq!(round_digit(bias_avg, -1), 0f64);
        assert_eq!(round_digit(bias_std_dev, -3), 0.01f64);

    }

    #[test]
    fn test_convolution_forward() {
        // B:1, C:2 H:7 W:7
        let value = NetworkBatchValueLayer::new(
            Array::from_shape_vec(
                (1,98),
                vec![
                    01f64, 02f64, 03f64, 04f64, 05f64, 06f64, 07f64,
                    11f64, 12f64, 13f64, 14f64, 15f64, 16f64, 17f64,
                    21f64, 22f64, 23f64, 24f64, 25f64, 26f64, 27f64,
                    31f64, 32f64, 33f64, 34f64, 35f64, 36f64, 37f64,
                    41f64, 42f64, 43f64, 44f64, 45f64, 46f64, 47f64,
                    51f64, 52f64, 53f64, 54f64, 55f64, 56f64, 57f64,
                    61f64, 62f64, 63f64, 64f64, 65f64, 66f64, 67f64,

                    101f64, 102f64, 103f64, 104f64, 105f64, 106f64, 107f64,
                    111f64, 112f64, 113f64, 114f64, 115f64, 116f64, 117f64,
                    121f64, 122f64, 123f64, 124f64, 125f64, 126f64, 127f64,
                    131f64, 132f64, 133f64, 134f64, 135f64, 136f64, 137f64,
                    141f64, 142f64, 143f64, 144f64, 145f64, 146f64, 147f64,
                    151f64, 152f64, 153f64, 154f64, 155f64, 156f64, 157f64,
                    161f64, 162f64, 163f64, 164f64, 165f64, 166f64, 167f64,
                ]
            ).ok().unwrap()
        );
        // FN:9, C:2 FH:3 FW:3
        let filter = NetworkBatchValueLayer::new(
            Array::from_shape_vec(
                (9,18),
                vec![
                    01f64, 02f64, 03f64, 04f64, 05f64, 06f64, 07f64, 08f64, 09f64,  10f64, 11f64, 12f64, 13f64, 14f64, 15f64, 16f64, 17f64, 18f64,
                    02f64, 03f64, 04f64, 05f64, 06f64, 07f64, 08f64, 09f64, 10f64,  11f64, 12f64, 13f64, 14f64, 15f64, 16f64, 17f64, 18f64, 19f64,
                    03f64, 04f64, 05f64, 06f64, 07f64, 08f64, 09f64, 10f64, 11f64,  12f64, 13f64, 14f64, 15f64, 16f64, 17f64, 18f64, 19f64, 20f64,
                    04f64, 05f64, 06f64, 07f64, 08f64, 09f64, 10f64, 11f64, 12f64,  13f64, 14f64, 15f64, 16f64, 17f64, 18f64, 19f64, 20f64, 21f64,
                    05f64, 06f64, 07f64, 08f64, 09f64, 10f64, 11f64, 12f64, 13f64,  14f64, 15f64, 16f64, 17f64, 18f64, 19f64, 20f64, 21f64, 22f64,
                    06f64, 07f64, 08f64, 09f64, 10f64, 11f64, 12f64, 13f64, 14f64,  15f64, 16f64, 17f64, 18f64, 19f64, 20f64, 21f64, 22f64, 23f64,
                    07f64, 08f64, 09f64, 10f64, 11f64, 12f64, 13f64, 14f64, 15f64,  16f64, 17f64, 18f64, 19f64, 20f64, 21f64, 22f64, 23f64, 24f64,
                    08f64, 09f64, 10f64, 11f64, 12f64, 13f64, 14f64, 15f64, 16f64,  17f64, 18f64, 19f64, 20f64, 21f64, 22f64, 23f64, 24f64, 25f64,
                    09f64, 10f64, 11f64, 12f64, 13f64, 14f64, 15f64, 16f64, 17f64,  18f64, 19f64, 20f64, 21f64, 22f64, 23f64, 24f64, 25f64, 26f64,

                ]
            ).ok().unwrap()
        );
        // FN:9
        let bias = NetworkBatchValueLayer::new(
            Array::from_shape_vec(
                (1,9),
                vec![
                    // 01f64,
                    // 02f64,
                    // 03f64,
                    // 04f64,
                    // 05f64,
                    // 06f64,
                    // 07f64,
                    // 08f64,
                    // 09f64,
                    0f64,
                    0f64,
                    0f64,
                    0f64,
                    0f64,
                    0f64,
                    0f64,
                    0f64,
                    0f64,
                ]
            ).ok().unwrap()
        );
        let stride = 2;
        let pad = 0;
        let mut conv = Convolution::new(value, filter, bias, (1, 2 ,7, 7), (1, 9, 3, 3), (9, 2 ,3, 3), stride, pad);

        let y = conv.forward(false);

        let verification_y_2d = 
            verification_forward(
                conv.x.forward(false),
                conv.filter.forward(false),
                conv.bias.forward(false),
                conv.x_shape,
                conv.y_shape,
                conv.filter_shape,
                conv.stride,
                conv.pad
            );
        assert_eq!(verification_y_2d, y);
    }

    #[test]
    fn test_convolution_im2col() {
        let data = Array::from_shape_vec(
            (1,2,7,7),
            vec![
                01f64, 02f64, 03f64, 04f64, 05f64, 06f64, 07f64,
                11f64, 12f64, 13f64, 14f64, 15f64, 16f64, 17f64,
                21f64, 22f64, 23f64, 24f64, 25f64, 26f64, 27f64,
                31f64, 32f64, 33f64, 34f64, 35f64, 36f64, 37f64,
                41f64, 42f64, 43f64, 44f64, 45f64, 46f64, 47f64,
                51f64, 52f64, 53f64, 54f64, 55f64, 56f64, 57f64,
                61f64, 62f64, 63f64, 64f64, 65f64, 66f64, 67f64,

                101f64, 102f64, 103f64, 104f64, 105f64, 106f64, 107f64,
                111f64, 112f64, 113f64, 114f64, 115f64, 116f64, 117f64,
                121f64, 122f64, 123f64, 124f64, 125f64, 126f64, 127f64,
                131f64, 132f64, 133f64, 134f64, 135f64, 136f64, 137f64,
                141f64, 142f64, 143f64, 144f64, 145f64, 146f64, 147f64,
                151f64, 152f64, 153f64, 154f64, 155f64, 156f64, 157f64,
                161f64, 162f64, 163f64, 164f64, 165f64, 166f64, 167f64,
            ]
        ).ok().unwrap();

        let expect = Array::from_shape_vec(
            (9, 18),
            vec![
                01f64, 02f64, 03f64, 11f64, 12f64, 13f64, 21f64, 22f64, 23f64,101f64, 102f64, 103f64, 111f64, 112f64, 113f64, 121f64, 122f64, 123f64,
                03f64, 04f64, 05f64, 13f64, 14f64, 15f64, 23f64, 24f64, 25f64,103f64, 104f64, 105f64, 113f64, 114f64, 115f64, 123f64, 124f64, 125f64,
                05f64, 06f64, 07f64, 15f64, 16f64, 17f64, 25f64, 26f64, 27f64,105f64, 106f64, 107f64, 115f64, 116f64, 117f64, 125f64, 126f64, 127f64,
                21f64, 22f64, 23f64, 31f64, 32f64, 33f64, 41f64, 42f64, 43f64,121f64, 122f64, 123f64, 131f64, 132f64, 133f64, 141f64, 142f64, 143f64,
                23f64, 24f64, 25f64, 33f64, 34f64, 35f64, 43f64, 44f64, 45f64,123f64, 124f64, 125f64, 133f64, 134f64, 135f64, 143f64, 144f64, 145f64,
                25f64, 26f64, 27f64, 35f64, 36f64, 37f64, 45f64, 46f64, 47f64,125f64, 126f64, 127f64, 135f64, 136f64, 137f64, 145f64, 146f64, 147f64, 
                41f64, 42f64, 43f64, 51f64, 52f64, 53f64, 61f64, 62f64, 63f64,141f64, 142f64, 143f64, 151f64, 152f64, 153f64, 161f64, 162f64, 163f64,
                43f64, 44f64, 45f64, 53f64, 54f64, 55f64, 63f64, 64f64, 65f64,143f64, 144f64, 145f64, 153f64, 154f64, 155f64, 163f64, 164f64, 165f64,
                45f64, 46f64, 47f64, 55f64, 56f64, 57f64, 65f64, 66f64, 67f64,145f64, 146f64, 147f64, 155f64, 156f64, 157f64, 165f64, 166f64, 167f64,
            ]
        ).ok().unwrap();

        let col = im2col(&data, 3, 3, 2, 0);

        assert_eq!(col, expect);

    }

    #[test]
    fn test_convolution_pad_array4() {
        let data = Array::from_shape_vec(
            (2,2,3,3),
            vec![
                // [
                    // [
                        1f64, 2f64, 3f64,
                        4f64, 5f64, 6f64,
                        7f64, 8f64, 9f64,
                    // ]
                    // [
                        9f64, 8f64, 7f64,
                        6f64, 5f64, 4f64,
                        3f64, 2f64, 1f64,
                    // ]
                // ]
                // [
                    // [
                        11f64, 12f64, 13f64,
                        14f64, 15f64, 16f64,
                        17f64, 18f64, 19f64,
                    // ]
                    // [
                        19f64, 18f64, 17f64,
                        16f64, 15f64, 14f64,
                        13f64, 12f64, 11f64,
                    // ]
                // ]
            ]
        ).ok().unwrap();

        let expect_data = Array::from_shape_vec(
            (5,5,6,6),
            vec![
                // [ pad
                    // [
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                    // ]
                    // [
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                    // ]
                    // [
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                    // ]
                    // [
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                    // ]
                    // [
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                    // ]
                // ]
                // [
                    // [ pad
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                    // ]
                    // [ pad
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                    // ]
                    // [
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64, // pad
                        0f64, 0f64, 1f64, 2f64, 3f64, 0f64,
                        0f64, 0f64, 4f64, 5f64, 6f64, 0f64,
                        0f64, 0f64, 7f64, 8f64, 9f64, 0f64,
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64, // pad
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64, // pad
                    // ]
                    // [
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64, // pad
                        0f64, 0f64, 9f64, 8f64, 7f64, 0f64,
                        0f64, 0f64, 6f64, 5f64, 4f64, 0f64,
                        0f64, 0f64, 3f64, 2f64, 1f64, 0f64,
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64, // pad
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64, // pad
                    // ]
                    // [ pad
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                    // ]
                /*],*/
                /*[*/
                    // [ pad
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                    // ]
                    // [ pad
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                    // ]
                    // [
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64, // pad
                        0f64, 0f64, 11f64, 12f64, 13f64, 0f64,
                        0f64, 0f64, 14f64, 15f64, 16f64, 0f64,
                        0f64, 0f64, 17f64, 18f64, 19f64, 0f64,
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64, // pad
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64, // pad
                    // ]
                    // [
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64, // pad
                        0f64, 0f64, 19f64, 18f64, 17f64, 0f64,
                        0f64, 0f64, 16f64, 15f64, 14f64, 0f64,
                        0f64, 0f64, 13f64, 12f64, 11f64, 0f64,
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64, // pad
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64, // pad
                    // ]
                    // [ pad
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                    // ]
                /*]*/
                // [ pad
                    // [
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                    // ]
                    // [
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                    // ]
                    // [
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                    // ]
                    // [
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                    // ]
                    // [
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                    // ]
                // ]
                // [ pad
                    // [
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                    // ]
                    // [
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                    // ]
                    // [
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                    // ]
                    // [
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                    // ]
                    // [
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                        0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                    // ]
                // ]
            ]
        ).ok().unwrap();

        let paded = pad_array4(&data, [(1, 2), (2, 1), (1, 2), (2, 1)]);

        assert_eq!(paded, expect_data);
    }

    fn verification_forward(
        x: Array2<f64>,
        filter: Array2<f64>,
        bias: Array2<f64>,
        x_shape: (usize, usize, usize, usize),
        y_shape: (usize, usize, usize, usize),
        filter_shape: (usize, usize, usize, usize),
        stride: usize,
        pad: usize,
    ) -> Array2<f64> {

        let x_2d = x;
        let x_4d = x_2d.to_shared().reshape(x_shape).to_owned();

        let filter_2d = filter;
        let filter_4d = filter_2d.to_shared().reshape(filter_shape).to_owned();
        
        let img = pad_array4(&x_4d, [(0,0), (0,0), (pad, pad), (pad, pad)]);

        let (x_b, x_c, x_h, x_w) = (x_shape);
        let (y_b, y_c, y_h, y_w) = (y_shape);
        let (filter_b, filter_c, filter_h, filter_w) = (filter_shape);

        let stride_count_h = (x_h + 2 * pad - filter_h) / stride + 1;
        let stride_count_w = (x_w + 2 * pad - filter_w) / stride + 1;

        let mut y_4d = Array4::<f64>::zeros(y_shape);

        println!("y_shape: {:?}", y_shape);

        for batch in 0..y_b {  
            for filter_index in 0..y_c {  
                let indexed_filter = filter_4d.index_axis(Axis(0), filter_index);
                let indexed_bias = bias[(0, filter_index)];
                for st_h in 0..stride_count_h {
                    let img_index_h = st_h * stride;
                    for st_w in 0..stride_count_w {
                        let img_index_w = st_w * stride;

                        let ranged_img = img.slice(s![batch, .., img_index_h..=img_index_h+stride, img_index_w..=img_index_w+stride]);

                        // println!("filter:\n{:?}", indexed_filter);
                        // println!("img:\n{:?}", ranged_img);

                        let fil_img = indexed_filter.to_owned() * ranged_img;

                        let mut indexed_filtered_img = 0f64;
                        for n in fil_img {
                            indexed_filtered_img += n;
                        }
                        indexed_filtered_img += indexed_bias;

                        // println!("y_index: {:?}", (batch, filter_index, st_h, st_w));
                        y_4d[(batch, filter_index, st_h, st_w)] = indexed_filtered_img;
                    }   
                }
            }
        }

        let y_2d = y_4d.to_shared().reshape((y_b, y_c*y_h*y_w)).to_owned();

        return y_2d;
    }
}
