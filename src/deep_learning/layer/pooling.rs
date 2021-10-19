use ndarray::{
    Array2,
    ArrayView1,
    Axis,
};

use crate::deep_learning::layer::*;
// use crate::deep_learning::optimizer::*;
// use crate::deep_learning::common::*;


pub struct Pooling {
    x: Box<dyn NetworkLayer>,
    y: Option<Array2<f64>>,
    x_shape: (usize, usize, usize, usize),
    filter_h: usize,
    filter_w: usize,
    stride: usize,
    padding: usize,
}
impl Pooling {
    pub fn new<TX>(x: TX, x_shape: (usize, usize, usize, usize), filter_h: usize, filter_w: usize, stride: usize, padding: usize)
        -> Pooling
        where   TX : NetworkLayer + 'static,
    {
        Pooling {
            x: Box::new(x),
            y: None,
            x_shape: x_shape,
            filter_h: filter_h,
            filter_w: filter_w,
            stride: stride,
            padding: padding,
        }
    }
}
impl NetworkLayer for Pooling {
    fn forward(&mut self, is_learning: bool) -> Array2<f64> {
        if self.y.is_none() {
            
            let x = self.x.forward(is_learning);
            let (batch_num, channel_num, x_h, x_w) = self.x_shape;
            let step_h = (x_h + 2 * self.padding - self.filter_h) / self.stride + 1;
            let step_w = (x_w + 2 * self.padding - self.filter_w) / self.stride + 1;

            let x_4d = x.to_shared().reshape(self.x_shape).to_owned();
            let col = im2col(&x_4d, self.filter_h, self.filter_w, self.stride, self.padding);

            let shaped_col = col.to_shared().reshape((batch_num*channel_num*step_h*step_w , self.filter_h*self.filter_w));

            let col_max = shaped_col.map_axis(
                Axis(1),
                |nums: ArrayView1<f64>| -> f64 {
                    let mut max = nums[0];
                    for n in nums {
                        if max < *n {
                            max = *n;
                        }
                    }
                    return max;
                }
            );

            let mut col_max_3d = col_max.to_shared().reshape((batch_num, step_h*step_w, channel_num)).to_owned();
            col_max_3d.swap_axes(1, 2);

            let y = col_max_3d.to_shared().reshape((batch_num, channel_num*step_h*step_w)).to_owned();
            self.y = Some(y);
        }
        self.y.clone().unwrap()
    }
    fn backward(&mut self, dout: Array2<f64>) {
        ;
    }
    fn set_value(&mut self, value: &Array2<f64>) {
        self.x.set_value(value);
        self.clean();
    }
    fn set_lbl(&mut self, value: &Array2<f64>) {
        self.x.set_lbl(value);
        self.clean();
    }
    fn clean(&mut self) {
        self.y = None;
    }
    fn plot(&self){
        self.x.plot();
    }
    fn weight_squared_sum(&self) -> f64 {
        return self.x.weight_squared_sum();
    }
    fn weight_sum(&self) -> f64 {
        return self.x.weight_sum();
    }
}

#[cfg(test)]
mod test {
    use super::*;

    use ndarray::{
        Array,
        arr2,
    };

    #[test]
    fn test_pooling_forward() {
        // B:2, C:2 H:6 W:6
        let value = DirectValue::new(
            Array::from_shape_vec(
                (2,72),
                vec![
                    01f64, 02f64, 03f64, 04f64, 05f64, 06f64,
                    11f64, 12f64, 13f64, 14f64, 15f64, 16f64,
                    21f64, 22f64, 23f64, 24f64, 25f64, 26f64,
                    31f64, 32f64, 33f64, 34f64, 35f64, 36f64,
                    41f64, 42f64, 43f64, 44f64, 45f64, 46f64,
                    51f64, 52f64, 53f64, 54f64, 55f64, 56f64,

                    101f64, 102f64, 103f64, 104f64, 105f64, 106f64,
                    111f64, 112f64, 113f64, 114f64, 115f64, 116f64,
                    121f64, 122f64, 123f64, 124f64, 125f64, 126f64,
                    131f64, 132f64, 133f64, 134f64, 135f64, 136f64,
                    141f64, 142f64, 143f64, 144f64, 145f64, 146f64,
                    151f64, 152f64, 153f64, 154f64, 155f64, 156f64,

                    201f64, 202f64, 203f64, 204f64, 205f64, 306f64,
                    211f64, 212f64, 213f64, 214f64, 215f64, 316f64,
                    221f64, 222f64, 223f64, 224f64, 225f64, 326f64,
                    231f64, 232f64, 233f64, 234f64, 235f64, 336f64,
                    241f64, 242f64, 243f64, 244f64, 245f64, 346f64,
                    251f64, 252f64, 253f64, 254f64, 255f64, 356f64,

                    401f64, 402f64, 403f64, 404f64, 405f64, 406f64,
                    411f64, 412f64, 413f64, 414f64, 415f64, 416f64,
                    421f64, 422f64, 423f64, 424f64, 425f64, 426f64,
                    431f64, 432f64, 433f64, 434f64, 435f64, 436f64,
                    441f64, 442f64, 443f64, 444f64, 445f64, 446f64,
                    451f64, 452f64, 453f64, 454f64, 455f64, 456f64,
                ]
            ).ok().unwrap()
        );

        let expect = Array::from_shape_vec(
            (2,18),
            vec![
                12f64, 14f64, 16f64,
                32f64, 34f64, 36f64,
                52f64, 54f64, 56f64,

                112f64, 114f64, 116f64,
                132f64, 134f64, 136f64,
                152f64, 154f64, 156f64,

                212f64, 214f64, 316f64,
                232f64, 234f64, 336f64,
                252f64, 254f64, 356f64,

                412f64, 414f64, 416f64,
                432f64, 434f64, 436f64,
                452f64, 454f64, 456f64,
            ]
        ).ok().unwrap();

        let filter_h = 2;
        let filter_w = 2;
        let stride = 2;
        let pad = 0;
        let mut pool = Pooling::new(value, (2, 2, 6, 6), filter_h, filter_w, stride, pad);

        let y = pool.forward(false);

        assert_eq!(y, expect);
    }

    #[test]
    fn aa() {
        let a = arr2(&
            [
                [1f64,2f64,3f64],
                [6f64,2f64,3f64],
                [6f64,9f64,3f64],
                [6f64,9f64,12f64],
            ]
        );

        let aa = a.map_axis(
            Axis(1),
            |nums: ArrayView1<f64>| -> f64 {
                let mut max = nums[0];
                for n in nums {
                    if max < *n {
                        max = *n;
                    }
                }
                return max;
            }
        );
        println!("{:?}", aa);
    }
}