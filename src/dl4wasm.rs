use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use image::*;
use js_sys::*;
use web_sys::{Element, CanvasRenderingContext2d, HtmlCanvasElement, ImageData};
use ndarray::{
    s,
    Array2
};

fn console_log(s: &str) {
    web_sys::console::log_1(&JsValue::from(s));
}

#[wasm_bindgen(start)]
pub fn run() {
    console_log("start!!!!!!!!!!!!!!!!!");
}

#[wasm_bindgen]
pub fn guess() {
    console_log("guess!!!!!!!!!!!!!!");
    let document = web_sys::window().unwrap().document().unwrap();
    let canvas: Element = document.get_element_by_id("mainCanvas").unwrap();
    console_log(&format!("id: {}", canvas.id()));
    let canvas: web_sys::HtmlCanvasElement = canvas
        .dyn_into::<web_sys::HtmlCanvasElement>()
        .map_err(|_| ())
        .unwrap();

    let context = canvas
        .get_context("2d")
        .unwrap()
        .unwrap()
        .dyn_into::<web_sys::CanvasRenderingContext2d>()
        .unwrap();

    let image_width = canvas.width();
    let image_height = canvas.height();

    let data: ImageData = context
        .get_image_data(0.0, 0.0, image_width as f64, image_height as f64)
        .unwrap();
    
    // sRGB
    let image_data_srgb = data.data().to_vec();

    let mut image_data_gray = Vec::<u8>::with_capacity((image_width*image_height) as usize);
    for i in 0..image_data_srgb.len()/4 {
        let mut max = image_data_srgb[i*4];
        for j in 1..4 {
            if max < image_data_srgb[i*4+j] {
                max = image_data_srgb[i*4+j]
            }
        }
        image_data_gray.push(max);
    }

    console_log(&format!("image_data_srgb: {:?}", image_data_srgb.len()));
    console_log(&format!("image_data_srgb: {:?}", image_data_srgb));
    console_log(&format!("image_data_gray: {:?}", image_data_gray.len()));
    console_log(&format!("image_data_gray: {:?}", image_data_gray));

    let img = Array2::from_shape_vec((image_width as usize, image_height as usize), image_data_gray).unwrap();
    let mut converted_img = Array2::<f64>::zeros((28, 28));

    // convert 28*28
    for hi in 0..28-1 {
        let hi_range_head = (image_height as f64 *  hi as f64         / 28f64).round() as usize;
        let hi_range_tail = (image_height as f64 * (hi as f64 + 1f64) / 28f64).round() as usize;
        let hi_range = hi_range_head..hi_range_tail;
        for wi in 0..28-1 {
            let wi_range_head = (image_width as f64 *  wi as f64          / 28f64).round() as usize;
            let wi_range_tail = (image_width as f64 * (wi as f64 + 1f64 ) / 28f64).round() as usize;
            let wi_range = wi_range_head..wi_range_tail;

            let l = img.slice(s![hi_range.clone(), wi_range]);
            let mut max = l[(0, 0)];
            for n in l {
                if max < *n {
                    max = *n;
                }
            }
            converted_img[(hi, wi)] = max as f64 / 255f64;
        }
    }7

    for i in 0..28 {
        let mut line = "".to_string();
        for j in 0..28 {
            if converted_img[(i, j)] > 0f64 {
                line.push_str("x");
            } else {
                line.push_str(" ");
            }
        }
        console_log(&line);
    }
}