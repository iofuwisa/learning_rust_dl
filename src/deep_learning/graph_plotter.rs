// use plotters::prelude::*;

// pub fn prot(data: &Vec<(&str, f64)>) -> Result<(), Box<dyn std::error::Error>> {


//     // x軸 : 日付のVector
//     let xs: Vec<&str> = data.iter()
//                             .map(|(x, _)| *x)
//                             .collect();
//     // y軸: 値のVector
//     let ys: Vec<f64> = data.iter()
//                             .map(|(_, y)| *y)
//                             .collect();

//     // image size
//     let image_width = 1080;
//     let image_height = 720;

//     // BitMapBackend for generate file
//     let root = BitMapBackend::new
//         ("plot.png", (image_width, image_height)).into_drawing_area();

//     // Background is white
//     root.fill(&WHITE)?;

//     // Calc max and min in y axis
//     let (y_min, y_max) = ys.iter()
//                          .fold(
//                            (0.0/0.0, 0.0/0.0),
//                            |(m,n), v| (v.min(m), v.max(n))
//                           );
    
//     let caption = "Sample Plot";
//     let font = ("sans-serif", 20);

//     let mut chart = ChartBuilder::on(&root)
//         .caption(caption, font.into_font()) // キャプションのフォントやサイズ
//         .margin(10)                         // 上下左右全ての余白
//         .x_label_area_size(16)              // x軸ラベル部分の余白
//         .y_label_area_size(42)              // y軸ラベル部分の余白
//         .build_cartesian_2d(                // x軸とy軸の数値の範囲を指定する
//             *xs.first().unwrap()..*xs.last().unwrap(), // x軸の範囲
//             y_min..y_max                               // y軸の範囲
//         )?;
    
//     // x軸y軸、グリッド線などを描画
//     chart.configure_mesh().draw()?;

//     // 折れ線グラフの定義＆描画
//     let line_series = LineSeries::new(
//                         xs.iter()
//                         .zip(ys.iter())
//                         .map(|(x, y)| (*x, *y)),
//                         &RED
//                     );
//     chart.draw_series(line_series)?;

//     Ok(())

// }