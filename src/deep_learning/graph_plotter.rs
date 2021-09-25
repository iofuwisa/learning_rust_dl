use plotters::prelude::*;

use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use rand_xorshift::XorShiftRng;

use num_traits::sign::Signed;

pub fn prot_correct_rate(correct_rates: Vec<f64>, path: &str) -> Result<(), Box<dyn std::error::Error>> {
    // image size
    let image_width = 1080;
    let image_height = 720;

    // BitMapBackend for generate file
    let root = BitMapBackend::new
        (path, (image_width, image_height)).into_drawing_area();

    // Background is white
    root.fill(&WHITE)?;
    
    let caption = "";
    let font = ("sans-serif", 20);

    let x_range = 0u32..(correct_rates.len() as u32);
    let y_range = 0f64..1f64;

    // Graph setting
    let mut chart = ChartBuilder::on(&root)
        .caption(caption, font.into_font())
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(
            x_range.clone(),
            y_range.clone(),
        )?;
    
    // Draw grid
    chart.configure_mesh().draw()?;

    // Add x data to correct rate
    let correct_rates_with_x = correct_rates.clone().into_iter()
        .zip(x_range.clone().collect::<Vec<u32>>().into_iter())
        .map(|(y, x)| (x, y));

    // Draw correct rate
    chart
        .draw_series(LineSeries::new(
            correct_rates_with_x,
            &RED
        ))?
        .label("correct answer rate")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    // Line setting
    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    Ok(())
}

pub fn prot_histogram(data: Vec<f64>, path: &str) -> Result<(), Box<dyn std::error::Error>> {
    // image size
    let image_width = 1080;
    let image_height = 720;

    // BitMapBackend for generate file
    let root = BitMapBackend::new
        (path, (image_width, image_height)).into_drawing_area();

    // Background is white
    root.fill(&WHITE)?;
    
    let caption = "";
    let font = ("sans-serif", 20);

    // let x_range = 0u32..(correct_rates.len() as u32);
    // let y_range = 0f64..1f64;

    // Graph setting
    let mut chart = ChartBuilder::on(&root)
        .caption(caption, font.into_font())
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(
            (-4f64..4f64).step(0.1).use_round().into_segmented(),
            0u32..5000u32
        )?;

    chart
        .configure_mesh()
        .disable_x_mesh()
        .bold_line_style(&WHITE.mix(0.3))
        .axis_desc_style(font)
        .draw()?;

    chart.draw_series(
        Histogram::vertical(&chart)
            .style(RED.mix(0.5).filled())
            .data(data.iter().map(|x: &f64| (*x, 1))),
    )?;

    Ok(())
}


#[cfg(test)]
mod graph_plotters_test {
    use super::*;

    use crate::deep_learning::common::*;
    
    #[test]
    fn test_prot_correct_rate() {
        let data = vec![0.1, 0.5, 0.8, 0.5, 0.0, 0.5];
        prot_correct_rate(data, "./plot.png");
    }

    #[test]
    fn test_prot_histogram() {
        let data = norm_random(100000);
        prot_histogram(data, "./plot.png");
    }
    
}