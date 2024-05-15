use std::f32::consts::PI;

use anyhow::Result;
use plotters::prelude::*;
use rust_math::{sin_p2_i32, sin_p3_16384, sin_p4_7384, sin_p5_51437};

fn main() -> Result<()> {
    const WIDTH: u32 = 1280;
    const HEIGHT: u32 = 720;
    const RIGHT: f32 = 3.2;
    const TOP: f32 = 1.1;
    const RESOLUTION: i32 = 100;
    const RESOLUTION_AS_F32: f32 = RESOLUTION as f32;
    const LAST: i32 = (RESOLUTION as f32 * RIGHT / PI + 1.0) as i32;
    const STRAIGHT: f32 = (1 << 16) as f32;
    const ONE: f32 = (1 << 30) as f32;

    let root = BitMapBackend::new("sin.png", (WIDTH, HEIGHT)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption("y = sin(x)", ("sans-serif", 50).into_font())
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(-RIGHT..RIGHT, -TOP..TOP)?;

    chart.configure_mesh().draw()?;

    let legend = |x, y, color| {
        PathElement::new(
            vec![(x, y), (x + 20, y)],
            ShapeStyle::from(color).stroke_width(5),
        )
    };

    chart
        .draw_series(LineSeries::new(
            (-LAST..=LAST).map(|x| {
                let y = sin_p2_i32((x as f32 * STRAIGHT / RESOLUTION_AS_F32) as i32) as f32 / ONE;
                let pos = (x as f32 / RESOLUTION_AS_F32 * PI, y);
                pos
            }),
            &BLUE,
        ))?
        .label("y = sin_p2_i32(x)")
        .legend(move |(x, y)| legend(x, y, &BLUE));
    chart
        .draw_series(LineSeries::new(
            (-LAST..=LAST).map(|x| {
                let y = sin_p3_16384((x as f32 * STRAIGHT / RESOLUTION_AS_F32) as i32) as f32 / ONE;
                let pos = (x as f32 / RESOLUTION_AS_F32 * PI, y);
                pos
            }),
            &CYAN,
        ))?
        .label("y = sin_p3_16384(x)")
        .legend(move |(x, y)| legend(x, y, &CYAN));
    chart
        .draw_series(LineSeries::new(
            (-LAST..=LAST).map(|x| {
                let y = sin_p4_7384((x as f32 * STRAIGHT / RESOLUTION_AS_F32) as i32) as f32 / ONE;
                let pos = (x as f32 / RESOLUTION_AS_F32 * PI, y);
                pos
            }),
            &GREEN,
        ))?
        .label("y = sin_p4_7384(x)")
        .legend(move |(x, y)| legend(x, y, &GREEN));
    chart
        .draw_series(LineSeries::new(
            (-LAST..=LAST).map(|x| {
                let y = sin_p5_51437((x as f32 * STRAIGHT / RESOLUTION_AS_F32) as i32) as f32 / ONE;
                let pos = (x as f32 / RESOLUTION_AS_F32 * PI, y);
                pos
            }),
            &YELLOW,
        ))?
        .label("y = sin_p5_51437(x)")
        .legend(move |(x, y)| legend(x, y, &YELLOW));
    chart
        .draw_series(LineSeries::new(
            (-LAST..=LAST).map(|x| {
                let y = (x as f32 / RESOLUTION_AS_F32 * PI).sin();
                let pos = (x as f32 / RESOLUTION_AS_F32 * PI, y);
                pos
            }),
            &RED,
        ))?
        .label("y = std::f32::sin(x)")
        .legend(move |(x, y)| legend(x, y, &RED));

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .label_font(("sans-serif", 20).into_font())
        .position(SeriesLabelPosition::LowerRight)
        .margin(20)
        .draw()?;

    root.present()?;

    Ok(())
}
