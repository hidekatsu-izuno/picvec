//! OKLab colour processing. All three coordinates are scaled by 100.
//! Thus white has L=100 and a distance of 1 means 0.01 in standard OKLab.

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Oklab {
    pub l: f32,
    pub a: f32,
    pub b: f32,
}

impl Oklab {
    pub fn distance(self, other: Self) -> f32 {
        ((self.l - other.l).powi(2) + (self.a - other.a).powi(2) + (self.b - other.b).powi(2))
            .sqrt()
    }
}

#[inline]
fn linear_channel(value: f32) -> f32 {
    if value <= 0.04045 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    }
}

#[inline]
fn nonlinear_channel(value: f32) -> f32 {
    if value <= 0.003_130_8 {
        12.92 * value
    } else {
        1.055 * value.powf(1.0 / 2.4) - 0.055
    }
}

/// Display sRGB to 100-scaled OKLab, using Ottosson's 2021 matrices.
/// https://bottosson.github.io/posts/oklab/
pub fn rgb_to_oklab(rgb: [f32; 3]) -> Oklab {
    let [r, g, b] = rgb.map(|value| linear_channel(value.clamp(0.0, 1.0)));
    let l = (0.412_221_46 * r + 0.536_332_55 * g + 0.051_445_995 * b).cbrt();
    let m = (0.211_903_5 * r + 0.680_699_5 * g + 0.107_396_96 * b).cbrt();
    let s = (0.088_302_46 * r + 0.281_718_85 * g + 0.629_978_7 * b).cbrt();
    Oklab {
        l: 100.0 * (0.210_454_26 * l + 0.793_617_8 * m - 0.004_072_047 * s),
        a: 100.0 * (1.977_998_5 * l - 2.428_592_2 * m + 0.450_593_7 * s),
        b: 100.0 * (0.025_904_037 * l + 0.782_771_77 * m - 0.808_675_77 * s),
    }
}

/// 100-scaled OKLab to display sRGB. Clip only after the inverse transform.
pub fn oklab_to_rgb(value: Oklab) -> [f32; 3] {
    let (l, a, b) = (value.l / 100.0, value.a / 100.0, value.b / 100.0);
    let l = (l + 0.396_337_78 * a + 0.215_803_76 * b).powi(3);
    let m = (value.l / 100.0 - 0.105_561_346 * a - 0.063_854_17 * b).powi(3);
    let s = (value.l / 100.0 - 0.089_484_18 * a - 1.291_485_5 * b).powi(3);
    [
        4.076_741_7 * l - 3.307_711_6 * m + 0.230_969_94 * s,
        -1.268_438 * l + 2.609_757_4 * m - 0.341_319_38 * s,
        -0.004_196_086_3 * l - 0.703_418_6 * m + 1.707_614_7 * s,
    ]
    .map(|channel| nonlinear_channel(channel).clamp(0.0, 1.0))
}

pub fn oklab_values_to_rgb(values: &[Oklab]) -> Vec<[f32; 3]> {
    values.iter().copied().map(oklab_to_rgb).collect()
}

/// Euclidean distance in 100-scaled OKLab, used by every perceptual gate.
pub fn delta_e_ok(first: Oklab, second: Oklab) -> f32 {
    first.distance(second)
}

pub fn delta_e_ok_pairs(first: &[Oklab], second: &[Oklab]) -> Vec<f32> {
    assert_eq!(first.len(), second.len());
    first
        .iter()
        .zip(second)
        .map(|(&a, &b)| delta_e_ok(a, b))
        .collect()
}

pub fn delta_e_ok_to_many(first: &[Oklab], second: Oklab) -> Vec<f32> {
    first.iter().map(|&a| delta_e_ok(a, second)).collect()
}

pub fn relative_luminance(rgb: [f32; 3]) -> f32 {
    0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
}

pub fn rgb_hex(rgb: [f32; 3]) -> String {
    format!(
        "#{:02x}{:02x}{:02x}",
        (rgb[0].clamp(0.0, 1.0) * 255.0).round() as u8,
        (rgb[1].clamp(0.0, 1.0) * 255.0).round() as u8,
        (rgb[2].clamp(0.0, 1.0) * 255.0).round() as u8
    )
}

#[cfg(test)]
include!("../tests/unit/color.rs");
