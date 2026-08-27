use std::f32::consts::PI;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Lab {
    pub l: f32,
    pub a: f32,
    pub b: f32,
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

pub fn rgb_to_lab(rgb: [f32; 3]) -> Lab {
    let r = linear_channel(rgb[0].clamp(0.0, 1.0));
    let g = linear_channel(rgb[1].clamp(0.0, 1.0));
    let b = linear_channel(rgb[2].clamp(0.0, 1.0));
    // Match skimage.color.rgb2lab, which is the numerical reference used by
    // raster2svg.  These coefficients intentionally differ by a few ulps
    // from the newer IEC matrix: Lab is rounded to integer histogram cells,
    // so substituting another valid matrix changes palette topology.
    let x = (0.412_453 * r + 0.357_58 * g + 0.180_423 * b) / 0.95047;
    let y = (0.212_671 * r + 0.715_16 * g + 0.072_169 * b) / 1.0;
    let z = (0.019_334 * r + 0.119_193 * g + 0.950_227 * b) / 1.08883;
    fn f(value: f32) -> f32 {
        if value > 0.008_856 {
            value.cbrt()
        } else {
            7.787 * value + 16.0 / 116.0
        }
    }
    let fx = f(x);
    let fy = f(y);
    let fz = f(z);
    Lab {
        l: 116.0 * fy - 16.0,
        a: 500.0 * (fx - fy),
        b: 200.0 * (fy - fz),
    }
}

/// Convert CIE Lab (D65) back to clipped sRGB.  This is the inverse of
/// `rgb_to_lab` and is used by the source antialias mixture model.
pub fn lab_to_rgb(lab: Lab) -> [f32; 3] {
    #[inline]
    fn inverse_f(value: f32) -> f32 {
        const DELTA: f32 = 6.0 / 29.0;
        if value > DELTA {
            value.powi(3)
        } else {
            3.0 * DELTA * DELTA * (value - 4.0 / 29.0)
        }
    }

    let fy = (lab.l + 16.0) / 116.0;
    let fx = fy + lab.a / 500.0;
    let fz = fy - lab.b / 200.0;
    let x = 0.95047 * inverse_f(fx);
    let y = inverse_f(fy);
    let z = 1.08883 * inverse_f(fz);
    // perceptual_pipeline uses its explicit float32 Lab-to-sRGB matrix here,
    // rather than skimage's older inverse matrix used by rgb2lab above.
    let r = 3.240_454_2 * x - 1.537_138_5 * y - 0.498_531_4 * z;
    let g = -0.969_266 * x + 1.876_010_8 * y + 0.041_556 * z;
    let b = 0.055_643_4 * x - 0.204_025_9 * y + 1.057_225_2 * z;
    [
        nonlinear_channel(r).clamp(0.0, 1.0),
        nonlinear_channel(g).clamp(0.0, 1.0),
        nonlinear_channel(b).clamp(0.0, 1.0),
    ]
}

#[inline]
pub fn delta_e76(first: Lab, second: Lab) -> f32 {
    ((first.l - second.l).powi(2) + (first.a - second.a).powi(2) + (first.b - second.b).powi(2))
        .sqrt()
}

/// CIEDE2000, used for the fidelity report and perceptual merge gates.
pub fn delta_e2000(first: Lab, second: Lab) -> f32 {
    let c1 = (first.a * first.a + first.b * first.b).sqrt();
    let c2 = (second.a * second.a + second.b * second.b).sqrt();
    let c_bar = (c1 + c2) * 0.5;
    let g = 0.5 * (1.0 - (c_bar.powi(7) / (c_bar.powi(7) + 25_f32.powi(7))).sqrt());
    let a1p = (1.0 + g) * first.a;
    let a2p = (1.0 + g) * second.a;
    let c1p = (a1p * a1p + first.b * first.b).sqrt();
    let c2p = (a2p * a2p + second.b * second.b).sqrt();
    fn hue(b: f32, a: f32) -> f32 {
        let value = b.atan2(a).to_degrees();
        if value < 0.0 {
            value + 360.0
        } else {
            value
        }
    }
    let h1p = if c1p <= 1e-12 { 0.0 } else { hue(first.b, a1p) };
    let h2p = if c2p <= 1e-12 {
        0.0
    } else {
        hue(second.b, a2p)
    };
    let dl = second.l - first.l;
    let dc = c2p - c1p;
    let dh_angle = if c1p * c2p <= 1e-12 {
        0.0
    } else if (h2p - h1p).abs() <= 180.0 {
        h2p - h1p
    } else if h2p <= h1p {
        h2p - h1p + 360.0
    } else {
        h2p - h1p - 360.0
    };
    let dh = 2.0 * (c1p * c2p).sqrt() * (0.5 * dh_angle.to_radians()).sin();
    let l_bar = (first.l + second.l) * 0.5;
    let c_bar_p = (c1p + c2p) * 0.5;
    let h_bar = if c1p * c2p <= 1e-12 {
        h1p + h2p
    } else if (h1p - h2p).abs() <= 180.0 {
        (h1p + h2p) * 0.5
    } else if h1p + h2p < 360.0 {
        (h1p + h2p + 360.0) * 0.5
    } else {
        (h1p + h2p - 360.0) * 0.5
    };
    let t = 1.0 - 0.17 * (h_bar - 30.0).to_radians().cos()
        + 0.24 * (2.0 * h_bar).to_radians().cos()
        + 0.32 * (3.0 * h_bar + 6.0).to_radians().cos()
        - 0.20 * (4.0 * h_bar - 63.0).to_radians().cos();
    let sl = 1.0 + 0.015 * (l_bar - 50.0).powi(2) / (20.0 + (l_bar - 50.0).powi(2)).sqrt();
    let sc = 1.0 + 0.045 * c_bar_p;
    let sh = 1.0 + 0.015 * c_bar_p * t;
    let delta_theta = 30.0 * (-((h_bar - 275.0) / 25.0).powi(2)).exp();
    let rc = 2.0 * (c_bar_p.powi(7) / (c_bar_p.powi(7) + 25_f32.powi(7))).sqrt();
    let rt = -rc * (2.0 * delta_theta * PI / 180.0).sin();
    let l_term = dl / sl;
    let c_term = dc / sc;
    let h_term = dh / sh;
    (l_term * l_term + c_term * c_term + h_term * h_term + rt * c_term * h_term)
        .max(0.0)
        .sqrt()
}

#[inline]
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
mod tests {
    use super::*;

    #[test]
    fn lab_reference_white_and_black() {
        let white = rgb_to_lab([1.0, 1.0, 1.0]);
        let black = rgb_to_lab([0.0, 0.0, 0.0]);
        assert!((white.l - 100.0).abs() < 0.01);
        assert!(black.l.abs() < 0.01);
    }

    #[test]
    fn ciede2000_is_symmetric() {
        let a = rgb_to_lab([0.8, 0.1, 0.2]);
        let b = rgb_to_lab([0.1, 0.4, 0.9]);
        assert!((delta_e2000(a, b) - delta_e2000(b, a)).abs() < 1e-4);
        assert!(delta_e2000(a, a) < 1e-5);
    }

    #[test]
    fn lab_round_trip_preserves_srgb() {
        for rgb in [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.82, 0.13, 0.41]] {
            let restored = lab_to_rgb(rgb_to_lab(rgb));
            for channel in 0..3 {
                assert!((restored[channel] - rgb[channel]).abs() < 2e-4);
            }
        }
    }

    #[test]
    fn lab_matches_skimage_float32_reference() {
        let references = [
            (
                [0.82, 0.13, 0.41],
                Lab {
                    l: 46.552_74,
                    a: 68.523_08,
                    b: 5.206_191_5,
                },
            ),
            (
                [0.1, 0.4, 0.9],
                Lab {
                    l: 46.174_625,
                    a: 26.251_226,
                    b: -70.543_816,
                },
            ),
        ];
        for (rgb, expected) in references {
            let actual = rgb_to_lab(rgb);
            assert!((actual.l - expected.l).abs() < 2e-4);
            assert!((actual.a - expected.a).abs() < 2e-4);
            assert!((actual.b - expected.b).abs() < 2e-4);
        }
    }
}
