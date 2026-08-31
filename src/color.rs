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
        const LINEAR_COEFFICIENT: f32 = f32::from_bits(0x3e03_8027);
        const LINEAR_OFFSET: f32 = f32::from_bits(0x3e0d_3dcb);
        if value > DELTA {
            value.powi(3)
        } else {
            LINEAR_COEFFICIENT * (value - LINEAR_OFFSET)
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

/// Match `skimage.color.lab2rgb` for float32 arrays.  The Paint fitter in the
/// Python reference uses skimage in both directions; its inverse matrix and
/// Lab low branch differ slightly from the preprocessing colour conversion.
pub fn skimage_lab_values_to_rgb(labs: &[Lab]) -> Vec<[f32; 3]> {
    let mut xyz = Vec::<[f32; 3]>::with_capacity(labs.len());
    let mut nonlinear = Vec::<f32>::new();
    let mut nonlinear_positions = Vec::<(usize, usize)>::new();
    for &lab in labs {
        let y = (lab.l + 16.0) / 116.0;
        let x = lab.a / 500.0 + y;
        let z = (y - lab.b / 200.0).max(0.0);
        xyz.push([x, y, z]);
    }
    for (row, value) in xyz.iter().enumerate() {
        for (channel, &entry) in value.iter().enumerate() {
            if entry > 0.206_896_6 {
                nonlinear_positions.push((row, channel));
                nonlinear.push(entry);
            }
        }
    }
    crate::elementary::pow_f32_in_place(&mut nonlinear, 3.0);
    let mut transformed = xyz.clone();
    for (row, value) in transformed.iter_mut().enumerate() {
        for channel in 0..3 {
            if let Some(position) = nonlinear_positions
                .iter()
                .position(|&candidate| candidate == (row, channel))
            {
                value[channel] = nonlinear[position];
            } else {
                value[channel] = (xyz[row][channel] - 16.0 / 116.0) / 7.787;
            }
            value[channel] *= [0.95047, 1.0, 1.08883][channel];
        }
    }
    let mut rgb = Vec::<[f32; 3]>::with_capacity(labs.len());
    for value in transformed {
        rgb.push([
            value[0] * 3.240_481_4 + value[1] * -1.537_151_6 + value[2] * -0.498_536_32,
            value[0] * -0.969_254_9 + value[1] * 1.875_99 + value[2] * 0.041_555_93,
            value[0] * 0.055_646_64 + value[1] * -0.204_041_35 + value[2] * 1.057_311,
        ]);
    }
    let mut gamma_values = Vec::<f32>::new();
    let mut gamma_positions = Vec::<(usize, usize)>::new();
    for (row, value) in rgb.iter().enumerate() {
        for (channel, &entry) in value.iter().enumerate() {
            if entry > 0.003_130_8 {
                gamma_positions.push((row, channel));
                gamma_values.push(entry);
            }
        }
    }
    crate::elementary::pow_f32_in_place(&mut gamma_values, 1.0 / 2.4);
    for (row, value) in rgb.iter_mut().enumerate() {
        for (channel, entry) in value.iter_mut().enumerate() {
            if let Some(position) = gamma_positions
                .iter()
                .position(|&candidate| candidate == (row, channel))
            {
                *entry = 1.055 * gamma_values[position] - 0.055;
            } else {
                *entry *= 12.92;
            }
            *entry = entry.clamp(0.0, 1.0);
        }
    }
    rgb
}

/// Convert a contiguous Lab image with the same float32 array operation
/// order used by `perceptual_pipeline.lab_to_srgb`.
pub fn lab_pixels_to_rgb(labs: &[Lab]) -> Vec<[f32; 3]> {
    const DELTA: f32 = f32::from_bits(0x3e53_dcb1);
    const LINEAR_COEFFICIENT: f32 = f32::from_bits(0x3e03_8027);
    const LINEAR_OFFSET: f32 = f32::from_bits(0x3e0d_3dcb);
    let mut f = Vec::<[f32; 3]>::with_capacity(labs.len());
    for &lab in labs {
        let fy = (lab.l + 16.0) / 116.0;
        f.push([lab.a / 500.0 + fy, fy, fy - lab.b / 200.0]);
    }
    let mut xyz = vec![[0.0_f32; 3]; labs.len()];
    for channel in 0..3 {
        let mut powered: Vec<f32> = f.iter().map(|value| value[channel]).collect();
        crate::elementary::pow_f32_in_place(&mut powered, 3.0);
        for (index, value) in f.iter().enumerate() {
            let nonlinear = if value[channel] > DELTA {
                powered[index]
            } else {
                LINEAR_COEFFICIENT * (value[channel] - LINEAR_OFFSET)
            };
            xyz[index][channel] = nonlinear * [0.95047, 1.0, 1.08883][channel];
        }
    }
    let mut rgb = Vec::<[f32; 3]>::with_capacity(labs.len());
    for value in xyz {
        let r = value[2].mul_add(
            -0.498_531_4,
            value[1].mul_add(-1.537_138_5, value[0] * 3.240_454_2),
        );
        let g = value[2].mul_add(
            0.041_556,
            value[1].mul_add(1.876_010_8, value[0] * -0.969_266),
        );
        let b = value[2].mul_add(
            1.057_225_2,
            value[1].mul_add(-0.204_025_9, value[0] * 0.055_643_4),
        );
        rgb.push([r, g, b]);
    }
    let mut powered: Vec<f32> = rgb
        .iter()
        .flat_map(|value| value.iter().map(|&channel| channel.max(0.0)))
        .collect();
    crate::elementary::pow_f32_in_place(&mut powered, 1.0 / 2.4);
    for (index, value) in rgb.iter_mut().enumerate() {
        for channel in 0..3 {
            let linear = value[channel];
            value[channel] = if linear <= 0.003_130_8 {
                12.92 * linear
            } else {
                1.055 * powered[index * 3 + channel] - 0.055
            }
            .clamp(0.0, 1.0);
        }
    }
    rgb
}

#[inline]
pub fn delta_e76(first: Lab, second: Lab) -> f32 {
    ((first.l - second.l).powi(2) + (first.a - second.a).powi(2) + (first.b - second.b).powi(2))
        .sqrt()
}

/// Symmetric CIE94-style local colour distance.
///
/// Bilateral smoothing compares only nearby pixels and needs a monotone range
/// weight rather than a reporting-grade colour difference.  This form avoids
/// the trigonometric CIEDE2000 path, stays entirely in float32, and is readily
/// vectorized by LLVM while retaining lightness/chroma/hue scaling.
#[inline]
pub fn delta_e94_local(first: Lab, second: Lab) -> f32 {
    let dl = first.l - second.l;
    let c1 = first.a.hypot(first.b);
    let c2 = second.a.hypot(second.b);
    let dc = c1 - c2;
    let da = first.a - second.a;
    let db = first.b - second.b;
    let dh_squared = (da * da + db * db - dc * dc).max(0.0);
    let chroma = 0.5 * (c1 + c2);
    let sc = 1.0 + 0.045 * chroma;
    let sh = 1.0 + 0.015 * chroma;
    (dl * dl + (dc / sc).powi(2) + dh_squared / (sh * sh)).sqrt()
}

/// CIEDE2000, used for the fidelity report and perceptual merge gates.
pub fn delta_e2000(first: Lab, second: Lab) -> f32 {
    delta_e2000_pairs(&[first], &[second])[0]
}

/// CIEDE2000 over contiguous pairs, evaluating `atan2` in portable SIMD
/// batches and preserving NumPy's float32 degree/radian constants.
pub fn delta_e2000_pairs(first: &[Lab], second: &[Lab]) -> Vec<f32> {
    assert_eq!(first.len(), second.len());
    delta_e2000_map(first, |index| second[index])
}

/// CIEDE2000 from every contiguous Lab value to one common reference.
///
/// Keeping the comparison in one batch lets the elementary functions process
/// complete SIMD vectors. Calling `delta_e2000` in a tight nearest-colour loop
/// would otherwise evaluate a padded vector for every candidate.
pub fn delta_e2000_to_many(first: &[Lab], second: Lab) -> Vec<f32> {
    delta_e2000_map(first, |_| second)
}

#[derive(Default)]
pub(crate) struct DeltaE2000Workspace {
    initial: Vec<(usize, Lab, Lab, f32, f32)>,
    atan_y_first: Vec<f32>,
    atan_x_first: Vec<f32>,
    atan_y_second: Vec<f32>,
    atan_x_second: Vec<f32>,
    atan_first: Vec<f32>,
    atan_second: Vec<f32>,
}

/// Return the exact first minimum produced by `delta_e2000_to_many` while
/// retaining all temporary SIMD buffers for the next palette lookup.
pub(crate) fn delta_e2000_nearest(
    first: &[Lab],
    second: Lab,
    maximum_distance: f32,
    workspace: &mut DeltaE2000Workspace,
) -> Option<(usize, f32)> {
    if first.is_empty() {
        return None;
    }
    workspace.initial.clear();
    workspace.atan_y_first.clear();
    workspace.atan_x_first.clear();
    workspace.atan_y_second.clear();
    workspace.atan_x_second.clear();
    workspace.initial.reserve(first.len());
    workspace.atan_y_first.reserve(first.len());
    workspace.atan_x_first.reserve(first.len());
    workspace.atan_y_second.reserve(first.len());
    workspace.atan_x_second.reserve(first.len());
    for (index, &value) in first.iter().enumerate() {
        // The CIEDE2000 chroma/hue quadratic is non-negative because
        // |R_T| <= 2, so |delta-L| / S_L is an exact lower bound. Palette
        // quantization only needs candidates inside its largest acceptance
        // radius; anything beyond this bound creates a new representative
        // regardless of its ordering among rejected colours.
        let l_bar = (value.l + second.l) * 0.5;
        let l_offset = l_bar - 50.0;
        let sl = 1.0 + 0.015 * l_offset.powi(2) / (20.0 + l_offset.powi(2)).sqrt();
        let lightness_lower_bound = (value.l - second.l).abs() / sl;
        if lightness_lower_bound > maximum_distance + 1e-4 {
            continue;
        }
        let c1 = value.a.hypot(value.b);
        let c2 = second.a.hypot(second.b);
        let c_bar = (c1 + c2) * 0.5;
        let g = 0.5 * (1.0 - (c_bar.powi(7) / (c_bar.powi(7) + 25_f32.powi(7))).sqrt());
        let a1p = (1.0 + g) * value.a;
        let a2p = (1.0 + g) * second.a;
        let c1p = a1p.hypot(value.b);
        let c2p = a2p.hypot(second.b);
        workspace.initial.push((index, value, second, c1p, c2p));
        workspace.atan_y_first.push(value.b);
        workspace.atan_x_first.push(a1p);
        workspace.atan_y_second.push(second.b);
        workspace.atan_x_second.push(a2p);
    }
    crate::elementary::atan2_f32_into(
        &workspace.atan_y_first,
        &workspace.atan_x_first,
        &mut workspace.atan_first,
    );
    crate::elementary::atan2_f32_into(
        &workspace.atan_y_second,
        &workspace.atan_x_second,
        &mut workspace.atan_second,
    );
    const RAD2DEG: f32 = 180.0_f32 / std::f32::consts::PI;
    const DEG2RAD: f32 = std::f32::consts::PI / 180.0_f32;
    let mut best = (usize::MAX, f32::INFINITY);
    for ((&(index, value, reference, c1p, c2p), &atan_first), &atan_second) in workspace
        .initial
        .iter()
        .zip(&workspace.atan_first)
        .zip(&workspace.atan_second)
    {
        let h1p = (atan_first * RAD2DEG) % 360.0;
        let h1p = if h1p < 0.0 { h1p + 360.0 } else { h1p };
        let h2p = (atan_second * RAD2DEG) % 360.0;
        let h2p = if h2p < 0.0 { h2p + 360.0 } else { h2p };
        let distance = delta_e2000_after_hue(value, reference, c1p, c2p, h1p, h2p, DEG2RAD);
        if distance <= maximum_distance && distance.total_cmp(&best.1).is_lt() {
            best = (index, distance);
        }
    }
    (best.0 != usize::MAX).then_some(best)
}

fn delta_e2000_map(first: &[Lab], second: impl Fn(usize) -> Lab) -> Vec<f32> {
    let mut initial = Vec::with_capacity(first.len());
    let mut atan_y_first = Vec::with_capacity(first.len());
    let mut atan_x_first = Vec::with_capacity(first.len());
    let mut atan_y_second = Vec::with_capacity(first.len());
    let mut atan_x_second = Vec::with_capacity(first.len());
    for (index, &first) in first.iter().enumerate() {
        let second = second(index);
        let c1 = first.a.hypot(first.b);
        let c2 = second.a.hypot(second.b);
        let c_bar = (c1 + c2) * 0.5;
        let g = 0.5 * (1.0 - (c_bar.powi(7) / (c_bar.powi(7) + 25_f32.powi(7))).sqrt());
        let a1p = (1.0 + g) * first.a;
        let a2p = (1.0 + g) * second.a;
        let c1p = a1p.hypot(first.b);
        let c2p = a2p.hypot(second.b);
        initial.push((first, second, c1p, c2p));
        atan_y_first.push(first.b);
        atan_x_first.push(a1p);
        atan_y_second.push(second.b);
        atan_x_second.push(a2p);
    }
    let atan_first = crate::elementary::atan2_f32(&atan_y_first, &atan_x_first);
    let atan_second = crate::elementary::atan2_f32(&atan_y_second, &atan_x_second);
    // NumPy defines RAD2DEG and DEG2RAD from float32 PI operands. Rust's
    // `to_degrees` uses a differently rounded precomputed constant.
    const RAD2DEG: f32 = 180.0_f32 / std::f32::consts::PI;
    const DEG2RAD: f32 = std::f32::consts::PI / 180.0_f32;
    initial
        .into_iter()
        .zip(atan_first.into_iter().zip(atan_second))
        .map(|((first, second, c1p, c2p), (atan_first, atan_second))| {
            let h1p = (atan_first * RAD2DEG) % 360.0;
            let h1p = if h1p < 0.0 { h1p + 360.0 } else { h1p };
            let h2p = (atan_second * RAD2DEG) % 360.0;
            let h2p = if h2p < 0.0 { h2p + 360.0 } else { h2p };
            delta_e2000_after_hue(first, second, c1p, c2p, h1p, h2p, DEG2RAD)
        })
        .collect()
}

#[inline]
fn delta_e2000_after_hue(
    first: Lab,
    second: Lab,
    c1p: f32,
    c2p: f32,
    h1p: f32,
    h2p: f32,
    deg2rad: f32,
) -> f32 {
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
    let dh = 2.0 * (c1p * c2p).sqrt() * (0.5 * dh_angle * deg2rad).sin();
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
    let t = 1.0 - 0.17 * ((h_bar - 30.0) * deg2rad).cos()
        + 0.24 * ((2.0 * h_bar) * deg2rad).cos()
        + 0.32 * ((3.0 * h_bar + 6.0) * deg2rad).cos()
        - 0.20 * ((4.0 * h_bar - 63.0) * deg2rad).cos();
    let sl = 1.0 + 0.015 * (l_bar - 50.0).powi(2) / (20.0 + (l_bar - 50.0).powi(2)).sqrt();
    let sc = 1.0 + 0.045 * c_bar_p;
    let sh = 1.0 + 0.015 * c_bar_p * t;
    let delta_theta = 30.0 * (-((h_bar - 275.0) / 25.0).powi(2)).exp();
    let rc = 2.0 * (c_bar_p.powi(7) / (c_bar_p.powi(7) + 25_f32.powi(7))).sqrt();
    let rt = -rc * (2.0 * delta_theta * deg2rad).sin();
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
    fn local_cie94_distance_is_symmetric_and_zero_on_identity() {
        let a = rgb_to_lab([0.82, 0.13, 0.41]);
        let b = rgb_to_lab([0.76, 0.18, 0.38]);
        assert!((delta_e94_local(a, b) - delta_e94_local(b, a)).abs() < 1e-6);
        assert!(delta_e94_local(a, a) < 1e-6);
        assert!(delta_e94_local(a, b).is_finite());
    }

    #[test]
    fn ciede2000_common_reference_batch_matches_scalar_dispatch() {
        let reference = rgb_to_lab([0.27, 0.61, 0.84]);
        let values: Vec<Lab> = (0..33)
            .map(|index| {
                let amount = index as f32 / 32.0;
                rgb_to_lab([amount, 1.0 - amount, 0.2 + 0.6 * amount])
            })
            .collect();
        let scalar: Vec<f32> = values
            .iter()
            .map(|&value| delta_e2000(value, reference))
            .collect();
        assert_eq!(delta_e2000_to_many(&values, reference), scalar);
    }

    #[test]
    fn reusable_nearest_matches_common_reference_batch() {
        let reference = rgb_to_lab([0.37, 0.21, 0.82]);
        let values: Vec<Lab> = (0..33)
            .map(|index| {
                let amount = index as f32 / 32.0;
                rgb_to_lab([amount, 0.8 - 0.6 * amount, 0.1 + 0.7 * amount])
            })
            .collect();
        let expected = delta_e2000_to_many(&values, reference)
            .into_iter()
            .enumerate()
            .min_by(|left, right| {
                left.1
                    .total_cmp(&right.1)
                    .then_with(|| left.0.cmp(&right.0))
            });
        let mut workspace = DeltaE2000Workspace::default();
        assert_eq!(
            delta_e2000_nearest(&values, reference, f32::INFINITY, &mut workspace),
            expected
        );
        assert_eq!(
            delta_e2000_nearest(&[], reference, f32::INFINITY, &mut workspace),
            None
        );
        assert_eq!(
            delta_e2000_nearest(&values, reference, f32::INFINITY, &mut workspace),
            expected
        );
    }

    #[test]
    fn lightness_bound_keeps_every_candidate_inside_radius() {
        let reference = Lab {
            l: 51.0,
            a: 28.0,
            b: -31.0,
        };
        let values = (-20..=120)
            .map(|lightness| Lab {
                l: lightness as f32,
                a: (lightness % 37) as f32 - 18.0,
                b: (lightness % 29) as f32 - 14.0,
            })
            .collect::<Vec<_>>();
        let distances = delta_e2000_to_many(&values, reference);
        let radius = 5.0;
        let expected = distances
            .iter()
            .enumerate()
            .filter(|&(_, distance)| *distance <= radius)
            .min_by(|left, right| left.1.total_cmp(right.1).then_with(|| left.0.cmp(&right.0)))
            .map(|(index, &distance)| (index, distance));
        let mut workspace = DeltaE2000Workspace::default();
        assert_eq!(
            delta_e2000_nearest(&values, reference, radius, &mut workspace),
            expected
        );
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
