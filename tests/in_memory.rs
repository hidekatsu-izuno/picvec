use image::{DynamicImage, ImageFormat, Rgba, RgbaImage};
use picvec::{vectorize, vectorize_bytes, Config};
use std::io::Cursor;

fn fixture() -> Vec<u8> {
    let image = RgbaImage::from_fn(24, 24, |x, y| {
        if (4..20).contains(&x) && (4..20).contains(&y) {
            Rgba([210, 40, 60, 128])
        } else {
            Rgba([0, 0, 0, 0])
        }
    });
    let mut bytes = Cursor::new(Vec::new());
    DynamicImage::ImageRgba8(image)
        .write_to(&mut bytes, ImageFormat::Png)
        .unwrap();
    bytes.into_inner()
}

#[test]
fn memory_and_file_conversion_match_including_transparency() {
    let bytes = fixture();
    let config = Config {
        adaptive_refinement: false,
        rayon_threads: 1,
        ..Config::default()
    };
    let (svg, summary) = vectorize_bytes(&bytes, &config).unwrap();
    let directory = tempfile::tempdir().unwrap();
    let input = directory.path().join("image.png");
    let output = directory.path().join("image.svg");
    std::fs::write(&input, bytes).unwrap();
    let native = vectorize(&input, &output, &config).unwrap();
    assert_eq!(svg, std::fs::read_to_string(output).unwrap());
    assert_eq!(summary.svg.objects, native.svg.objects);
    assert!(summary.source_alpha.detected);
    assert!(summary.output.as_os_str().is_empty());
    assert!(!svg.contains("<mask"));
    assert!(!svg.contains(" mask="));
    assert!(svg.contains("fill-opacity="));
    for scale in [1, 4] {
        let tree = resvg::usvg::Tree::from_str(&svg, &resvg::usvg::Options::default()).unwrap();
        let mut pixels = resvg::tiny_skia::Pixmap::new(24 * scale, 24 * scale).unwrap();
        resvg::render(
            &tree,
            resvg::tiny_skia::Transform::from_scale(scale as f32, scale as f32),
            &mut pixels.as_mut(),
        );
        assert_eq!(pixels.pixel(0, 0).unwrap().alpha(), 0);
        let center = pixels.pixel(12 * scale, 12 * scale).unwrap();
        assert!((120..=136).contains(&center.alpha()));
        let color = center.demultiply();
        assert!((color.red() as i32 - 210).abs() < 12);
        assert!((color.green() as i32 - 40).abs() < 12);
        assert!((color.blue() as i32 - 60).abs() < 12);
    }
}

#[test]
fn memory_conversion_rejects_invalid_input_and_decode_limits() {
    assert!(vectorize_bytes(b"not an image", &Config::default()).is_err());
    let config = Config {
        maximum_decode_bytes: 1024 * 1024,
        ..Config::default()
    };
    let image = DynamicImage::new_rgba8(1024, 1024);
    let mut bytes = Cursor::new(Vec::new());
    image.write_to(&mut bytes, ImageFormat::Png).unwrap();
    assert!(vectorize_bytes(bytes.get_ref(), &config).is_err());
    let config = Config {
        maximum_dimension: 0,
        ..Config::default()
    };
    assert!(vectorize_bytes(&fixture(), &config).is_err());
}
