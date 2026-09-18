//! Offline evaluation only. Detect on the source, recognize the same rectangles
//! in every candidate, including candidates where a glyph has disappeared.
use std::{error::Error, fs};

use ocrs_cjk::{ImageSource, OcrEngine, OcrEngineParams};
use rten_imageproc::BoundingRect;
use serde_json::{json, Value};

const REVISION: &str = "2f72f76ac2152ee1112ee6b6e36048bab069f391";

fn positions(length: u32, tile: u32) -> Vec<u32> {
    let mut out = vec![0];
    while *out.last().unwrap() + tile < length {
        out.push((out.last().unwrap() + tile * 3 / 4).min(length - tile));
    }
    out
}

fn intersection(a: [f32; 4], b: [f32; 4]) -> f32 {
    (a[2].min(b[2]) - a[0].max(b[0])).max(0.0) * (a[3].min(b[3]) - a[1].max(b[1])).max(0.0)
}

fn main() -> Result<(), Box<dyn Error>> {
    let path = std::env::args()
        .nth(1)
        .ok_or("Usage: picvec-ocr-eval config.json")?;
    let config: Value = serde_json::from_slice(&fs::read(path)?)?;
    let required = |key: &str| -> Result<&str, Box<dyn Error>> {
        config[key]
            .as_str()
            .ok_or_else(|| format!("Missing string: {key}").into())
    };
    let engine = OcrEngine::new(OcrEngineParams {
        detection_model: Some(rten::Model::load_file(required("detection_model")?)?),
        recognition_model: Some(rten::Model::load_file(required("recognition_model")?)?),
        alphabet: Some(fs::read_to_string(required("alphabet")?)?),
        ..Default::default()
    })?;
    let source = image::open(required("source")?)?.into_rgb8();
    let (width, height) = source.dimensions();
    let mut candidates = Vec::new();
    for (name, path) in config["candidates"]
        .as_object()
        .ok_or("Missing candidates")?
    {
        let image =
            image::open(path.as_str().ok_or("Candidate path must be a string")?)?.into_rgb8();
        if image.dimensions() != source.dimensions() {
            return Err(format!("{name}: render at the source dimensions").into());
        }
        candidates.push((name, image));
    }
    let tile = config["tile_size"].as_u64().unwrap_or(1024) as u32;
    if tile < 64 {
        return Err("tile_size must be at least 64".into());
    }
    let max_height = config["max_height"].as_f64().unwrap_or(64.0) as f32;
    let mut boxes: Vec<[f32; 4]> = Vec::new();
    let mut regions = Vec::new();
    for y in positions(height, tile) {
        for x in positions(width, tile) {
            let (w, h) = (tile.min(width - x), tile.min(height - y));
            let crop = image::imageops::crop_imm(&source, x, y, w, h).to_image();
            let input = engine.prepare_input(ImageSource::from_bytes(crop.as_raw(), (w, h))?)?;
            let detected = engine.detect_words(&input)?;
            let mut lines = Vec::new();
            let mut current_boxes = Vec::new();
            for rect in detected {
                let b = rect.bounding_rect();
                if b.height() > max_height || b.height() < 5.0 {
                    continue;
                }
                // Overlap lets a neighbouring tile own lines cut by this crop.
                if (x > 0 && b.left() < 8.0)
                    || (y > 0 && b.top() < 8.0)
                    || (x + w < width && b.right() > w as f32 - 8.0)
                    || (y + h < height && b.bottom() > h as f32 - 8.0)
                {
                    continue;
                }
                let global = [
                    b.left() + x as f32,
                    b.top() + y as f32,
                    b.right() + x as f32,
                    b.bottom() + y as f32,
                ];
                if boxes.iter().any(|&other| {
                    let area = |v: [f32; 4]| (v[2] - v[0]) * (v[3] - v[1]);
                    intersection(global, other) > 0.6 * area(global).min(area(other))
                }) {
                    continue;
                }
                boxes.push(global);
                current_boxes.push(global);
                // PP-OCR detects text runs. Do not join distant diagram labels
                // just because they happen to share a baseline.
                lines.push(vec![rect]);
            }
            if lines.is_empty() {
                continue;
            }
            let reference = engine.recognize_text(&input, &lines)?;
            let mut texts = Vec::new();
            for (name, candidate) in &candidates {
                let crop = image::imageops::crop_imm(candidate, x, y, w, h).to_image();
                let input =
                    engine.prepare_input(ImageSource::from_bytes(crop.as_raw(), (w, h))?)?;
                texts.push((name, engine.recognize_text(&input, &lines)?));
            }
            for (i, bbox) in current_boxes.into_iter().enumerate() {
                let reference = reference[i]
                    .as_ref()
                    .map(ToString::to_string)
                    .unwrap_or_default();
                let mut variants = serde_json::Map::new();
                for (name, lines) in &texts {
                    variants.insert(
                        name.to_string(),
                        json!(lines[i]
                            .as_ref()
                            .map(ToString::to_string)
                            .unwrap_or_default()),
                    );
                }
                regions.push(json!({"bbox": bbox, "reference": reference, "texts": variants}));
            }
            eprintln!("tile {x},{y}: {} source regions", lines.len());
        }
    }
    fs::write(
        required("output")?,
        serde_json::to_vec_pretty(&json!({
            "ocrs_cjk_revision": REVISION, "config": config,
            "width": width, "height": height, "regions": regions,
            "note": "Source OCR is a noisy reference, not a transcription. Empty candidate results are retained."
        }))?,
    )?;
    Ok(())
}
