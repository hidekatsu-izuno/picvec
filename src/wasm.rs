//! Browser entry point. Rayon uses its single-thread fallback inside a Web Worker.
use wasm_bindgen::prelude::*;

/// Convert PNG/JPEG bytes to SVG. Limits bound browser memory use before decoding.
#[wasm_bindgen]
pub fn convert_image(
    input: &[u8],
    maximum_dimension: u32,
    remove_background: bool,
) -> std::result::Result<String, JsValue> {
    if input.len() > 20 * 1024 * 1024 {
        return Err(JsValue::from_str("Choose an image smaller than 20 MiB."));
    }
    if !(64..=1024).contains(&maximum_dimension) {
        return Err(JsValue::from_str(
            "Processing size must be between 64 and 1024 pixels.",
        ));
    }
    let config = crate::Config {
        maximum_input_dimension: 8192,
        maximum_input_pixels: 16_000_000,
        maximum_decode_bytes: 128 * 1024 * 1024,
        maximum_dimension,
        auto_dimension: false,
        adaptive_refinement: false,
        remove_chroma_key_background: remove_background,
        rayon_threads: 1,
        ..crate::Config::default()
    };
    crate::vectorize_bytes(input, &config)
        .map(|(svg, _)| svg)
        .map_err(|error| JsValue::from_str(&error.to_string()))
}

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console, js_name = error)]
    fn console_error(message: &str);
}

#[wasm_bindgen(start)]
pub fn start() {
    std::panic::set_hook(Box::new(|info| console_error(&info.to_string())));
}
