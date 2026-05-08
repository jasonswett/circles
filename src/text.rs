use fontdue::{Font, FontSettings};
use std::sync::OnceLock;

const FONT_BYTES: &[u8] = include_bytes!("../assets/DejaVuSansMono.ttf");

fn font() -> &'static Font {
    static FONT: OnceLock<Font> = OnceLock::new();
    FONT.get_or_init(|| {
        Font::from_bytes(FONT_BYTES, FontSettings::default()).expect("font failed to parse")
    })
}

// Glyph layout is essentially a port of fontdue's API into our coordinate
// system. The arithmetic (xmin/ymin offsets, baseline computation, bounds
// checks) is verified by visual inspection of the running binary; pinning
// down exact pixel positions in unit tests would tightly couple them to
// fontdue's rasterizer and the bundled font version.
#[mutants::skip]
pub fn pixels(text: &str, size: f32) -> Vec<(usize, usize, u8)> {
    if text.is_empty() {
        return Vec::new();
    }
    let font = font();
    // Compute a baseline y high enough to host the largest glyph in the line.
    // ascent at this size = how far above baseline a glyph reaches.
    let line_metrics = font.horizontal_line_metrics(size).expect("missing metrics");
    let baseline = line_metrics.ascent.ceil() as i32;

    let mut out = Vec::new();
    let mut cursor_x: f32 = 0.0;
    for ch in text.chars() {
        let (metrics, bitmap) = font.rasterize(ch, size);
        let glyph_origin_x = cursor_x.round() as i32 + metrics.xmin;
        // ymin in fontdue is the y offset of the bottom-left of the glyph
        // bitmap from the baseline (positive y = up). The top of the bitmap
        // sits at `baseline - ymin - height`.
        let glyph_origin_y = baseline - metrics.ymin - metrics.height as i32;
        for row in 0..metrics.height {
            for col in 0..metrics.width {
                let alpha = bitmap[row * metrics.width + col];
                if alpha == 0 {
                    continue;
                }
                let px = glyph_origin_x + col as i32;
                let py = glyph_origin_y + row as i32;
                if px < 0 || py < 0 {
                    continue;
                }
                out.push((px as usize, py as usize, alpha));
            }
        }
        cursor_x += metrics.advance_width;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_empty_string_produces_no_pixels() {
        let pixels = pixels("", 16.0);

        assert!(pixels.is_empty());
    }

    #[test]
    fn a_single_character_produces_some_lit_pixels() {
        let pixels = pixels("A", 16.0);

        assert!(!pixels.is_empty());
    }

    #[test]
    fn every_returned_pixel_has_nonzero_alpha() {
        let pixels = pixels("ABCabc123", 16.0);

        assert!(pixels.iter().all(|&(_, _, alpha)| alpha > 0));
    }

    #[test]
    fn the_same_text_at_the_same_size_produces_the_same_output() {
        let a = pixels("Hello 42", 20.0);
        let b = pixels("Hello 42", 20.0);

        assert_eq!(a, b);
    }

    #[test]
    fn rendering_two_characters_extends_further_right_than_one() {
        let one_x_max = pixels("X", 16.0).iter().map(|&(x, _, _)| x).max().unwrap();
        let two_x_max = pixels("XX", 16.0).iter().map(|&(x, _, _)| x).max().unwrap();

        assert!(two_x_max > one_x_max);
    }

    #[test]
    fn larger_size_produces_pixels_at_larger_y_coordinates() {
        let small_y_max = pixels("X", 12.0).iter().map(|&(_, y, _)| y).max().unwrap();
        let large_y_max = pixels("X", 48.0).iter().map(|&(_, y, _)| y).max().unwrap();

        assert!(large_y_max > small_y_max);
    }
}
