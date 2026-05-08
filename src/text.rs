const GLYPH_WIDTH: usize = 3;
const GLYPH_HEIGHT: usize = 5;
const GLYPH_GAP: usize = 1;

const DIGIT_GLYPHS: [[u8; GLYPH_HEIGHT]; 10] = [
    [0b111, 0b101, 0b101, 0b101, 0b111], // 0
    [0b010, 0b110, 0b010, 0b010, 0b111], // 1
    [0b111, 0b001, 0b111, 0b100, 0b111], // 2
    [0b111, 0b001, 0b111, 0b001, 0b111], // 3
    [0b101, 0b101, 0b111, 0b001, 0b001], // 4
    [0b111, 0b100, 0b111, 0b001, 0b111], // 5
    [0b111, 0b100, 0b111, 0b101, 0b111], // 6
    [0b111, 0b001, 0b010, 0b010, 0b010], // 7
    [0b111, 0b101, 0b111, 0b101, 0b111], // 8
    [0b111, 0b101, 0b111, 0b001, 0b111], // 9
];

fn glyph_for(ch: char) -> Option<[u8; GLYPH_HEIGHT]> {
    ch.to_digit(10).map(|d| DIGIT_GLYPHS[d as usize])
}

pub fn pixels(text: &str, scale: usize) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut cursor_x = 0;
    for ch in text.chars() {
        if let Some(glyph) = glyph_for(ch) {
            for (row, bits) in glyph.iter().enumerate() {
                for col in 0..GLYPH_WIDTH {
                    if bits & (1 << (GLYPH_WIDTH - 1 - col)) != 0 {
                        for sy in 0..scale {
                            for sx in 0..scale {
                                out.push((cursor_x + col * scale + sx, row * scale + sy));
                            }
                        }
                    }
                }
            }
        }
        cursor_x += (GLYPH_WIDTH + GLYPH_GAP) * scale;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    mod pixels {
        use super::*;

        #[test]
        fn rendering_an_empty_string_produces_no_pixels() {
            let pixels = pixels("", 1);

            assert!(pixels.is_empty());
        }

        #[test]
        fn the_digit_one_lights_up_the_top_center_pixel() {
            // Digit '1' glyph (3 cols × 5 rows) has bit set at row 0, col 1.
            let pixels = pixels("1", 1);

            assert!(pixels.contains(&(1, 0)));
        }

        #[test]
        fn the_digit_zero_does_not_light_up_the_center_pixel() {
            // '0' is a hollow ring; the inner cell (1, 2) should be dark.
            let pixels = pixels("0", 1);

            assert!(!pixels.contains(&(1, 2)));
        }

        #[test]
        fn at_scale_two_each_glyph_pixel_becomes_a_2x2_block() {
            let pixels = pixels("1", 2);

            // The single lit pixel at (1, 0) becomes a 2x2 block at (2..4, 0..2).
            assert!(pixels.contains(&(2, 0)));
            assert!(pixels.contains(&(3, 0)));
            assert!(pixels.contains(&(2, 1)));
            assert!(pixels.contains(&(3, 1)));
        }

        #[test]
        fn at_scale_two_the_bottom_row_lands_at_y_eight_or_nine() {
            // The glyph is 5 rows tall. At scale 2 the bottom row maps to y = 8 or 9.
            // '1' row 4 = 0b111 (all cols lit), so col 0 at scale 2 sy=1 → (0, 9).
            // Under a row*scale → row/scale mutation, y would never exceed 3.
            let pixels = pixels("1", 2);

            assert!(pixels.contains(&(0, 9)));
        }

        #[test]
        fn rendering_two_digits_offsets_the_second_to_the_right_of_the_first() {
            // Glyph width 3 + 1px gap = 4 columns per glyph at scale 1. The first
            // pixel of the second '1' is at column 4 + 1 = 5.
            let pixels = pixels("11", 1);

            assert!(pixels.contains(&(5, 0)));
        }

        #[test]
        fn at_scale_two_the_cursor_advances_by_eight_columns_per_glyph() {
            // (3 + 1) * 2 = 8 columns per glyph slot. The lit top pixel of '1' is
            // at glyph-local col 1, so the second '1' starts its lit pixel at
            // x = 8 + 1*2 = 10 (a 2x2 block at x=10,11).
            let pixels = pixels("11", 2);

            assert!(pixels.contains(&(10, 0)));
            assert!(pixels.contains(&(11, 0)));
        }

        #[test]
        fn unsupported_characters_advance_the_cursor_without_drawing() {
            // A space advances the cursor; the digit after it lands at the second
            // position. With glyph_width 3 + 1 gap = 4 cols per slot at scale 1,
            // the '1' after the space starts at col 4, with its lit pixel at col 5.
            let pixels = pixels(" 1", 1);

            assert!(pixels.contains(&(5, 0)));
        }
    }
}
