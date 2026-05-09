use crate::{Critter, Pellet, PELLET_COLOR, PELLET_RADIUS};

pub const OUTLINE_COLOR: u32 = 0x00_00_FF;
pub const ZERO_ENERGY_COLOR: u32 = 0x40_40_40;
pub const STOLEN_FROM_COLOR: u32 = 0xFF_00_00;
pub const OUTLINE_THICKNESS: i32 = 2;
pub const FRONT_DOT_RADIUS: i32 = 4;

struct Ring {
    cx: i32,
    cy: i32,
    radius: i32,
    inner_squared: i32,
}

struct Canvas<'a> {
    buffer: &'a mut [u32],
    width: usize,
    height: usize,
}

pub struct Renderer;

impl Renderer {
    pub fn draw(critter: &Critter, radius: i32, buffer: &mut [u32], width: usize, height: usize) {
        let cx = critter.x();
        let cy = critter.y();
        let inner_radius = radius - OUTLINE_THICKNESS;
        let mut canvas = Canvas {
            buffer,
            width,
            height,
        };
        let color = if critter.is_being_stolen_from() {
            STOLEN_FROM_COLOR
        } else {
            energy_color(critter.energy(), critter.initial_energy())
        };

        let body = Ring {
            cx,
            cy,
            radius,
            inner_squared: inner_radius * inner_radius,
        };
        Self::fill_ring_with_wrap(&body, &mut canvas, color);

        let heading = critter.heading();
        let (offset_x, offset_y) = heading.offset();
        let raw_dot_offset = radius - FRONT_DOT_RADIUS;
        let dot_offset = if heading.is_diagonal() {
            ((raw_dot_offset as f32) * std::f32::consts::FRAC_1_SQRT_2).round() as i32
        } else {
            raw_dot_offset
        };
        let dot = Ring {
            cx: cx + offset_x * dot_offset,
            cy: cy + offset_y * dot_offset,
            radius: FRONT_DOT_RADIUS,
            inner_squared: -1,
        };
        Self::fill_ring_with_wrap(&dot, &mut canvas, color);
    }

    pub fn draw_pellet(pellet: &Pellet, buffer: &mut [u32], width: usize, height: usize) {
        let mut canvas = Canvas {
            buffer,
            width,
            height,
        };
        let disc = Ring {
            cx: pellet.x,
            cy: pellet.y,
            radius: PELLET_RADIUS,
            inner_squared: -1,
        };
        Self::fill_ring_with_wrap(&disc, &mut canvas, PELLET_COLOR);
    }

    // The `+` mutations on the offset additions below are equivalent: the loops
    // iterate over the symmetric set {-w, 0, w}, which is closed under negation,
    // so any sign flip produces the same set of draws in a different order.
    #[mutants::skip]
    fn fill_ring_with_wrap(ring: &Ring, canvas: &mut Canvas, color: u32) {
        let w = canvas.width as i32;
        let h = canvas.height as i32;
        for dx in [-w, 0, w] {
            for dy in [-h, 0, h] {
                let shifted = Ring {
                    cx: ring.cx + dx,
                    cy: ring.cy + dy,
                    radius: ring.radius,
                    inner_squared: ring.inner_squared,
                };
                Self::fill_ring(&shifted, canvas, color);
            }
        }
    }

    fn fill_ring(ring: &Ring, canvas: &mut Canvas, color: u32) {
        let outer_squared = ring.radius * ring.radius;
        for y in (ring.cy - ring.radius)..=(ring.cy + ring.radius) {
            if y < 0 || y >= canvas.height as i32 {
                continue;
            }
            for x in (ring.cx - ring.radius)..=(ring.cx + ring.radius) {
                if x < 0 || x >= canvas.width as i32 {
                    continue;
                }
                let dx = x - ring.cx;
                let dy = y - ring.cy;
                let distance_squared = dx * dx + dy * dy;
                if distance_squared <= outer_squared && distance_squared > ring.inner_squared {
                    canvas.buffer[y as usize * canvas.width + x as usize] = color;
                }
            }
        }
    }
}

fn energy_color(energy: u32, initial_energy: u32) -> u32 {
    let ratio = if initial_energy == 0 {
        0.0
    } else {
        energy as f32 / initial_energy as f32
    };
    interpolate_color(ZERO_ENERGY_COLOR, OUTLINE_COLOR, ratio)
}

fn interpolate_color(from: u32, to: u32, ratio: f32) -> u32 {
    let [_, from_r, from_g, from_b] = from.to_be_bytes();
    let [_, to_r, to_g, to_b] = to.to_be_bytes();
    let r = lerp(from_r, to_r, ratio);
    let g = lerp(from_g, to_g, ratio);
    let b = lerp(from_b, to_b, ratio);
    u32::from_be_bytes([0, r, g, b])
}

fn lerp(from: u8, to: u8, ratio: f32) -> u8 {
    let from = from as f32;
    let to = to as f32;
    (from + (to - from) * ratio).round() as u8
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Critter, Genome, Heading, Instruction};

    mod interpolate_color_tests {
        use crate::renderer::interpolate_color;

        #[test]
        fn at_ratio_zero_it_returns_the_from_color() {
            assert_eq!(interpolate_color(0xAB_CD_EF, 0x12_34_56, 0.0), 0xAB_CD_EF);
        }

        #[test]
        fn at_ratio_one_it_returns_the_to_color() {
            assert_eq!(interpolate_color(0xAB_CD_EF, 0x12_34_56, 1.0), 0x12_34_56);
        }

        #[test]
        fn each_channel_is_interpolated_independently() {
            // from = (0, 100, 200), to = (200, 100, 0), ratio = 0.5.
            // r: 0 + (200 - 0) * 0.5 = 100
            // g: 100 + (100 - 100) * 0.5 = 100
            // b: 200 + (0 - 200) * 0.5 = 100
            assert_eq!(interpolate_color(0x0064C8, 0xC86400, 0.5), 0x646464);
        }
    }

    const RADIUS: i32 = 20;
    const CANVAS: usize = 200;
    const CENTER: i32 = 50;
    const NEAR_TOP: i32 = RADIUS;
    const NEAR_LEFT: i32 = RADIUS;
    const NEAR_BOTTOM: i32 = CANVAS as i32 - 1 - RADIUS;
    const NEAR_RIGHT: i32 = CANVAS as i32 - 1 - RADIUS;

    mod draw {
        use super::*;

        #[test]
        fn the_center_of_the_critter_is_not_filled() {
            let critter = stationary_critter(CENTER, CENTER, Heading::North);

            let buffer = render(&critter);

            assert_eq!(pixel_at(&buffer, CENTER, CENTER), 0);
        }

        #[test]
        fn a_point_well_inside_the_outline_is_not_filled() {
            // 5 pixels in from center is well inside the inner_radius of 18.
            let critter = stationary_critter(CENTER, CENTER, Heading::North);

            let buffer = render(&critter);

            assert_eq!(pixel_at(&buffer, CENTER + 5, CENTER + 5), 0);
        }

        #[test]
        fn the_pixel_just_inside_the_outer_radius_is_drawn_in_outline_color() {
            let critter = stationary_critter(CENTER, CENTER, Heading::North);

            let buffer = render(&critter);

            // The outer edge: at distance radius - 1, distance² = (radius-1)² ≤ radius².
            assert_eq!(
                pixel_at(&buffer, CENTER + RADIUS - 1, CENTER),
                OUTLINE_COLOR
            );
        }

        #[test]
        fn the_pixel_at_the_inner_radius_is_not_filled() {
            // The inner_radius is `radius - thickness`. A pixel exactly at distance
            // `inner_radius` has distance² == inner_squared, which fails the strict
            // `distance_squared > inner_squared` check, so it's outside the ring.
            let critter = stationary_critter(CENTER, CENTER, Heading::North);

            let buffer = render(&critter);

            assert_eq!(
                pixel_at(&buffer, CENTER + RADIUS - OUTLINE_THICKNESS, CENTER),
                0
            );
        }

        #[test]
        fn the_pixel_one_step_outside_the_inner_radius_is_drawn_in_outline_color() {
            let critter = stationary_critter(CENTER, CENTER, Heading::North);

            let buffer = render(&critter);

            // One pixel further out than the inner_radius is the innermost lit pixel.
            assert_eq!(
                pixel_at(&buffer, CENTER + RADIUS - OUTLINE_THICKNESS + 1, CENTER),
                OUTLINE_COLOR
            );
        }

        #[test]
        fn a_point_outside_the_outer_radius_is_not_drawn() {
            let critter = stationary_critter(CENTER, CENTER, Heading::North);

            let buffer = render(&critter);

            assert_eq!(pixel_at(&buffer, CENTER + RADIUS + 1, CENTER), 0);
        }

        #[test]
        fn the_front_dot_is_drawn_north_of_center_when_facing_north() {
            let critter = stationary_critter(CENTER, CENTER, Heading::North);

            let buffer = render(&critter);

            assert_eq!(
                pixel_at(&buffer, CENTER, CENTER - RADIUS + FRONT_DOT_RADIUS),
                OUTLINE_COLOR
            );
        }

        #[test]
        fn the_front_dot_extends_to_its_full_radius_inside_the_ring() {
            // The pixel at the dot's bottom edge sits inside the ring's hollow center,
            // so it's only lit if the dot itself is at full radius.
            let critter = stationary_critter(CENTER, CENTER, Heading::North);

            let buffer = render(&critter);

            let dot_center_y = CENTER - RADIUS + FRONT_DOT_RADIUS;
            assert_eq!(
                pixel_at(&buffer, CENTER, dot_center_y + FRONT_DOT_RADIUS),
                OUTLINE_COLOR
            );
        }

        #[test]
        fn the_front_dot_is_drawn_east_of_center_when_facing_east() {
            let critter = stationary_critter(CENTER, CENTER, Heading::East);

            let buffer = render(&critter);

            assert_eq!(
                pixel_at(&buffer, CENTER + RADIUS - FRONT_DOT_RADIUS, CENTER),
                OUTLINE_COLOR
            );
        }

        #[test]
        fn the_front_dot_is_drawn_south_of_center_when_facing_south() {
            let critter = stationary_critter(CENTER, CENTER, Heading::South);

            let buffer = render(&critter);

            assert_eq!(
                pixel_at(&buffer, CENTER, CENTER + RADIUS - FRONT_DOT_RADIUS),
                OUTLINE_COLOR
            );
        }

        #[test]
        fn the_front_dot_is_drawn_west_of_center_when_facing_west() {
            let critter = stationary_critter(CENTER, CENTER, Heading::West);

            let buffer = render(&critter);

            assert_eq!(
                pixel_at(&buffer, CENTER - RADIUS + FRONT_DOT_RADIUS, CENTER),
                OUTLINE_COLOR
            );
        }

        #[test]
        fn the_front_dot_is_drawn_at_the_diagonal_offset_when_facing_northeast() {
            // For a diagonal heading, the dot's offset from center is scaled by √2/2
            // along each axis. With (RADIUS - FRONT_DOT_RADIUS) = 16, scaled = round(11.31) = 11.
            const DIAGONAL_DOT_OFFSET: i32 = 11;
            let critter = stationary_critter(CENTER, CENTER, Heading::NorthEast);

            let buffer = render(&critter);

            assert_eq!(
                pixel_at(
                    &buffer,
                    CENTER + DIAGONAL_DOT_OFFSET,
                    CENTER - DIAGONAL_DOT_OFFSET
                ),
                OUTLINE_COLOR
            );
        }

        #[test]
        fn the_front_dot_is_drawn_at_the_diagonal_offset_when_facing_southwest() {
            const DIAGONAL_DOT_OFFSET: i32 = 11;
            let critter = stationary_critter(CENTER, CENTER, Heading::SouthWest);

            let buffer = render(&critter);

            assert_eq!(
                pixel_at(
                    &buffer,
                    CENTER - DIAGONAL_DOT_OFFSET,
                    CENTER + DIAGONAL_DOT_OFFSET
                ),
                OUTLINE_COLOR
            );
        }

        #[test]
        fn the_ring_extends_all_the_way_to_the_top_edge_when_the_critter_is_against_it() {
            let critter = stationary_critter(CENTER, NEAR_TOP, Heading::East);

            let buffer = render(&critter);

            assert_eq!(pixel_at(&buffer, CENTER, 0), OUTLINE_COLOR);
        }

        #[test]
        fn the_ring_extends_all_the_way_to_the_left_edge_when_the_critter_is_against_it() {
            let critter = stationary_critter(NEAR_LEFT, CENTER, Heading::North);

            let buffer = render(&critter);

            assert_eq!(pixel_at(&buffer, 0, CENTER), OUTLINE_COLOR);
        }

        #[test]
        fn the_ring_extends_all_the_way_to_the_right_edge_when_the_critter_is_against_it() {
            let critter = stationary_critter(NEAR_RIGHT, CENTER, Heading::North);

            let buffer = render(&critter);

            assert_eq!(pixel_at(&buffer, CANVAS as i32 - 1, CENTER), OUTLINE_COLOR);
        }

        #[test]
        fn the_ring_extends_all_the_way_to_the_bottom_edge_when_the_critter_is_against_it() {
            let critter = stationary_critter(CENTER, NEAR_BOTTOM, Heading::North);

            let buffer = render(&critter);

            assert_eq!(pixel_at(&buffer, CENTER, CANVAS as i32 - 1), OUTLINE_COLOR);
        }

        #[test]
        fn the_ring_is_still_drawn_when_part_of_the_critter_is_above_the_top_edge() {
            // Critter centered at y=0: top half of the ring is off-canvas, bottom half visible.
            let critter = stationary_critter(CENTER, 0, Heading::East);

            let buffer = render(&critter);

            // The bottom of the ring (at distance RADIUS below center) is on-canvas.
            assert_eq!(pixel_at(&buffer, CENTER, RADIUS), OUTLINE_COLOR);
        }

        #[test]
        fn the_ring_is_still_drawn_when_part_of_the_critter_is_left_of_the_left_edge() {
            let critter = stationary_critter(0, CENTER, Heading::North);

            let buffer = render(&critter);

            assert_eq!(pixel_at(&buffer, RADIUS, CENTER), OUTLINE_COLOR);
        }

        #[test]
        fn the_ring_is_still_drawn_when_part_of_the_critter_is_right_of_the_right_edge() {
            let critter = stationary_critter(CANVAS as i32 - 1, CENTER, Heading::North);

            let buffer = render(&critter);

            // The left side of the ring (at distance RADIUS to the left) is on-canvas.
            assert_eq!(
                pixel_at(&buffer, CANVAS as i32 - 1 - RADIUS, CENTER),
                OUTLINE_COLOR
            );
        }

        #[test]
        fn the_ring_is_still_drawn_when_part_of_the_critter_is_below_the_bottom_edge() {
            let critter = stationary_critter(CENTER, CANVAS as i32 - 1, Heading::North);

            let buffer = render(&critter);

            assert_eq!(
                pixel_at(&buffer, CENTER, CANVAS as i32 - 1 - RADIUS),
                OUTLINE_COLOR
            );
        }

        mod color {
            use super::*;
            use crate::Instruction;

            const INITIAL_ENERGY: u32 = 100;
            const ON_RING_X_OFFSET: i32 = RADIUS - 1;

            #[test]
            fn a_critter_at_full_energy_renders_in_pure_blue() {
                let critter = critter_with_energy(INITIAL_ENERGY);

                let buffer = render(&critter);

                assert_eq!(
                    pixel_at(&buffer, CENTER + ON_RING_X_OFFSET, CENTER),
                    0x00_00_FF
                );
            }

            #[test]
            fn a_critter_at_zero_energy_renders_in_ghostly_gray() {
                let critter = critter_with_energy(0);

                let buffer = render(&critter);

                assert_eq!(
                    pixel_at(&buffer, CENTER + ON_RING_X_OFFSET, CENTER),
                    0x40_40_40
                );
            }

            #[test]
            fn a_critter_at_half_energy_renders_halfway_between_gray_and_blue() {
                // Halfway between (64, 64, 64) and (0, 0, 255) is (32, 32, 160) when rounded.
                let critter = critter_with_energy(INITIAL_ENERGY / 2);

                let buffer = render(&critter);

                assert_eq!(
                    pixel_at(&buffer, CENTER + ON_RING_X_OFFSET, CENTER),
                    0x20_20_A0
                );
            }

            #[test]
            fn a_critter_with_energy_above_initial_still_renders_in_pure_blue() {
                // With the cap on gain_energy removed, a critter can stockpile
                // past its initial_energy. The color must clamp at pure blue.
                let mut critter = critter_with_energy(INITIAL_ENERGY);
                critter.gain_energy(INITIAL_ENERGY); // total = 2 × INITIAL_ENERGY

                let buffer = render(&critter);

                assert_eq!(
                    pixel_at(&buffer, CENTER + ON_RING_X_OFFSET, CENTER),
                    0x00_00_FF
                );
            }

            #[test]
            fn a_critter_being_stolen_from_renders_in_red_regardless_of_energy() {
                let mut critter = critter_with_energy(INITIAL_ENERGY);
                critter.mark_being_stolen_from_for(10);

                let buffer = render(&critter);

                assert_eq!(
                    pixel_at(&buffer, CENTER + ON_RING_X_OFFSET, CENTER),
                    0xFF_00_00
                );
            }

            fn critter_with_energy(current_energy: u32) -> Critter {
                let mut critter = Critter::with_genome(
                    CENTER,
                    CENTER,
                    Heading::North,
                    1,
                    1,
                    INITIAL_ENERGY,
                    0,
                    Genome::all(Instruction::DoNothing),
                );
                for _ in 0..(INITIAL_ENERGY - current_energy) {
                    critter.tick(true);
                }
                critter
            }
        }

        mod wrap_rendering {
            use super::*;

            #[test]
            fn a_critter_against_the_right_edge_renders_pixels_on_the_left_edge_too() {
                // Critter at x = CANVAS - 1: ring extends from x = CANVAS - 1 - RADIUS to
                // x = CANVAS - 1 + RADIUS, the latter wrapping into [0, RADIUS - 1].
                let critter = stationary_critter(CANVAS as i32 - 1, CENTER, Heading::North);

                let buffer = render(&critter);

                // A pixel that would be lit on the unwrapped circle's right side now
                // appears on the left edge.
                assert_eq!(pixel_at(&buffer, RADIUS - 2, CENTER), OUTLINE_COLOR);
            }

            #[test]
            fn a_critter_against_the_bottom_edge_renders_pixels_on_the_top_edge_too() {
                let critter = stationary_critter(CENTER, CANVAS as i32 - 1, Heading::North);

                let buffer = render(&critter);

                assert_eq!(pixel_at(&buffer, CENTER, RADIUS - 2), OUTLINE_COLOR);
            }
        }

        mod pellet {
            use super::*;
            use crate::{Pellet, PELLET_COLOR, PELLET_RADIUS};

            #[test]
            fn the_center_of_a_pellet_is_drawn_in_pellet_color() {
                let pellet = Pellet {
                    x: CENTER,
                    y: CENTER,
                };
                let buffer = render_pellet(&pellet);

                assert_eq!(pixel_at(&buffer, CENTER, CENTER), PELLET_COLOR);
            }

            #[test]
            fn the_pellet_is_drawn_as_a_filled_disc_of_pellet_radius() {
                let pellet = Pellet {
                    x: CENTER,
                    y: CENTER,
                };
                let buffer = render_pellet(&pellet);

                // Edge of the disc — at distance PELLET_RADIUS from center, on-axis.
                assert_eq!(
                    pixel_at(&buffer, CENTER + PELLET_RADIUS, CENTER),
                    PELLET_COLOR
                );
                // One past the edge — should not be drawn.
                assert_eq!(pixel_at(&buffer, CENTER + PELLET_RADIUS + 1, CENTER), 0);
            }

            #[test]
            fn drawing_a_pellet_does_not_panic_when_partly_off_canvas() {
                let pellet = Pellet { x: 0, y: 0 };
                let _ = render_pellet(&pellet);
            }

            fn render_pellet(pellet: &Pellet) -> Vec<u32> {
                let mut buffer = vec![0u32; CANVAS * CANVAS];
                Renderer::draw_pellet(pellet, &mut buffer, CANVAS, CANVAS);
                buffer
            }
        }

        // Helpers below the tests, in keeping with hiding incidental detail.

        fn stationary_critter(x: i32, y: i32, heading: Heading) -> Critter {
            Critter::with_genome(
                x,
                y,
                heading,
                1,
                1,
                u32::MAX,
                0,
                Genome::all(Instruction::DoNothing),
            )
        }

        fn render(critter: &Critter) -> Vec<u32> {
            let mut buffer = vec![0u32; CANVAS * CANVAS];
            Renderer::draw(critter, RADIUS, &mut buffer, CANVAS, CANVAS);
            buffer
        }

        fn pixel_at(buffer: &[u32], x: i32, y: i32) -> u32 {
            buffer[y as usize * CANVAS + x as usize]
        }
    }
}
