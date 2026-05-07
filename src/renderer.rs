use crate::Critter;

pub const OUTLINE_COLOR: u32 = 0x00_00_FF;
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

        let body = Ring {
            cx,
            cy,
            radius,
            inner_squared: inner_radius * inner_radius,
        };
        Self::fill_ring(&body, &mut canvas);

        let (offset_x, offset_y) = critter.heading().offset();
        let dot = Ring {
            cx: cx + offset_x * (radius - FRONT_DOT_RADIUS),
            cy: cy + offset_y * (radius - FRONT_DOT_RADIUS),
            radius: FRONT_DOT_RADIUS,
            inner_squared: -1,
        };
        Self::fill_ring(&dot, &mut canvas);
    }

    fn fill_ring(ring: &Ring, canvas: &mut Canvas) {
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
                    canvas.buffer[y as usize * canvas.width + x as usize] = OUTLINE_COLOR;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Critter, Heading};

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

        // Helpers below the tests, in keeping with hiding incidental detail.

        fn stationary_critter(x: i32, y: i32, heading: Heading) -> Critter {
            Critter::new(x, y, heading, vec![], 1, 1)
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
