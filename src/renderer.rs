use crate::Critter;

pub const OUTLINE_COLOR: u32 = 0x00_00_FF;
pub const OUTLINE_THICKNESS: i32 = 2;
pub const FRONT_DOT_RADIUS: i32 = 4;

pub struct Renderer;

impl Renderer {
    pub fn draw(critter: &Critter, radius: i32, buffer: &mut [u32], width: usize, height: usize) {
        let cx = critter.x();
        let cy = critter.y();
        let outer_squared = radius * radius;
        let inner_radius = radius - OUTLINE_THICKNESS;
        let inner_squared = inner_radius * inner_radius;

        let x_min = (cx - radius).max(0);
        let y_min = (cy - radius).max(0);
        let x_max = (cx + radius).min(width as i32 - 1);
        let y_max = (cy + radius).min(height as i32 - 1);

        for y in y_min..=y_max {
            for x in x_min..=x_max {
                let dx = x - cx;
                let dy = y - cy;
                let distance_squared = dx * dx + dy * dy;
                if distance_squared <= outer_squared && distance_squared > inner_squared {
                    buffer[y as usize * width + x as usize] = OUTLINE_COLOR;
                }
            }
        }

        let (offset_x, offset_y) = critter.heading().offset();
        let front_x = cx + offset_x * (radius - FRONT_DOT_RADIUS);
        let front_y = cy + offset_y * (radius - FRONT_DOT_RADIUS);
        let dot_radius_squared = FRONT_DOT_RADIUS * FRONT_DOT_RADIUS;

        let dot_x_min = (front_x - FRONT_DOT_RADIUS).max(0);
        let dot_y_min = (front_y - FRONT_DOT_RADIUS).max(0);
        let dot_x_max = (front_x + FRONT_DOT_RADIUS).min(width as i32 - 1);
        let dot_y_max = (front_y + FRONT_DOT_RADIUS).min(height as i32 - 1);

        for y in dot_y_min..=dot_y_max {
            for x in dot_x_min..=dot_x_max {
                let dx = x - front_x;
                let dy = y - front_y;
                if dx * dx + dy * dy <= dot_radius_squared {
                    buffer[y as usize * width + x as usize] = OUTLINE_COLOR;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Critter, Heading, Instruction};

    const RADIUS: i32 = 20;

    fn render(critter: &Critter, width: usize, height: usize) -> Vec<u32> {
        let mut buffer = vec![0u32; width * height];
        Renderer::draw(critter, RADIUS, &mut buffer, width, height);
        buffer
    }

    mod draw {
        use super::*;

        #[test]
        fn the_center_of_the_critter_is_not_filled() {
            let critter = Critter::new(50, 50, Heading::North, vec![Instruction::DoNothing], 1, 1);
            let buffer = render(&critter, 200, 200);
            assert_eq!(buffer[50 * 200 + 50], 0);
        }

        #[test]
        fn a_point_well_inside_the_outline_is_not_filled() {
            let critter = Critter::new(50, 50, Heading::North, vec![Instruction::DoNothing], 1, 1);
            let buffer = render(&critter, 200, 200);
            assert_eq!(buffer[55 * 200 + 55], 0);
        }

        #[test]
        fn a_point_on_the_outline_is_drawn_in_outline_color() {
            let critter = Critter::new(50, 50, Heading::North, vec![Instruction::DoNothing], 1, 1);
            let buffer = render(&critter, 200, 200);
            let on_ring_x = (50 + RADIUS - 1) as usize;
            assert_eq!(buffer[50 * 200 + on_ring_x], OUTLINE_COLOR);
        }

        #[test]
        fn a_point_just_inside_the_outline_thickness_is_drawn_in_outline_color() {
            let critter = Critter::new(50, 50, Heading::North, vec![Instruction::DoNothing], 1, 1);
            let buffer = render(&critter, 200, 200);
            let just_inside_x = (50 + RADIUS - OUTLINE_THICKNESS + 1) as usize;
            assert_eq!(buffer[50 * 200 + just_inside_x], OUTLINE_COLOR);
        }

        #[test]
        fn a_point_outside_the_radius_is_not_drawn() {
            let critter = Critter::new(50, 50, Heading::North, vec![Instruction::DoNothing], 1, 1);
            let buffer = render(&critter, 200, 200);
            assert_eq!(buffer[10 * 200 + 10], 0);
        }

        #[test]
        fn the_front_dot_is_drawn_in_the_same_color_as_the_outline() {
            let critter = Critter::new(50, 50, Heading::North, vec![Instruction::DoNothing], 1, 1);
            let buffer = render(&critter, 200, 200);
            let front_y = (50 - RADIUS + FRONT_DOT_RADIUS) as usize;
            assert_eq!(buffer[front_y * 200 + 50], OUTLINE_COLOR);
        }

        #[test]
        fn the_front_dot_is_drawn_east_when_heading_is_east() {
            let critter = Critter::new(50, 50, Heading::East, vec![Instruction::DoNothing], 1, 1);
            let buffer = render(&critter, 200, 200);
            let front_x = (50 + RADIUS - FRONT_DOT_RADIUS) as usize;
            assert_eq!(buffer[50 * 200 + front_x], OUTLINE_COLOR);
        }

        #[test]
        fn out_of_bounds_pixels_are_not_drawn() {
            let critter = Critter::new(2, 2, Heading::North, vec![Instruction::DoNothing], 1, 1);
            let buffer = render(&critter, 50, 50);
            assert_eq!(buffer.len(), 50 * 50);
        }
    }
}
