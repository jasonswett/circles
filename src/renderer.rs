use crate::Critter;

pub const BODY_COLOR: u32 = 0xFF_FF_FF;
pub const FRONT_COLOR: u32 = 0xFF_00_00;
pub const FRONT_DOT_RADIUS: i32 = 2;

pub struct Renderer;

impl Renderer {
    pub fn draw(critter: &Critter, radius: i32, buffer: &mut [u32], width: usize, height: usize) {
        let cx = critter.x();
        let cy = critter.y();
        let radius_squared = radius * radius;

        let x_min = (cx - radius).max(0);
        let y_min = (cy - radius).max(0);
        let x_max = (cx + radius).min(width as i32 - 1);
        let y_max = (cy + radius).min(height as i32 - 1);

        for y in y_min..=y_max {
            for x in x_min..=x_max {
                let dx = x - cx;
                let dy = y - cy;
                if dx * dx + dy * dy <= radius_squared {
                    buffer[y as usize * width + x as usize] = BODY_COLOR;
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
                    buffer[y as usize * width + x as usize] = FRONT_COLOR;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Critter, Heading, Instruction};

    const BODY_COLOR: u32 = 0xFF_FF_FF;
    const FRONT_COLOR: u32 = 0xFF_00_00;
    const RADIUS: i32 = 10;

    fn render(critter: &Critter, width: usize, height: usize) -> Vec<u32> {
        let mut buffer = vec![0u32; width * height];
        Renderer::draw(critter, RADIUS, &mut buffer, width, height);
        buffer
    }

    mod draw {
        use super::*;

        #[test]
        fn the_center_of_the_critter_is_drawn_in_body_color() {
            let critter = Critter::new(50, 50, Heading::North, vec![Instruction::DoNothing], 1, 1);
            let buffer = render(&critter, 100, 100);
            assert_eq!(buffer[50 * 100 + 50], BODY_COLOR);
        }

        #[test]
        fn a_point_outside_the_radius_is_not_drawn() {
            let critter = Critter::new(50, 50, Heading::North, vec![Instruction::DoNothing], 1, 1);
            let buffer = render(&critter, 100, 100);
            assert_eq!(buffer[10 * 100 + 10], 0);
        }

        #[test]
        fn the_front_dot_is_drawn_north_when_heading_is_north() {
            let critter = Critter::new(50, 50, Heading::North, vec![Instruction::DoNothing], 1, 1);
            let buffer = render(&critter, 100, 100);
            let front_y = (50 - RADIUS + 2) as usize;
            assert_eq!(buffer[front_y * 100 + 50], FRONT_COLOR);
        }

        #[test]
        fn the_front_dot_is_drawn_east_when_heading_is_east() {
            let critter = Critter::new(50, 50, Heading::East, vec![Instruction::DoNothing], 1, 1);
            let buffer = render(&critter, 100, 100);
            let front_x = (50 + RADIUS - 2) as usize;
            assert_eq!(buffer[50 * 100 + front_x], FRONT_COLOR);
        }

        #[test]
        fn out_of_bounds_pixels_are_not_drawn() {
            let critter = Critter::new(2, 2, Heading::North, vec![Instruction::DoNothing], 1, 1);
            let buffer = render(&critter, 50, 50);
            assert_eq!(buffer.len(), 50 * 50);
        }
    }
}
