/// Where on the world the screen is looking. The world can be bigger than
/// the screen, so the screen shows a window onto it whose top-left corner
/// this tracks. The world wraps, and so does the camera: panning off one
/// edge brings it round to the other.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Camera {
    x: i32,
    y: i32,
    world_width: usize,
    world_height: usize,
}

impl Camera {
    pub const fn new(world_width: usize, world_height: usize) -> Self {
        Self {
            x: 0,
            y: 0,
            world_width,
            world_height,
        }
    }

    pub fn x(&self) -> i32 {
        self.x
    }

    pub fn y(&self) -> i32 {
        self.y
    }

    pub fn world_width(&self) -> usize {
        self.world_width
    }

    pub fn world_height(&self) -> usize {
        self.world_height
    }

    pub fn pan(&mut self, dx: i32, dy: i32) {
        self.x = (self.x + dx).rem_euclid(self.world_width as i32);
        self.y = (self.y + dy).rem_euclid(self.world_height as i32);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_new_camera_looks_at_the_worlds_origin() {
        let camera = Camera::new(600, 400);

        assert_eq!((camera.x(), camera.y()), (0, 0));
    }

    #[test]
    fn panning_moves_where_the_camera_looks() {
        let mut camera = Camera::new(600, 400);

        camera.pan(30, -20);

        assert_eq!((camera.x(), camera.y()), (30, 380));
    }

    #[test]
    fn panning_past_the_worlds_edge_comes_round_the_other_side() {
        // The world is a torus, so a camera that keeps going right arrives
        // back where it started rather than looking at nothing.
        let mut camera = Camera::new(600, 400);

        camera.pan(590, 0);
        camera.pan(20, 0);

        assert_eq!(camera.x(), 10);
    }

    #[test]
    fn the_camera_knows_how_big_the_world_is() {
        let camera = Camera::new(600, 400);

        assert_eq!((camera.world_width(), camera.world_height()), (600, 400));
    }
}
