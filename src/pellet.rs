/// A drifting morsel of food. Pellets are emitted from the world's central
/// emitter and coast outward forever, wrapping at the edges. Position is
/// floating point so a pellet can move slower than one pixel per tick.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pellet {
    pub x: f32,
    pub y: f32,
    pub dx: f32,
    pub dy: f32,
}

impl Pellet {
    /// A motionless pellet at a whole-pixel position. Test-only: real pellets
    /// are emitted with a heading by the world's emitter.
    #[cfg(test)]
    pub fn at(x: i32, y: i32) -> Self {
        Self {
            x: x as f32,
            y: y as f32,
            dx: 0.0,
            dy: 0.0,
        }
    }

    /// Advances the pellet along its heading, wrapping around the world.
    pub fn drift(&mut self, width: f32, height: f32) {
        self.x = (self.x + self.dx).rem_euclid(width);
        self.y = (self.y + self.dy).rem_euclid(height);
    }
}

pub const PELLET_RADIUS: i32 = 2;
/// Fastest a pellet may drift, in pixels per tick. Slow enough that food
/// lingers near the emitter rather than crossing the world in moments.
pub const PELLET_MAX_DRIFT: f32 = 0.35;
/// Slowest a pellet may drift. Nonzero so no pellet parks on the emitter.
pub const PELLET_MIN_DRIFT: f32 = 0.05;
pub const PELLET_COLOR: u32 = 0x00_FF_00;
pub const PELLET_ENERGY: u32 = 100;
