/// A drifting morsel of food. Pellets are emitted from the world's central
/// emitter and coast outward forever, wrapping at the edges. Position is
/// floating point so a pellet can move slower than one pixel per tick.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pellet {
    pub x: f32,
    pub y: f32,
    pub dx: f32,
    pub dy: f32,
    /// Poison kills any critter it touches, whether or not the critter meant
    /// to eat it. Critters cannot see it coming: nothing in their sensorium
    /// distinguishes poison from food.
    pub poisonous: bool,
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
            poisonous: false,
        }
    }

    /// A motionless poison pellet at a whole-pixel position. Test-only.
    #[cfg(test)]
    pub fn poison_at(x: i32, y: i32) -> Self {
        Self {
            poisonous: true,
            ..Self::at(x, y)
        }
    }

    /// What this pellet is drawn in: poison is red, food green.
    pub fn color(&self) -> u32 {
        if self.poisonous {
            POISON_COLOR
        } else {
            PELLET_COLOR
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
pub const PELLET_MAX_DRIFT: f32 = 0.7;
/// Slowest a pellet may drift. Nonzero so no pellet parks on the emitter.
pub const PELLET_MIN_DRIFT: f32 = 0.1;
pub const PELLET_COLOR: u32 = 0x00_FF_00;
pub const POISON_COLOR: u32 = 0xFF_00_00;
/// One pellet in this many is poison rather than food.
pub const PELLETS_PER_POISON: usize = 20;
pub const PELLET_ENERGY: u32 = 100;
