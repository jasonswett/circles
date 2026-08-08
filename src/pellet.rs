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
    /// Ticks since this pellet was emitted. Food spoils, so a pellet that
    /// goes uneaten eventually vanishes.
    pub age: u32,
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
            age: 0,
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

    /// Whether the pellet has spoiled and should be removed from the world.
    pub fn is_spoiled(&self) -> bool {
        self.age >= PELLET_LIFETIME_TICKS
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
pub const PELLETS_PER_POISON: usize = 50;
pub const PELLET_ENERGY: u32 = 100;
/// How long a pellet lasts before spoiling away. At ~60 FPS this is sixty
/// seconds, so uneaten food does not accumulate indefinitely.
pub const PELLET_LIFETIME_TICKS: u32 = 3600;

#[cfg(test)]
mod tests {
    use super::*;

    fn aged(age: u32) -> Pellet {
        let mut pellet = Pellet::at(0, 0);
        pellet.age = age;
        pellet
    }

    mod is_spoiled {
        use super::*;

        #[test]
        fn a_fresh_pellet_has_not_spoiled() {
            assert!(!aged(0).is_spoiled());
        }

        #[test]
        fn a_pellet_short_of_its_lifetime_has_not_spoiled() {
            assert!(!aged(PELLET_LIFETIME_TICKS - 1).is_spoiled());
        }

        #[test]
        fn a_pellet_that_reaches_its_lifetime_has_spoiled() {
            assert!(aged(PELLET_LIFETIME_TICKS).is_spoiled());
        }
    }
}
