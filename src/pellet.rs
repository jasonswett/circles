#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Pellet {
    pub x: i32,
    pub y: i32,
}

pub const PELLET_RADIUS: i32 = 4;
pub const PELLET_COLOR: u32 = 0x00_FF_00;
pub const PELLET_ENERGY: u32 = 100;
