mod critter;
mod heading;
mod instruction;
mod pellet;
mod renderer;
mod world;

pub use critter::Critter;
pub use heading::Heading;
pub use instruction::Instruction;
pub use pellet::{Pellet, PELLET_COLOR, PELLET_ENERGY, PELLET_RADIUS};
pub use renderer::Renderer;
pub use world::{World, CRITTER_RADIUS};
