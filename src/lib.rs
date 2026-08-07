mod cli;
mod critter;
mod elapsed_format;
mod fps_counter;
mod genome;
mod heading;
mod instruction;
mod pellet;
mod population_growth_detector;
mod renderer;
mod replenish_countdown;
mod snapshot;
mod stagnation_detector;
mod text;
mod world;

pub use cli::{parse as parse_cli, CliError, Startup};
pub use critter::{Critter, MAX_CRITTER_ENERGY};
pub use elapsed_format::format_elapsed;
pub use fps_counter::FpsCounter;
pub use genome::{Genome, GenomeParseError};
pub use heading::Heading;
pub use instruction::Instruction;
pub use pellet::{
    Pellet, EMITTER_COLOR, EMITTER_RADIUS, PELLET_COLOR, PELLET_ENERGY, PELLET_MAX_DRIFT,
    PELLET_MIN_DRIFT, PELLET_RADIUS,
};
pub use population_growth_detector::PopulationGrowthDetector;
pub use renderer::Renderer;
pub use replenish_countdown::{format_minutes_seconds, frames_until_next_replenish};
pub use snapshot::format_block as format_snapshot_block;
pub use stagnation_detector::StagnationDetector;
pub use text::pixels as text_pixels;
pub use world::{World, CRITTER_RADIUS};
