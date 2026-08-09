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
mod snapshot;
mod stagnation_detector;
mod text;
mod world;
mod world_record;

pub use cli::{parse as parse_cli, CliError, Startup};
pub use critter::{
    Critter, CRITTER_RADIUS, MAX_CRITTER_ENERGY, REFERENCE_ENERGY, SPLIT_DURATION_TICKS,
};
pub use elapsed_format::format_elapsed;
pub use fps_counter::FpsCounter;
pub use genome::{Genome, GenomeParseError, Senses};
pub use heading::Heading;
pub use instruction::Instruction;
pub use pellet::{
    Pellet, PELLETS_PER_POISON, PELLET_COLOR, PELLET_ENERGY, PELLET_LIFESPAN_TICKS,
    PELLET_MAX_DRIFT, PELLET_MIN_DRIFT, PELLET_RADIUS, POISON_COLOR,
};
pub use population_growth_detector::PopulationGrowthDetector;
pub use renderer::Renderer;
pub use snapshot::format_block as format_snapshot_block;
pub use stagnation_detector::StagnationDetector;
pub use text::pixels as text_pixels;
pub use world::{World, PELLET_BATCH_SIZE};
pub use world_record::WorldRecord;
