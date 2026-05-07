use crate::{Critter, Heading, Instruction};
use rand::Rng;

pub const CRITTER_RADIUS: i32 = 20;

const NUM_CRITTERS: usize = 8;
const INITIAL_ENERGY: u32 = 60;
const TICKS_PER_INSTRUCTION: u32 = 15;
const INSTRUCTION_LIST_LENGTH: usize = 4;
const STEP_SIZE: i32 = 25;

pub struct World {
    critters: Vec<Critter>,
}

impl World {
    pub fn new<R: Rng>(width: usize, height: usize, rng: &mut R) -> Self {
        let critters = (0..NUM_CRITTERS)
            .map(|_| spawn_critter(width, height, rng))
            .collect();
        Self { critters }
    }

    pub fn critters(&self) -> &[Critter] {
        &self.critters
    }

    pub fn tick(&mut self) {
        for critter in &mut self.critters {
            critter.tick();
        }
    }

    pub fn reset<R: Rng>(&mut self, width: usize, height: usize, rng: &mut R) {
        self.critters = (0..NUM_CRITTERS)
            .map(|_| spawn_critter(width, height, rng))
            .collect();
    }
}

fn spawn_critter<R: Rng>(width: usize, height: usize, rng: &mut R) -> Critter {
    let instructions = Instruction::random_list(rng, INSTRUCTION_LIST_LENGTH);
    let x = rng.gen_range(CRITTER_RADIUS..(width as i32 - CRITTER_RADIUS));
    let y = rng.gen_range(CRITTER_RADIUS..(height as i32 - CRITTER_RADIUS));
    Critter::new(
        x,
        y,
        Heading::random(rng),
        instructions,
        TICKS_PER_INSTRUCTION,
        STEP_SIZE,
        INITIAL_ENERGY,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    const TEST_WIDTH: usize = 200;
    const TEST_HEIGHT: usize = 200;

    mod new {
        use super::*;

        #[test]
        fn it_creates_the_configured_number_of_critters() {
            let mut rng = StdRng::seed_from_u64(0);

            let world = World::new(TEST_WIDTH, TEST_HEIGHT, &mut rng);

            assert_eq!(world.critters().len(), NUM_CRITTERS);
        }

        #[test]
        fn each_critter_starts_with_full_energy() {
            let mut rng = StdRng::seed_from_u64(0);

            let world = World::new(TEST_WIDTH, TEST_HEIGHT, &mut rng);

            assert!(world
                .critters()
                .iter()
                .all(|c| c.energy() == c.initial_energy()));
        }

        #[test]
        fn every_critter_spawns_fully_inside_the_world_bounds() {
            let mut rng = StdRng::seed_from_u64(0);

            let world = World::new(TEST_WIDTH, TEST_HEIGHT, &mut rng);

            for critter in world.critters() {
                assert!(critter.x() >= CRITTER_RADIUS);
                assert!(critter.x() < TEST_WIDTH as i32 - CRITTER_RADIUS);
                assert!(critter.y() >= CRITTER_RADIUS);
                assert!(critter.y() < TEST_HEIGHT as i32 - CRITTER_RADIUS);
            }
        }
    }

    mod tick {
        use super::*;

        #[test]
        fn it_advances_each_critter_by_one_tick() {
            let mut rng = StdRng::seed_from_u64(0);
            let mut world = World::new(TEST_WIDTH, TEST_HEIGHT, &mut rng);
            // After enough ticks (>= TICKS_PER_INSTRUCTION) every critter must have
            // executed at least one instruction, so its energy must have decreased.
            let initial_energies: Vec<u32> = world.critters().iter().map(|c| c.energy()).collect();

            for _ in 0..TICKS_PER_INSTRUCTION {
                world.tick();
            }

            for (critter, initial) in world.critters().iter().zip(initial_energies) {
                assert!(critter.energy() < initial);
            }
        }
    }

    mod reset {
        use super::*;

        #[test]
        fn it_replaces_the_critters() {
            let mut rng = StdRng::seed_from_u64(0);
            let mut world = World::new(TEST_WIDTH, TEST_HEIGHT, &mut rng);
            for _ in 0..TICKS_PER_INSTRUCTION {
                world.tick();
            }
            let depleted_energies: Vec<u32> = world.critters().iter().map(|c| c.energy()).collect();

            world.reset(TEST_WIDTH, TEST_HEIGHT, &mut rng);

            let fresh_energies: Vec<u32> = world.critters().iter().map(|c| c.energy()).collect();
            assert_ne!(fresh_energies, depleted_energies);
            assert!(fresh_energies.iter().all(|&e| e == INITIAL_ENERGY));
        }
    }
}
