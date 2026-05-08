use crate::{Critter, Heading, Instruction, Pellet, PELLET_RADIUS};
use rand::Rng;

pub const CRITTER_RADIUS: i32 = 20;

const NUM_CRITTERS: usize = 8;
const NUM_PELLETS: usize = 30;
const INITIAL_ENERGY: u32 = 60;
const TICKS_PER_INSTRUCTION: u32 = 15;
const INSTRUCTION_LIST_LENGTH: usize = 4;
const STEP_SIZE: i32 = 25;

pub struct World {
    critters: Vec<Critter>,
    pellets: Vec<Pellet>,
}

impl World {
    pub fn new<R: Rng>(width: usize, height: usize, rng: &mut R) -> Self {
        let critters = (0..NUM_CRITTERS)
            .map(|_| spawn_critter(width, height, rng))
            .collect();
        let pellets = (0..NUM_PELLETS)
            .map(|_| spawn_pellet(width, height, rng))
            .collect();
        Self { critters, pellets }
    }

    #[cfg(test)]
    pub fn with_critters_and_pellets(critters: Vec<Critter>, pellets: Vec<Pellet>) -> Self {
        Self { critters, pellets }
    }

    pub fn critters(&self) -> &[Critter] {
        &self.critters
    }

    pub fn pellets(&self) -> &[Pellet] {
        &self.pellets
    }

    pub fn tick(&mut self) {
        for critter in &mut self.critters {
            critter.tick();
        }
        self.consume_pellets();
    }

    fn consume_pellets(&mut self) {
        let eat_distance_squared =
            (CRITTER_RADIUS + PELLET_RADIUS) * (CRITTER_RADIUS + PELLET_RADIUS);
        for critter in &mut self.critters {
            if critter.energy() >= critter.initial_energy() {
                continue;
            }
            self.pellets.retain(|pellet| {
                if critter.energy() >= critter.initial_energy() {
                    return true;
                }
                let dx = critter.x() - pellet.x;
                let dy = critter.y() - pellet.y;
                if dx * dx + dy * dy < eat_distance_squared {
                    critter.gain_energy(crate::PELLET_ENERGY);
                    false
                } else {
                    true
                }
            });
        }
    }

    pub fn reset<R: Rng>(&mut self, width: usize, height: usize, rng: &mut R) {
        self.critters = (0..NUM_CRITTERS)
            .map(|_| spawn_critter(width, height, rng))
            .collect();
        self.pellets = (0..NUM_PELLETS)
            .map(|_| spawn_pellet(width, height, rng))
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

fn spawn_pellet<R: Rng>(width: usize, height: usize, rng: &mut R) -> Pellet {
    Pellet {
        x: rng.gen_range(PELLET_RADIUS..(width as i32 - PELLET_RADIUS)),
        y: rng.gen_range(PELLET_RADIUS..(height as i32 - PELLET_RADIUS)),
    }
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
            // Construct with no pellets so eating doesn't interfere with the
            // per-instruction energy decrement.
            let mut rng = StdRng::seed_from_u64(0);
            let world_with_critters = World::new(TEST_WIDTH, TEST_HEIGHT, &mut rng);
            let critters: Vec<Critter> = world_with_critters.critters().to_vec();
            let mut world = World::with_critters_and_pellets(critters.clone(), vec![]);
            let initial_energies: Vec<u32> = critters.iter().map(|c| c.energy()).collect();

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

    mod eating {
        use super::*;
        use crate::{Critter, Heading, Instruction, Pellet, PELLET_ENERGY};

        const HUNGRY_INITIAL: u32 = 60;
        const STARTING_ENERGY: u32 = 30;

        fn hungry_critter(x: i32, y: i32) -> Critter {
            let mut critter = Critter::new(
                x,
                y,
                Heading::North,
                vec![Instruction::DoNothing],
                u32::MAX, // never executes
                1,
                HUNGRY_INITIAL,
            );
            critter.lose_energy(HUNGRY_INITIAL - STARTING_ENERGY);
            critter
        }

        #[test]
        fn a_critter_overlapping_a_pellet_consumes_it() {
            let critter = hungry_critter(100, 100);
            let pellet = Pellet { x: 100, y: 100 };
            let mut world = World::with_critters_and_pellets(vec![critter], vec![pellet]);

            world.tick();

            assert_eq!(world.pellets().len(), 0);
        }

        #[test]
        fn eating_a_pellet_increases_energy_by_the_pellet_energy_amount() {
            let critter = hungry_critter(100, 100);
            let pellet = Pellet { x: 100, y: 100 };
            let mut world = World::with_critters_and_pellets(vec![critter], vec![pellet]);

            world.tick();

            assert_eq!(
                world.critters()[0].energy(),
                STARTING_ENERGY + PELLET_ENERGY
            );
        }

        #[test]
        fn a_critter_that_does_not_overlap_a_pellet_leaves_it_alone() {
            // Centers far apart: critter at (100, 100), pellet at (300, 100).
            let critter = hungry_critter(100, 100);
            let pellet = Pellet { x: 300, y: 100 };
            let mut world = World::with_critters_and_pellets(vec![critter], vec![pellet]);

            world.tick();

            assert_eq!(world.pellets().len(), 1);
            assert_eq!(world.critters()[0].energy(), STARTING_ENERGY);
        }

        #[test]
        fn a_pellet_just_inside_the_eating_distance_is_consumed() {
            // Eating distance is CRITTER_RADIUS + PELLET_RADIUS = 24.
            // Place pellet at distance 23 — strictly less than 24, eaten.
            let critter = hungry_critter(100, 100);
            let pellet = Pellet {
                x: 100 + (CRITTER_RADIUS + PELLET_RADIUS - 1),
                y: 100,
            };
            let mut world = World::with_critters_and_pellets(vec![critter], vec![pellet]);

            world.tick();

            assert_eq!(world.pellets().len(), 0);
        }

        #[test]
        fn a_pellet_at_exactly_the_eating_distance_is_not_consumed() {
            // Distance equals CRITTER_RADIUS + PELLET_RADIUS — circles tangent, not overlapping.
            let critter = hungry_critter(100, 100);
            let pellet = Pellet {
                x: 100 + CRITTER_RADIUS + PELLET_RADIUS,
                y: 100,
            };
            let mut world = World::with_critters_and_pellets(vec![critter], vec![pellet]);

            world.tick();

            assert_eq!(world.pellets().len(), 1);
        }

        #[test]
        fn a_full_critter_does_not_consume_a_pellet() {
            let full = Critter::new(
                100,
                100,
                Heading::North,
                vec![Instruction::DoNothing],
                u32::MAX,
                1,
                HUNGRY_INITIAL,
            );
            let pellet = Pellet { x: 100, y: 100 };
            let mut world = World::with_critters_and_pellets(vec![full], vec![pellet]);

            world.tick();

            assert_eq!(world.pellets().len(), 1);
            assert_eq!(world.critters()[0].energy(), HUNGRY_INITIAL);
        }

        #[test]
        fn eating_caps_energy_at_initial_energy() {
            // Critter just below full: only has room for a partial pellet.
            let mut critter = Critter::new(
                100,
                100,
                Heading::North,
                vec![Instruction::DoNothing],
                u32::MAX,
                1,
                HUNGRY_INITIAL,
            );
            critter.lose_energy(2); // energy = 58, space for 2 of the 10 pellet energy
            let pellet = Pellet { x: 100, y: 100 };
            let mut world = World::with_critters_and_pellets(vec![critter], vec![pellet]);

            world.tick();

            assert_eq!(world.critters()[0].energy(), HUNGRY_INITIAL);
        }
    }

    mod pellets {
        use super::*;
        use crate::PELLET_RADIUS;

        #[test]
        fn the_world_scatters_the_configured_number_of_pellets_on_creation() {
            let mut rng = StdRng::seed_from_u64(0);

            let world = World::new(TEST_WIDTH, TEST_HEIGHT, &mut rng);

            assert_eq!(world.pellets().len(), NUM_PELLETS);
        }

        #[test]
        fn every_pellet_spawns_fully_inside_the_world_bounds() {
            // Use multiple seeds so we exhaust the rng range and reliably catch
            // boundary-mutation bugs in the spawn rectangle.
            for seed in 0..50 {
                let mut rng = StdRng::seed_from_u64(seed);
                let world = World::new(TEST_WIDTH, TEST_HEIGHT, &mut rng);
                for pellet in world.pellets() {
                    assert!(pellet.x >= PELLET_RADIUS);
                    assert!(pellet.x < TEST_WIDTH as i32 - PELLET_RADIUS);
                    assert!(pellet.y >= PELLET_RADIUS);
                    assert!(pellet.y < TEST_HEIGHT as i32 - PELLET_RADIUS);
                }
            }
        }

        #[test]
        fn pellets_spread_across_the_full_world_width_and_height() {
            // Across many spawn batches, at least one pellet must land in the right
            // half and at least one in the bottom half — proving the spawn range
            // is the full canvas, not a clipped corner.
            let mut any_right = false;
            let mut any_bottom = false;
            let half_width = (TEST_WIDTH as i32) / 2;
            let half_height = (TEST_HEIGHT as i32) / 2;
            for seed in 0..50 {
                let mut rng = StdRng::seed_from_u64(seed);
                let world = World::new(TEST_WIDTH, TEST_HEIGHT, &mut rng);
                for pellet in world.pellets() {
                    if pellet.x >= half_width {
                        any_right = true;
                    }
                    if pellet.y >= half_height {
                        any_bottom = true;
                    }
                }
            }
            assert!(any_right);
            assert!(any_bottom);
        }

        #[test]
        fn reset_re_scatters_the_pellets() {
            let mut rng = StdRng::seed_from_u64(0);
            let mut world = World::new(TEST_WIDTH, TEST_HEIGHT, &mut rng);
            let original: Vec<_> = world.pellets().to_vec();

            world.reset(TEST_WIDTH, TEST_HEIGHT, &mut rng);

            assert_eq!(world.pellets().len(), NUM_PELLETS);
            assert_ne!(world.pellets(), original.as_slice());
        }
    }
}
