use crate::{Critter, Heading, Instruction, Pellet, PELLET_RADIUS};
use rand::Rng;

pub const CRITTER_RADIUS: i32 = 10;

const NUM_CRITTERS: usize = 16;
const NUM_PELLETS: usize = 120;
const INITIAL_ENERGY: u32 = 60;
const TICKS_PER_INSTRUCTION: u32 = 5;
const INSTRUCTION_LIST_LENGTH: usize = 8;
const STEP_SIZE: i32 = 12;

pub struct World {
    width: usize,
    height: usize,
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
        Self {
            width,
            height,
            critters,
            pellets,
        }
    }

    #[cfg(test)]
    pub fn with_critters_and_pellets(
        width: usize,
        height: usize,
        critters: Vec<Critter>,
        pellets: Vec<Pellet>,
    ) -> Self {
        Self {
            width,
            height,
            critters,
            pellets,
        }
    }

    pub fn critters(&self) -> &[Critter] {
        &self.critters
    }

    pub fn pellets(&self) -> &[Pellet] {
        &self.pellets
    }

    pub fn tick(&mut self) {
        let mut children = Vec::new();
        for critter in &mut self.critters {
            if let Some(mut child) = critter.tick() {
                child.wrap_position(self.width as i32, self.height as i32);
                children.push(child);
            }
            critter.wrap_position(self.width as i32, self.height as i32);
        }
        self.critters.extend(children);
        self.consume_pellets();
    }

    fn consume_pellets(&mut self) {
        let eat_distance_squared =
            (CRITTER_RADIUS + PELLET_RADIUS) * (CRITTER_RADIUS + PELLET_RADIUS);
        let width = self.width as i32;
        let height = self.height as i32;
        for critter in &mut self.critters {
            if critter.energy() >= critter.initial_energy() {
                continue;
            }
            self.pellets.retain(|pellet| {
                if critter.energy() >= critter.initial_energy() {
                    return true;
                }
                let dx = toroidal_delta(critter.x(), pellet.x, width);
                let dy = toroidal_delta(critter.y(), pellet.y, height);
                if dx * dx + dy * dy < eat_distance_squared {
                    critter.gain_energy(crate::PELLET_ENERGY);
                    false
                } else {
                    true
                }
            });
        }
    }

    pub fn reset<R: Rng>(&mut self, rng: &mut R) {
        self.critters = (0..NUM_CRITTERS)
            .map(|_| spawn_critter(self.width, self.height, rng))
            .collect();
        self.pellets = (0..NUM_PELLETS)
            .map(|_| spawn_pellet(self.width, self.height, rng))
            .collect();
    }
}

fn toroidal_delta(a: i32, b: i32, size: i32) -> i32 {
    let raw = (a - b).rem_euclid(size);
    if raw <= size / 2 {
        raw
    } else {
        raw - size
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
            // Use multiple seeds so we exhaust the rng range and reliably catch
            // boundary-mutation bugs in the spawn rectangle.
            for seed in 0..50 {
                let mut rng = StdRng::seed_from_u64(seed);
                let world = World::new(TEST_WIDTH, TEST_HEIGHT, &mut rng);
                for critter in world.critters() {
                    assert!(critter.x() >= CRITTER_RADIUS);
                    assert!(critter.x() < TEST_WIDTH as i32 - CRITTER_RADIUS);
                    assert!(critter.y() >= CRITTER_RADIUS);
                    assert!(critter.y() < TEST_HEIGHT as i32 - CRITTER_RADIUS);
                }
            }
        }

        #[test]
        fn critters_spread_across_the_full_world_width_and_height() {
            // Across many spawn batches, at least one critter must land in the
            // right half and at least one in the bottom half — proving the spawn
            // range is the full canvas, not a clipped corner.
            let mut any_right = false;
            let mut any_bottom = false;
            let half_width = (TEST_WIDTH as i32) / 2;
            let half_height = (TEST_HEIGHT as i32) / 2;
            for seed in 0..50 {
                let mut rng = StdRng::seed_from_u64(seed);
                let world = World::new(TEST_WIDTH, TEST_HEIGHT, &mut rng);
                for critter in world.critters() {
                    if critter.x() >= half_width {
                        any_right = true;
                    }
                    if critter.y() >= half_height {
                        any_bottom = true;
                    }
                }
            }
            assert!(any_right);
            assert!(any_bottom);
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
            let mut world =
                World::with_critters_and_pellets(TEST_WIDTH, TEST_HEIGHT, critters.clone(), vec![]);
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
        fn after_reset_every_critter_is_at_full_energy() {
            let mut rng = StdRng::seed_from_u64(0);
            let mut world =
                World::with_critters_and_pellets(TEST_WIDTH, TEST_HEIGHT, vec![], vec![]);

            world.reset(&mut rng);

            assert!(world
                .critters()
                .iter()
                .all(|c| c.energy() == INITIAL_ENERGY));
        }

        #[test]
        fn after_reset_the_critter_positions_change() {
            let mut rng = StdRng::seed_from_u64(0);
            let mut world = World::new(TEST_WIDTH, TEST_HEIGHT, &mut rng);
            let original_positions: Vec<(i32, i32)> =
                world.critters().iter().map(|c| (c.x(), c.y())).collect();

            world.reset(&mut rng);

            let new_positions: Vec<(i32, i32)> =
                world.critters().iter().map(|c| (c.x(), c.y())).collect();
            assert_ne!(new_positions, original_positions);
        }
    }

    mod toroidal_delta_tests {
        use super::super::toroidal_delta;

        #[test]
        fn equal_values_have_zero_delta() {
            assert_eq!(toroidal_delta(50, 50, 200), 0);
        }

        #[test]
        fn small_positive_delta_is_returned_as_is() {
            assert_eq!(toroidal_delta(50, 30, 200), 20);
        }

        #[test]
        fn small_negative_delta_is_returned_as_negative() {
            assert_eq!(toroidal_delta(30, 50, 200), -20);
        }

        #[test]
        fn delta_larger_than_half_size_wraps_to_a_smaller_negative_value() {
            // Half-size = 100. raw = (10 - 190).rem_euclid(200) = 20.
            // 20 < 100, so the result is 20 — but that's the unwrapped path.
            // For 150: raw = (190 - 40).rem_euclid(200) = 150. 150 > 100, returns 150 - 200 = -50.
            assert_eq!(toroidal_delta(190, 40, 200), -50);
        }

        #[test]
        fn delta_just_above_half_size_wraps_to_negative() {
            // (5 - 100).rem_euclid(200) = 105. > 100 → returns 105 - 200 = -95.
            assert_eq!(toroidal_delta(5, 100, 200), -95);
        }
    }

    mod splitting {
        use super::*;
        use crate::{Critter, Heading, Instruction};

        #[test]
        fn a_critter_that_splits_appears_twice_in_the_critter_list_after_a_tick() {
            let splitter =
                Critter::new(100, 100, Heading::North, vec![Instruction::Split], 1, 1, 60);
            let mut world =
                World::with_critters_and_pellets(TEST_WIDTH, TEST_HEIGHT, vec![splitter], vec![]);

            world.tick();

            assert_eq!(world.critters().len(), 2);
        }
    }

    mod wrapping {
        use super::*;
        use crate::{Critter, Heading, Instruction};

        #[test]
        fn a_critter_that_walks_past_the_right_edge_wraps_to_the_left() {
            let critter = Critter::new(
                TEST_WIDTH as i32 - 1,
                50,
                Heading::East,
                vec![Instruction::MoveForward],
                1,
                1,
                u32::MAX,
            );
            let mut world =
                World::with_critters_and_pellets(TEST_WIDTH, TEST_HEIGHT, vec![critter], vec![]);

            world.tick();

            assert_eq!(world.critters()[0].x(), 0);
        }

        #[test]
        fn a_critter_that_walks_past_the_top_edge_wraps_to_the_bottom() {
            let critter = Critter::new(
                50,
                0,
                Heading::North,
                vec![Instruction::MoveForward],
                1,
                1,
                u32::MAX,
            );
            let mut world =
                World::with_critters_and_pellets(TEST_WIDTH, TEST_HEIGHT, vec![critter], vec![]);

            world.tick();

            assert_eq!(world.critters()[0].y(), TEST_HEIGHT as i32 - 1);
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
            let mut world = World::with_critters_and_pellets(
                TEST_WIDTH,
                TEST_HEIGHT,
                vec![critter],
                vec![pellet],
            );

            world.tick();

            assert_eq!(world.pellets().len(), 0);
        }

        #[test]
        fn eating_a_pellet_increases_energy_by_the_pellet_energy_amount() {
            let critter = hungry_critter(100, 100);
            let pellet = Pellet { x: 100, y: 100 };
            let mut world = World::with_critters_and_pellets(
                TEST_WIDTH,
                TEST_HEIGHT,
                vec![critter],
                vec![pellet],
            );

            world.tick();

            assert_eq!(
                world.critters()[0].energy(),
                STARTING_ENERGY + PELLET_ENERGY
            );
        }

        #[test]
        fn a_critter_that_does_not_overlap_a_pellet_leaves_it_alone() {
            // Critter at (50, 100), pellet at (100, 100): dx = 50, no wrap is shorter.
            let critter = hungry_critter(50, 100);
            let pellet = Pellet { x: 100, y: 100 };
            let mut world = World::with_critters_and_pellets(
                TEST_WIDTH,
                TEST_HEIGHT,
                vec![critter],
                vec![pellet],
            );

            world.tick();

            assert_eq!(world.pellets().len(), 1);
            assert_eq!(world.critters()[0].energy(), STARTING_ENERGY);
        }

        #[test]
        fn a_pellet_just_inside_the_eating_distance_is_consumed() {
            // Eating distance is CRITTER_RADIUS + PELLET_RADIUS.
            // Place pellet at distance 23 — strictly less than 24, eaten.
            let critter = hungry_critter(100, 100);
            let pellet = Pellet {
                x: 100 + (CRITTER_RADIUS + PELLET_RADIUS - 1),
                y: 100,
            };
            let mut world = World::with_critters_and_pellets(
                TEST_WIDTH,
                TEST_HEIGHT,
                vec![critter],
                vec![pellet],
            );

            world.tick();

            assert_eq!(world.pellets().len(), 0);
        }

        #[test]
        fn a_pellet_outside_the_eating_distance_along_a_dominant_axis_is_not_consumed() {
            // Asymmetric placement (small dx, large dy) so both axes' squared
            // contributions are distinguishable: the distance formula must square
            // each component, not just sum them or treat them as identical.
            // With dx=2, dy=15: dx² + dy² = 4 + 225 = 229 > eat_distance_squared.
            let critter = hungry_critter(100, 100);
            let pellet = Pellet { x: 102, y: 115 };
            let mut world = World::with_critters_and_pellets(
                TEST_WIDTH,
                TEST_HEIGHT,
                vec![critter],
                vec![pellet],
            );

            world.tick();

            assert_eq!(world.pellets().len(), 1);
        }

        #[test]
        fn a_pellet_at_exactly_the_eating_distance_is_not_consumed() {
            // Distance equals CRITTER_RADIUS + PELLET_RADIUS — circles tangent, not overlapping.
            let critter = hungry_critter(100, 100);
            let pellet = Pellet {
                x: 100 + CRITTER_RADIUS + PELLET_RADIUS,
                y: 100,
            };
            let mut world = World::with_critters_and_pellets(
                TEST_WIDTH,
                TEST_HEIGHT,
                vec![critter],
                vec![pellet],
            );

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
            let mut world =
                World::with_critters_and_pellets(TEST_WIDTH, TEST_HEIGHT, vec![full], vec![pellet]);

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
            let mut world = World::with_critters_and_pellets(
                TEST_WIDTH,
                TEST_HEIGHT,
                vec![critter],
                vec![pellet],
            );

            world.tick();

            assert_eq!(world.critters()[0].energy(), HUNGRY_INITIAL);
        }

        #[test]
        fn a_critter_near_the_left_edge_can_eat_a_pellet_near_the_right_edge_via_wrap() {
            // Critter at x=2, pellet at x=TEST_WIDTH-2: euclidean distance ≈ 196,
            // but wrapped (toroidal) distance is just 4 — well within eating range.
            let critter = hungry_critter(2, 100);
            let pellet = Pellet {
                x: TEST_WIDTH as i32 - 2,
                y: 100,
            };
            let mut world = World::with_critters_and_pellets(
                TEST_WIDTH,
                TEST_HEIGHT,
                vec![critter],
                vec![pellet],
            );

            world.tick();

            assert_eq!(world.pellets().len(), 0);
        }

        #[test]
        fn a_critter_near_the_top_edge_can_eat_a_pellet_near_the_bottom_edge_via_wrap() {
            let critter = hungry_critter(100, 2);
            let pellet = Pellet {
                x: 100,
                y: TEST_HEIGHT as i32 - 2,
            };
            let mut world = World::with_critters_and_pellets(
                TEST_WIDTH,
                TEST_HEIGHT,
                vec![critter],
                vec![pellet],
            );

            world.tick();

            assert_eq!(world.pellets().len(), 0);
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

            world.reset(&mut rng);

            assert_eq!(world.pellets().len(), NUM_PELLETS);
            assert_ne!(world.pellets(), original.as_slice());
        }
    }
}
