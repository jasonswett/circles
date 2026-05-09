use crate::{Critter, Heading, Pellet, PELLET_RADIUS};
use rand::Rng;

pub const CRITTER_RADIUS: i32 = 10;

const NUM_CRITTERS: usize = 100;
const NUM_PELLETS: usize = 1000;
pub const MIN_POPULATION: usize = 20;
const INITIAL_ENERGY: u32 = 60;
const TICKS_PER_INSTRUCTION: u32 = 5;
const STEP_SIZE: i32 = 12;

pub struct World {
    width: usize,
    height: usize,
    critters: Vec<Critter>,
    pellets: Vec<Pellet>,
    original_total_energy: u32,
    generation: u32,
}

impl World {
    pub fn new<R: Rng>(width: usize, height: usize, rng: &mut R) -> Self {
        let critters: Vec<Critter> = (0..NUM_CRITTERS)
            .map(|_| spawn_critter(width, height, rng))
            .collect();
        let pellets: Vec<Pellet> = (0..NUM_PELLETS)
            .map(|_| spawn_pellet(width, height, rng))
            .collect();
        let original_total_energy = critter_total_energy(&critters) + pellet_total_energy(&pellets);
        Self {
            width,
            height,
            critters,
            pellets,
            original_total_energy,
            generation: 1,
        }
    }

    #[cfg(test)]
    pub fn with_critters_and_pellets(
        width: usize,
        height: usize,
        critters: Vec<Critter>,
        pellets: Vec<Pellet>,
    ) -> Self {
        let original_total_energy = critter_total_energy(&critters) + pellet_total_energy(&pellets);
        Self {
            width,
            height,
            critters,
            pellets,
            original_total_energy,
            generation: 1,
        }
    }

    pub fn original_total_energy(&self) -> u32 {
        self.original_total_energy
    }

    pub fn replenish_pellets<R: Rng>(&mut self, rng: &mut R) {
        let current = self.total_energy();
        if current >= self.original_total_energy {
            return;
        }
        let deficit = self.original_total_energy - current;
        // Round up: add enough pellets so the new total is at least the target.
        let pellets_needed = deficit.div_ceil(crate::PELLET_ENERGY);
        for _ in 0..pellets_needed {
            self.pellets
                .push(spawn_pellet(self.width, self.height, rng));
        }
    }

    pub fn critters(&self) -> &[Critter] {
        &self.critters
    }

    pub fn pellets(&self) -> &[Pellet] {
        &self.pellets
    }

    pub fn total_energy(&self) -> u32 {
        let critter_energy: u32 = self.critters.iter().map(|c| c.energy()).sum();
        let pellet_energy = self.pellets.len() as u32 * crate::PELLET_ENERGY;
        critter_energy + pellet_energy
    }

    pub fn tick(&mut self, allow_split: bool) {
        let mut children = Vec::new();
        for critter in &mut self.critters {
            if let Some(mut child) = critter.tick(allow_split) {
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
            if critter.energy() >= crate::MAX_CRITTER_ENERGY {
                continue;
            }
            self.pellets.retain(|pellet| {
                if critter.energy() >= crate::MAX_CRITTER_ENERGY {
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

    pub fn reap_dead_critters(&mut self) {
        self.critters.retain(|c| c.energy() > 0);
    }

    pub fn population_too_low(&self) -> bool {
        self.critters.len() < MIN_POPULATION
    }

    pub fn generation(&self) -> u32 {
        self.generation
    }

    pub fn reset<R: Rng>(&mut self, rng: &mut R) {
        self.critters = (0..NUM_CRITTERS)
            .map(|_| spawn_critter(self.width, self.height, rng))
            .collect();
        self.pellets = (0..NUM_PELLETS)
            .map(|_| spawn_pellet(self.width, self.height, rng))
            .collect();
        self.generation += 1;
    }
}

fn critter_total_energy(critters: &[Critter]) -> u32 {
    critters.iter().map(|c| c.energy()).sum()
}

fn pellet_total_energy(pellets: &[Pellet]) -> u32 {
    pellets.len() as u32 * crate::PELLET_ENERGY
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
    let x = rng.gen_range(CRITTER_RADIUS..(width as i32 - CRITTER_RADIUS));
    let y = rng.gen_range(CRITTER_RADIUS..(height as i32 - CRITTER_RADIUS));
    Critter::new(
        x,
        y,
        Heading::random(rng),
        TICKS_PER_INSTRUCTION,
        STEP_SIZE,
        INITIAL_ENERGY,
        rng.gen(),
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
        fn ticking_a_world_decreases_total_critter_energy() {
            // Construct with no pellets so eating doesn't interfere with the
            // per-instruction energy decrement. Some critters may stall on a
            // Split they can't afford, but the total across all critters must
            // still drop because at least some will execute non-Split instructions.
            let mut rng = StdRng::seed_from_u64(0);
            let world_with_critters = World::new(TEST_WIDTH, TEST_HEIGHT, &mut rng);
            let critters: Vec<Critter> = world_with_critters.critters().to_vec();
            let mut world =
                World::with_critters_and_pellets(TEST_WIDTH, TEST_HEIGHT, critters.clone(), vec![]);
            let initial_total: u32 = critters.iter().map(|c| c.energy()).sum();

            for _ in 0..TICKS_PER_INSTRUCTION {
                world.tick(true);
            }

            let final_total: u32 = world.critters().iter().map(|c| c.energy()).sum();
            assert!(final_total < initial_total);
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
        use crate::{Critter, Genome, Heading, Instruction};

        #[test]
        fn a_critter_that_splits_appears_twice_in_the_critter_list_after_a_tick() {
            let splitter = Critter::with_genome(
                100,
                100,
                Heading::North,
                1,
                1,
                60,
                0,
                Genome::all(Instruction::Split),
            );
            let mut world =
                World::with_critters_and_pellets(TEST_WIDTH, TEST_HEIGHT, vec![splitter], vec![]);

            world.tick(true);

            assert_eq!(world.critters().len(), 2);
        }
    }

    mod wrapping {
        use super::*;
        use crate::{Critter, Genome, Heading, Instruction};

        #[test]
        fn a_critter_that_walks_past_the_right_edge_wraps_to_the_left() {
            let critter = Critter::with_genome(
                TEST_WIDTH as i32 - 1,
                50,
                Heading::East,
                1,
                1,
                u32::MAX,
                0,
                Genome::all(Instruction::MoveForward),
            );
            let mut world =
                World::with_critters_and_pellets(TEST_WIDTH, TEST_HEIGHT, vec![critter], vec![]);

            world.tick(true);

            assert_eq!(world.critters()[0].x(), 0);
        }

        #[test]
        fn a_critter_that_walks_past_the_top_edge_wraps_to_the_bottom() {
            let critter = Critter::with_genome(
                50,
                0,
                Heading::North,
                1,
                1,
                u32::MAX,
                0,
                Genome::all(Instruction::MoveForward),
            );
            let mut world =
                World::with_critters_and_pellets(TEST_WIDTH, TEST_HEIGHT, vec![critter], vec![]);

            world.tick(true);

            assert_eq!(world.critters()[0].y(), TEST_HEIGHT as i32 - 1);
        }
    }

    mod eating {
        use super::*;
        use crate::{Critter, Genome, Heading, Instruction, Pellet, PELLET_ENERGY};

        const HUNGRY_INITIAL: u32 = 200;
        const STARTING_ENERGY: u32 = 10;

        fn hungry_critter(x: i32, y: i32) -> Critter {
            let mut critter = Critter::with_genome(
                x,
                y,
                Heading::North,
                u32::MAX, // never executes
                1,
                HUNGRY_INITIAL,
                0,
                Genome::all(Instruction::DoNothing),
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

            world.tick(true);

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

            world.tick(true);

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

            world.tick(true);

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

            world.tick(true);

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

            world.tick(true);

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

            world.tick(true);

            assert_eq!(world.pellets().len(), 1);
        }

        #[test]
        fn a_critter_at_the_energy_cap_does_not_eat_an_overlapping_pellet() {
            let mut critter = Critter::with_genome(
                100,
                100,
                Heading::North,
                u32::MAX,
                1,
                100,
                0,
                Genome::all(Instruction::DoNothing),
            );
            critter.gain_energy(crate::MAX_CRITTER_ENERGY); // saturates at MAX
            assert_eq!(critter.energy(), crate::MAX_CRITTER_ENERGY);
            let pellet = Pellet { x: 100, y: 100 };
            let mut world = World::with_critters_and_pellets(
                TEST_WIDTH,
                TEST_HEIGHT,
                vec![critter],
                vec![pellet],
            );

            world.tick(true);

            assert_eq!(world.pellets().len(), 1);
            assert_eq!(world.critters()[0].energy(), crate::MAX_CRITTER_ENERGY);
        }

        #[test]
        fn eating_can_push_energy_past_initial_energy() {
            // Eating no longer caps at initial_energy: a critter can stockpile.
            let critter = Critter::with_genome(
                100,
                100,
                Heading::North,
                u32::MAX,
                1,
                HUNGRY_INITIAL,
                0,
                Genome::all(Instruction::DoNothing),
            );
            let pellet = Pellet { x: 100, y: 100 };
            let mut world = World::with_critters_and_pellets(
                TEST_WIDTH,
                TEST_HEIGHT,
                vec![critter],
                vec![pellet],
            );

            world.tick(true);

            assert_eq!(world.critters()[0].energy(), HUNGRY_INITIAL + PELLET_ENERGY);
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

            world.tick(true);

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

            world.tick(true);

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

    mod total_energy {
        use super::*;
        use crate::{Critter, Genome, Heading, Instruction, Pellet, PELLET_ENERGY};

        #[test]
        fn an_empty_world_has_zero_total_energy() {
            let world = World::with_critters_and_pellets(TEST_WIDTH, TEST_HEIGHT, vec![], vec![]);

            assert_eq!(world.total_energy(), 0);
        }

        #[test]
        fn total_energy_sums_each_critters_current_energy() {
            let critter_a = Critter::with_genome(
                50,
                50,
                Heading::North,
                1,
                1,
                30,
                0,
                Genome::all(Instruction::DoNothing),
            );
            let critter_b = Critter::with_genome(
                70,
                70,
                Heading::North,
                1,
                1,
                25,
                0,
                Genome::all(Instruction::DoNothing),
            );
            let world = World::with_critters_and_pellets(
                TEST_WIDTH,
                TEST_HEIGHT,
                vec![critter_a, critter_b],
                vec![],
            );

            assert_eq!(world.total_energy(), 55);
        }

        #[test]
        fn total_energy_counts_each_pellet_at_pellet_energy_value() {
            let pellets = vec![
                Pellet { x: 10, y: 10 },
                Pellet { x: 20, y: 20 },
                Pellet { x: 30, y: 30 },
            ];
            let world = World::with_critters_and_pellets(TEST_WIDTH, TEST_HEIGHT, vec![], pellets);

            assert_eq!(world.total_energy(), 3 * PELLET_ENERGY);
        }

        #[test]
        fn total_energy_combines_critters_and_pellets() {
            let critter = Critter::with_genome(
                50,
                50,
                Heading::North,
                1,
                1,
                40,
                0,
                Genome::all(Instruction::DoNothing),
            );
            let pellet = Pellet { x: 20, y: 20 };
            let world = World::with_critters_and_pellets(
                TEST_WIDTH,
                TEST_HEIGHT,
                vec![critter],
                vec![pellet],
            );

            assert_eq!(world.total_energy(), 40 + PELLET_ENERGY);
        }
    }

    mod reap_dead_critters {
        use super::*;
        use crate::{Critter, Genome, Heading, Instruction};

        fn critter_with_energy(x: i32, y: i32, energy: u32) -> Critter {
            let mut critter = Critter::with_genome(
                x,
                y,
                Heading::North,
                u32::MAX,
                1,
                100,
                0,
                Genome::all(Instruction::DoNothing),
            );
            critter.lose_energy(100 - energy);
            critter
        }

        #[test]
        fn it_removes_critters_whose_energy_is_zero() {
            let dead = critter_with_energy(50, 50, 0);
            let mut world =
                World::with_critters_and_pellets(TEST_WIDTH, TEST_HEIGHT, vec![dead], vec![]);

            world.reap_dead_critters();

            assert!(world.critters().is_empty());
        }

        #[test]
        fn it_leaves_critters_with_energy_above_zero() {
            let alive = critter_with_energy(50, 50, 1);
            let mut world =
                World::with_critters_and_pellets(TEST_WIDTH, TEST_HEIGHT, vec![alive], vec![]);

            world.reap_dead_critters();

            assert_eq!(world.critters().len(), 1);
        }

        #[test]
        fn it_removes_only_the_dead_critters_in_a_mixed_population() {
            let alive = critter_with_energy(50, 50, 5);
            let dead = critter_with_energy(60, 60, 0);
            let also_alive = critter_with_energy(70, 70, 10);
            let mut world = World::with_critters_and_pellets(
                TEST_WIDTH,
                TEST_HEIGHT,
                vec![alive, dead, also_alive],
                vec![],
            );

            world.reap_dead_critters();

            let energies: Vec<u32> = world.critters().iter().map(|c| c.energy()).collect();
            assert_eq!(energies, vec![5, 10]);
        }

        #[test]
        fn it_does_not_remove_pellets() {
            let dead = critter_with_energy(50, 50, 0);
            let pellet = Pellet { x: 30, y: 30 };
            let mut world =
                World::with_critters_and_pellets(TEST_WIDTH, TEST_HEIGHT, vec![dead], vec![pellet]);

            world.reap_dead_critters();

            assert_eq!(world.pellets().len(), 1);
        }
    }

    mod original_total_energy {
        use super::*;

        #[test]
        fn it_returns_the_total_energy_present_at_construction() {
            let mut rng = StdRng::seed_from_u64(0);

            let world = World::new(TEST_WIDTH, TEST_HEIGHT, &mut rng);

            assert_eq!(world.original_total_energy(), world.total_energy());
        }

        #[test]
        fn it_does_not_change_after_critters_lose_energy() {
            let mut rng = StdRng::seed_from_u64(0);
            let mut world = World::new(TEST_WIDTH, TEST_HEIGHT, &mut rng);
            let original = world.original_total_energy();

            for _ in 0..100 {
                world.tick(true);
            }

            assert_eq!(world.original_total_energy(), original);
            assert!(world.total_energy() < original);
        }
    }

    mod replenish_pellets {
        use super::*;
        use crate::{Critter, Genome, Heading, Instruction, PELLET_ENERGY};

        fn empty_world() -> World {
            World::with_critters_and_pellets(TEST_WIDTH, TEST_HEIGHT, vec![], vec![])
        }

        fn world_with_target(target: u32) -> World {
            // Construct an "empty" world and then make the target deterministic
            // by directly using with_critters_and_pellets with a known
            // composition that achieves total_energy == target.
            let mut world = empty_world();
            let pellets_needed = target / PELLET_ENERGY;
            let mut rng = StdRng::seed_from_u64(0);
            for _ in 0..pellets_needed {
                world
                    .pellets
                    .push(super::spawn_pellet(TEST_WIDTH, TEST_HEIGHT, &mut rng));
            }
            world.original_total_energy = world.total_energy();
            world
        }

        #[test]
        fn it_does_not_add_pellets_when_total_energy_is_already_at_or_above_original() {
            let target = 10 * PELLET_ENERGY;
            let mut world = world_with_target(target);
            let pellets_before = world.pellets().len();
            let mut rng = StdRng::seed_from_u64(1);

            world.replenish_pellets(&mut rng);

            assert_eq!(world.pellets().len(), pellets_before);
        }

        #[test]
        fn it_adds_enough_pellets_to_bring_total_energy_back_to_the_original() {
            let target = 10 * PELLET_ENERGY;
            let mut world = world_with_target(target);
            // Drain energy by removing pellets directly.
            world.pellets.clear();
            let mut rng = StdRng::seed_from_u64(1);

            world.replenish_pellets(&mut rng);

            assert!(world.total_energy() >= target);
        }

        #[test]
        fn it_does_not_add_substantially_more_pellets_than_needed() {
            // After replenish, total should be within one pellet's worth of the
            // target, not (current + target). Test with a partially-full world.
            let target = 10 * PELLET_ENERGY;
            let mut world = world_with_target(target);
            // Remove half the pellets so current > 0 but < target.
            world.pellets.truncate(5);
            let mut rng = StdRng::seed_from_u64(1);

            world.replenish_pellets(&mut rng);

            assert!(world.total_energy() < target + PELLET_ENERGY);
        }

        #[test]
        fn replenished_pellets_land_inside_the_world_bounds() {
            let mut world = empty_world();
            world.original_total_energy = 50 * PELLET_ENERGY;
            for seed in 0..20 {
                let mut rng = StdRng::seed_from_u64(seed);
                world.pellets.clear();
                world.replenish_pellets(&mut rng);
                for pellet in world.pellets() {
                    assert!(pellet.x >= PELLET_RADIUS);
                    assert!(pellet.x < TEST_WIDTH as i32 - PELLET_RADIUS);
                    assert!(pellet.y >= PELLET_RADIUS);
                    assert!(pellet.y < TEST_HEIGHT as i32 - PELLET_RADIUS);
                }
            }
        }

        #[test]
        fn replenishment_accounts_for_critter_energy_too() {
            // The original target = critter_energy + pellet_energy. If a critter
            // exists with substantial energy, fewer pellets are needed to top up.
            let mut world = empty_world();
            let critter = Critter::with_genome(
                50,
                50,
                Heading::North,
                u32::MAX,
                1,
                100,
                0,
                Genome::all(Instruction::DoNothing),
            );
            world.critters.push(critter);
            // Target = 100 (critter) + 5 pellets * 100 (pellet energy hypothetical) — let's be concrete:
            world.original_total_energy = 100 + 5 * PELLET_ENERGY;
            let mut rng = StdRng::seed_from_u64(0);

            world.replenish_pellets(&mut rng);

            assert!(world.total_energy() >= world.original_total_energy);
        }
    }

    mod population_too_low {
        use super::*;
        use crate::{Critter, Genome, Heading, Instruction};

        fn world_with_critter_count(count: usize) -> World {
            let critters = (0..count)
                .map(|i| {
                    Critter::with_genome(
                        i as i32,
                        0,
                        Heading::North,
                        1,
                        1,
                        100,
                        i as u64,
                        Genome::all(Instruction::DoNothing),
                    )
                })
                .collect();
            World::with_critters_and_pellets(TEST_WIDTH, TEST_HEIGHT, critters, vec![])
        }

        #[test]
        fn it_returns_false_when_population_equals_the_minimum() {
            let world = world_with_critter_count(MIN_POPULATION);

            assert!(!world.population_too_low());
        }

        #[test]
        fn it_returns_true_when_population_is_just_below_the_minimum() {
            let world = world_with_critter_count(MIN_POPULATION - 1);

            assert!(world.population_too_low());
        }

        #[test]
        fn it_returns_false_when_population_is_well_above_the_minimum() {
            let world = world_with_critter_count(MIN_POPULATION + 30);

            assert!(!world.population_too_low());
        }
    }

    mod generation {
        use super::*;

        #[test]
        fn a_new_world_starts_at_generation_one() {
            let mut rng = StdRng::seed_from_u64(0);

            let world = World::new(TEST_WIDTH, TEST_HEIGHT, &mut rng);

            assert_eq!(world.generation(), 1);
        }

        #[test]
        fn each_reset_increments_the_generation() {
            let mut rng = StdRng::seed_from_u64(0);
            let mut world = World::new(TEST_WIDTH, TEST_HEIGHT, &mut rng);

            world.reset(&mut rng);

            assert_eq!(world.generation(), 2);
        }

        #[test]
        fn the_generation_count_persists_across_multiple_resets() {
            let mut rng = StdRng::seed_from_u64(0);
            let mut world = World::new(TEST_WIDTH, TEST_HEIGHT, &mut rng);

            world.reset(&mut rng);
            world.reset(&mut rng);
            world.reset(&mut rng);

            assert_eq!(world.generation(), 4);
        }
    }
}
