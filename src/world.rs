use crate::{Critter, Genome, Heading, Pellet, PELLET_RADIUS};
use rand::Rng;

pub const CRITTER_RADIUS: i32 = 10;

const NUM_CRITTERS: usize = 1000;
const NUM_PELLETS: usize = 1000;
pub const MIN_POPULATION: usize = 20;
const INITIAL_ENERGY: u32 = 60;
const TICKS_PER_INSTRUCTION: u32 = 5;
const STEP_SIZE: i32 = 12;
// How many critters to refresh per overlap-detection call. The detector cycles
// through the population round-robin: critters not reached on a given call
// keep their previous flag until they come up again.
const OVERLAP_DETECTION_BUDGET: usize = 20;
// How many ticks a critter stays marked as overlapping after a confirmed
// detection. Smooths out the visual flicker caused by the round-robin budget.
const OVERLAP_INDICATOR_LINGER_TICKS: u32 = 30;
// How many ticks a victim stays marked as "being stolen from" after a
// successful energy transfer. Brief — theft is supposed to look like a flash.
const STOLEN_FROM_INDICATOR_LINGER_TICKS: u32 = 10;

pub struct World {
    width: usize,
    height: usize,
    critters: Vec<Critter>,
    pellets: Vec<Pellet>,
    original_total_energy: u32,
    generation: u32,
    overlap_detection_cursor: usize,
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
            overlap_detection_cursor: 0,
        }
    }

    /// Build a world where every critter starts with the given seed genome
    /// instead of a freshly randomized one. Positions, headings, and pellets
    /// are randomized as in `new`.
    pub fn with_seed_genome<R: Rng>(
        width: usize,
        height: usize,
        seed_genome: Genome,
        rng: &mut R,
    ) -> Self {
        let critters: Vec<Critter> = (0..NUM_CRITTERS)
            .map(|_| spawn_critter_with_genome(width, height, seed_genome.clone(), rng))
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
            overlap_detection_cursor: 0,
        }
    }

    /// Returns the genome with the most copies among the current critters.
    /// Ties are broken by first-seen order in `critters()`. Returns `None` if
    /// there are no critters.
    pub fn dominant_genome(&self) -> Option<&Genome> {
        // Vec<(&Genome, usize)> preserves first-seen order naturally; we walk
        // the critters once, incrementing the count for any matching entry or
        // appending a new one. Linear in the number of unique genomes — fine
        // for this scale.
        let mut tally: Vec<(&Genome, usize)> = Vec::new();
        for critter in &self.critters {
            let genome = critter.genome();
            if let Some(entry) = tally.iter_mut().find(|(g, _)| *g == genome) {
                entry.1 += 1;
            } else {
                tally.push((genome, 1));
            }
        }
        // Manual fold instead of `max_by_key`: the standard iterator method
        // keeps the *last* maximum on ties, but we want the first-seen
        // winner. Walk left-to-right, replacing only on a strict increase.
        let mut best: Option<(&Genome, usize)> = None;
        for (genome, count) in tally {
            match best {
                Some((_, best_count)) if count <= best_count => {}
                _ => best = Some((genome, count)),
            }
        }
        best.map(|(genome, _)| genome)
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
            overlap_detection_cursor: 0,
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
        let mut eater_indices: Vec<usize> = Vec::new();
        for (index, critter) in self.critters.iter_mut().enumerate() {
            critter.age_overlap_indicator();
            critter.age_being_stolen_from_indicator();
            let outcome = critter.tick(allow_split);
            if let Some(mut child) = outcome.child {
                child.wrap_position(self.width as i32, self.height as i32);
                children.push(child);
            }
            if outcome.attempted_eat {
                eater_indices.push(index);
            }
            critter.wrap_position(self.width as i32, self.height as i32);
        }
        self.critters.extend(children);
        self.resolve_eats(&eater_indices);
        self.detect_critter_overlaps();
    }

    fn resolve_eats(&mut self, eater_indices: &[usize]) {
        let count = self.critters.len();
        let pellet_eat_distance_squared =
            (CRITTER_RADIUS + PELLET_RADIUS) * (CRITTER_RADIUS + PELLET_RADIUS);
        let critter_eat_distance_squared = (2 * CRITTER_RADIUS) * (2 * CRITTER_RADIUS);
        let width = self.width as i32;
        let height = self.height as i32;

        for &eater_index in eater_indices {
            if eater_index >= count {
                continue;
            }
            if self.critters[eater_index].energy() == 0 {
                continue;
            }
            let eater_x = self.critters[eater_index].x();
            let eater_y = self.critters[eater_index].y();

            // Look for an overlapping pellet first; eat it if found.
            let pellet_position = self.pellets.iter().position(|pellet| {
                let dx = toroidal_delta(eater_x, pellet.x, width);
                let dy = toroidal_delta(eater_y, pellet.y, height);
                dx * dx + dy * dy < pellet_eat_distance_squared
            });
            if let Some(position) = pellet_position {
                self.pellets.swap_remove(position);
                self.critters[eater_index].gain_energy(crate::PELLET_ENERGY);
                continue;
            }

            // No pellet in range — drain the first overlapping critter, if
            // any. The eater's gain saturates at MAX_CRITTER_ENERGY; whatever
            // doesn't fit is lost.
            let victim_index = (0..count).find(|&victim_index| {
                if victim_index == eater_index {
                    return false;
                }
                if self.critters[victim_index].energy() == 0 {
                    return false;
                }
                let dx = toroidal_delta(eater_x, self.critters[victim_index].x(), width);
                let dy = toroidal_delta(eater_y, self.critters[victim_index].y(), height);
                dx * dx + dy * dy < critter_eat_distance_squared
            });
            if let Some(victim_index) = victim_index {
                let victim_energy = self.critters[victim_index].energy();
                let gained = self.critters[eater_index].gain_energy_saturating(victim_energy);
                self.critters[victim_index].lose_energy(gained);
                self.critters[victim_index]
                    .mark_being_stolen_from_for(STOLEN_FROM_INDICATOR_LINGER_TICKS);
            }
        }
    }

    pub fn detect_critter_overlaps(&mut self) {
        let count = self.critters.len();
        if count < 2 {
            return;
        }
        let overlap_distance_squared = (2 * CRITTER_RADIUS) * (2 * CRITTER_RADIUS);
        let width = self.width as i32;
        let height = self.height as i32;
        let budget = OVERLAP_DETECTION_BUDGET.min(count);

        for offset in 0..budget {
            let i = (self.overlap_detection_cursor + offset) % count;
            if self.critters[i].energy() == 0 {
                continue;
            }
            for j in 0..count {
                if i == j {
                    continue;
                }
                if self.critters[j].energy() == 0 {
                    continue;
                }
                let dx = toroidal_delta(self.critters[i].x(), self.critters[j].x(), width);
                let dy = toroidal_delta(self.critters[i].y(), self.critters[j].y(), height);
                if dx * dx + dy * dy < overlap_distance_squared {
                    let i_color = self.critters[i].genome_color();
                    let j_color = self.critters[j].genome_color();
                    self.critters[i].mark_overlapping_critter_for(OVERLAP_INDICATOR_LINGER_TICKS);
                    self.critters[i].record_overlap_color(j_color);
                    self.critters[j].mark_overlapping_critter_for(OVERLAP_INDICATOR_LINGER_TICKS);
                    self.critters[j].record_overlap_color(i_color);
                }
            }
        }

        // The inner `(cursor + offset) % count` indexing normalizes the cursor
        // on every read, so we don't need to keep the cursor itself bounded;
        // wrapping_add makes the eventual overflow explicit and harmless.
        self.overlap_detection_cursor = self.overlap_detection_cursor.wrapping_add(budget);
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

fn spawn_critter_with_genome<R: Rng>(
    width: usize,
    height: usize,
    genome: Genome,
    rng: &mut R,
) -> Critter {
    let x = rng.gen_range(CRITTER_RADIUS..(width as i32 - CRITTER_RADIUS));
    let y = rng.gen_range(CRITTER_RADIUS..(height as i32 - CRITTER_RADIUS));
    Critter::with_genome(
        x,
        y,
        Heading::random(rng),
        TICKS_PER_INSTRUCTION,
        STEP_SIZE,
        INITIAL_ENERGY,
        rng.gen(),
        genome,
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
        use crate::{Critter, Genome, Heading, Instruction};

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

        #[test]
        fn the_overlap_indicator_decays_to_off_over_enough_ticks_with_no_overlap() {
            // A lone critter has nothing to overlap with. Mark it for a few
            // linger ticks; after enough world ticks, the indicator should be
            // gone — proving World::tick ages the indicator.
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
            critter.mark_overlapping_critter_for(3);
            let mut world =
                World::with_critters_and_pellets(TEST_WIDTH, TEST_HEIGHT, vec![critter], vec![]);

            for _ in 0..3 {
                world.tick(true);
            }

            assert!(!world.critters()[0].is_overlapping_critter());
        }

        #[test]
        fn the_being_stolen_from_indicator_decays_to_off_over_enough_ticks() {
            // A lone critter cannot be stolen from. Mark it for a few linger
            // ticks; after enough world ticks the indicator should be gone —
            // proving World::tick ages this indicator too.
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
            critter.mark_being_stolen_from_for(3);
            let mut world =
                World::with_critters_and_pellets(TEST_WIDTH, TEST_HEIGHT, vec![critter], vec![]);

            for _ in 0..3 {
                world.tick(true);
            }

            assert!(!world.critters()[0].is_being_stolen_from());
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
        use crate::{
            Critter, Genome, Heading, Instruction, Pellet, EAT_ATTEMPT_COST, PELLET_ENERGY,
        };

        const HUNGRY_INITIAL: u32 = 200;
        const STARTING_ENERGY: u32 = 20;
        // Energy after a single Eat firing tick where no pellet was found —
        // the eat-attempt cost plus the base 1-energy tick cost.
        const STARTING_AFTER_FAILED_EAT: u32 = STARTING_ENERGY - EAT_ATTEMPT_COST - 1;

        // A critter whose genome decodes to Eat at every cursor and which fires
        // an instruction every tick. Energy is set just above zero so that
        // gaining a pellet is observable without hitting the cap.
        fn eating_critter(x: i32, y: i32) -> Critter {
            let mut critter = Critter::with_genome(
                x,
                y,
                Heading::North,
                1, // fire every tick
                1,
                HUNGRY_INITIAL,
                0,
                Genome::all(Instruction::Eat),
            );
            critter.lose_energy(HUNGRY_INITIAL - STARTING_ENERGY);
            critter
        }

        // A critter whose genome decodes to DoNothing at every cursor — used
        // to confirm that critters which never execute Eat do not consume
        // pellets, even when overlapping one.
        fn idle_critter(x: i32, y: i32) -> Critter {
            let mut critter = Critter::with_genome(
                x,
                y,
                Heading::North,
                1,
                1,
                HUNGRY_INITIAL,
                0,
                Genome::all(Instruction::DoNothing),
            );
            critter.lose_energy(HUNGRY_INITIAL - STARTING_ENERGY);
            critter
        }

        fn world_with(critter: Critter, pellet: Pellet) -> World {
            World::with_critters_and_pellets(TEST_WIDTH, TEST_HEIGHT, vec![critter], vec![pellet])
        }

        #[test]
        fn a_critter_executing_eat_while_overlapping_a_pellet_consumes_it() {
            let mut world = world_with(eating_critter(100, 100), Pellet { x: 100, y: 100 });

            world.tick(true);

            assert_eq!(world.pellets().len(), 0);
        }

        #[test]
        fn an_eater_drained_to_zero_by_the_attempt_cost_eats_nothing() {
            let critter = Critter::with_genome(
                100,
                100,
                Heading::North,
                1,
                1,
                EAT_ATTEMPT_COST,
                0,
                Genome::all(Instruction::Eat),
            );
            let mut world = world_with(critter, Pellet { x: 100, y: 100 });

            world.tick(true);

            assert_eq!(world.pellets().len(), 1);
            assert_eq!(world.critters()[0].energy(), 0);
        }

        #[test]
        fn eating_a_pellet_increases_energy_by_the_pellet_energy_amount() {
            // Start, minus eat-attempt cost, minus the base tick cost, plus
            // the pellet's energy.
            let mut world = world_with(eating_critter(100, 100), Pellet { x: 100, y: 100 });

            world.tick(true);

            assert_eq!(
                world.critters()[0].energy(),
                STARTING_AFTER_FAILED_EAT + PELLET_ENERGY
            );
        }

        #[test]
        fn a_critter_that_does_not_execute_eat_leaves_an_overlapping_pellet_alone() {
            let mut world = world_with(idle_critter(100, 100), Pellet { x: 100, y: 100 });

            world.tick(true);

            assert_eq!(world.pellets().len(), 1);
            assert_eq!(world.critters()[0].energy(), STARTING_ENERGY - 1);
        }

        #[test]
        fn an_eating_critter_that_does_not_overlap_a_pellet_leaves_it_alone() {
            let mut world = world_with(eating_critter(50, 100), Pellet { x: 100, y: 100 });

            world.tick(true);

            assert_eq!(world.pellets().len(), 1);
            assert_eq!(world.critters()[0].energy(), STARTING_AFTER_FAILED_EAT);
        }

        #[test]
        fn a_pellet_just_inside_the_eating_distance_is_consumed() {
            let pellet = Pellet {
                x: 100 + (CRITTER_RADIUS + PELLET_RADIUS - 1),
                y: 100,
            };
            let mut world = world_with(eating_critter(100, 100), pellet);

            world.tick(true);

            assert_eq!(world.pellets().len(), 0);
        }

        #[test]
        fn a_pellet_outside_the_eating_distance_along_a_dominant_axis_is_not_consumed() {
            // dx=2, dy=15: dx² + dy² = 229 > (CRITTER_RADIUS+PELLET_RADIUS)² = 196.
            let mut world = world_with(eating_critter(100, 100), Pellet { x: 102, y: 115 });

            world.tick(true);

            assert_eq!(world.pellets().len(), 1);
        }

        #[test]
        fn a_pellet_at_exactly_the_eating_distance_is_not_consumed() {
            let pellet = Pellet {
                x: 100 + CRITTER_RADIUS + PELLET_RADIUS,
                y: 100,
            };
            let mut world = world_with(eating_critter(100, 100), pellet);

            world.tick(true);

            assert_eq!(world.pellets().len(), 1);
        }

        #[test]
        fn eating_can_push_energy_past_initial_energy() {
            // A critter at full initial energy that successfully eats ends up
            // above its initial — there is no cap at initial_energy, only at
            // MAX_CRITTER_ENERGY.
            let critter = Critter::with_genome(
                100,
                100,
                Heading::North,
                1,
                1,
                HUNGRY_INITIAL,
                0,
                Genome::all(Instruction::Eat),
            );
            let mut world = world_with(critter, Pellet { x: 100, y: 100 });

            world.tick(true);

            assert_eq!(
                world.critters()[0].energy(),
                HUNGRY_INITIAL - EAT_ATTEMPT_COST - 1 + PELLET_ENERGY
            );
        }

        #[test]
        fn a_critter_near_the_left_edge_can_eat_a_pellet_near_the_right_edge_via_wrap() {
            let pellet = Pellet {
                x: TEST_WIDTH as i32 - 2,
                y: 100,
            };
            let mut world = world_with(eating_critter(2, 100), pellet);

            world.tick(true);

            assert_eq!(world.pellets().len(), 0);
        }

        #[test]
        fn a_critter_near_the_top_edge_can_eat_a_pellet_near_the_bottom_edge_via_wrap() {
            let pellet = Pellet {
                x: 100,
                y: TEST_HEIGHT as i32 - 2,
            };
            let mut world = world_with(eating_critter(100, 2), pellet);

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

    mod detect_critter_overlaps {
        use super::*;
        use crate::{Critter, Genome, Heading, Instruction};

        fn idle_critter_at(x: i32, y: i32) -> Critter {
            Critter::with_genome(
                x,
                y,
                Heading::North,
                u32::MAX,
                1,
                100,
                0,
                Genome::all(Instruction::DoNothing),
            )
        }

        #[test]
        fn two_overlapping_critters_both_get_their_flag_set() {
            let a = idle_critter_at(100, 100);
            let b = idle_critter_at(105, 100);
            let mut world =
                World::with_critters_and_pellets(TEST_WIDTH, TEST_HEIGHT, vec![a, b], vec![]);

            world.detect_critter_overlaps();

            assert!(world.critters()[0].is_overlapping_critter());
            assert!(world.critters()[1].is_overlapping_critter());
        }

        #[test]
        fn each_critter_records_the_genome_color_of_the_critter_it_overlaps() {
            // The two idle critters share the same DoNothing genome and so
            // share the same genome color. Each ends up recording that color.
            let a = idle_critter_at(100, 100);
            let b = idle_critter_at(105, 100);
            let expected_color = a.genome_color();
            let mut world =
                World::with_critters_and_pellets(TEST_WIDTH, TEST_HEIGHT, vec![a, b], vec![]);

            world.detect_critter_overlaps();

            assert_eq!(
                world.critters()[0].most_recent_overlap_color(),
                Some(expected_color)
            );
            assert_eq!(
                world.critters()[1].most_recent_overlap_color(),
                Some(expected_color)
            );
        }

        #[test]
        fn two_non_overlapping_critters_keep_their_flags_clear() {
            // Distance 50 between centers, well outside 2 * CRITTER_RADIUS = 20.
            let a = idle_critter_at(50, 100);
            let b = idle_critter_at(100, 100);
            let mut world =
                World::with_critters_and_pellets(TEST_WIDTH, TEST_HEIGHT, vec![a, b], vec![]);

            world.detect_critter_overlaps();

            assert!(!world.critters()[0].is_overlapping_critter());
            assert!(!world.critters()[1].is_overlapping_critter());
        }

        #[test]
        fn critters_that_touch_at_the_overlap_threshold_are_not_marked_overlapping() {
            // Distance equals 2 * CRITTER_RADIUS = 20: tangent, not overlapping.
            let a = idle_critter_at(100, 100);
            let b = idle_critter_at(120, 100);
            let mut world =
                World::with_critters_and_pellets(TEST_WIDTH, TEST_HEIGHT, vec![a, b], vec![]);

            world.detect_critter_overlaps();

            assert!(!world.critters()[0].is_overlapping_critter());
            assert!(!world.critters()[1].is_overlapping_critter());
        }

        #[test]
        fn an_overlap_with_a_zero_energy_critter_marks_neither_critter() {
            let a = idle_critter_at(100, 100);
            let mut b = idle_critter_at(105, 100);
            b.lose_energy(b.energy());
            assert_eq!(b.energy(), 0);
            let mut world =
                World::with_critters_and_pellets(TEST_WIDTH, TEST_HEIGHT, vec![a, b], vec![]);

            world.detect_critter_overlaps();

            assert!(!world.critters()[0].is_overlapping_critter());
            assert!(!world.critters()[1].is_overlapping_critter());
        }

        #[test]
        fn two_critters_just_inside_the_overlap_threshold_are_both_marked() {
            // Centers 19 apart: distance² = 361, strictly less than the
            // (2 * CRITTER_RADIUS)² = 400 threshold, so the pair counts as
            // overlapping. Sitting just inside the boundary kills threshold
            // mutations whose squared cutoff drops well below 361.
            let a = idle_critter_at(100, 100);
            let b = idle_critter_at(119, 100);
            let mut world =
                World::with_critters_and_pellets(TEST_WIDTH, TEST_HEIGHT, vec![a, b], vec![]);

            world.detect_critter_overlaps();

            assert!(world.critters()[0].is_overlapping_critter());
            assert!(world.critters()[1].is_overlapping_critter());
        }

        #[test]
        fn an_asymmetric_pair_outside_the_threshold_along_one_axis_is_not_marked() {
            // dx = 10, dy = 18: distance² = 100 + 324 = 424 ≥ 400, so the pair
            // does not overlap. Asymmetry between the axes ensures the squared-
            // distance computation must square each component separately rather
            // than sum, subtract, or otherwise collapse them into one value.
            let a = idle_critter_at(100, 100);
            let b = idle_critter_at(110, 118);
            let mut world =
                World::with_critters_and_pellets(TEST_WIDTH, TEST_HEIGHT, vec![a, b], vec![]);

            world.detect_critter_overlaps();

            assert!(!world.critters()[0].is_overlapping_critter());
            assert!(!world.critters()[1].is_overlapping_critter());
        }

        #[test]
        fn an_overlapping_pair_at_indices_well_past_zero_is_still_marked() {
            // The detector's outer cursor steps i across the population. A
            // mutation that collapses i to a constant (e.g. `(cursor + offset)
            // / count` instead of `% count`) would only ever look at critter
            // zero. Hide the overlapping pair near the end of the list so the
            // cursor must actually walk through the indices to find it.
            let pair_first_index = OVERLAP_DETECTION_BUDGET - 2;
            let stride: i32 = 40;
            let per_row: usize = (TEST_WIDTH as i32 / stride) as usize;
            let mut critters: Vec<Critter> = (0..OVERLAP_DETECTION_BUDGET)
                .map(|i| {
                    let col = (i % per_row) as i32;
                    let row = (i / per_row) as i32;
                    idle_critter_at(20 + col * stride, 20 + row * stride)
                })
                .collect();
            // Place the last two critters on top of each other so they overlap.
            let overlap_x = 100;
            let overlap_y = 100;
            critters[pair_first_index] = idle_critter_at(overlap_x, overlap_y);
            critters[pair_first_index + 1] = idle_critter_at(overlap_x + 5, overlap_y);
            let mut world =
                World::with_critters_and_pellets(TEST_WIDTH, TEST_HEIGHT, critters, vec![]);

            world.detect_critter_overlaps();

            assert!(world.critters()[pair_first_index].is_overlapping_critter());
            assert!(world.critters()[pair_first_index + 1].is_overlapping_critter());
        }

        #[test]
        fn the_first_detection_call_does_not_visit_critters_beyond_its_budget() {
            // With a population larger than the budget, the first detect call
            // can only sweep the first OVERLAP_DETECTION_BUDGET critters. An
            // overlapping pair placed beyond that should remain unmarked.
            let world = world_with_overlapping_pair_at(OVERLAP_DETECTION_BUDGET);

            let world = run_detection_once(world);

            assert!(!world.critters()[OVERLAP_DETECTION_BUDGET].is_overlapping_critter());
            assert!(!world.critters()[OVERLAP_DETECTION_BUDGET + 1].is_overlapping_critter());
        }

        #[test]
        fn the_second_detection_call_visits_critters_the_first_call_did_not_reach() {
            // After the first call the cursor advances past the initial sweep,
            // so a second call covers the next slice — including a pair placed
            // beyond the first budget window.
            let world = world_with_overlapping_pair_at(OVERLAP_DETECTION_BUDGET);

            let world = run_detection_twice(world);

            assert!(world.critters()[OVERLAP_DETECTION_BUDGET].is_overlapping_critter());
            assert!(world.critters()[OVERLAP_DETECTION_BUDGET + 1].is_overlapping_critter());
        }

        #[test]
        fn two_zero_energy_critters_overlapping_each_other_mark_neither() {
            let mut a = idle_critter_at(100, 100);
            a.lose_energy(a.energy());
            let mut b = idle_critter_at(105, 100);
            b.lose_energy(b.energy());
            let mut world =
                World::with_critters_and_pellets(TEST_WIDTH, TEST_HEIGHT, vec![a, b], vec![]);

            world.detect_critter_overlaps();

            assert!(!world.critters()[0].is_overlapping_critter());
            assert!(!world.critters()[1].is_overlapping_critter());
        }

        fn world_with_overlapping_pair_at(pair_first_index: usize) -> World {
            // Build a population large enough that the pair sits beyond the
            // first detection budget, with all other critters spaced far apart
            // on a grid so they never overlap. The overlap pair sits at an
            // off-grid position so no grid critter accidentally overlaps it.
            let critter_count = pair_first_index + 2;
            let stride: i32 = 40;
            let per_row: usize = (TEST_WIDTH as i32 / stride) as usize;
            let mut critters: Vec<Critter> = (0..critter_count)
                .map(|i| {
                    let col = (i % per_row) as i32;
                    let row = (i / per_row) as i32;
                    idle_critter_at(20 + col * stride, 20 + row * stride)
                })
                .collect();
            // Off-grid coordinates: midpoint between four grid critters, so
            // no grid neighbor sits within the overlap radius.
            let overlap_x = 40;
            let overlap_y = 40;
            critters[pair_first_index] = idle_critter_at(overlap_x, overlap_y);
            critters[pair_first_index + 1] = idle_critter_at(overlap_x + 5, overlap_y);
            World::with_critters_and_pellets(TEST_WIDTH, TEST_HEIGHT, critters, vec![])
        }

        fn run_detection_once(mut world: World) -> World {
            world.detect_critter_overlaps();
            world
        }

        fn run_detection_twice(mut world: World) -> World {
            world.detect_critter_overlaps();
            world.detect_critter_overlaps();
            world
        }
    }

    mod eating_critters {
        use super::*;
        use crate::{
            Critter, Genome, Heading, Instruction, Pellet, EAT_ATTEMPT_COST, MAX_CRITTER_ENERGY,
        };

        const HUNGRY_INITIAL: u32 = 200;
        const STARTING_ENERGY: u32 = 20;
        // Energy after a single Eat firing tick where no transfer happened —
        // the eat-attempt cost plus the base 1-energy tick cost.
        const STARTING_AFTER_FAILED_EAT: u32 = STARTING_ENERGY - EAT_ATTEMPT_COST - 1;

        fn eating_critter_with_energy(x: i32, y: i32, energy: u32) -> Critter {
            Critter::with_genome(
                x,
                y,
                Heading::North,
                1,
                1,
                energy,
                0,
                Genome::all(Instruction::Eat),
            )
        }

        fn eating_critter(x: i32, y: i32) -> Critter {
            let mut critter = eating_critter_with_energy(x, y, HUNGRY_INITIAL);
            critter.lose_energy(HUNGRY_INITIAL - STARTING_ENERGY);
            critter
        }

        // A passive critter that does not execute Eat itself. Its energy is
        // set explicitly so it can serve as a victim of nearby eaters.
        fn idle_critter_with_energy(x: i32, y: i32, energy: u32) -> Critter {
            Critter::with_genome(
                x,
                y,
                Heading::North,
                u32::MAX,
                1,
                energy,
                0,
                Genome::all(Instruction::DoNothing),
            )
        }

        #[test]
        fn an_eater_with_no_pellet_in_range_drains_a_touching_critter() {
            let eater = eating_critter(100, 100);
            let victim = idle_critter_with_energy(105, 100, 80);
            let mut world = World::with_critters_and_pellets(
                TEST_WIDTH,
                TEST_HEIGHT,
                vec![eater, victim],
                vec![],
            );

            world.tick(true);

            assert_eq!(world.critters()[0].energy(), STARTING_AFTER_FAILED_EAT + 80);
            assert_eq!(world.critters()[1].energy(), 0);
        }

        #[test]
        fn an_eater_drains_only_what_fits_under_its_cap_when_eating_a_critter() {
            let eater = eating_critter_with_energy(100, 100, MAX_CRITTER_ENERGY - 30);
            let victim = idle_critter_with_energy(105, 100, 100);
            let mut world = World::with_critters_and_pellets(
                TEST_WIDTH,
                TEST_HEIGHT,
                vec![eater, victim],
                vec![],
            );

            world.tick(true);

            // Pre-resolve: eater pays the eat-attempt cost and the base
            // 1-energy tick cost, widening its headroom below the cap.
            // Eater caps at MAX; victim drops by exactly that headroom.
            assert_eq!(world.critters()[0].energy(), MAX_CRITTER_ENERGY);
            assert_eq!(
                world.critters()[1].energy(),
                100 - (30 + EAT_ATTEMPT_COST + 1)
            );
        }

        #[test]
        fn a_drained_victim_is_marked_as_being_stolen_from() {
            let eater = eating_critter(100, 100);
            let victim = idle_critter_with_energy(105, 100, 80);
            let mut world = World::with_critters_and_pellets(
                TEST_WIDTH,
                TEST_HEIGHT,
                vec![eater, victim],
                vec![],
            );

            world.tick(true);

            assert!(world.critters()[1].is_being_stolen_from());
        }

        #[test]
        fn an_eater_prefers_a_pellet_over_a_touching_critter_when_both_are_in_range() {
            let eater = eating_critter(100, 100);
            let victim = idle_critter_with_energy(105, 100, 80);
            let pellet = Pellet { x: 100, y: 100 };
            let mut world = World::with_critters_and_pellets(
                TEST_WIDTH,
                TEST_HEIGHT,
                vec![eater, victim],
                vec![pellet],
            );

            world.tick(true);

            assert_eq!(world.pellets().len(), 0);
            // Victim untouched because the pellet was eaten instead.
            assert_eq!(world.critters()[1].energy(), 80);
        }

        #[test]
        fn an_eater_does_not_drain_a_critter_outside_the_overlap_radius() {
            let eater = eating_critter(100, 100);
            // 50 px away — well outside 2 * CRITTER_RADIUS = 20.
            let bystander = idle_critter_with_energy(150, 100, 80);
            let mut world = World::with_critters_and_pellets(
                TEST_WIDTH,
                TEST_HEIGHT,
                vec![eater, bystander],
                vec![],
            );

            world.tick(true);

            assert_eq!(world.critters()[0].energy(), STARTING_AFTER_FAILED_EAT);
            assert_eq!(world.critters()[1].energy(), 80);
        }

        #[test]
        fn an_eater_does_not_drain_a_zero_energy_victim() {
            let eater = eating_critter(100, 100);
            let mut victim = idle_critter_with_energy(105, 100, 1);
            victim.lose_energy(1);
            let mut world = World::with_critters_and_pellets(
                TEST_WIDTH,
                TEST_HEIGHT,
                vec![eater, victim],
                vec![],
            );

            world.tick(true);

            assert_eq!(world.critters()[0].energy(), STARTING_AFTER_FAILED_EAT);
        }

        #[test]
        fn a_lone_eater_does_not_drain_itself() {
            let eater = eating_critter(100, 100);
            let mut world =
                World::with_critters_and_pellets(TEST_WIDTH, TEST_HEIGHT, vec![eater], vec![]);

            world.tick(true);

            assert_eq!(world.critters()[0].energy(), STARTING_AFTER_FAILED_EAT);
        }

        #[test]
        fn a_victim_just_inside_the_critter_eat_radius_is_drained() {
            // Centers 19 apart: distance² = 361 < (2 * CRITTER_RADIUS)² = 400.
            // Sits inside the eat radius; mutations that shrink the threshold
            // below 361 would mistakenly classify the pair as out of range.
            let eater = eating_critter(100, 100);
            let victim = idle_critter_with_energy(119, 100, 80);
            let mut world = World::with_critters_and_pellets(
                TEST_WIDTH,
                TEST_HEIGHT,
                vec![eater, victim],
                vec![],
            );

            world.tick(true);

            assert_eq!(world.critters()[1].energy(), 0);
        }

        #[test]
        fn a_critter_at_exactly_the_eat_distance_is_not_drained() {
            // Distance equals 2 * CRITTER_RADIUS = 20 — circles tangent, not
            // overlapping. The strict `<` comparison must reject this pair.
            let eater = eating_critter(100, 100);
            let victim = idle_critter_with_energy(120, 100, 80);
            let mut world = World::with_critters_and_pellets(
                TEST_WIDTH,
                TEST_HEIGHT,
                vec![eater, victim],
                vec![],
            );

            world.tick(true);

            assert_eq!(world.critters()[1].energy(), 80);
        }

        #[test]
        fn an_asymmetric_pair_outside_the_critter_eat_radius_is_not_drained() {
            // dx = 10, dy = 18: distance² = 100 + 324 = 424 ≥ 400, so the pair
            // is not within eat range. Asymmetry between the axes proves the
            // squared-distance computation must square each component
            // independently rather than summing or subtracting them.
            let eater = eating_critter(100, 100);
            let victim = idle_critter_with_energy(110, 118, 80);
            let mut world = World::with_critters_and_pellets(
                TEST_WIDTH,
                TEST_HEIGHT,
                vec![eater, victim],
                vec![],
            );

            world.tick(true);

            assert_eq!(world.critters()[1].energy(), 80);
        }

        #[test]
        fn draining_a_critter_works_across_the_toroidal_wrap() {
            let eater = eating_critter(2, 100);
            let victim = idle_critter_with_energy(TEST_WIDTH as i32 - 2, 100, 80);
            let mut world = World::with_critters_and_pellets(
                TEST_WIDTH,
                TEST_HEIGHT,
                vec![eater, victim],
                vec![],
            );

            world.tick(true);

            assert_eq!(world.critters()[1].energy(), 0);
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

    mod dominant_genome {
        use super::*;
        use crate::{Critter, Genome, Heading, Instruction};

        fn critter_with(x: i32, genome: Genome) -> Critter {
            Critter::with_genome(x, 0, Heading::East, 1, 1, 10, 0, genome)
        }

        #[test]
        fn it_returns_none_when_the_world_has_no_critters() {
            let world = World::with_critters_and_pellets(100, 100, Vec::new(), Vec::new());

            assert!(world.dominant_genome().is_none());
        }

        #[test]
        fn it_returns_the_genome_shared_by_the_majority() {
            let majority = Genome::all(Instruction::TurnLeft);
            let minority = Genome::all(Instruction::TurnRight);
            let critters = vec![
                critter_with(0, majority.clone()),
                critter_with(1, minority.clone()),
                critter_with(2, majority.clone()),
                critter_with(3, majority.clone()),
            ];
            let world = World::with_critters_and_pellets(100, 100, critters, Vec::new());

            assert_eq!(world.dominant_genome(), Some(&majority));
        }

        #[test]
        fn it_picks_the_genome_with_the_higher_count_even_when_seen_later() {
            // The minority appears first; the majority appears later. The
            // accessor must follow the count, not the encounter order.
            let later_majority = Genome::all(Instruction::TurnLeft);
            let early_minority = Genome::all(Instruction::TurnRight);
            let critters = vec![
                critter_with(0, early_minority.clone()),
                critter_with(1, later_majority.clone()),
                critter_with(2, later_majority.clone()),
                critter_with(3, later_majority.clone()),
            ];
            let world = World::with_critters_and_pellets(100, 100, critters, Vec::new());

            assert_eq!(world.dominant_genome(), Some(&later_majority));
        }

        #[test]
        fn it_counts_more_than_two_copies_correctly() {
            // Three of one genome beats two of another. Catches mutations
            // that turn the count increment into a no-op (all counts stay
            // at 1) or that mishandle the equality check (every critter
            // becomes its own entry with count 1).
            let three_copy = Genome::all(Instruction::TurnLeft);
            let two_copy = Genome::all(Instruction::TurnRight);
            let critters = vec![
                critter_with(0, two_copy.clone()),
                critter_with(1, three_copy.clone()),
                critter_with(2, two_copy.clone()),
                critter_with(3, three_copy.clone()),
                critter_with(4, three_copy.clone()),
            ];
            let world = World::with_critters_and_pellets(100, 100, critters, Vec::new());

            assert_eq!(world.dominant_genome(), Some(&three_copy));
        }

        #[test]
        fn ties_are_broken_by_first_seen_in_the_critter_list() {
            let first = Genome::all(Instruction::TurnLeft);
            let second = Genome::all(Instruction::TurnRight);
            let critters = vec![
                critter_with(0, first.clone()),
                critter_with(1, second.clone()),
                critter_with(2, first.clone()),
                critter_with(3, second.clone()),
            ];
            let world = World::with_critters_and_pellets(100, 100, critters, Vec::new());

            assert_eq!(world.dominant_genome(), Some(&first));
        }
    }

    mod with_seed_genome {
        use super::*;
        use crate::{Genome, Instruction};

        #[test]
        fn every_critter_in_the_world_starts_with_the_given_genome() {
            let seed = Genome::all(Instruction::Split);
            let mut rng = StdRng::seed_from_u64(0);

            let world = World::with_seed_genome(TEST_WIDTH, TEST_HEIGHT, seed.clone(), &mut rng);

            assert!(!world.critters().is_empty());
            for critter in world.critters() {
                assert_eq!(critter.genome(), &seed);
            }
        }

        #[test]
        fn it_creates_the_usual_number_of_pellets() {
            let seed = Genome::all(Instruction::Split);
            let mut rng = StdRng::seed_from_u64(0);

            let world = World::with_seed_genome(TEST_WIDTH, TEST_HEIGHT, seed, &mut rng);

            assert_eq!(world.pellets().len(), 1000);
        }

        #[test]
        fn every_critter_spawns_fully_inside_the_world_bounds() {
            // Same invariant the default-spawn path enforces: positions are
            // drawn from [CRITTER_RADIUS, width - CRITTER_RADIUS) so a
            // critter's circle never crosses the edge.
            let seed = Genome::all(Instruction::Split);
            let mut rng = StdRng::seed_from_u64(0);

            let world = World::with_seed_genome(TEST_WIDTH, TEST_HEIGHT, seed, &mut rng);

            for critter in world.critters() {
                assert!(critter.x() >= CRITTER_RADIUS);
                assert!(critter.x() < TEST_WIDTH as i32 - CRITTER_RADIUS);
                assert!(critter.y() >= CRITTER_RADIUS);
                assert!(critter.y() < TEST_HEIGHT as i32 - CRITTER_RADIUS);
            }
        }

        #[test]
        fn critters_spread_across_the_full_world_width_and_height() {
            // Mirrors the same invariant the default-spawn path is tested
            // against: across many spawns, at least one critter lands in the
            // right half and at least one in the bottom half. Pins the spawn
            // range to the full canvas rather than a clipped corner.
            let seed = Genome::all(Instruction::Split);
            let mut any_right = false;
            let mut any_bottom = false;
            let half_width = (TEST_WIDTH as i32) / 2;
            let half_height = (TEST_HEIGHT as i32) / 2;
            for rng_seed in 0..50 {
                let mut rng = StdRng::seed_from_u64(rng_seed);
                let world =
                    World::with_seed_genome(TEST_WIDTH, TEST_HEIGHT, seed.clone(), &mut rng);
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
}
