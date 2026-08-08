use crate::{
    Critter, Genome, Heading, Pellet, PELLETS_PER_POISON, PELLET_MAX_DRIFT, PELLET_MIN_DRIFT,
    PELLET_RADIUS,
};
use rand::Rng;

pub const CRITTER_RADIUS: i32 = 6;

// The population a world grows toward. It does not start there: spawning
// thousands of critters at once stalls the first seconds of a run, so a world
// begins at SEED_POPULATION and is topped up gradually while the frame rate
// holds.
const NUM_CRITTERS: usize = 4000;
// How many critters a world starts with. Above MIN_POPULATION so a freshly
// seeded world is not immediately judged too small and reset.
const SEED_POPULATION: usize = 100;
// How much food a world's energy budget is sized for. Not a ceiling on
// pellets: nothing caps the larder, which settles wherever consumption meets
// the feed rate.
const NUM_PELLETS: usize = 4000;
// How many pellets a single replenishment call may add. Small, because the
// call happens often: food seeps into the world a few at a time rather than
// a whole deficit's worth arriving at once.
pub const PELLET_BATCH_SIZE: usize = 10;
// How often a replenishment begins. At ~60 FPS this is sixty seconds: a
// world starts feeding on this cadence and keeps at it until its energy is
// restored, however long that takes.
const FEEDING_INTERVAL_FRAMES: u32 = 3600;
pub const MIN_POPULATION: usize = 20;
const INITIAL_ENERGY: u32 = 60;
const TICKS_PER_INSTRUCTION: u32 = 5;
const STEP_SIZE: i32 = 5;
// How many critters to refresh per overlap-detection call. The detector cycles
// through the population round-robin: critters not reached on a given call
// keep their previous flag until they come up again.
const OVERLAP_DETECTION_BUDGET: usize = 20;
// How many ticks a critter stays marked as overlapping after a confirmed
// detection. Smooths out the visual flicker caused by the round-robin budget.
const OVERLAP_INDICATOR_LINGER_TICKS: u32 = 30;
// How many ticks a victim stays marked as "being eaten" after it is killed.
// Brief — the kill is supposed to look like a flash.
const EATEN_INDICATOR_LINGER_TICKS: u32 = 10;

/// What the world is doing about food right now.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeedingState {
    /// Taking food on, until the world's energy is restored.
    Filling,
    /// Resting, with this many frames until feeding begins again.
    Waiting(u32),
}

pub struct World {
    width: usize,
    height: usize,
    critters: Vec<Critter>,
    pellets: Vec<Pellet>,
    original_total_energy: u32,
    generation: u32,
    overlap_detection_cursor: usize,
    /// How many pellets this world has emitted, so every PELLETS_PER_POISON
    /// of them can be poison.
    pellets_emitted: usize,
    /// Frames until the next replenishment, or zero while feeding.
    frames_until_feeding: u32,
    /// The genome new critters are seeded with, if this world was started
    /// from one. None means each newcomer gets a fresh random genome.
    seed_genome: Option<Genome>,
}

impl World {
    pub fn new<R: Rng>(width: usize, height: usize, rng: &mut R) -> Self {
        let critters: Vec<Critter> = (0..SEED_POPULATION)
            .map(|_| spawn_critter(width, height, rng))
            .collect();
        // Budget for what the world is growing toward, not what it starts
        // with: both the population and the larder fill up over time, and a
        // budget sized to the opening moment would never top either up.
        let original_total_energy = full_population_energy() + full_larder_energy();
        Self {
            width,
            height,
            critters,
            pellets: Vec::new(),
            original_total_energy,
            generation: 1,
            overlap_detection_cursor: 0,
            pellets_emitted: 0,
            frames_until_feeding: 0,
            seed_genome: None,
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
        let critters: Vec<Critter> = (0..SEED_POPULATION)
            .map(|_| spawn_critter_with_genome(width, height, seed_genome.clone(), rng))
            .collect();
        let original_total_energy = full_population_energy() + full_larder_energy();
        Self {
            width,
            height,
            critters,
            pellets: Vec::new(),
            original_total_energy,
            generation: 1,
            overlap_detection_cursor: 0,
            pellets_emitted: 0,
            frames_until_feeding: 0,
            seed_genome: Some(seed_genome),
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
            pellets_emitted: 0,
            frames_until_feeding: 0,
            seed_genome: None,
        }
    }

    pub fn original_total_energy(&self) -> u32 {
        self.original_total_energy
    }

    /// Where the world stands on food: taking it on, or resting with this
    /// many frames until it begins again.
    pub fn feeding_state(&self) -> FeedingState {
        if self.frames_until_feeding > 0 {
            FeedingState::Waiting(self.frames_until_feeding)
        } else {
            FeedingState::Filling
        }
    }

    /// Takes on food for this frame. A world feeds until its energy is
    /// restored, then waits FEEDING_INTERVAL_FRAMES before the next
    /// replenishment begins.
    pub fn feed<R: Rng>(&mut self, rng: &mut R) {
        if self.frames_until_feeding > 0 {
            self.frames_until_feeding -= 1;
            return;
        }
        self.replenish_pellets(PELLET_BATCH_SIZE, rng);
        if !self.needs_more_food() {
            self.frames_until_feeding = FEEDING_INTERVAL_FRAMES;
        }
    }

    /// Whether the world holds less energy than its budget allows. Counts
    /// energy inside critters as well as on the ground, so a well-fed
    /// population slows feeding by itself -- the budget is a thermostat on
    /// the whole world rather than a ceiling on pellets.
    pub fn needs_more_food(&self) -> bool {
        self.total_energy() < self.original_total_energy
    }

    /// Adds up to `count` pellets, stopping once the world's energy budget is
    /// met. What limits food is the budget, the length of a feeding run, and
    /// what the critters eat.
    pub fn replenish_pellets<R: Rng>(&mut self, count: usize, rng: &mut R) {
        for _ in 0..count {
            if !self.needs_more_food() {
                return;
            }
            self.pellets_emitted += 1;
            let poisonous = self.pellets_emitted.is_multiple_of(PELLETS_PER_POISON);
            self.pellets.push(spawn_pellet_of_kind(
                self.width,
                self.height,
                poisonous,
                rng,
            ));
        }
    }

    /// Kills any critter touching poison, and consumes the poison with it.
    /// Contact alone is fatal: a critter need not be trying to eat, and
    /// nothing in its sensorium warns it.
    fn resolve_poison(&mut self) {
        let touch_distance_squared =
            (CRITTER_RADIUS + PELLET_RADIUS) * (CRITTER_RADIUS + PELLET_RADIUS);
        let (width, height) = (self.width as i32, self.height as i32);
        let mut spent: Vec<usize> = Vec::new();

        for critter in &mut self.critters {
            if critter.energy() == 0 {
                continue;
            }
            let touched = self.pellets.iter().position(|pellet| {
                if !pellet.poisonous {
                    return false;
                }
                let dx = toroidal_delta(critter.x(), pellet.x.round() as i32, width);
                let dy = toroidal_delta(critter.y(), pellet.y.round() as i32, height);
                dx * dx + dy * dy < touch_distance_squared
            });
            if let Some(index) = touched {
                critter.die();
                if !spent.contains(&index) {
                    spent.push(index);
                }
            }
        }

        spent.sort_unstable_by(|a, b| b.cmp(a));
        for index in spent {
            self.pellets.swap_remove(index);
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
            critter.age_being_eaten_indicator();
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
        let (width, height) = (self.width as f32, self.height as f32);
        for pellet in &mut self.pellets {
            pellet.drift(width, height);
            pellet.age += 1;
        }
        self.pellets.retain(|pellet| !pellet.is_spoiled());
        self.resolve_poison();
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
                let dx = toroidal_delta(eater_x, pellet.x.round() as i32, width);
                let dy = toroidal_delta(eater_y, pellet.y.round() as i32, height);
                dx * dx + dy * dy < pellet_eat_distance_squared
            });
            if let Some(position) = pellet_position {
                self.pellets.swap_remove(position);
                self.critters[eater_index].gain_energy(crate::PELLET_ENERGY);
                continue;
            }

            // No pellet in range — kill the first overlapping critter, if
            // any. Eating a critter is lethal: the victim loses everything
            // even though the eater's gain saturates at MAX_CRITTER_ENERGY;
            // whatever doesn't fit is destroyed (replenishment later returns
            // it to the world as pellets).
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
                self.critters[eater_index].gain_energy(victim_energy);
                self.critters[victim_index].die();
                self.critters[victim_index].mark_being_eaten_for(EATEN_INDICATOR_LINGER_TICKS);
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

    /// Adds up to `count` more critters, stopping at the target population.
    /// The caller decides when the world can afford to grow. Critters carry
    /// the world's seed genome when it has one, so a `--genome` run stays a
    /// monoculture as it ramps rather than diluting with random newcomers.
    pub fn seed_more_critters<R: Rng>(&mut self, count: usize, rng: &mut R) {
        let room = NUM_CRITTERS.saturating_sub(self.critters.len());
        for _ in 0..count.min(room) {
            let critter = match &self.seed_genome {
                Some(genome) => {
                    spawn_critter_with_genome(self.width, self.height, genome.clone(), rng)
                }
                None => spawn_critter(self.width, self.height, rng),
            };
            self.critters.push(critter);
        }
    }

    /// Whether the world has finished ramping up to its target population.
    pub fn is_fully_seeded(&self) -> bool {
        self.critters.len() >= NUM_CRITTERS
    }

    pub fn population_too_low(&self) -> bool {
        self.critters.len() < MIN_POPULATION
    }

    pub fn generation(&self) -> u32 {
        self.generation
    }

    pub fn reset<R: Rng>(&mut self, rng: &mut R) {
        self.critters = (0..SEED_POPULATION)
            .map(|_| spawn_critter(self.width, self.height, rng))
            .collect();
        self.pellets.clear();
        self.frames_until_feeding = 0;
        self.generation += 1;
    }
}

#[cfg(test)]
fn critter_total_energy(critters: &[Critter]) -> u32 {
    critters.iter().map(|c| c.energy()).sum()
}

#[cfg(test)]
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

/// The energy a full target population would hold. Used to size a world's
/// budget up front, so replenishment aims at the population being grown
/// toward rather than whatever happens to be alive at construction.
/// The energy a world's food stocks hold when full. Like the population
/// budget, this is what the world is filling toward rather than what it
/// holds, so a world that starts empty still knows to feed itself.
fn full_larder_energy() -> u32 {
    NUM_PELLETS as u32 * crate::PELLET_ENERGY
}

fn full_population_energy() -> u32 {
    NUM_CRITTERS as u32 * INITIAL_ENERGY
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

/// Emits a pellet from the world's center on a random heading. Food streams
/// outward from one source rather than appearing evenly across the world, so
/// where a critter forages matters.
#[cfg(test)]
fn spawn_pellet<R: Rng>(width: usize, height: usize, rng: &mut R) -> Pellet {
    spawn_pellet_of_kind(width, height, false, rng)
}

fn spawn_pellet_of_kind<R: Rng>(
    width: usize,
    height: usize,
    poisonous: bool,
    rng: &mut R,
) -> Pellet {
    let angle = rng.gen_range(0.0..std::f32::consts::TAU);
    let speed = rng.gen_range(PELLET_MIN_DRIFT..=PELLET_MAX_DRIFT);
    Pellet {
        x: width as f32 / 2.0,
        y: height as f32 / 2.0,
        dx: angle.cos() * speed,
        dy: angle.sin() * speed,
        poisonous,
        age: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    // Feeds a world the way the main loop does. Bounded so a broken feed
    // fails the test rather than hanging it.
    fn feed_a_while<R: Rng>(world: &mut World, rng: &mut R) {
        for _ in 0..(NUM_PELLETS / PELLET_BATCH_SIZE + 2) {
            world.feed(rng);
        }
    }

    const TEST_WIDTH: usize = 200;
    const TEST_HEIGHT: usize = 200;

    mod new {
        use super::*;

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
        fn the_being_eaten_indicator_decays_to_off_over_enough_ticks() {
            // A lone critter cannot be eaten. Mark it for a few linger
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
            critter.mark_being_eaten_for(3);
            let mut world =
                World::with_critters_and_pellets(TEST_WIDTH, TEST_HEIGHT, vec![critter], vec![]);

            for _ in 0..3 {
                world.tick(true);
            }

            assert!(!world.critters()[0].is_being_eaten());
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
        fn a_critter_that_splits_appears_twice_once_its_division_finishes() {
            // Well fed: a division costs the attempt plus one energy per
            // tick of its duration, so a critter needs reserves to see one
            // through.
            let splitter = Critter::with_genome(
                100,
                100,
                Heading::North,
                1,
                1,
                crate::MAX_CRITTER_ENERGY,
                0,
                Genome::all(Instruction::Split),
            );
            let mut world =
                World::with_critters_and_pellets(TEST_WIDTH, TEST_HEIGHT, vec![splitter], vec![]);

            // Division takes time, so the child arrives some ticks later.
            for _ in 0..=crate::SPLIT_DURATION_TICKS {
                world.tick(true);
            }

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
        // Energy after a single Eat firing tick where no pellet was found —
        // just the base 1-energy tick cost is paid since Eat itself is free.
        const STARTING_AFTER_FAILED_EAT: u32 = STARTING_ENERGY - 1;

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
            let mut world = world_with(eating_critter(100, 100), Pellet::at(100, 100));

            world.tick(true);

            assert_eq!(world.pellets().len(), 0);
        }

        #[test]
        fn eating_a_pellet_increases_energy_by_the_pellet_energy_amount() {
            // Start, minus eat-attempt cost, minus the base tick cost, plus
            // the pellet's energy.
            let mut world = world_with(eating_critter(100, 100), Pellet::at(100, 100));

            world.tick(true);

            assert_eq!(
                world.critters()[0].energy(),
                STARTING_AFTER_FAILED_EAT + PELLET_ENERGY
            );
        }

        #[test]
        fn a_critter_that_does_not_execute_eat_leaves_an_overlapping_pellet_alone() {
            let mut world = world_with(idle_critter(100, 100), Pellet::at(100, 100));

            world.tick(true);

            assert_eq!(world.pellets().len(), 1);
            assert_eq!(world.critters()[0].energy(), STARTING_ENERGY - 1);
        }

        #[test]
        fn an_eating_critter_that_does_not_overlap_a_pellet_leaves_it_alone() {
            let mut world = world_with(eating_critter(50, 100), Pellet::at(100, 100));

            world.tick(true);

            assert_eq!(world.pellets().len(), 1);
            assert_eq!(world.critters()[0].energy(), STARTING_AFTER_FAILED_EAT);
        }

        #[test]
        fn a_pellet_just_inside_the_eating_distance_is_consumed() {
            let pellet = Pellet::at(100 + (CRITTER_RADIUS + PELLET_RADIUS - 1), 100);
            let mut world = world_with(eating_critter(100, 100), pellet);

            world.tick(true);

            assert_eq!(world.pellets().len(), 0);
        }

        #[test]
        fn a_pellet_outside_the_eating_distance_along_a_dominant_axis_is_not_consumed() {
            // dx=2, dy=15: dx² + dy² = 229 > (CRITTER_RADIUS+PELLET_RADIUS)² = 196.
            let mut world = world_with(eating_critter(100, 100), Pellet::at(102, 115));

            world.tick(true);

            assert_eq!(world.pellets().len(), 1);
        }

        #[test]
        fn a_pellet_at_exactly_the_eating_distance_is_not_consumed() {
            let pellet = Pellet::at(100 + CRITTER_RADIUS + PELLET_RADIUS, 100);
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
            let mut world = world_with(critter, Pellet::at(100, 100));

            world.tick(true);

            assert_eq!(
                world.critters()[0].energy(),
                HUNGRY_INITIAL - 1 + PELLET_ENERGY
            );
        }

        #[test]
        fn a_critter_near_the_left_edge_can_eat_a_pellet_near_the_right_edge_via_wrap() {
            let pellet = Pellet::at(TEST_WIDTH as i32 - 2, 100);
            let mut world = world_with(eating_critter(2, 100), pellet);

            world.tick(true);

            assert_eq!(world.pellets().len(), 0);
        }

        #[test]
        fn a_critter_near_the_top_edge_can_eat_a_pellet_near_the_bottom_edge_via_wrap() {
            let pellet = Pellet::at(100, TEST_HEIGHT as i32 - 2);
            let mut world = world_with(eating_critter(100, 2), pellet);

            world.tick(true);

            assert_eq!(world.pellets().len(), 0);
        }
    }

    mod pellets {
        use super::*;

        #[test]
        fn the_world_feeds_itself_from_empty() {
            let mut rng = StdRng::seed_from_u64(0);
            let mut world = World::new(TEST_WIDTH, TEST_HEIGHT, &mut rng);

            feed_a_while(&mut world, &mut rng);

            // Deliveries recur, so repeated runs do fill the larder.
            assert!(!world.pellets().is_empty());
        }

        #[test]
        fn every_pellet_stays_inside_the_world_as_it_drifts() {
            // Pellets are emitted at the center and wrap at the edges, so no
            // amount of drifting should carry one outside the world.
            for seed in 0..20 {
                let mut rng = StdRng::seed_from_u64(seed);
                let mut world = World::new(TEST_WIDTH, TEST_HEIGHT, &mut rng);
                for _ in 0..200 {
                    world.tick(true);
                }
                for pellet in world.pellets() {
                    assert!(pellet.x >= 0.0 && pellet.x < TEST_WIDTH as f32);
                    assert!(pellet.y >= 0.0 && pellet.y < TEST_HEIGHT as f32);
                }
            }
        }

        #[test]
        fn drifting_pellets_spread_away_from_the_emitter_in_every_direction() {
            // Given time to travel, the batch should reach all four quadrants
            // rather than streaming off in one direction.
            let mut rng = StdRng::seed_from_u64(0);
            let mut world = World::new(TEST_WIDTH, TEST_HEIGHT, &mut rng);
            feed_a_while(&mut world, &mut rng);
            let (cx, cy) = (TEST_WIDTH as f32 / 2.0, TEST_HEIGHT as f32 / 2.0);

            for _ in 0..100 {
                world.tick(true);
            }

            let quadrants: std::collections::HashSet<(bool, bool)> = world
                .pellets()
                .iter()
                .map(|pellet| (pellet.x > cx, pellet.y > cy))
                .collect();
            assert_eq!(quadrants.len(), 4);
        }

        #[test]
        fn reset_empties_the_larder_and_feeds_it_afresh() {
            let mut rng = StdRng::seed_from_u64(0);
            let mut world = World::new(TEST_WIDTH, TEST_HEIGHT, &mut rng);
            feed_a_while(&mut world, &mut rng);
            let original: Vec<_> = world.pellets().to_vec();

            world.reset(&mut rng);
            assert!(world.pellets().is_empty());

            feed_a_while(&mut world, &mut rng);
            assert!(!world.pellets().is_empty());
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
            let pellets = vec![Pellet::at(10, 10), Pellet::at(20, 20), Pellet::at(30, 30)];
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
            let pellet = Pellet::at(20, 20);
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
            let pellet = Pellet::at(30, 30);
            let mut world =
                World::with_critters_and_pellets(TEST_WIDTH, TEST_HEIGHT, vec![dead], vec![pellet]);

            world.reap_dead_critters();

            assert_eq!(world.pellets().len(), 1);
        }
    }

    mod original_total_energy {
        use super::*;

        #[test]
        fn it_exceeds_the_energy_present_in_a_freshly_seeded_world() {
            // The budget is sized for the target population, so a world that
            // has only just started seeding holds less than its budget — the
            // gap is the room it has to grow into.
            let mut rng = StdRng::seed_from_u64(0);

            let world = World::new(TEST_WIDTH, TEST_HEIGHT, &mut rng);

            assert!(world.original_total_energy() > world.total_energy());
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
        use crate::{Critter, Genome, Heading, Instruction, MAX_CRITTER_ENERGY, PELLET_ENERGY};

        // Empty of critters and food, but budgeted for a full world, so it
        // is hungry. The test-only constructor sizes its budget from actual
        // contents, which for an empty world would be zero.
        fn empty_world() -> World {
            let mut world =
                World::with_critters_and_pellets(TEST_WIDTH, TEST_HEIGHT, vec![], vec![]);
            world.original_total_energy = full_population_energy() + full_larder_energy();
            world
        }

        #[test]
        fn a_new_world_starts_empty_of_food_and_feeds_like_any_other() {
            // The opening larder is filled by the same metered spells as
            // every later one, not handed over whole at construction.
            let mut rng = StdRng::seed_from_u64(0);

            let world = World::new(TEST_WIDTH, TEST_HEIGHT, &mut rng);

            assert!(world.pellets().is_empty());
        }

        #[test]
        fn a_reset_world_starts_empty_of_food_too() {
            let mut rng = StdRng::seed_from_u64(0);
            let mut world = World::new(TEST_WIDTH, TEST_HEIGHT, &mut rng);
            for _ in 0..500 {
                world.feed(&mut rng);
            }

            world.reset(&mut rng);

            assert!(world.pellets().is_empty());
        }

        #[test]
        fn a_new_worlds_budget_covers_a_full_larder() {
            // Sized for the food the world is filling toward rather than what
            // it holds, or an empty world would consider itself stocked.
            let mut rng = StdRng::seed_from_u64(0);

            let world = World::new(TEST_WIDTH, TEST_HEIGHT, &mut rng);

            assert_eq!(
                world.original_total_energy(),
                full_population_energy() + full_larder_energy()
            );
        }

        #[test]
        fn a_world_with_no_pause_left_is_filling_rather_than_waiting() {
            // The boundary: zero frames of pause means feeding, not a wait
            // of zero length.
            let world = empty_world();

            assert_eq!(world.feeding_state(), FeedingState::Filling);
        }

        #[test]
        fn a_hungry_world_keeps_feeding_until_its_energy_is_restored() {
            // No timer bounds the run: it lasts exactly as long as the world
            // is short of energy.
            let mut world = empty_world();
            let target = 4 * PELLET_BATCH_SIZE;
            world.original_total_energy = target as u32 * PELLET_ENERGY;
            let mut rng = StdRng::seed_from_u64(0);

            // Exactly enough calls to fill it; the wait begins on the last.
            for _ in 0..4 {
                world.feed(&mut rng);
            }

            assert_eq!(world.pellets().len(), target);
            assert_eq!(
                world.feeding_state(),
                FeedingState::Waiting(FEEDING_INTERVAL_FRAMES)
            );
        }

        #[test]
        fn a_restored_world_pauses_before_feeding_again() {
            let mut world = empty_world();
            world.original_total_energy = 2 * PELLET_ENERGY;
            let mut rng = StdRng::seed_from_u64(0);
            for _ in 0..4 {
                world.feed(&mut rng);
            }
            let after_run = world.pellets().len();

            // Drain it: the pause holds even though the world is hungry again.
            world.pellets.clear();
            world.feed(&mut rng);

            assert_eq!(after_run, 2);
            assert!(world.pellets().is_empty());
            assert!(matches!(world.feeding_state(), FeedingState::Waiting(_)));
        }

        #[test]
        fn the_pause_gives_way_to_another_run() {
            let mut world = empty_world();
            world.original_total_energy = 2 * PELLET_ENERGY;
            let mut rng = StdRng::seed_from_u64(0);
            // One call fills it (batch of 10 covers the 2 it needs) and
            // starts the pause.
            world.feed(&mut rng);
            world.pellets.clear();

            // One call per paused frame runs the pause out; the next feeds.
            for _ in 0..FEEDING_INTERVAL_FRAMES {
                world.feed(&mut rng);
            }
            assert!(world.pellets().is_empty());

            world.feed(&mut rng);

            assert!(!world.pellets().is_empty());
        }

        #[test]
        fn a_world_below_its_energy_budget_takes_on_food() {
            let world = empty_world();

            assert!(world.needs_more_food());
        }

        #[test]
        fn a_world_at_its_energy_budget_takes_on_none() {
            let mut world = empty_world();
            world.original_total_energy = 0;

            assert!(!world.needs_more_food());
        }

        #[test]
        fn energy_inside_critters_counts_toward_the_budget() {
            // This is what makes the budget a thermostat rather than a
            // ceiling on pellets: a world whose critters are fat stops
            // feeding even with bare ground.
            let mut world = empty_world();
            world.original_total_energy = MAX_CRITTER_ENERGY;
            assert!(world.needs_more_food());

            world.critters.push(Critter::with_genome(
                10,
                10,
                Heading::North,
                u32::MAX,
                1,
                MAX_CRITTER_ENERGY,
                0,
                Genome::all(Instruction::DoNothing),
            ));

            assert!(world.pellets().is_empty());
            assert!(!world.needs_more_food());
        }

        #[test]
        fn replenishing_stops_at_the_energy_budget() {
            let mut world = empty_world();
            world.original_total_energy = 3 * PELLET_ENERGY;
            let mut rng = StdRng::seed_from_u64(0);

            world.replenish_pellets(100, &mut rng);

            assert_eq!(world.pellets().len(), 3);
        }

        #[test]
        fn replenishing_adds_only_a_small_batch_at_a_time() {
            // Food trickles in the way critters do, rather than a whole
            // deficit's worth landing in one frame.
            let mut world = empty_world();
            world.original_total_energy = 500 * PELLET_ENERGY;
            let mut rng = StdRng::seed_from_u64(0);

            world.replenish_pellets(PELLET_BATCH_SIZE, &mut rng);

            assert_eq!(world.pellets().len(), PELLET_BATCH_SIZE);
        }

        #[test]
        fn repeated_small_batches_accumulate_without_bound() {
            // Nothing caps the larder: what limits food in a running world is
            // the length of a feeding run and what the critters eat.
            let mut world = empty_world();
            let mut rng = StdRng::seed_from_u64(0);
            let batches = NUM_PELLETS / PELLET_BATCH_SIZE + 2;

            for _ in 0..batches {
                world.replenish_pellets(PELLET_BATCH_SIZE, &mut rng);
            }

            assert_eq!(world.pellets().len(), batches * PELLET_BATCH_SIZE);
        }

        // Fills a world the way the main loop does: many small batches.
        // Bounded rather than looping on a condition, so a broken
        // condition fails the test instead of hanging it.
        fn fill_to_target<R: Rng>(world: &mut World, rng: &mut R) {
            for _ in 0..(NUM_PELLETS / PELLET_BATCH_SIZE + 2) {
                world.replenish_pellets(PELLET_BATCH_SIZE, rng);
            }
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
        fn it_adds_enough_pellets_to_bring_total_energy_back_to_the_original() {
            let target = 10 * PELLET_ENERGY;
            let mut world = world_with_target(target);
            // Drain energy by removing pellets directly.
            world.pellets.clear();
            let mut rng = StdRng::seed_from_u64(1);

            fill_to_target(&mut world, &mut rng);

            assert!(world.total_energy() >= target);
        }

        #[test]
        fn replenished_pellets_emerge_from_the_emitter() {
            // Replenishment uses the same emitter as the initial batch, so new
            // food arrives from the center rather than appearing underfoot.
            let mut world = empty_world();
            world.original_total_energy = 50 * PELLET_ENERGY;
            for seed in 0..20 {
                let mut rng = StdRng::seed_from_u64(seed);
                world.pellets.clear();
                world.replenish_pellets(PELLET_BATCH_SIZE, &mut rng);
                for pellet in world.pellets() {
                    assert_eq!(pellet.x, TEST_WIDTH as f32 / 2.0);
                    assert_eq!(pellet.y, TEST_HEIGHT as f32 / 2.0);
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

            fill_to_target(&mut world, &mut rng);

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
            // Distance 50 between centers, well outside 2 * CRITTER_RADIUS.
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
            // Distance equals exactly 2 * CRITTER_RADIUS: tangent, not
            // overlapping. The strict `<` comparison must reject this pair.
            let a = idle_critter_at(100, 100);
            let b = idle_critter_at(100 + 2 * CRITTER_RADIUS, 100);
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
            // Centers one pixel closer than the 2 * CRITTER_RADIUS threshold,
            // so the pair counts as overlapping. Sitting just inside the
            // boundary kills threshold mutations whose squared cutoff drops
            // below this separation.
            let a = idle_critter_at(100, 100);
            let b = idle_critter_at(100 + 2 * CRITTER_RADIUS - 1, 100);
            let mut world =
                World::with_critters_and_pellets(TEST_WIDTH, TEST_HEIGHT, vec![a, b], vec![]);

            world.detect_critter_overlaps();

            assert!(world.critters()[0].is_overlapping_critter());
            assert!(world.critters()[1].is_overlapping_critter());
        }

        #[test]
        fn an_asymmetric_pair_outside_the_threshold_along_one_axis_is_not_marked() {
            // dx = R, dy = 2R - 1, so distance² = R² + (2R-1)², which clears
            // the (2R)² threshold by a hair for any radius. Asymmetry between
            // the axes ensures the squared-distance computation must square
            // each component separately rather than sum, subtract, or
            // otherwise collapse them into one value.
            let a = idle_critter_at(100, 100);
            let b = idle_critter_at(100 + CRITTER_RADIUS, 100 + 2 * CRITTER_RADIUS - 1);
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
        use crate::{Critter, Genome, Heading, Instruction, Pellet, MAX_CRITTER_ENERGY};

        const HUNGRY_INITIAL: u32 = 200;
        const STARTING_ENERGY: u32 = 10;
        // Energy after a single Eat firing tick where no transfer happened —
        // just the base 1-energy tick cost is paid since Eat itself is free.
        const STARTING_AFTER_FAILED_EAT: u32 = STARTING_ENERGY - 1;

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
        fn an_eater_that_caps_out_mid_meal_still_kills_its_victim() {
            let eater = eating_critter_with_energy(100, 100, MAX_CRITTER_ENERGY - 30);
            let victim = idle_critter_with_energy(105, 100, 100);
            let mut world = World::with_critters_and_pellets(
                TEST_WIDTH,
                TEST_HEIGHT,
                vec![eater, victim],
                vec![],
            );

            world.tick(true);

            // The surplus the eater couldn't absorb is destroyed, not left
            // with the victim.
            assert_eq!(world.critters()[0].energy(), MAX_CRITTER_ENERGY);
            assert_eq!(world.critters()[1].energy(), 0);
        }

        #[test]
        fn an_eaten_victim_is_marked_as_being_eaten() {
            let eater = eating_critter(100, 100);
            let victim = idle_critter_with_energy(105, 100, 80);
            let mut world = World::with_critters_and_pellets(
                TEST_WIDTH,
                TEST_HEIGHT,
                vec![eater, victim],
                vec![],
            );

            world.tick(true);

            assert!(world.critters()[1].is_being_eaten());
        }

        #[test]
        fn an_eater_prefers_a_pellet_over_a_touching_critter_when_both_are_in_range() {
            let eater = eating_critter(100, 100);
            let victim = idle_critter_with_energy(105, 100, 80);
            let pellet = Pellet::at(100, 100);
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
            // 50 px away — well outside 2 * CRITTER_RADIUS.
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
            // Centers one pixel closer than the 2 * CRITTER_RADIUS eat radius.
            // Mutations that shrink the threshold below this separation would
            // mistakenly classify the pair as out of range.
            let eater = eating_critter(100, 100);
            let victim = idle_critter_with_energy(100 + 2 * CRITTER_RADIUS - 1, 100, 80);
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
            // Distance equals exactly 2 * CRITTER_RADIUS — circles tangent,
            // not overlapping. The strict `<` comparison must reject this pair.
            let eater = eating_critter(100, 100);
            let victim = idle_critter_with_energy(100 + 2 * CRITTER_RADIUS, 100, 80);
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
            // dx = R, dy = 2R - 1, so distance² = R² + (2R-1)², just past the
            // (2R)² eat threshold for any radius. Asymmetry between the axes
            // proves the squared-distance computation must square each
            // component independently rather than summing or subtracting them.
            let eater = eating_critter(100, 100);
            let victim =
                idle_critter_with_energy(100 + CRITTER_RADIUS, 100 + 2 * CRITTER_RADIUS - 1, 80);
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

    mod poison {
        use super::*;
        use crate::{Critter, Genome, Heading, Instruction, PELLETS_PER_POISON};
        use rand::rngs::StdRng;
        use rand::SeedableRng;

        fn critter_at(x: i32, y: i32) -> Critter {
            Critter::with_genome(
                x,
                y,
                Heading::North,
                u32::MAX, // never fires, so any death is from contact alone
                1,
                60,
                0,
                Genome::all(Instruction::DoNothing),
            )
        }

        #[test]
        fn a_fixed_share_of_emitted_pellets_is_poison() {
            let mut world =
                World::with_critters_and_pellets(TEST_WIDTH, TEST_HEIGHT, vec![], vec![]);
            world.original_total_energy = full_population_energy() + full_larder_energy();
            let mut rng = StdRng::seed_from_u64(0);

            world.replenish_pellets(PELLETS_PER_POISON * 3, &mut rng);

            let poison = world.pellets().iter().filter(|p| p.poisonous).count();
            assert_eq!(poison, 3);
        }

        #[test]
        fn a_critter_touching_poison_dies() {
            // No eating involved: the critter never fires an instruction.
            let mut world = World::with_critters_and_pellets(
                TEST_WIDTH,
                TEST_HEIGHT,
                vec![critter_at(100, 100)],
                vec![Pellet::poison_at(100, 100)],
            );

            world.tick(true);

            assert_eq!(world.critters()[0].energy(), 0);
        }

        #[test]
        fn touching_poison_consumes_it() {
            let mut world = World::with_critters_and_pellets(
                TEST_WIDTH,
                TEST_HEIGHT,
                vec![critter_at(100, 100)],
                vec![Pellet::poison_at(100, 100)],
            );

            world.tick(true);

            assert!(world.pellets().is_empty());
        }

        #[test]
        fn poison_just_inside_the_touch_radius_kills() {
            // One pixel closer than the boundary. Pins where the radius
            // falls, which a poison pellet sitting on the critter does not.
            let touch = CRITTER_RADIUS + PELLET_RADIUS;
            let mut world = World::with_critters_and_pellets(
                TEST_WIDTH,
                TEST_HEIGHT,
                vec![critter_at(100, 100)],
                vec![Pellet::poison_at(100 + touch - 1, 100)],
            );

            world.tick(true);

            assert_eq!(world.critters()[0].energy(), 0);
        }

        #[test]
        fn poison_at_exactly_the_touch_radius_does_not_kill() {
            // Tangent, not overlapping: the comparison is strict.
            let touch = CRITTER_RADIUS + PELLET_RADIUS;
            let mut world = World::with_critters_and_pellets(
                TEST_WIDTH,
                TEST_HEIGHT,
                vec![critter_at(100, 100)],
                vec![Pellet::poison_at(100 + touch, 100)],
            );

            world.tick(true);

            assert!(world.critters()[0].energy() > 0);
        }

        #[test]
        fn poison_offset_on_both_axes_is_measured_by_true_distance() {
            // dx = dy = touch - 1 puts the poison well beyond the radius
            // diagonally even though each axis alone is inside it. Squaring
            // and summing both axes is what tells them apart.
            let touch = CRITTER_RADIUS + PELLET_RADIUS;
            let mut world = World::with_critters_and_pellets(
                TEST_WIDTH,
                TEST_HEIGHT,
                vec![critter_at(100, 100)],
                vec![Pellet::poison_at(100 + touch - 1, 100 + touch - 1)],
            );

            world.tick(true);

            assert!(world.critters()[0].energy() > 0);
        }

        #[test]
        fn poison_kills_across_the_toroidal_wrap() {
            // The critter sits against the left edge, the poison against the
            // right: they touch the short way round.
            let mut world = World::with_critters_and_pellets(
                TEST_WIDTH,
                TEST_HEIGHT,
                vec![critter_at(1, 100)],
                vec![Pellet::poison_at(TEST_WIDTH as i32 - 2, 100)],
            );

            world.tick(true);

            assert_eq!(world.critters()[0].energy(), 0);
        }

        #[test]
        fn a_critter_clear_of_poison_lives() {
            let mut world = World::with_critters_and_pellets(
                TEST_WIDTH,
                TEST_HEIGHT,
                vec![critter_at(100, 100)],
                vec![Pellet::poison_at(160, 100)],
            );

            world.tick(true);

            assert!(world.critters()[0].energy() > 0);
            assert_eq!(world.pellets().len(), 1);
        }

        #[test]
        fn ordinary_food_is_harmless_to_touch() {
            let mut world = World::with_critters_and_pellets(
                TEST_WIDTH,
                TEST_HEIGHT,
                vec![critter_at(100, 100)],
                vec![Pellet::at(100, 100)],
            );

            world.tick(true);

            assert!(world.critters()[0].energy() > 0);
            assert_eq!(world.pellets().len(), 1);
        }
    }

    mod emanating_pellets {
        use super::*;

        use rand::rngs::StdRng;
        use rand::SeedableRng;

        #[test]
        fn a_spawned_pellet_starts_at_the_center_of_the_world() {
            let mut rng = StdRng::seed_from_u64(0);

            let pellet = spawn_pellet(TEST_WIDTH, TEST_HEIGHT, &mut rng);

            assert_eq!(pellet.x.round() as i32, TEST_WIDTH as i32 / 2);
            assert_eq!(pellet.y.round() as i32, TEST_HEIGHT as i32 / 2);
        }

        #[test]
        fn spawned_pellets_head_off_in_a_range_of_directions() {
            // Every pellet leaves the emitter on its own heading, so a batch
            // spreads rather than travelling as one clump.
            let mut rng = StdRng::seed_from_u64(0);

            let headings: Vec<(i32, i32)> = (0..20)
                .map(|_| {
                    let pellet = spawn_pellet(TEST_WIDTH, TEST_HEIGHT, &mut rng);
                    // Scale up so distinct slow velocities do not round together.
                    ((pellet.dx * 1000.0) as i32, (pellet.dy * 1000.0) as i32)
                })
                .collect();

            let distinct: std::collections::HashSet<(i32, i32)> =
                headings.iter().copied().collect();
            assert!(
                distinct.len() > 10,
                "expected varied headings, got {} distinct out of 20",
                distinct.len()
            );
        }

        #[test]
        fn a_spawned_pellet_drifts_slowly() {
            // Slow enough that pellets linger near the emitter rather than
            // shooting to the edge in a moment.
            let mut rng = StdRng::seed_from_u64(0);

            for _ in 0..20 {
                let pellet = spawn_pellet(TEST_WIDTH, TEST_HEIGHT, &mut rng);

                let speed = (pellet.dx * pellet.dx + pellet.dy * pellet.dy).sqrt();
                assert!(
                    speed > 0.0 && speed <= PELLET_MAX_DRIFT,
                    "speed {speed} outside the expected drift range"
                );
            }
        }

        #[test]
        fn ticking_ages_the_pellets() {
            let mut rng = StdRng::seed_from_u64(0);
            let mut world = World::new(TEST_WIDTH, TEST_HEIGHT, &mut rng);
            world.feed(&mut rng);

            world.tick(true);

            assert_eq!(world.pellets()[0].age, 1);
        }

        #[test]
        fn a_spoiled_pellet_is_removed_from_the_world() {
            let mut rng = StdRng::seed_from_u64(0);
            let mut world = World::new(TEST_WIDTH, TEST_HEIGHT, &mut rng);
            world.feed(&mut rng);
            let before = world.pellets().len();
            for pellet in &mut world.pellets {
                pellet.age = crate::PELLET_LIFETIME_TICKS - 1;
            }

            world.tick(true);

            assert!(before > 0);
            assert!(world.pellets().is_empty());
        }

        #[test]
        fn spoilage_leaves_younger_pellets_alone() {
            let mut rng = StdRng::seed_from_u64(0);
            let mut world = World::new(TEST_WIDTH, TEST_HEIGHT, &mut rng);
            world.feed(&mut rng);
            let doomed = world.pellets().len() / 2;
            for pellet in world.pellets.iter_mut().take(doomed) {
                pellet.age = crate::PELLET_LIFETIME_TICKS - 1;
            }
            let total = world.pellets().len();

            world.tick(true);

            assert_eq!(world.pellets().len(), total - doomed);
        }

        #[test]
        fn ticking_moves_a_pellet_along_its_heading() {
            let mut rng = StdRng::seed_from_u64(0);
            let mut world = World::new(TEST_WIDTH, TEST_HEIGHT, &mut rng);
            world.feed(&mut rng);
            let before = world.pellets()[0];

            world.tick(true);

            let after = world.pellets()[0];
            assert_eq!(after.x, before.x + before.dx);
            assert_eq!(after.y, before.y + before.dy);
        }

        #[test]
        fn a_drifting_pellet_wraps_around_the_world_edge() {
            let mut rng = StdRng::seed_from_u64(0);
            let mut world = World::new(TEST_WIDTH, TEST_HEIGHT, &mut rng);
            world.feed(&mut rng);
            world.pellets[0] = Pellet {
                x: TEST_WIDTH as f32 - 0.5,
                y: 10.0,
                dx: 1.0,
                dy: 0.0,
                poisonous: false,
                age: 0,
            };

            world.tick(true);

            assert!(world.pellets()[0].x < 1.0);
        }
    }

    mod seeding {
        use super::*;
        use crate::Instruction;
        use rand::rngs::StdRng;
        use rand::SeedableRng;

        #[test]
        fn a_new_world_starts_with_a_seed_population_rather_than_the_full_target() {
            let mut rng = StdRng::seed_from_u64(0);

            let world = World::new(TEST_WIDTH, TEST_HEIGHT, &mut rng);

            assert_eq!(world.critters().len(), SEED_POPULATION);
        }

        #[test]
        fn a_new_worlds_energy_budget_covers_the_full_target_population() {
            // Replenishment tops the world back up to its original budget, so
            // that budget must reflect the population the world is growing
            // toward — not the handful it starts with, which would starve it.
            let mut rng = StdRng::seed_from_u64(0);

            let world = World::new(TEST_WIDTH, TEST_HEIGHT, &mut rng);

            let full_population_energy = NUM_CRITTERS as u32 * INITIAL_ENERGY;
            assert!(world.original_total_energy >= full_population_energy);
        }

        #[test]
        fn seeding_more_critters_adds_them_to_the_population() {
            let mut rng = StdRng::seed_from_u64(0);
            let mut world = World::new(TEST_WIDTH, TEST_HEIGHT, &mut rng);
            let before = world.critters().len();

            world.seed_more_critters(10, &mut rng);

            assert_eq!(world.critters().len(), before + 10);
        }

        #[test]
        fn seeding_stops_at_the_target_population() {
            let mut rng = StdRng::seed_from_u64(0);
            let mut world = World::new(TEST_WIDTH, TEST_HEIGHT, &mut rng);

            world.seed_more_critters(NUM_CRITTERS * 2, &mut rng);

            assert_eq!(world.critters().len(), NUM_CRITTERS);
        }

        #[test]
        fn a_seed_genome_world_seeds_newcomers_with_that_same_genome() {
            // A --genome run should stay a monoculture as it ramps, rather
            // than diluting itself with randomly generated newcomers.
            let mut rng = StdRng::seed_from_u64(0);
            let seed = Genome::all(Instruction::Split);
            let mut world =
                World::with_seed_genome(TEST_WIDTH, TEST_HEIGHT, seed.clone(), &mut rng);

            world.seed_more_critters(10, &mut rng);

            assert!(world
                .critters()
                .iter()
                .all(|critter| *critter.genome() == seed));
        }

        #[test]
        fn a_world_without_a_seed_genome_seeds_newcomers_randomly() {
            let mut rng = StdRng::seed_from_u64(0);
            let mut world = World::new(TEST_WIDTH, TEST_HEIGHT, &mut rng);

            world.seed_more_critters(10, &mut rng);

            let first = world.critters()[0].genome().clone();
            assert!(world
                .critters()
                .iter()
                .any(|critter| *critter.genome() != first));
        }

        #[test]
        fn a_world_at_its_target_population_is_done_seeding() {
            let mut rng = StdRng::seed_from_u64(0);
            let mut world = World::new(TEST_WIDTH, TEST_HEIGHT, &mut rng);
            assert!(!world.is_fully_seeded());

            world.seed_more_critters(NUM_CRITTERS, &mut rng);

            assert!(world.is_fully_seeded());
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
        fn it_feeds_itself_from_empty_like_any_other_world() {
            let seed = Genome::all(Instruction::Split);
            let mut rng = StdRng::seed_from_u64(0);
            let mut world = World::with_seed_genome(TEST_WIDTH, TEST_HEIGHT, seed, &mut rng);

            feed_a_while(&mut world, &mut rng);

            assert!(!world.pellets().is_empty());
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
