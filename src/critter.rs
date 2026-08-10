use crate::{Genome, Heading, Instruction, Senses};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use std::collections::VecDeque;

// The energy a critter is reckoned against rather than a ceiling it cannot
// pass. Nothing stops a critter banking more; this is the scale that senses
// and size are measured on, so a critter beyond it simply reads as full.
pub const MAX_CRITTER_ENERGY: u32 = 1000;
// A critter's area is proportional to its energy, so its radius goes as the
// square root: four times the energy makes a critter twice as wide, not four
// times. Size is therefore a reading of how well fed a critter is, visible at
// a glance without any readout.
//
// CRITTER_RADIUS is the size a critter is for free, whatever its energy --
// the size every critter used to be. Energy buys area on top of that, so the
// newly born and the nearly spent are still fully visible bodies rather than
// specks, and being well fed shows as growth beyond the ordinary rather than
// as the difference between existing and not.
pub const CRITTER_RADIUS: i32 = 6;
// The energy that doubles a critter's area, taking it from the free size to
// about one and a half times as wide. Larger values make energy count for
// less, flattening the range of sizes the world shows.
pub const REFERENCE_ENERGY: u32 = 250;
// Energy charged when a critter actually fires Split, regardless of whether
// the action succeeds (insufficient energy to halve, blocked by allow_split,
// etc.). Stops the "split, eat baby, repeat" exploit by making each
// reproduction attempt cost something the parent can't recover by eating
// the child.
pub const SPLIT_ATTEMPT_COST: u32 = 5;
// How long a division takes. A critter that fires Split is committed for this
// many ticks: it cannot act, so it cannot feed, but it still burns energy.
// Reproduction is therefore a gamble rather than a free action -- a critter
// that starves partway through dies with nothing to show for it.
pub const SPLIT_DURATION_TICKS: u32 = 1;
// How many slots a skip moves the playhead. Fixed for now: evolution
// controls whether a critter jumps, through the usual weights and sigmoid,
// but not yet how far.
pub const SKIP_DISTANCE: usize = 4;
// What a step costs at each pace. Moving fast covers twice the ground of
// moving slow, but the cost of covering it rises with the square of the
// pace, the way drag does in any real medium: twice the speed, four times
// the price. Ground is therefore bought at a worsening exchange rate, so
// hurrying is a choice with a real cost rather than a strictly better
// option.
// What a critter pays out of its own reserves each turn it acts, whatever it
// does with the turn. Running a large body costs more than running a small
// one, so the charge is the same share at any size and nobody outgrows it --
// with nothing capping what a critter can bank, a flat cost alone would be a
// rounding error to the rich. Charged for acting rather than for moving, so
// that standing still is not a way to avoid it.
pub const UPKEEP_PERCENT: u32 = 1;
pub const MOVE_SLOW_COST: u32 = 1;
pub const MOVE_FAST_COST: u32 = 4;
// How much further a fast step carries than a slow one.
const FAST_STEP_MULTIPLIER: i32 = 2;

/// How far each turn instruction turns a critter, positive being clockwise.
/// A reversal is the same turn whichever way it is taken.
fn turn_degrees(instruction: Instruction) -> f32 {
    match instruction {
        Instruction::TurnLeft15 => -15.0,
        Instruction::TurnLeft45 => -45.0,
        Instruction::TurnLeft90 => -90.0,
        Instruction::TurnRight15 => 15.0,
        Instruction::TurnRight45 => 45.0,
        Instruction::TurnRight90 => 90.0,
        _ => 180.0,
    }
}

/// What a critter's tick produced this turn. World inspects this after each
/// critter ticks to add any newborn child to the population and to know
/// whether the critter attempted to eat (which the world resolves later
/// because the critter doesn't see its neighbors or the pellets).
#[derive(Default)]
pub struct TickOutcome {
    pub child: Option<Critter>,
    pub attempted_eat: bool,
}

#[derive(Clone)]
pub struct Critter {
    x: i32,
    y: i32,
    heading: Heading,
    genome: Genome,
    genome_cursor: usize,
    last_executed: Option<Instruction>,
    ticks_per_instruction: u32,
    tick_counter: u32,
    next_fire_threshold: u32,
    step_size: i32,
    energy: u32,
    initial_energy: u32,
    overlap_indicator_ticks: u32,
    being_eaten_indicator_ticks: u32,
    most_recent_overlap_color: Option<u32>,
    // The instructions this critter most recently executed, newest last.
    // Runtime state rather than genome: a child starts with no memory of its
    // parent's actions. Trimmed to the genome's history window.
    recent_actions: VecDeque<Instruction>,
    /// Ticks left in an in-progress division, or zero when not dividing.
    dividing_ticks_remaining: u32,
    /// How many ticks this critter has lived. Its own, not inherited: a child
    /// begins at nothing however old its parent is.
    age: u32,
    rng: SmallRng,
}

impl Critter {
    pub fn new(
        x: i32,
        y: i32,
        heading: Heading,
        ticks_per_instruction: u32,
        step_size: i32,
        initial_energy: u32,
        seed: u64,
    ) -> Self {
        let mut rng = SmallRng::seed_from_u64(seed);
        let genome = Genome::random(&mut rng);
        Self::new_with(
            x,
            y,
            heading,
            genome,
            ticks_per_instruction,
            step_size,
            initial_energy,
            rng,
        )
    }

    /// Build a critter with a specific genome instead of a freshly randomized
    /// one. Used by tests and by the seed-genome world constructor.
    pub fn with_genome(
        x: i32,
        y: i32,
        heading: Heading,
        ticks_per_instruction: u32,
        step_size: i32,
        initial_energy: u32,
        seed: u64,
        genome: Genome,
    ) -> Self {
        let rng = SmallRng::seed_from_u64(seed);
        Self::new_with(
            x,
            y,
            heading,
            genome,
            ticks_per_instruction,
            step_size,
            initial_energy,
            rng,
        )
    }

    fn new_with(
        x: i32,
        y: i32,
        heading: Heading,
        genome: Genome,
        ticks_per_instruction: u32,
        step_size: i32,
        initial_energy: u32,
        rng: SmallRng,
    ) -> Self {
        Self {
            x,
            y,
            heading,
            genome,
            genome_cursor: 0,
            last_executed: None,
            ticks_per_instruction,
            tick_counter: 0,
            next_fire_threshold: ticks_per_instruction,
            step_size,
            energy: initial_energy,
            initial_energy,
            overlap_indicator_ticks: 0,
            being_eaten_indicator_ticks: 0,
            most_recent_overlap_color: None,
            recent_actions: VecDeque::new(),
            dividing_ticks_remaining: 0,
            age: 0,
            rng,
        }
    }

    pub fn x(&self) -> i32 {
        self.x
    }

    pub fn y(&self) -> i32 {
        self.y
    }

    pub fn heading(&self) -> Heading {
        self.heading
    }

    pub fn energy(&self) -> u32 {
        self.energy
    }

    /// How many ticks this critter has lived.
    pub fn age(&self) -> u32 {
        self.age
    }

    pub fn initial_energy(&self) -> u32 {
        self.initial_energy
    }

    pub fn genome(&self) -> &Genome {
        &self.genome
    }

    /// How wide this critter is, in pixels. Area is the free allowance plus
    /// the critter's energy, so the radius grows as the square root: four
    /// times the energy makes a critter twice as wide, not four times.
    pub fn radius(&self) -> i32 {
        let energy = self.energy as f32;
        let reference = REFERENCE_ENERGY as f32;
        let scaled = CRITTER_RADIUS as f32 * ((energy + reference) / reference).sqrt();
        scaled.round() as i32
    }

    /// A color derived from the genome bytes — identical for genomes with
    /// identical bytes, distinct (with very high probability) for any
    /// difference. Lets the renderer paint each lineage in its own color.
    pub fn genome_color(&self) -> u32 {
        self.genome.digest_color()
    }

    pub fn gain_energy(&mut self, amount: u32) {
        self.energy = self.energy.saturating_add(amount);
    }

    pub fn lose_energy(&mut self, amount: u32) {
        self.energy = self.energy.saturating_sub(amount);
    }

    /// Rolls the predator's own dice against a percentage risk, using its
    /// own rng so the outcome is the critter's rather than the world's.
    pub fn roll_predation_death(&mut self, percent: u32) -> bool {
        self.rng.gen_range(0..100) < percent
    }

    /// Kills the critter outright. A zero-energy critter can't act and the
    /// reaper removes it on its next pass.
    pub fn die(&mut self) {
        self.energy = 0;
    }

    /// Whether the critter is partway through a division, and so committed:
    /// it fires no instructions until the division finishes or it dies.
    pub fn is_dividing(&self) -> bool {
        self.dividing_ticks_remaining > 0
    }

    pub fn is_overlapping_critter(&self) -> bool {
        self.overlap_indicator_ticks > 0
    }

    pub fn mark_overlapping_critter_for(&mut self, ticks: u32) {
        self.overlap_indicator_ticks = ticks;
    }

    pub fn age_overlap_indicator(&mut self) {
        self.overlap_indicator_ticks = self.overlap_indicator_ticks.saturating_sub(1);
    }

    pub fn is_being_eaten(&self) -> bool {
        self.being_eaten_indicator_ticks > 0
    }

    pub fn mark_being_eaten_for(&mut self, ticks: u32) {
        self.being_eaten_indicator_ticks = ticks;
    }

    pub fn age_being_eaten_indicator(&mut self) {
        self.being_eaten_indicator_ticks = self.being_eaten_indicator_ticks.saturating_sub(1);
    }

    /// The genome color of the most recently detected overlapping critter,
    /// or None if no overlap has ever been seen. The world updates this each
    /// time it confirms an overlap. The genome can use it to compute a
    /// dissimilarity factor against `genome_color()`, letting evolution
    /// discover behaviors like "treat closely related critters differently
    /// from strangers."
    pub fn most_recent_overlap_color(&self) -> Option<u32> {
        self.most_recent_overlap_color
    }

    pub fn record_overlap_color(&mut self, color: u32) {
        self.most_recent_overlap_color = Some(color);
    }

    pub fn wrap_position(&mut self, width: i32, height: i32) {
        self.x = self.x.rem_euclid(width);
        self.y = self.y.rem_euclid(height);
    }

    pub fn tick(&mut self, allow_split: bool) -> TickOutcome {
        // Counted on every tick the critter exists, including the ones where
        // it does not fire, so age measures time lived rather than actions
        // taken.
        self.age = self.age.saturating_add(1);
        self.tick_counter += 1;
        if self.tick_counter < self.next_fire_threshold {
            return TickOutcome::default();
        }
        self.tick_counter = 0;
        self.next_fire_threshold = jitter_threshold(&mut self.rng, self.ticks_per_instruction);

        if self.energy == 0 {
            return TickOutcome::default();
        }

        if self.is_dividing() {
            return self.continue_dividing();
        }

        let instruction = self.genome.decode_at(self.genome_cursor);

        // Each instruction is gated by the genome's sigmoid: the probability of
        // acting is sigmoid((energy - threshold) / softness) for that
        // instruction's per-critter parameters. A "no" still consumes one
        // energy and `last_executed` is left untouched so RepeatPreviousMove
        // keeps referring to whatever did execute last.
        let senses = Senses {
            energy: self.energy,
            touching_critter: self.is_overlapping_critter(),
            // Black when nothing has been touched, so an untouched critter
            // senses no colour rather than some arbitrary one.
            touched_color: self.most_recent_overlap_color.unwrap_or(0),
            recent_repetition: self.recent_repetition_of(instruction),
            age: self.age,
        };
        let probability = self.genome.probability_of_acting(instruction, &senses);
        let split_blocked = instruction == Instruction::Split && !allow_split;
        let acted = !split_blocked && self.roll_against(probability);
        let outcome = if acted {
            self.remember_action(instruction);
            self.execute(instruction)
        } else {
            TickOutcome::default()
        };
        // A skip that fires moves the playhead instead of advancing it; every
        // other outcome, including a skip whose roll failed, walks on by one.
        // decode_at handles wrap-around, so the cursor can grow without an
        // explicit modulo.
        self.genome_cursor = match instruction {
            Instruction::SkipAhead if acted => self.genome_cursor + SKIP_DISTANCE,
            Instruction::SkipBack if acted => self.genome_cursor.saturating_sub(SKIP_DISTANCE),
            _ => self.genome_cursor + 1,
        };
        self.energy = self.energy.saturating_sub(self.upkeep());
        outcome
    }

    /// The share of the critter's remembered actions that were `instruction`,
    /// in [0, 1]. Zero when the genome's window is zero or nothing is
    /// remembered yet, so a critter with no memory is unaffected by history.
    fn recent_repetition_of(&self, instruction: Instruction) -> f32 {
        let window = self.genome.history_window();
        if window == 0 || self.recent_actions.is_empty() {
            return 0.0;
        }
        let matching = self
            .recent_actions
            .iter()
            .filter(|&&remembered| remembered == instruction)
            .count();
        matching as f32 / self.recent_actions.len() as f32
    }

    /// Records an executed instruction, dropping whatever has aged out of the
    /// genome's window. Only executed instructions are remembered: an
    /// instruction whose sigmoid roll failed never happened.
    fn remember_action(&mut self, instruction: Instruction) {
        let window = self.genome.history_window();
        if window == 0 {
            self.recent_actions.clear();
            return;
        }
        self.recent_actions.push_back(instruction);
        // Drop whatever aged out. A push adds at most one, so this removes at
        // most one — expressed as a bounded drain rather than a loop that
        // could spin if the comparison were wrong.
        let overflow = self.recent_actions.len().saturating_sub(window);
        self.recent_actions.drain(..overflow);
    }

    // The `<` vs `<=` mutation is an equivalent mutant for our continuous f32
    // rolls: `rng.gen::<f32>()` produces values in [0, 1), so the boundary case
    // `roll == probability` happens with probability 0 and is unobservable.
    #[mutants::skip]
    fn roll_against(&mut self, probability: f32) -> bool {
        let roll: f32 = self.rng.gen();
        roll < probability
    }

    /// What a turn costs a critter holding `energy`, simply for acting.
    /// Exposed so callers and tests can reckon in intent rather than
    /// arithmetic.
    //
    // The `>` is an equivalent mutant: integer division means the share is
    // zero below one hundred energy and UPKEEP_PERCENT at or above it, so it
    // is never exactly one and `>=` would pick the same branch every time.
    #[mutants::skip]
    pub const fn upkeep_for(energy: u32) -> u32 {
        let share = energy / 100 * UPKEEP_PERCENT;
        if share > 1 {
            share
        } else {
            1
        }
    }

    /// What this turn costs the critter simply for acting: a share of its own
    /// energy, never less than one so a spent critter still winds down.
    /// Divides before multiplying, since a percentage taken the other way
    /// round overflows for a critter holding near u32::MAX.
    fn upkeep(&self) -> u32 {
        Self::upkeep_for(self.energy)
    }

    /// The share of its own energy a critter spends to carry itself one step.
    //
    // The `*` is an equivalent mutant whenever UPKEEP_PERCENT is one, since
    // multiplying and dividing by one agree. It is a tuning knob rather than a
    // fixed part of the design, so the equivalence comes and goes with it.
    #[mutants::skip]
    fn hauling_for(energy: u32) -> u32 {
        energy / 100 * UPKEEP_PERCENT
    }

    /// Moves `multiplier` steps along the heading, charging `cost` plus a
    /// share of the critter's own energy for the distance covered. A critter
    /// that cannot cover the fare pays what it has and moves anyway.
    fn travel(&mut self, multiplier: i32, cost: u32, instruction: Instruction) -> TickOutcome {
        // Charged per unit of distance, so a longer stride costs proportionally
        // more of the body being hauled.
        let hauling = Self::hauling_for(self.energy) * multiplier as u32;
        // Charged against what the critter has rather than refused outright.
        // A critter that could not afford a step used to stay where it was,
        // which is a trap and not a cost: movement is the only way to reach
        // food, so anything too poor to move could never recover and simply
        // stood still until something ate it. Being poor should slow a
        // critter down, not strand it.
        self.energy = self.energy.saturating_sub(cost + hauling);
        // Rounded to whole pixels, so a critter on a shallow angle drifts
        // rather than tracking the line exactly. The unit vector is already
        // one long, so nothing has to correct a diagonal.
        let (dx, dy) = self.heading.unit();
        let step = (self.step_size * multiplier) as f32;
        self.x += (dx * step).round() as i32;
        self.y += (dy * step).round() as i32;
        self.last_executed = Some(instruction);
        TickOutcome::default()
    }

    fn execute(&mut self, instruction: Instruction) -> TickOutcome {
        match instruction {
            Instruction::MoveSlow => self.travel(1, MOVE_SLOW_COST, Instruction::MoveSlow),
            Instruction::MoveFast => {
                self.travel(FAST_STEP_MULTIPLIER, MOVE_FAST_COST, Instruction::MoveFast)
            }
            Instruction::TurnLeft15
            | Instruction::TurnLeft45
            | Instruction::TurnLeft90
            | Instruction::TurnRight15
            | Instruction::TurnRight45
            | Instruction::TurnRight90
            | Instruction::TurnAbout => {
                self.heading = self.heading.turned_right(turn_degrees(instruction));
                self.last_executed = Some(instruction);
                TickOutcome::default()
            }
            Instruction::DoNothing => {
                self.last_executed = Some(Instruction::DoNothing);
                TickOutcome::default()
            }
            Instruction::RepeatPreviousMove => {
                if let Some(previous) = self.last_executed {
                    self.execute(previous)
                } else {
                    TickOutcome::default()
                }
            }
            Instruction::Split => {
                self.energy = self.energy.saturating_sub(SPLIT_ATTEMPT_COST);
                self.dividing_ticks_remaining = SPLIT_DURATION_TICKS;
                self.last_executed = Some(Instruction::Split);
                TickOutcome::default()
            }
            Instruction::SkipAhead | Instruction::SkipBack => {
                // The playhead move happens in `tick`, which owns the cursor.
                self.last_executed = Some(instruction);
                TickOutcome::default()
            }
            Instruction::Eat => {
                // The critter can't see its surroundings, so it only signals
                // the intent. World::tick resolves which pellet or critter
                // (if any) the critter is touching and consumes it.
                self.last_executed = Some(Instruction::Eat);
                TickOutcome {
                    attempted_eat: true,
                    ..TickOutcome::default()
                }
            }
        }
    }

    /// Advances an in-progress division by one tick, yielding the child on
    /// the tick it completes. The energy cost of the tick is charged either
    /// way, so a critter can starve mid-division.
    fn continue_dividing(&mut self) -> TickOutcome {
        self.dividing_ticks_remaining -= 1;
        // Dividing is work, and the critter is committed to it, so it pays the
        // same upkeep as a turn spent doing anything else.
        self.energy = self.energy.saturating_sub(self.upkeep());
        if self.is_dividing() || self.energy == 0 {
            return TickOutcome::default();
        }
        let child = self.spawn_child();
        self.energy /= 2;
        TickOutcome {
            child: Some(child),
            ..TickOutcome::default()
        }
    }

    fn spawn_child(&mut self) -> Critter {
        let (dx, dy) = self.heading.unit();
        let offset = self.step_size as f32;
        let child_seed: u64 = self.rng.gen();
        let mut child_rng = SmallRng::seed_from_u64(child_seed);
        // Children's first firing is jittered, so they desynchronize from the
        // parent immediately.
        let child_threshold = jitter_threshold(&mut child_rng, self.ticks_per_instruction);
        let mut child_genome = self.genome.clone();
        // How readily a split mutates the child is itself encoded in the
        // parent's genome, so mutability evolves. Every split mutates: each
        // bit is considered independently, the way replication errors happen
        // per site rather than per offspring.
        child_genome.mutate(&mut child_rng, self.genome.mutation_rate());
        Critter {
            x: self.x - (dx * offset).round() as i32,
            y: self.y - (dy * offset).round() as i32,
            heading: self.heading,
            genome: child_genome,
            genome_cursor: self.genome_cursor,
            last_executed: None,
            ticks_per_instruction: self.ticks_per_instruction,
            tick_counter: 0,
            next_fire_threshold: child_threshold,
            step_size: self.step_size,
            energy: self.energy / 2,
            initial_energy: self.initial_energy,
            age: 0,
            overlap_indicator_ticks: 0,
            being_eaten_indicator_ticks: 0,
            most_recent_overlap_color: None,
            recent_actions: VecDeque::new(),
            dividing_ticks_remaining: 0,
            rng: child_rng,
        }
    }
}

fn jitter_threshold<R: Rng>(rng: &mut R, base: u32) -> u32 {
    if base <= 1 {
        return base;
    }
    rng.gen_range(1..=(2 * base - 1))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Instruction, EAST, NORTH, NORTH_WEST, SOUTH, SOUTH_EAST};

    // Fires Split and runs the division out, returning the child. Since
    // division takes time, a single tick no longer yields one.
    fn divide_fully(critter: &mut Critter) -> Critter {
        for _ in 0..=SPLIT_DURATION_TICKS {
            if let Some(child) = critter.tick(true).child {
                return child;
            }
        }
        panic!("division never finished");
    }

    const START_X: i32 = 10;
    const START_Y: i32 = 10;
    const TICKS_PER_INSTRUCTION: u32 = 30;

    mod new {
        use super::*;

        #[test]
        fn it_starts_at_the_given_position() {
            let critter = Critter::with_genome(
                START_X,
                START_Y,
                NORTH,
                1,
                1,
                u32::MAX,
                0,
                Genome::all(Instruction::DoNothing),
            );

            assert_eq!((critter.x(), critter.y()), (START_X, START_Y));
        }

        #[test]
        fn it_starts_with_the_given_heading() {
            let critter = Critter::with_genome(
                START_X,
                START_Y,
                NORTH,
                1,
                1,
                u32::MAX,
                0,
                Genome::all(Instruction::DoNothing),
            );

            assert_eq!(critter.heading(), NORTH);
        }
    }

    mod overlapping_critter_indicator {
        use super::*;

        fn fresh_critter() -> Critter {
            Critter::with_genome(
                START_X,
                START_Y,
                NORTH,
                1,
                1,
                u32::MAX,
                0,
                Genome::all(Instruction::DoNothing),
            )
        }

        #[test]
        fn a_freshly_created_critter_is_not_marked_as_overlapping_another_critter() {
            let critter = fresh_critter();

            assert!(!critter.is_overlapping_critter());
        }

        #[test]
        fn marking_the_critter_with_a_positive_tick_count_makes_it_report_as_overlapping() {
            let mut critter = fresh_critter();

            critter.mark_overlapping_critter_for(30);

            assert!(critter.is_overlapping_critter());
        }

        #[test]
        fn marking_with_zero_ticks_leaves_the_critter_not_overlapping() {
            let mut critter = fresh_critter();

            critter.mark_overlapping_critter_for(0);

            assert!(!critter.is_overlapping_critter());
        }

        #[test]
        fn aging_decrements_the_remaining_ticks_by_one() {
            let mut critter = fresh_critter();
            critter.mark_overlapping_critter_for(2);

            critter.age_overlap_indicator();

            assert!(critter.is_overlapping_critter());
        }

        #[test]
        fn aging_until_the_counter_runs_out_clears_the_overlapping_state() {
            let mut critter = fresh_critter();
            critter.mark_overlapping_critter_for(2);

            critter.age_overlap_indicator();
            critter.age_overlap_indicator();

            assert!(!critter.is_overlapping_critter());
        }

        #[test]
        fn aging_when_already_at_zero_does_not_underflow() {
            let mut critter = fresh_critter();

            critter.age_overlap_indicator();

            assert!(!critter.is_overlapping_critter());
        }
    }

    mod genome_color {
        use super::*;
        use rand::rngs::SmallRng;
        use rand::SeedableRng;

        #[test]
        fn it_matches_the_underlying_genomes_digest_color() {
            let mut rng = SmallRng::seed_from_u64(42);
            let genome = Genome::random(&mut rng);
            let critter =
                Critter::with_genome(START_X, START_Y, NORTH, 1, 1, 100, 0, genome.clone());

            assert_eq!(critter.genome_color(), genome.digest_color());
        }
    }

    mod most_recent_overlap_color {
        use super::*;

        fn fresh_critter() -> Critter {
            Critter::with_genome(
                START_X,
                START_Y,
                NORTH,
                1,
                1,
                100,
                0,
                Genome::all(Instruction::DoNothing),
            )
        }

        #[test]
        fn a_freshly_created_critter_has_no_recorded_overlap_color() {
            let critter = fresh_critter();

            assert!(critter.most_recent_overlap_color().is_none());
        }

        #[test]
        fn recording_an_overlap_color_makes_it_retrievable() {
            let mut critter = fresh_critter();

            critter.record_overlap_color(0xAB_CD_EF);

            assert_eq!(critter.most_recent_overlap_color(), Some(0xAB_CD_EF));
        }

        #[test]
        fn a_later_recording_overwrites_an_earlier_one() {
            let mut critter = fresh_critter();
            critter.record_overlap_color(0x11_22_33);

            critter.record_overlap_color(0xAA_BB_CC);

            assert_eq!(critter.most_recent_overlap_color(), Some(0xAA_BB_CC));
        }
    }

    mod being_eaten_indicator {
        use super::*;

        fn fresh_critter() -> Critter {
            Critter::with_genome(
                START_X,
                START_Y,
                NORTH,
                1,
                1,
                u32::MAX,
                0,
                Genome::all(Instruction::DoNothing),
            )
        }

        #[test]
        fn a_freshly_created_critter_is_not_marked_as_being_eaten() {
            let critter = fresh_critter();

            assert!(!critter.is_being_eaten());
        }

        #[test]
        fn marking_the_critter_for_a_positive_tick_count_makes_it_report_as_being_eaten() {
            let mut critter = fresh_critter();

            critter.mark_being_eaten_for(10);

            assert!(critter.is_being_eaten());
        }

        #[test]
        fn aging_until_the_counter_runs_out_clears_the_being_eaten_state() {
            let mut critter = fresh_critter();
            critter.mark_being_eaten_for(2);

            critter.age_being_eaten_indicator();
            critter.age_being_eaten_indicator();

            assert!(!critter.is_being_eaten());
        }

        #[test]
        fn aging_when_already_at_zero_does_not_underflow() {
            let mut critter = fresh_critter();

            critter.age_being_eaten_indicator();

            assert!(!critter.is_being_eaten());
        }
    }

    mod tick {
        use super::*;

        #[test]
        fn it_does_not_execute_an_instruction_before_n_ticks_have_passed() {
            let mut critter = Critter::with_genome(
                START_X,
                START_Y,
                NORTH,
                TICKS_PER_INSTRUCTION,
                1,
                u32::MAX,
                0,
                Genome::all(Instruction::MoveSlow),
            );

            for _ in 0..(TICKS_PER_INSTRUCTION - 1) {
                critter.tick(true);
            }

            assert_eq!(critter.y(), START_Y);
        }

        #[test]
        fn it_executes_an_instruction_on_the_nth_tick() {
            let mut critter = Critter::with_genome(
                START_X,
                START_Y,
                NORTH,
                TICKS_PER_INSTRUCTION,
                1,
                u32::MAX,
                0,
                Genome::all(Instruction::MoveSlow),
            );

            for _ in 0..TICKS_PER_INSTRUCTION {
                critter.tick(true);
            }

            assert_eq!(critter.y(), START_Y - 1);
        }

        #[test]
        fn move_forward_moves_one_pixel_in_the_heading_direction_with_unit_step_size() {
            let mut critter = Critter::with_genome(
                START_X,
                START_Y,
                EAST,
                1,
                1,
                u32::MAX,
                0,
                Genome::all(Instruction::MoveSlow),
            );

            critter.tick(true);

            assert_eq!((critter.x(), critter.y()), (START_X + 1, START_Y));
        }

        #[test]
        fn move_forward_advances_x_by_the_configured_step_size() {
            const STEP_SIZE: i32 = 25;
            let mut critter = Critter::with_genome(
                START_X,
                START_Y,
                EAST,
                1,
                STEP_SIZE,
                u32::MAX,
                0,
                Genome::all(Instruction::MoveSlow),
            );

            critter.tick(true);

            assert_eq!(critter.x(), START_X + STEP_SIZE);
        }

        #[test]
        fn move_forward_advances_y_by_the_configured_step_size() {
            const STEP_SIZE: i32 = 25;
            let mut critter = Critter::with_genome(
                START_X,
                START_Y,
                SOUTH,
                1,
                STEP_SIZE,
                u32::MAX,
                0,
                Genome::all(Instruction::MoveSlow),
            );

            critter.tick(true);

            assert_eq!(critter.y(), START_Y + STEP_SIZE);
        }

        #[test]
        fn move_forward_on_a_diagonal_scales_the_step_by_root_two_over_two() {
            // step_size 10, scaled by √2/2 ≈ 0.707 and rounded → 7.
            const STEP_SIZE: i32 = 10;
            const DIAGONAL_STEP: i32 = 7;
            let mut critter = Critter::with_genome(
                START_X,
                START_Y,
                SOUTH_EAST,
                1,
                STEP_SIZE,
                u32::MAX,
                0,
                Genome::all(Instruction::MoveSlow),
            );

            critter.tick(true);

            assert_eq!(
                (critter.x(), critter.y()),
                (START_X + DIAGONAL_STEP, START_Y + DIAGONAL_STEP)
            );
        }

        #[test]
        fn move_forward_on_a_northwest_diagonal_subtracts_the_scaled_step_from_each_axis() {
            const STEP_SIZE: i32 = 10;
            const DIAGONAL_STEP: i32 = 7;
            let mut critter = Critter::with_genome(
                START_X,
                START_Y,
                NORTH_WEST,
                1,
                STEP_SIZE,
                u32::MAX,
                0,
                Genome::all(Instruction::MoveSlow),
            );

            critter.tick(true);

            assert_eq!(
                (critter.x(), critter.y()),
                (START_X - DIAGONAL_STEP, START_Y - DIAGONAL_STEP)
            );
        }

        #[test]
        fn each_turn_instruction_turns_its_own_amount() {
            // A sharp turn is one instruction rather than six fine ones. Both
            // reach the same heading, but a manoeuvre held in one slot
            // survives mutation where one spread over six does not.
            for (instruction, degrees) in [
                (Instruction::TurnLeft15, -15.0),
                (Instruction::TurnLeft45, -45.0),
                (Instruction::TurnLeft90, -90.0),
                (Instruction::TurnRight15, 15.0),
                (Instruction::TurnRight45, 45.0),
                (Instruction::TurnRight90, 90.0),
                (Instruction::TurnAbout, 180.0),
            ] {
                let mut critter = Critter::with_genome(
                    START_X,
                    START_Y,
                    NORTH,
                    1,
                    1,
                    u32::MAX,
                    0,
                    Genome::all(instruction),
                );

                critter.tick(true);

                let expected = Heading::from_degrees(degrees);
                let gap = (critter.heading().radians() - expected.radians()).abs();
                let gap = gap.min(std::f32::consts::TAU - gap);
                assert!(
                    gap < 1e-4,
                    "{instruction:?} should have turned {degrees} degrees, faced {}",
                    critter.heading().radians().to_degrees()
                );
            }
        }

        #[test]
        fn several_fine_turns_reach_the_same_heading_as_one_sharp_one() {
            // What the sharp turns buy is slots and ticks, not headings that
            // were otherwise out of reach.
            let mut fine = Critter::with_genome(
                START_X,
                START_Y,
                NORTH,
                1,
                1,
                u32::MAX,
                0,
                Genome::all(Instruction::TurnRight15),
            );
            let mut sharp = Critter::with_genome(
                START_X,
                START_Y,
                NORTH,
                1,
                1,
                u32::MAX,
                0,
                Genome::all(Instruction::TurnRight45),
            );

            for _ in 0..3 {
                fine.tick(true);
            }
            sharp.tick(true);

            let gap = (fine.heading().radians() - sharp.heading().radians()).abs();
            assert!(gap < 1e-4, "three fifteens should equal one forty-five");
        }

        #[test]
        fn turn_left_does_not_change_the_position() {
            let mut critter = Critter::with_genome(
                START_X,
                START_Y,
                NORTH,
                1,
                1,
                u32::MAX,
                0,
                Genome::all(Instruction::TurnLeft15),
            );

            critter.tick(true);

            assert_eq!((critter.x(), critter.y()), (START_X, START_Y));
        }

        #[test]
        fn turn_right_does_not_change_the_position() {
            let mut critter = Critter::with_genome(
                START_X,
                START_Y,
                NORTH,
                1,
                1,
                u32::MAX,
                0,
                Genome::all(Instruction::TurnRight15),
            );

            critter.tick(true);

            assert_eq!((critter.x(), critter.y()), (START_X, START_Y));
        }

        #[test]
        fn do_nothing_leaves_position_unchanged() {
            let mut critter = Critter::with_genome(
                START_X,
                START_Y,
                NORTH,
                1,
                1,
                u32::MAX,
                0,
                Genome::all(Instruction::DoNothing),
            );

            critter.tick(true);

            assert_eq!((critter.x(), critter.y()), (START_X, START_Y));
        }

        #[test]
        fn do_nothing_leaves_heading_unchanged() {
            let mut critter = Critter::with_genome(
                START_X,
                START_Y,
                NORTH,
                1,
                1,
                u32::MAX,
                0,
                Genome::all(Instruction::DoNothing),
            );

            critter.tick(true);

            assert_eq!(critter.heading(), NORTH);
        }

        #[test]
        fn each_tick_consumes_the_next_instruction_in_the_list() {
            let mut critter = Critter::with_genome(
                START_X,
                START_Y,
                EAST,
                1,
                1,
                u32::MAX,
                0,
                Genome::from_instructions(&[Instruction::MoveSlow, Instruction::TurnRight15]),
            );

            critter.tick(true);
            critter.tick(true);

            assert_eq!(critter.x(), START_X + 1);
            assert_eq!(critter.heading(), EAST.turned_right(15.0));
        }

        #[test]
        fn the_instruction_list_loops_when_exhausted() {
            let mut critter = Critter::with_genome(
                START_X,
                START_Y,
                EAST,
                1,
                1,
                u32::MAX,
                0,
                Genome::all(Instruction::MoveSlow),
            );

            critter.tick(true);
            critter.tick(true);
            critter.tick(true);

            assert_eq!(critter.x(), START_X + 3);
        }
    }

    mod age {
        use super::*;

        fn a_critter() -> Critter {
            Critter::with_genome(
                START_X,
                START_Y,
                NORTH,
                1,
                1,
                u32::MAX,
                0,
                Genome::all(Instruction::DoNothing),
            )
        }

        #[test]
        fn a_new_critter_has_lived_no_ticks() {
            let critter = a_critter();

            assert_eq!(critter.age(), 0);
        }

        #[test]
        fn a_critter_ages_a_tick_at_a_time() {
            let mut critter = a_critter();

            critter.tick(true);
            critter.tick(true);
            critter.tick(true);

            assert_eq!(critter.age(), 3);
        }

        #[test]
        fn a_child_is_born_with_no_age_of_its_own() {
            // Age is the critter's own, not something inherited: a child
            // starts its life at nothing however old its parent is.
            let mut parent = Critter::with_genome(
                START_X,
                START_Y,
                NORTH,
                1,
                1,
                MAX_CRITTER_ENERGY,
                0,
                Genome::all(Instruction::Split),
            );

            let child = divide_fully(&mut parent);

            assert!(parent.age() > 0, "the parent should have lived a while");
            assert_eq!(child.age(), 0);
        }
    }

    mod upkeep {
        use super::*;

        fn critter_doing(instruction: Instruction, energy: u32) -> Critter {
            Critter::with_genome(
                START_X,
                START_Y,
                NORTH,
                1,
                5,
                energy,
                0,
                Genome::all(instruction),
            )
        }

        #[test]
        fn acting_costs_a_share_of_a_critters_energy_whatever_the_action() {
            // The charge is for being a large critter doing something, not for
            // any particular something: standing still is not a way out of it.
            for instruction in [
                Instruction::DoNothing,
                Instruction::TurnLeft15,
                Instruction::TurnRight15,
                Instruction::RepeatPreviousMove,
                Instruction::SkipAhead,
                Instruction::SkipBack,
                Instruction::Eat,
            ] {
                let start = 10_000;
                let mut critter = critter_doing(instruction, start);

                critter.tick(true);

                let spent = start - critter.energy();
                let expected = start / 100 * UPKEEP_PERCENT;
                assert!(
                    spent >= expected,
                    "{instruction:?} should cost at least {expected}, cost {spent}"
                );
            }
        }

        #[test]
        fn a_rich_critter_pays_more_to_act_than_a_poor_one() {
            let mut poor = critter_doing(Instruction::DoNothing, 200);
            let mut rich = critter_doing(Instruction::DoNothing, 20_000);

            poor.tick(true);
            rich.tick(true);

            let poor_spent = 200 - poor.energy();
            let rich_spent = 20_000 - rich.energy();
            assert!(
                rich_spent > poor_spent * 10,
                "the rich critter should pay far more: {rich_spent} vs {poor_spent}"
            );
        }

        #[test]
        fn upkeep_is_the_same_share_at_any_size() {
            let share = |energy: u32| {
                let mut critter = critter_doing(Instruction::DoNothing, energy);
                critter.tick(true);
                (energy - critter.energy()) as f32 / energy as f32
            };

            let modest = share(10_000);
            let vast = share(100_000);
            let expected = UPKEEP_PERCENT as f32 / 100.0;

            assert!(
                (modest - expected).abs() < 0.005 && (vast - expected).abs() < 0.005,
                "expected a {expected} share, got {modest} and {vast}"
            );
        }

        #[test]
        fn standing_still_is_cheaper_than_travelling() {
            // Upkeep applies to everything, but moving still costs more on top
            // of it, so the choice of what to do with a turn still matters.
            let start = 10_000;
            let mut idle = critter_doing(Instruction::DoNothing, start);
            let mut walker = critter_doing(Instruction::MoveSlow, start);

            idle.tick(true);
            walker.tick(true);

            assert!(walker.energy() < idle.energy());
        }

        #[test]
        fn upkeep_never_falls_below_one() {
            // A critter too poor for its share to round to anything still
            // winds down, so nothing can idle indefinitely on a scrap.
            for energy in 1..=20u32 {
                assert!(
                    Critter::upkeep_for(energy) >= 1,
                    "upkeep at {energy} should be at least one"
                );
            }
            // The floor holds right up to where the share overtakes it.
            assert_eq!(Critter::upkeep_for(99), 1);
            assert_eq!(Critter::upkeep_for(100), UPKEEP_PERCENT);
        }

        #[test]
        fn a_poor_critter_still_pays_something_to_act() {
            let mut critter = critter_doing(Instruction::DoNothing, 5);

            critter.tick(true);

            assert!(critter.energy() < 5);
        }
    }

    mod speeds {
        use super::*;

        const STEP: i32 = 5;
        const PLENTY: u32 = 10_000;

        fn critter_running(instruction: Instruction, energy: u32) -> Critter {
            Critter::with_genome(
                START_X,
                START_Y,
                NORTH,
                1,
                STEP,
                energy,
                0,
                Genome::all(instruction),
            )
        }

        #[test]
        fn moving_fast_covers_twice_the_ground_of_moving_slow() {
            let mut slow = critter_running(Instruction::MoveSlow, PLENTY);
            let mut fast = critter_running(Instruction::MoveFast, PLENTY);

            slow.tick(true);
            fast.tick(true);

            let slow_distance = START_Y - slow.y();
            let fast_distance = START_Y - fast.y();
            assert_eq!(fast_distance, slow_distance * 2);
        }

        #[test]
        fn moving_fast_costs_four_times_what_moving_slow_costs() {
            // Twice the speed for four times the price: the cost of covering
            // ground rises with the square of the pace, the way drag does.
            //
            // Probed with a critter too poor for the proportional charge to
            // round to anything, so what is measured is the flat cost alone.
            const TOO_POOR_TO_HAUL: u32 = 40;
            let mut slow = critter_running(Instruction::MoveSlow, TOO_POOR_TO_HAUL);
            let mut fast = critter_running(Instruction::MoveFast, TOO_POOR_TO_HAUL);

            slow.tick(true);
            fast.tick(true);

            let slow_spent = TOO_POOR_TO_HAUL - slow.energy();
            let fast_spent = TOO_POOR_TO_HAUL - fast.energy();
            // Both also pay the flat cost of living for the tick.
            let upkeep = Critter::upkeep_for(TOO_POOR_TO_HAUL);
            assert_eq!(fast_spent - upkeep, (slow_spent - upkeep) * 4);
        }

        #[test]
        fn hurrying_is_dearer_per_unit_of_ground_covered() {
            // The property that makes the choice a real one: speed buys
            // distance at a worsening exchange rate rather than a fixed one.
            //
            // Measured against the flat cost alone, on a critter too poor for
            // the proportional charges to round to anything. Those are levied
            // once a turn rather than once a step, so on a rich critter a
            // longer stride spreads the turn's upkeep over more ground and
            // hides the penalty that the flat cost expresses.
            const TOO_POOR_TO_HAUL: u32 = 40;
            let mut slow = critter_running(Instruction::MoveSlow, TOO_POOR_TO_HAUL);
            let mut fast = critter_running(Instruction::MoveFast, TOO_POOR_TO_HAUL);

            slow.tick(true);
            fast.tick(true);

            let upkeep = Critter::upkeep_for(TOO_POOR_TO_HAUL) as f32;
            let slow_rate = (TOO_POOR_TO_HAUL - slow.energy()) as f32 - upkeep;
            let fast_rate = (TOO_POOR_TO_HAUL - fast.energy()) as f32 - upkeep;
            let slow_per_pixel = slow_rate / (START_Y - slow.y()) as f32;
            let fast_per_pixel = fast_rate / (START_Y - fast.y()) as f32;
            assert!(
                fast_per_pixel > slow_per_pixel,
                "fast should cost more per pixel: {fast_per_pixel} vs {slow_per_pixel}"
            );
        }

        #[test]
        fn a_rich_critter_pays_more_to_move_than_a_poor_one() {
            // Hauling a large body costs more than hauling a small one, the
            // way an animal's cost of travel goes with its mass.
            let mut poor = critter_running(Instruction::MoveSlow, 200);
            let mut rich = critter_running(Instruction::MoveSlow, 20_000);

            poor.tick(true);
            rich.tick(true);

            let poor_spent = 200 - poor.energy();
            let rich_spent = 20_000 - rich.energy();
            assert!(
                rich_spent > poor_spent * 10,
                "the rich critter should pay far more: {rich_spent} vs {poor_spent}"
            );
        }

        #[test]
        fn moving_costs_the_same_share_of_energy_at_any_size() {
            // Proportional rather than flat: what changes with wealth is the
            // amount, not the share, so no reserve makes travel free.
            let share = |energy: u32| {
                let mut critter = critter_running(Instruction::MoveSlow, energy);
                critter.tick(true);
                (energy - critter.energy()) as f32 / energy as f32
            };

            let modest = share(10_000);
            let vast = share(100_000);
            // A walker pays the hauling charge and then the turn's upkeep on
            // what is left, so the two compound rather than simply adding.
            let rate = UPKEEP_PERCENT as f32 / 100.0;
            let expected = rate + (1.0 - rate) * rate;

            // Pinned against the expected share, not merely against each
            // other: two costs of zero would agree just as closely.
            assert!(
                (modest - expected).abs() < 0.005 && (vast - expected).abs() < 0.005,
                "expected a {expected} share, got {modest} and {vast}"
            );
        }

        #[test]
        fn a_poor_critter_still_pays_something_to_move() {
            // The flat cost survives underneath, so travel is never free even
            // for a critter with almost nothing.
            let mut critter = critter_running(Instruction::MoveSlow, 50);

            critter.tick(true);

            assert!(critter.energy() < 50);
        }

        #[test]
        fn covering_more_ground_costs_more_of_a_rich_critters_energy() {
            // The proportional charge is per unit of distance, so hurrying
            // costs a large critter more than walking does.
            let mut walker = critter_running(Instruction::MoveSlow, 20_000);
            let mut runner = critter_running(Instruction::MoveFast, 20_000);

            walker.tick(true);
            runner.tick(true);

            let walked = 20_000 - walker.energy();
            let ran = 20_000 - runner.energy();
            assert!(ran > walked, "running should cost more: {ran} vs {walked}");
        }

        #[test]
        fn a_critter_too_spent_to_pay_still_moves() {
            // Being poor slows a critter down; it does not strand it. Movement
            // is the only way to reach food, so a critter refused a step it
            // could not afford could never recover -- it would stand still
            // until something ate it.
            let mut critter = critter_running(Instruction::MoveFast, MOVE_FAST_COST);

            critter.tick(true);

            assert!(critter.y() < START_Y, "it should have moved anyway");
        }

        #[test]
        fn a_longer_stride_hauls_proportionally_more() {
            // The hauling charge is per unit of distance, so a fast step costs
            // twice the share a slow one does. Probed high enough that the
            // share rounds to something whatever the rate is set to.
            let rich = 100_000;
            let mut walker = critter_running(Instruction::MoveSlow, rich);
            let mut runner = critter_running(Instruction::MoveFast, rich);

            walker.tick(true);
            runner.tick(true);

            // The turn's upkeep is charged on what is left after hauling, so
            // the two do not separate cleanly. What the extra stride costs is
            // the difference between the paces, and that must be one whole
            // hauling charge.
            let walked = rich - walker.energy();
            let ran = rich - runner.energy();
            let extra_stride = ran - walked - (MOVE_FAST_COST - MOVE_SLOW_COST);
            let one_haul = rich / 100 * UPKEEP_PERCENT;
            let upkeep_relief = one_haul / 100 * UPKEEP_PERCENT;
            assert_eq!(extra_stride, one_haul - upkeep_relief);
        }

        #[test]
        fn a_critter_that_overpays_for_a_step_is_left_with_nothing() {
            let mut critter = critter_running(Instruction::MoveFast, MOVE_FAST_COST - 1);

            critter.tick(true);

            assert_eq!(critter.energy(), 0);
        }

        #[test]
        fn a_critter_too_spent_to_hurry_can_still_walk() {
            // Slow movement remains available to a critter that cannot afford
            // to run, so exhaustion changes the pace rather than stopping it.
            let mut critter = critter_running(Instruction::MoveSlow, MOVE_FAST_COST);

            critter.tick(true);

            assert!(critter.y() < START_Y);
        }
    }

    mod repeat_previous_move {
        use super::*;

        #[test]
        fn it_re_executes_the_previously_executed_instruction() {
            let mut critter = Critter::with_genome(
                START_X,
                START_Y,
                EAST,
                1,
                1,
                u32::MAX,
                0,
                Genome::from_instructions(&[
                    Instruction::MoveSlow,
                    Instruction::RepeatPreviousMove,
                ]),
            );

            critter.tick(true);
            critter.tick(true);

            assert_eq!(critter.x(), START_X + 2);
        }

        #[test]
        fn at_start_with_no_previous_move_it_does_not_change_position() {
            let mut critter = Critter::with_genome(
                START_X,
                START_Y,
                EAST,
                1,
                1,
                u32::MAX,
                0,
                Genome::all(Instruction::RepeatPreviousMove),
            );

            critter.tick(true);

            assert_eq!((critter.x(), critter.y()), (START_X, START_Y));
        }

        #[test]
        fn at_start_with_no_previous_move_it_does_not_change_heading() {
            let mut critter = Critter::with_genome(
                START_X,
                START_Y,
                EAST,
                1,
                1,
                u32::MAX,
                0,
                Genome::all(Instruction::RepeatPreviousMove),
            );

            critter.tick(true);

            assert_eq!(critter.heading(), EAST);
        }

        #[test]
        fn repeating_a_repeat_re_executes_the_underlying_move_a_third_time() {
            // Three ticks of [TurnRight15, Repeat, Repeat] turn fifteen
            // degrees apiece. The third tick must reach back through two
            // repeats to find the turn.
            let mut critter = Critter::with_genome(
                START_X,
                START_Y,
                EAST,
                1,
                1,
                u32::MAX,
                0,
                Genome::from_instructions(&[
                    Instruction::TurnRight15,
                    Instruction::RepeatPreviousMove,
                    Instruction::RepeatPreviousMove,
                ]),
            );

            critter.tick(true);
            critter.tick(true);
            critter.tick(true);

            // Three turns of fifteen degrees from where it started.
            assert_eq!(critter.heading(), EAST.turned_right(45.0));
        }
    }

    mod energy {
        use super::*;

        const INITIAL_ENERGY: u32 = 42;

        #[test]
        fn a_new_critter_starts_with_the_given_initial_energy() {
            let critter = Critter::with_genome(
                START_X,
                START_Y,
                NORTH,
                1,
                1,
                INITIAL_ENERGY,
                0,
                Genome::all(Instruction::DoNothing),
            );

            assert_eq!(critter.energy(), INITIAL_ENERGY);
        }

        #[test]
        fn initial_energy_reports_the_value_passed_at_construction() {
            let critter = Critter::with_genome(
                START_X,
                START_Y,
                NORTH,
                1,
                1,
                INITIAL_ENERGY,
                0,
                Genome::all(Instruction::DoNothing),
            );

            assert_eq!(critter.initial_energy(), INITIAL_ENERGY);
        }

        #[test]
        fn initial_energy_does_not_decrease_when_energy_is_consumed() {
            let mut critter = Critter::with_genome(
                START_X,
                START_Y,
                NORTH,
                1,
                1,
                INITIAL_ENERGY,
                0,
                Genome::all(Instruction::DoNothing),
            );

            critter.tick(true);

            assert_eq!(critter.initial_energy(), INITIAL_ENERGY);
        }

        #[test]
        fn executing_an_instruction_decrements_energy_by_one() {
            let mut critter = Critter::with_genome(
                START_X,
                START_Y,
                NORTH,
                1,
                1,
                INITIAL_ENERGY,
                0,
                Genome::all(Instruction::DoNothing),
            );

            critter.tick(true);

            assert_eq!(
                critter.energy(),
                INITIAL_ENERGY - Critter::upkeep_for(INITIAL_ENERGY)
            );
        }

        #[test]
        fn ticks_before_the_threshold_do_not_decrement_energy() {
            let mut critter = Critter::with_genome(
                START_X,
                START_Y,
                NORTH,
                TICKS_PER_INSTRUCTION,
                1,
                INITIAL_ENERGY,
                0,
                Genome::all(Instruction::DoNothing),
            );

            for _ in 0..(TICKS_PER_INSTRUCTION - 1) {
                critter.tick(true);
            }

            assert_eq!(critter.energy(), INITIAL_ENERGY);
        }

        #[test]
        fn at_zero_energy_move_forward_does_not_move_the_critter() {
            let mut critter = Critter::with_genome(
                START_X,
                START_Y,
                EAST,
                1,
                1,
                0,
                0,
                Genome::all(Instruction::MoveSlow),
            );

            critter.tick(true);

            assert_eq!((critter.x(), critter.y()), (START_X, START_Y));
        }

        #[test]
        fn ticking_at_zero_energy_keeps_energy_at_zero() {
            let mut critter = Critter::with_genome(
                START_X,
                START_Y,
                EAST,
                1,
                1,
                0,
                0,
                Genome::all(Instruction::MoveSlow),
            );

            for _ in 0..10 {
                critter.tick(true);
            }

            assert_eq!(critter.energy(), 0);
        }
    }

    mod radius {
        use super::*;

        fn critter_with_energy(energy: u32) -> Critter {
            Critter::with_genome(
                0,
                0,
                NORTH,
                1,
                1,
                energy,
                0,
                Genome::all(Instruction::DoNothing),
            )
        }

        #[test]
        fn a_critter_with_nothing_is_still_the_free_size() {
            // Size is not earned from zero: every critter gets CRITTER_RADIUS
            // for nothing, and energy buys growth beyond it.
            let critter = critter_with_energy(0);

            assert_eq!(critter.radius(), CRITTER_RADIUS);
        }

        #[test]
        fn energy_makes_a_critter_larger_than_the_free_size() {
            let free = critter_with_energy(0);
            let fed = critter_with_energy(REFERENCE_ENERGY);

            assert!(fed.radius() > free.radius());
        }

        #[test]
        fn a_critters_area_tracks_its_energy_across_the_range() {
            // The property being encoded, checked at every energy rather than
            // at one convenient pair: area over energy is a constant. Radii
            // are whole pixels, so each one is within rounding of the ideal
            // rather than exactly on it.
            for energy in (0..=MAX_CRITTER_ENERGY).step_by(37) {
                let reference = REFERENCE_ENERGY as f32;
                let ideal =
                    CRITTER_RADIUS as f32 * ((energy as f32 + reference) / reference).sqrt();
                let actual = critter_with_energy(energy).radius() as f32;

                assert!(
                    (actual - ideal).abs() <= 0.5,
                    "at energy {energy}: radius {actual}, ideal {ideal}"
                );
            }
        }

        #[test]
        fn a_critters_area_is_its_energy_plus_the_free_allowance() {
            // The law itself: area over (energy + allowance) is constant, so
            // the same energy adds the same area wherever a critter starts.
            let probe = |energy: u32| {
                let r = critter_with_energy(energy).radius() as f32;
                r * r / (energy + REFERENCE_ENERGY) as f32
            };

            let lean = probe(0);
            let middling = probe(REFERENCE_ENERGY * 2);
            let full = probe(MAX_CRITTER_ENERGY);

            assert!(
                (middling - lean).abs() < 0.02 && (full - lean).abs() < 0.02,
                "area per unit should be constant, got {lean} {middling} {full}"
            );
        }

        #[test]
        fn a_critter_keeps_growing_past_the_reference_energy() {
            let full = critter_with_energy(MAX_CRITTER_ENERGY);
            let overfull = critter_with_energy(MAX_CRITTER_ENERGY * 4);

            assert!(overfull.radius() > full.radius());
        }
    }

    mod wrap_position {
        use super::*;

        const WIDTH: i32 = 100;
        const HEIGHT: i32 = 100;

        fn make_critter_at(x: i32, y: i32) -> Critter {
            Critter::with_genome(
                x,
                y,
                NORTH,
                1,
                1,
                u32::MAX,
                0,
                Genome::all(Instruction::DoNothing),
            )
        }

        #[test]
        fn a_position_already_in_bounds_is_unchanged() {
            let mut critter = make_critter_at(50, 50);

            critter.wrap_position(WIDTH, HEIGHT);

            assert_eq!((critter.x(), critter.y()), (50, 50));
        }

        #[test]
        fn x_past_the_right_edge_wraps_to_the_left() {
            let mut critter = make_critter_at(WIDTH + 5, 50);

            critter.wrap_position(WIDTH, HEIGHT);

            assert_eq!(critter.x(), 5);
        }

        #[test]
        fn y_past_the_bottom_edge_wraps_to_the_top() {
            let mut critter = make_critter_at(50, HEIGHT + 5);

            critter.wrap_position(WIDTH, HEIGHT);

            assert_eq!(critter.y(), 5);
        }

        #[test]
        fn negative_x_wraps_to_the_right_side() {
            let mut critter = make_critter_at(-1, 50);

            critter.wrap_position(WIDTH, HEIGHT);

            assert_eq!(critter.x(), WIDTH - 1);
        }

        #[test]
        fn negative_y_wraps_to_the_bottom_side() {
            let mut critter = make_critter_at(50, -1);

            critter.wrap_position(WIDTH, HEIGHT);

            assert_eq!(critter.y(), HEIGHT - 1);
        }

        #[test]
        fn x_far_past_the_right_edge_wraps_modulo_width() {
            let mut critter = make_critter_at(WIDTH * 3 + 7, 50);

            critter.wrap_position(WIDTH, HEIGHT);

            assert_eq!(critter.x(), 7);
        }
    }

    mod skipping {
        use super::*;
        use crate::Genome;

        // A critter whose stream is the given instructions, firing every tick
        // and always acting.
        fn player(instructions: &[Instruction]) -> Critter {
            Critter::with_genome(
                START_X,
                START_Y,
                NORTH,
                1,
                1,
                MAX_CRITTER_ENERGY,
                0,
                Genome::from_instructions(instructions),
            )
        }

        #[test]
        fn skipping_ahead_moves_the_playhead_past_slots() {
            // The instruction after SkipAhead is not the one that runs next:
            // the playhead lands SKIP_DISTANCE further on.
            let mut critter = player(&[Instruction::SkipAhead]);
            let before = critter.genome_cursor;

            critter.tick(true);

            assert_eq!(critter.genome_cursor, before + SKIP_DISTANCE);
        }

        #[test]
        fn skipping_back_moves_the_playhead_toward_earlier_slots() {
            let mut critter = player(&[Instruction::SkipBack]);
            // Walk the cursor forward so there is somewhere to go back to.
            critter.genome_cursor = SKIP_DISTANCE * 2;
            let before = critter.genome_cursor;

            critter.tick(true);

            assert_eq!(critter.genome_cursor, before - SKIP_DISTANCE);
        }

        #[test]
        fn skipping_back_from_the_start_does_not_underflow() {
            let mut critter = player(&[Instruction::SkipBack]);
            critter.genome_cursor = 0;

            critter.tick(true);

            assert_eq!(critter.genome_cursor, 0);
        }

        #[test]
        fn a_skip_whose_roll_fails_walks_on_by_one() {
            // The jump is gated like any other instruction: inputs decide
            // whether the playhead moves, which is the point of the design.
            // A never-act genome leaves every roll failing.
            let mut genome = Genome::from_instructions(&[Instruction::SkipAhead]);
            genome.set_never_act_header();
            let mut critter =
                Critter::with_genome(START_X, START_Y, NORTH, 1, 1, MAX_CRITTER_ENERGY, 0, genome);
            let before = critter.genome_cursor;

            critter.tick(true);

            assert_eq!(critter.genome_cursor, before + 1);
        }

        #[test]
        fn a_skip_back_whose_roll_fails_also_walks_on_by_one() {
            let mut genome = Genome::from_instructions(&[Instruction::SkipBack]);
            genome.set_never_act_header();
            let mut critter =
                Critter::with_genome(START_X, START_Y, NORTH, 1, 1, MAX_CRITTER_ENERGY, 0, genome);
            critter.genome_cursor = SKIP_DISTANCE * 2;
            let before = critter.genome_cursor;

            critter.tick(true);

            assert_eq!(critter.genome_cursor, before + 1);
        }

        #[test]
        fn an_ordinary_instruction_advances_the_playhead_by_one() {
            let mut critter = player(&[Instruction::MoveSlow]);
            let before = critter.genome_cursor;

            critter.tick(true);

            assert_eq!(critter.genome_cursor, before + 1);
        }

        #[test]
        fn a_skip_changes_which_instruction_runs_next() {
            // The point of the whole thing: a jump lands the playhead on a
            // different instruction than the walk would have reached.
            let mut skipper = player(&[
                Instruction::SkipAhead,
                Instruction::TurnLeft15,
                Instruction::TurnRight15,
            ]);
            let mut walker = player(&[
                Instruction::DoNothing,
                Instruction::TurnLeft15,
                Instruction::TurnRight15,
            ]);

            skipper.tick(true);
            walker.tick(true);

            assert_ne!(skipper.genome_cursor, walker.genome_cursor);
        }
    }

    mod predation_risk {
        use super::*;
        use crate::Genome;

        fn roller(seed: u64) -> Critter {
            Critter::with_genome(
                START_X,
                START_Y,
                NORTH,
                1,
                1,
                60,
                seed,
                Genome::all(Instruction::DoNothing),
            )
        }

        #[test]
        fn a_zero_percent_risk_never_kills() {
            // The boundary: a roll of 0 must fall outside a 0% risk, so the
            // comparison is strict.
            let mut any = false;
            for seed in 0..200 {
                if roller(seed).roll_predation_death(0) {
                    any = true;
                }
            }

            assert!(!any);
        }

        #[test]
        fn a_hundred_percent_risk_always_kills() {
            let mut all = true;
            for seed in 0..200 {
                if !roller(seed).roll_predation_death(100) {
                    all = false;
                }
            }

            assert!(all);
        }
    }

    mod action_history {
        use super::*;
        use crate::Genome;

        fn critter_with_window(window_bits: u32, instruction: Instruction) -> Critter {
            let mut genome = Genome::all(instruction);
            genome.set_history_window_bits(window_bits);
            Critter::with_genome(START_X, START_Y, NORTH, 1, 1, MAX_CRITTER_ENERGY, 0, genome)
        }

        #[test]
        fn a_critter_remembers_no_more_actions_than_its_window_allows() {
            // Three set bits mean a window of three, so a critter that acts
            // ten times still remembers only its three most recent actions.
            let mut critter = critter_with_window(0b111, Instruction::MoveSlow);

            for _ in 0..10 {
                critter.tick(true);
            }

            assert_eq!(critter.recent_actions.len(), 3);
        }

        #[test]
        fn a_critter_whose_window_is_zero_remembers_nothing() {
            let mut critter = critter_with_window(0, Instruction::MoveSlow);

            for _ in 0..10 {
                critter.tick(true);
            }

            assert!(critter.recent_actions.is_empty());
        }

        #[test]
        fn repetition_is_the_share_of_remembered_actions_that_match() {
            // Two of the four remembered actions are MoveSlow, so the
            // share is one half — not a count, and not a share of the window.
            let mut critter = critter_with_window(0b1111, Instruction::MoveSlow);
            critter.recent_actions.clear();
            critter.recent_actions.push_back(Instruction::MoveSlow);
            critter.recent_actions.push_back(Instruction::TurnLeft15);
            critter.recent_actions.push_back(Instruction::MoveSlow);
            critter.recent_actions.push_back(Instruction::Eat);

            assert_eq!(critter.recent_repetition_of(Instruction::MoveSlow), 0.5);
            assert_eq!(critter.recent_repetition_of(Instruction::TurnLeft15), 0.25);
            assert_eq!(critter.recent_repetition_of(Instruction::Split), 0.0);
        }

        #[test]
        fn repetition_is_zero_when_nothing_is_remembered_yet() {
            let critter = critter_with_window(0b1111, Instruction::MoveSlow);

            assert_eq!(critter.recent_repetition_of(Instruction::MoveSlow), 0.0);
        }

        #[test]
        fn repetition_is_zero_when_the_window_is_closed() {
            // Even with actions in the buffer, a zero window means history is
            // not consulted at all.
            let mut critter = critter_with_window(0, Instruction::MoveSlow);
            critter.recent_actions.push_back(Instruction::MoveSlow);

            assert_eq!(critter.recent_repetition_of(Instruction::MoveSlow), 0.0);
        }

        #[test]
        fn a_full_buffer_of_one_instruction_gives_complete_repetition() {
            let mut critter = critter_with_window(0b11, Instruction::MoveSlow);
            critter.recent_actions.push_back(Instruction::Eat);
            critter.recent_actions.push_back(Instruction::Eat);

            assert_eq!(critter.recent_repetition_of(Instruction::Eat), 1.0);
        }

        #[test]
        fn a_child_starts_with_no_memory_of_its_parents_actions() {
            // History is runtime state, not genome, so it is not inherited.
            let mut parent = critter_with_window(0b1111, Instruction::Split);
            parent.tick(true);
            assert!(!parent.recent_actions.is_empty());

            let child = divide_fully(&mut parent);

            assert!(child.recent_actions.is_empty());
        }
    }

    mod split {
        use super::*;
        use crate::Genome;

        const INITIAL_ENERGY: u32 = 60;
        // The split flow: pay SPLIT_ATTEMPT_COST and the firing turn's upkeep,
        // then upkeep again for each tick of the division, and finally halve
        // what remains between parent and child. Upkeep is a share of what the
        // critter is holding at the time, so it is walked rather than
        // multiplied out.
        const SPLITTER_ENERGY: u32 = 2 * INITIAL_ENERGY;

        fn energy_at_division_end() -> u32 {
            let mut energy = SPLITTER_ENERGY - SPLIT_ATTEMPT_COST;
            energy -= Critter::upkeep_for(energy);
            for _ in 0..SPLIT_DURATION_TICKS {
                energy -= Critter::upkeep_for(energy);
            }
            energy
        }

        fn splitter() -> Critter {
            let mut critter = Critter::with_genome(
                START_X,
                START_Y,
                NORTH,
                1,
                1,
                INITIAL_ENERGY,
                0,
                Genome::all(Instruction::Split),
            );
            critter.gain_energy(SPLITTER_ENERGY - INITIAL_ENERGY);
            critter
        }

        #[test]
        fn firing_split_yields_no_child_that_tick() {
            // Division takes time: firing Split commits the critter to it
            // rather than producing a child on the spot.
            let mut critter = splitter();

            let outcome = critter.tick(true);

            assert!(outcome.child.is_none());
            assert!(critter.is_dividing());
        }

        #[test]
        fn a_child_arrives_once_the_division_has_run_its_course() {
            let mut critter = splitter();
            critter.tick(true);

            let child = (0..SPLIT_DURATION_TICKS)
                .find_map(|_| critter.tick(true).child)
                .expect("division should finish");

            assert!(!critter.is_dividing());
            assert!(child.energy() > 0);
        }

        #[test]
        fn a_dividing_critter_does_not_act() {
            // It is committed: no moving, so no foraging either. Holds however
            // brief the division is, so the loop runs the whole of it rather
            // than stopping a tick short to catch it mid-division.
            let mut critter = splitter();
            critter.tick(true);
            let (x, y) = (critter.x(), critter.y());

            // Tick through the whole division. Position is compared across
            // all of it rather than only on ticks that leave the critter
            // still dividing, which at a one-tick duration is none of them.
            assert!(critter.is_dividing());
            for _ in 0..SPLIT_DURATION_TICKS {
                critter.tick(true);
            }

            assert_eq!((critter.x(), critter.y()), (x, y));
        }

        #[test]
        fn a_dividing_critter_still_burns_energy() {
            let mut critter = splitter();
            critter.tick(true);
            let before = critter.energy();

            critter.tick(true);

            assert!(critter.energy() < before);
        }

        #[test]
        fn a_critter_that_starves_mid_division_produces_no_child() {
            // The gamble is real: energy spent dividing is lost if the
            // critter cannot see it through.
            let mut critter = splitter();
            critter.tick(true);
            critter.lose_energy(critter.energy() - 1);

            let children: Vec<_> = (0..(SPLIT_DURATION_TICKS * 2))
                .filter_map(|_| critter.tick(true).child)
                .collect();

            assert!(children.is_empty());
            assert_eq!(critter.energy(), 0);
        }

        #[test]
        fn dividing_to_completion_returns_a_child_critter() {
            let mut critter = splitter();

            let child = divide_fully(&mut critter);

            assert!(child.energy() > 0);
        }

        #[test]
        fn the_child_receives_half_of_the_parents_post_attempt_energy() {
            let mut parent = splitter();

            let child = divide_fully(&mut parent);

            assert_eq!(child.energy(), energy_at_division_end() / 2);
        }

        #[test]
        fn firing_split_charges_the_attempt_cost_without_halving_yet() {
            // The halving waits for the division to finish; firing only
            // commits the critter and charges the attempt.
            let mut parent = splitter();

            parent.tick(true);

            assert_eq!(
                parent.energy(),
                SPLITTER_ENERGY - SPLIT_ATTEMPT_COST - Critter::upkeep_for(SPLITTER_ENERGY)
            );
        }

        #[test]
        fn the_parent_keeps_half_of_what_survives_the_division() {
            let mut parent = splitter();

            divide_fully(&mut parent);

            assert_eq!(parent.energy(), energy_at_division_end() / 2);
        }

        #[test]
        fn splitting_costs_at_least_the_split_attempt_cost_total_energy() {
            // Total energy across parent and child after a split is bounded
            // above by pre-split energy minus the attempt cost. This is the
            // friction that keeps "split, eat baby, repeat" from being free.
            let mut parent = splitter();
            let parent_before = parent.energy();

            let child_energy = divide_fully(&mut parent).energy();
            let parent_after = parent.energy();

            assert!(parent_after + child_energy <= parent_before - SPLIT_ATTEMPT_COST);
        }

        #[test]
        fn the_child_inherits_the_parents_heading() {
            let mut parent = splitter();

            let child = divide_fully(&mut parent);

            assert_eq!(child.heading(), NORTH);
        }

        #[test]
        fn the_child_inherits_the_parents_initial_energy() {
            let mut parent = splitter();

            let child = divide_fully(&mut parent);

            assert_eq!(child.initial_energy(), INITIAL_ENERGY);
        }

        #[test]
        fn the_child_spawns_one_step_behind_the_parent() {
            // Parent facing North at (10, 10) with step_size 1: child should appear
            // one pixel south, at (10, 11) — directly behind.
            let mut parent = splitter();

            let child = divide_fully(&mut parent);

            assert_eq!((child.x(), child.y()), (START_X, START_Y + 1));
        }

        #[test]
        fn a_child_spawned_facing_east_appears_one_step_west_of_the_parent() {
            const STEP_SIZE: i32 = 5;
            let mut parent = Critter::with_genome(
                START_X,
                START_Y,
                EAST,
                1,
                STEP_SIZE,
                INITIAL_ENERGY,
                0,
                Genome::all(Instruction::Split),
            );
            parent.gain_energy(INITIAL_ENERGY);

            let child = divide_fully(&mut parent);

            assert_eq!((child.x(), child.y()), (START_X - STEP_SIZE, START_Y));
        }

        #[test]
        fn a_child_spawned_facing_southeast_uses_the_diagonal_scaled_offset() {
            // step_size 10, scaled by √2/2 ≈ 0.707 and rounded → 7. Child appears
            // northwest of parent (opposite of SouthEast), so at (parent - 7, parent - 7).
            const STEP_SIZE: i32 = 10;
            const DIAGONAL_OFFSET: i32 = 7;
            let mut parent = Critter::with_genome(
                START_X,
                START_Y,
                SOUTH_EAST,
                1,
                STEP_SIZE,
                INITIAL_ENERGY,
                0,
                Genome::all(Instruction::Split),
            );
            parent.gain_energy(INITIAL_ENERGY);

            let child = divide_fully(&mut parent);

            assert_eq!(
                (child.x(), child.y()),
                (START_X - DIAGONAL_OFFSET, START_Y - DIAGONAL_OFFSET)
            );
        }

        #[test]
        fn the_child_can_tick_independently_after_being_spawned() {
            // The child should have its own tick_counter starting fresh, and inherit
            // the parent's instruction list — so it can move on its own.
            let mut parent = Critter::with_genome(
                START_X,
                START_Y,
                EAST,
                1,
                1,
                INITIAL_ENERGY,
                0,
                Genome::from_instructions(&[Instruction::Split, Instruction::MoveSlow]),
            );
            parent.gain_energy(INITIAL_ENERGY);

            let mut child = divide_fully(&mut parent);
            let initial_child_x = child.x();
            child.tick(true); // executes the second instruction (MoveSlow)

            assert_eq!(child.x(), initial_child_x + 1);
        }

        #[test]
        fn move_forward_does_not_produce_a_child() {
            let mut critter = Critter::with_genome(
                START_X,
                START_Y,
                EAST,
                1,
                1,
                INITIAL_ENERGY,
                0,
                Genome::all(Instruction::MoveSlow),
            );

            assert!(critter.tick(true).child.is_none());
        }

        #[test]
        fn turn_left_does_not_produce_a_child() {
            let mut critter = Critter::with_genome(
                START_X,
                START_Y,
                NORTH,
                1,
                1,
                INITIAL_ENERGY,
                0,
                Genome::all(Instruction::TurnLeft15),
            );

            assert!(critter.tick(true).child.is_none());
        }

        #[test]
        fn do_nothing_does_not_produce_a_child() {
            let mut critter = Critter::with_genome(
                START_X,
                START_Y,
                NORTH,
                1,
                1,
                INITIAL_ENERGY,
                0,
                Genome::all(Instruction::DoNothing),
            );

            assert!(critter.tick(true).child.is_none());
        }

        #[test]
        fn a_tick_before_the_threshold_does_not_produce_a_child() {
            let mut critter = Critter::with_genome(
                START_X,
                START_Y,
                NORTH,
                10,
                1,
                INITIAL_ENERGY,
                0,
                Genome::all(Instruction::Split),
            );

            assert!(critter.tick(true).child.is_none());
        }

        #[test]
        fn a_split_when_repeating_a_previous_split_still_returns_a_child() {
            let mut parent = Critter::with_genome(
                START_X,
                START_Y,
                NORTH,
                1,
                1,
                INITIAL_ENERGY,
                0,
                Genome::from_instructions(&[Instruction::Split, Instruction::RepeatPreviousMove]),
            );
            // Need enough energy for two splits: pre-split must be ≥ 2 × initial.
            // After the first split, parent will have ~half — so we start with
            // 4 × initial to leave enough for the second split.
            parent.gain_energy(3 * INITIAL_ENERGY);

            divide_fully(&mut parent); // first split
            let second_child = divide_fully(&mut parent); // repeat → split again

            assert!(second_child.energy() > 0);
        }
    }

    mod mutation_on_split {
        use super::*;
        use crate::genome::MUTATION_RATE_BITS;
        use crate::Genome;

        // Build a parent that splits every tick, with a known starting genome
        // and high enough energy to keep splitting. Returns the parent's genome
        // and the genome of one resulting child.
        fn split_once(seed: u64) -> (Genome, Genome) {
            let mut parent = Critter::with_genome(
                10,
                10,
                NORTH,
                1,
                1,
                10,
                seed,
                Genome::all(Instruction::Split),
            );
            parent.gain_energy(MAX_CRITTER_ENERGY - 10);
            let parent_genome = parent.genome().clone();
            let child = divide_fully(&mut parent);
            (parent_genome, child.genome().clone())
        }

        // Splits a parent whose genome carries the given mutation-rate bits,
        // returning how many of `attempts` children differ from the parent.
        fn mutated_children_of_rate(rate_bits: u32, attempts: u64) -> u32 {
            (0..attempts)
                .filter(|&seed| {
                    let mut genome = Genome::all(Instruction::Split);
                    genome.set_mutation_rate_bits(rate_bits);
                    let mut parent = Critter::with_genome(10, 10, NORTH, 1, 1, 10, seed, genome);
                    parent.gain_energy(MAX_CRITTER_ENERGY - 10);
                    let parent_genome = parent.genome().clone();
                    let child = divide_fully(&mut parent);
                    *child.genome() != parent_genome
                })
                .count() as u32
        }

        // How many bits differ between a parent and each of its children.
        fn child_bit_distances(rate_bits: u32, attempts: u64) -> Vec<u32> {
            (0..attempts)
                .map(|seed| {
                    let mut genome = Genome::all(Instruction::Split);
                    genome.set_mutation_rate_bits(rate_bits);
                    let mut parent = Critter::with_genome(10, 10, NORTH, 1, 1, 10, seed, genome);
                    parent.gain_energy(MAX_CRITTER_ENERGY - 10);
                    let parent_genome = parent.genome().clone();
                    let child = divide_fully(&mut parent);
                    parent_genome
                        .to_bits()
                        .chars()
                        .zip(child.genome().to_bits().chars())
                        .filter(|(a, b)| a != b)
                        .count() as u32
                })
                .collect()
        }

        #[test]
        fn a_mutation_changes_only_a_few_bits_at_a_time() {
            // Each bit is considered independently, so a mutation is a small
            // edit rather than a burst: the genome drifts, it does not lurch.
            let max_rate_bits = u32::MAX >> (32 - MUTATION_RATE_BITS);
            let distances = child_bit_distances(max_rate_bits, 400);

            let worst = distances.iter().copied().max().unwrap_or(0);
            assert!(
                worst < 10,
                "expected mutations to change only a few bits, worst was {worst}"
            );
        }

        #[test]
        fn a_parent_whose_genome_encodes_no_mutation_produces_only_clones() {
            assert_eq!(mutated_children_of_rate(0, 200), 0);
        }

        #[test]
        fn a_parent_whose_genome_encodes_the_maximum_rate_mutates_far_more_often() {
            let never = mutated_children_of_rate(0, 200);
            let often = mutated_children_of_rate(u32::MAX >> (32 - MUTATION_RATE_BITS), 200);

            assert!(
                often > never + 20,
                "expected the max-rate parent to mutate far more than the zero-rate parent, \
                 got {often} vs {never} out of 200"
            );
        }

        #[test]
        fn a_child_inherits_the_parents_genome_when_no_mutation_fires() {
            let (parent, child) = split_once(0);

            assert_eq!(parent, child);
        }

        #[test]
        fn a_parent_encoding_a_middling_rate_mutates_some_children_but_not_most() {
            // Half the rate field's range. Mutation should fire sometimes and
            // not most of the time, which pins the mapping from bits to
            // probability without asserting an exact count.
            const HALF_RATE_BITS: u32 = 1 << (MUTATION_RATE_BITS - 1);
            let mutated = mutated_children_of_rate(HALF_RATE_BITS, 200);

            assert!(
                (1..200).contains(&mutated),
                "expected some but not all of 200 children to mutate, got {mutated}"
            );
        }
    }

    mod allow_split_gate {
        use super::*;
        use crate::Genome;

        fn ready_to_split(seed: u64) -> Critter {
            let mut critter = Critter::with_genome(
                10,
                10,
                NORTH,
                1,
                1,
                10,
                seed,
                Genome::all(Instruction::Split),
            );
            critter.gain_energy(MAX_CRITTER_ENERGY - 10);
            critter
        }

        #[test]
        fn when_split_is_disallowed_no_child_is_produced() {
            let mut parent = ready_to_split(0);

            let outcome = parent.tick(false);

            assert!(outcome.child.is_none());
        }

        #[test]
        fn when_split_is_disallowed_the_parent_still_pays_the_turns_upkeep() {
            let mut parent = ready_to_split(0);
            let energy_before = parent.energy();

            parent.tick(false);

            assert_eq!(
                parent.energy(),
                energy_before - Critter::upkeep_for(energy_before)
            );
        }

        #[test]
        fn when_split_is_allowed_a_child_is_still_produced() {
            let mut parent = ready_to_split(0);

            let child = divide_fully(&mut parent);

            assert!(child.energy() > 0);
        }
    }

    mod jitter_threshold_tests {
        use super::super::jitter_threshold;
        use rand::rngs::SmallRng;
        use rand::SeedableRng;

        #[test]
        fn at_base_zero_it_returns_zero() {
            let mut rng = SmallRng::seed_from_u64(0);

            assert_eq!(jitter_threshold(&mut rng, 0), 0);
        }

        #[test]
        fn at_base_one_it_returns_one() {
            let mut rng = SmallRng::seed_from_u64(0);

            assert_eq!(jitter_threshold(&mut rng, 1), 1);
        }

        #[test]
        fn for_a_base_above_one_results_span_one_to_two_base_minus_one() {
            // Over many draws at base=5, the lowest result must be 1 and the
            // highest must be 9. This pins down both endpoints of the range.
            const BASE: u32 = 5;
            let mut rng = SmallRng::seed_from_u64(0);
            let mut min_seen = u32::MAX;
            let mut max_seen = 0;
            for _ in 0..1000 {
                let v = jitter_threshold(&mut rng, BASE);
                min_seen = min_seen.min(v);
                max_seen = max_seen.max(v);
            }

            assert_eq!(min_seen, 1);
            assert_eq!(max_seen, 2 * BASE - 1);
        }
    }

    mod jitter {
        use super::*;

        const BASE_TICKS: u32 = 5;

        fn movement_only_critter(seed: u64) -> Critter {
            Critter::with_genome(
                START_X,
                START_Y,
                EAST,
                BASE_TICKS,
                1,
                u32::MAX,
                seed,
                Genome::all(Instruction::MoveSlow),
            )
        }

        #[test]
        fn two_critters_with_different_seeds_desynchronize_after_their_first_firing() {
            let mut a = movement_only_critter(1);
            let mut b = movement_only_critter(2);

            // Both fire at the deterministic initial threshold (BASE_TICKS).
            for _ in 0..BASE_TICKS {
                a.tick(true);
                b.tick(true);
            }
            // After the first firing each gets a jittered next threshold drawn
            // from its own rng — over enough ticks, their x values must diverge.
            for _ in 0..(BASE_TICKS * 4) {
                a.tick(true);
                b.tick(true);
            }

            assert_ne!(a.x(), b.x());
        }

        #[test]
        fn the_initial_firing_uses_the_deterministic_ticks_per_instruction_threshold() {
            // The first firing must happen at exactly BASE_TICKS ticks regardless
            // of seed; subsequent firings are the ones that vary.
            let mut critter = movement_only_critter(0);

            for _ in 0..(BASE_TICKS - 1) {
                critter.tick(true);
            }
            let before = critter.x();
            critter.tick(true);
            let after = critter.x();

            assert_eq!(before, START_X);
            assert_eq!(after, START_X + 1);
        }
    }

    mod energy_cap {
        use super::*;

        #[test]
        fn gain_energy_is_not_capped() {
            let mut critter = Critter::with_genome(
                0,
                0,
                NORTH,
                1,
                1,
                MAX_CRITTER_ENERGY - 10,
                0,
                Genome::all(Instruction::DoNothing),
            );

            critter.gain_energy(1_000);

            assert_eq!(critter.energy(), MAX_CRITTER_ENERGY - 10 + 1_000);
        }

        #[test]
        fn gain_energy_below_the_cap_increases_normally() {
            let mut critter = Critter::with_genome(
                0,
                0,
                NORTH,
                1,
                1,
                100,
                0,
                Genome::all(Instruction::DoNothing),
            );

            critter.gain_energy(50);

            assert_eq!(critter.energy(), 150);
        }
    }
}
