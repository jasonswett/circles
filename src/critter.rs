use crate::{Genome, Heading, Instruction};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

pub const MAX_CRITTER_ENERGY: u32 = 500;
const MUTATION_CHANCE: f32 = 0.1;
const BIT_FLIP_RATE: f32 = 0.01;

/// What a critter's tick produced this turn. World inspects this after each
/// critter ticks to add any newborn child to the population and to know
/// whether the critter attempted a steal (which the world resolves later
/// because the critter doesn't see its neighbors).
#[derive(Default)]
pub struct TickOutcome {
    pub child: Option<Critter>,
    pub attempted_steal: bool,
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
    being_stolen_from_indicator_ticks: u32,
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

    /// Test-only: build a critter with a specific genome.
    #[cfg(test)]
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
            being_stolen_from_indicator_ticks: 0,
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

    pub fn initial_energy(&self) -> u32 {
        self.initial_energy
    }

    #[cfg(test)]
    pub fn genome(&self) -> &Genome {
        &self.genome
    }

    pub fn gain_energy(&mut self, amount: u32) {
        self.energy = self.energy.saturating_add(amount).min(MAX_CRITTER_ENERGY);
    }

    /// Adds energy up to the cap and returns how much was actually accepted.
    /// Useful when the caller needs to know whether a top-up filled the
    /// critter so it can move on to the next source (e.g. a stealer that
    /// drains victims one at a time and must stop once it caps out).
    pub fn gain_energy_saturating(&mut self, amount: u32) -> u32 {
        let headroom = MAX_CRITTER_ENERGY.saturating_sub(self.energy);
        let gained = headroom.min(amount);
        self.energy += gained;
        gained
    }

    pub fn lose_energy(&mut self, amount: u32) {
        self.energy = self.energy.saturating_sub(amount);
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

    pub fn is_being_stolen_from(&self) -> bool {
        self.being_stolen_from_indicator_ticks > 0
    }

    pub fn mark_being_stolen_from_for(&mut self, ticks: u32) {
        self.being_stolen_from_indicator_ticks = ticks;
    }

    pub fn age_being_stolen_from_indicator(&mut self) {
        self.being_stolen_from_indicator_ticks =
            self.being_stolen_from_indicator_ticks.saturating_sub(1);
    }

    pub fn wrap_position(&mut self, width: i32, height: i32) {
        self.x = self.x.rem_euclid(width);
        self.y = self.y.rem_euclid(height);
    }

    pub fn tick(&mut self, allow_split: bool) -> TickOutcome {
        self.tick_counter += 1;
        if self.tick_counter < self.next_fire_threshold {
            return TickOutcome::default();
        }
        self.tick_counter = 0;
        self.next_fire_threshold = jitter_threshold(&mut self.rng, self.ticks_per_instruction);

        if self.energy == 0 {
            return TickOutcome::default();
        }

        let instruction = self.genome.decode_at(self.genome_cursor);
        // decode_at handles wrap-around itself, so the cursor can grow without
        // an explicit modulo. usize will not overflow within any realistic run.
        self.genome_cursor += 1;

        // Each instruction is gated by the genome's sigmoid: the probability of
        // acting is sigmoid((energy - threshold) / softness) for that
        // instruction's per-critter parameters. A "no" still consumes one
        // energy and `last_executed` is left untouched so RepeatPreviousMove
        // keeps referring to whatever did execute last.
        let probability = self.genome.probability_of_acting(
            instruction,
            self.energy,
            self.is_overlapping_critter(),
        );
        let split_blocked = instruction == Instruction::Split && !allow_split;
        let outcome = if !split_blocked && self.roll_against(probability) {
            self.execute(instruction)
        } else {
            TickOutcome::default()
        };
        self.energy = self.energy.saturating_sub(1);
        outcome
    }

    // The `<` vs `<=` mutation is an equivalent mutant for our continuous f32
    // rolls: `rng.gen::<f32>()` produces values in [0, 1), so the boundary case
    // `roll == probability` happens with probability 0 and is unobservable.
    #[mutants::skip]
    fn roll_against(&mut self, probability: f32) -> bool {
        let roll: f32 = self.rng.gen();
        roll < probability
    }

    fn execute(&mut self, instruction: Instruction) -> TickOutcome {
        match instruction {
            Instruction::MoveForward => {
                let (dx, dy) = self.heading.offset();
                let step = if self.heading.is_diagonal() {
                    ((self.step_size as f32) * std::f32::consts::FRAC_1_SQRT_2).round() as i32
                } else {
                    self.step_size
                };
                self.x += dx * step;
                self.y += dy * step;
                self.last_executed = Some(Instruction::MoveForward);
                TickOutcome::default()
            }
            Instruction::TurnLeft => {
                self.heading = self.heading.turn_left();
                self.last_executed = Some(Instruction::TurnLeft);
                TickOutcome::default()
            }
            Instruction::TurnRight => {
                self.heading = self.heading.turn_right();
                self.last_executed = Some(Instruction::TurnRight);
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
                let child = self.spawn_child();
                self.energy /= 2;
                self.last_executed = Some(Instruction::Split);
                TickOutcome {
                    child: Some(child),
                    attempted_steal: false,
                }
            }
            Instruction::Steal => {
                // The critter doesn't see its neighbors, so it can't actually
                // steal here; it only signals the intent. World::tick collects
                // these intents and resolves them after every critter has
                // ticked, so all critters draw from a consistent snapshot.
                self.last_executed = Some(Instruction::Steal);
                TickOutcome {
                    child: None,
                    attempted_steal: true,
                }
            }
        }
    }

    fn spawn_child(&mut self) -> Critter {
        let (dx, dy) = self.heading.offset();
        let offset = if self.heading.is_diagonal() {
            ((self.step_size as f32) * std::f32::consts::FRAC_1_SQRT_2).round() as i32
        } else {
            self.step_size
        };
        let child_seed: u64 = self.rng.gen();
        let mut child_rng = SmallRng::seed_from_u64(child_seed);
        // Children's first firing is jittered, so they desynchronize from the
        // parent immediately.
        let child_threshold = jitter_threshold(&mut child_rng, self.ticks_per_instruction);
        let mut child_genome = self.genome.clone();
        // Each split has a small chance of mutating the child's genome; when
        // it does, each bit independently flips with a smaller probability.
        if self.roll_against(MUTATION_CHANCE) {
            child_genome.mutate(&mut child_rng, BIT_FLIP_RATE);
        }
        Critter {
            x: self.x - dx * offset,
            y: self.y - dy * offset,
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
            overlap_indicator_ticks: 0,
            being_stolen_from_indicator_ticks: 0,
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
    use crate::{Heading, Instruction};

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
                Heading::North,
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
                Heading::North,
                1,
                1,
                u32::MAX,
                0,
                Genome::all(Instruction::DoNothing),
            );

            assert_eq!(critter.heading(), Heading::North);
        }
    }

    mod overlapping_critter_indicator {
        use super::*;

        fn fresh_critter() -> Critter {
            Critter::with_genome(
                START_X,
                START_Y,
                Heading::North,
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

    mod being_stolen_from_indicator {
        use super::*;

        fn fresh_critter() -> Critter {
            Critter::with_genome(
                START_X,
                START_Y,
                Heading::North,
                1,
                1,
                u32::MAX,
                0,
                Genome::all(Instruction::DoNothing),
            )
        }

        #[test]
        fn a_freshly_created_critter_is_not_marked_as_being_stolen_from() {
            let critter = fresh_critter();

            assert!(!critter.is_being_stolen_from());
        }

        #[test]
        fn marking_the_critter_for_a_positive_tick_count_makes_it_report_as_being_stolen_from() {
            let mut critter = fresh_critter();

            critter.mark_being_stolen_from_for(10);

            assert!(critter.is_being_stolen_from());
        }

        #[test]
        fn aging_until_the_counter_runs_out_clears_the_being_stolen_from_state() {
            let mut critter = fresh_critter();
            critter.mark_being_stolen_from_for(2);

            critter.age_being_stolen_from_indicator();
            critter.age_being_stolen_from_indicator();

            assert!(!critter.is_being_stolen_from());
        }

        #[test]
        fn aging_when_already_at_zero_does_not_underflow() {
            let mut critter = fresh_critter();

            critter.age_being_stolen_from_indicator();

            assert!(!critter.is_being_stolen_from());
        }
    }

    mod tick {
        use super::*;

        #[test]
        fn it_does_not_execute_an_instruction_before_n_ticks_have_passed() {
            let mut critter = Critter::with_genome(
                START_X,
                START_Y,
                Heading::North,
                TICKS_PER_INSTRUCTION,
                1,
                u32::MAX,
                0,
                Genome::all(Instruction::MoveForward),
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
                Heading::North,
                TICKS_PER_INSTRUCTION,
                1,
                u32::MAX,
                0,
                Genome::all(Instruction::MoveForward),
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
                Heading::East,
                1,
                1,
                u32::MAX,
                0,
                Genome::all(Instruction::MoveForward),
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
                Heading::East,
                1,
                STEP_SIZE,
                u32::MAX,
                0,
                Genome::all(Instruction::MoveForward),
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
                Heading::South,
                1,
                STEP_SIZE,
                u32::MAX,
                0,
                Genome::all(Instruction::MoveForward),
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
                Heading::SouthEast,
                1,
                STEP_SIZE,
                u32::MAX,
                0,
                Genome::all(Instruction::MoveForward),
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
                Heading::NorthWest,
                1,
                STEP_SIZE,
                u32::MAX,
                0,
                Genome::all(Instruction::MoveForward),
            );

            critter.tick(true);

            assert_eq!(
                (critter.x(), critter.y()),
                (START_X - DIAGONAL_STEP, START_Y - DIAGONAL_STEP)
            );
        }

        #[test]
        fn turn_left_changes_the_heading() {
            let mut critter = Critter::with_genome(
                START_X,
                START_Y,
                Heading::North,
                1,
                1,
                u32::MAX,
                0,
                Genome::all(Instruction::TurnLeft),
            );

            critter.tick(true);

            assert_eq!(critter.heading(), Heading::NorthWest);
        }

        #[test]
        fn turn_left_does_not_change_the_position() {
            let mut critter = Critter::with_genome(
                START_X,
                START_Y,
                Heading::North,
                1,
                1,
                u32::MAX,
                0,
                Genome::all(Instruction::TurnLeft),
            );

            critter.tick(true);

            assert_eq!((critter.x(), critter.y()), (START_X, START_Y));
        }

        #[test]
        fn turn_right_changes_the_heading() {
            let mut critter = Critter::with_genome(
                START_X,
                START_Y,
                Heading::North,
                1,
                1,
                u32::MAX,
                0,
                Genome::all(Instruction::TurnRight),
            );

            critter.tick(true);

            assert_eq!(critter.heading(), Heading::NorthEast);
        }

        #[test]
        fn turn_right_does_not_change_the_position() {
            let mut critter = Critter::with_genome(
                START_X,
                START_Y,
                Heading::North,
                1,
                1,
                u32::MAX,
                0,
                Genome::all(Instruction::TurnRight),
            );

            critter.tick(true);

            assert_eq!((critter.x(), critter.y()), (START_X, START_Y));
        }

        #[test]
        fn do_nothing_leaves_position_unchanged() {
            let mut critter = Critter::with_genome(
                START_X,
                START_Y,
                Heading::North,
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
                Heading::North,
                1,
                1,
                u32::MAX,
                0,
                Genome::all(Instruction::DoNothing),
            );

            critter.tick(true);

            assert_eq!(critter.heading(), Heading::North);
        }

        #[test]
        fn each_tick_consumes_the_next_instruction_in_the_list() {
            let mut critter = Critter::with_genome(
                START_X,
                START_Y,
                Heading::East,
                1,
                1,
                u32::MAX,
                0,
                Genome::from_instructions(&[Instruction::MoveForward, Instruction::TurnRight]),
            );

            critter.tick(true);
            critter.tick(true);

            assert_eq!(critter.x(), START_X + 1);
            assert_eq!(critter.heading(), Heading::SouthEast);
        }

        #[test]
        fn the_instruction_list_loops_when_exhausted() {
            let mut critter = Critter::with_genome(
                START_X,
                START_Y,
                Heading::East,
                1,
                1,
                u32::MAX,
                0,
                Genome::all(Instruction::MoveForward),
            );

            critter.tick(true);
            critter.tick(true);
            critter.tick(true);

            assert_eq!(critter.x(), START_X + 3);
        }
    }

    mod repeat_previous_move {
        use super::*;

        #[test]
        fn it_re_executes_the_previously_executed_instruction() {
            let mut critter = Critter::with_genome(
                START_X,
                START_Y,
                Heading::East,
                1,
                1,
                u32::MAX,
                0,
                Genome::from_instructions(&[
                    Instruction::MoveForward,
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
                Heading::East,
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
                Heading::East,
                1,
                1,
                u32::MAX,
                0,
                Genome::all(Instruction::RepeatPreviousMove),
            );

            critter.tick(true);

            assert_eq!(critter.heading(), Heading::East);
        }

        #[test]
        fn repeating_a_repeat_re_executes_the_underlying_move_a_third_time() {
            // After three ticks with [TurnRight, Repeat, Repeat] (each TurnRight = 45°):
            //   tick 1: TurnRight        — East -> SouthEast
            //   tick 2: Repeat -> Right  — SouthEast -> South
            //   tick 3: Repeat -> Right  — South -> SouthWest
            // The third tick must reach back through two repeats to find TurnRight.
            let mut critter = Critter::with_genome(
                START_X,
                START_Y,
                Heading::East,
                1,
                1,
                u32::MAX,
                0,
                Genome::from_instructions(&[
                    Instruction::TurnRight,
                    Instruction::RepeatPreviousMove,
                    Instruction::RepeatPreviousMove,
                ]),
            );

            critter.tick(true);
            critter.tick(true);
            critter.tick(true);

            assert_eq!(critter.heading(), Heading::SouthWest);
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
                Heading::North,
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
                Heading::North,
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
                Heading::North,
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
                Heading::North,
                1,
                1,
                INITIAL_ENERGY,
                0,
                Genome::all(Instruction::DoNothing),
            );

            critter.tick(true);

            assert_eq!(critter.energy(), INITIAL_ENERGY - 1);
        }

        #[test]
        fn ticks_before_the_threshold_do_not_decrement_energy() {
            let mut critter = Critter::with_genome(
                START_X,
                START_Y,
                Heading::North,
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
                Heading::East,
                1,
                1,
                0,
                0,
                Genome::all(Instruction::MoveForward),
            );

            critter.tick(true);

            assert_eq!((critter.x(), critter.y()), (START_X, START_Y));
        }

        #[test]
        fn ticking_at_zero_energy_keeps_energy_at_zero() {
            let mut critter = Critter::with_genome(
                START_X,
                START_Y,
                Heading::East,
                1,
                1,
                0,
                0,
                Genome::all(Instruction::MoveForward),
            );

            for _ in 0..10 {
                critter.tick(true);
            }

            assert_eq!(critter.energy(), 0);
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
                Heading::North,
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

    mod split {
        use super::*;
        use crate::Genome;

        const INITIAL_ENERGY: u32 = 60;
        // Pre-split energy is the parent's energy at the moment of splitting; the
        // child gets half and the parent retains half (minus the instruction cost).
        const SPLITTER_ENERGY: u32 = 2 * INITIAL_ENERGY;

        fn splitter() -> Critter {
            let mut critter = Critter::with_genome(
                START_X,
                START_Y,
                Heading::North,
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
        fn ticking_a_split_instruction_returns_a_child_critter() {
            let mut critter = splitter();

            let outcome = critter.tick(true);

            assert!(outcome.child.is_some());
        }

        #[test]
        fn the_child_receives_half_of_the_parents_pre_split_energy() {
            let mut parent = splitter();

            let child = parent.tick(true).child.unwrap();

            assert_eq!(child.energy(), SPLITTER_ENERGY / 2);
        }

        #[test]
        fn the_parent_keeps_half_of_its_pre_split_energy_minus_the_instruction_cost() {
            let mut parent = splitter();

            parent.tick(true);

            assert_eq!(parent.energy(), SPLITTER_ENERGY / 2 - 1);
        }

        #[test]
        fn the_child_inherits_the_parents_heading() {
            let mut parent = splitter();

            let child = parent.tick(true).child.unwrap();

            assert_eq!(child.heading(), Heading::North);
        }

        #[test]
        fn the_child_inherits_the_parents_initial_energy() {
            let mut parent = splitter();

            let child = parent.tick(true).child.unwrap();

            assert_eq!(child.initial_energy(), INITIAL_ENERGY);
        }

        #[test]
        fn the_child_spawns_one_step_behind_the_parent() {
            // Parent facing North at (10, 10) with step_size 1: child should appear
            // one pixel south, at (10, 11) — directly behind.
            let mut parent = splitter();

            let child = parent.tick(true).child.unwrap();

            assert_eq!((child.x(), child.y()), (START_X, START_Y + 1));
        }

        #[test]
        fn a_child_spawned_facing_east_appears_one_step_west_of_the_parent() {
            const STEP_SIZE: i32 = 5;
            let mut parent = Critter::with_genome(
                START_X,
                START_Y,
                Heading::East,
                1,
                STEP_SIZE,
                INITIAL_ENERGY,
                0,
                Genome::all(Instruction::Split),
            );
            parent.gain_energy(INITIAL_ENERGY);

            let child = parent.tick(true).child.unwrap();

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
                Heading::SouthEast,
                1,
                STEP_SIZE,
                INITIAL_ENERGY,
                0,
                Genome::all(Instruction::Split),
            );
            parent.gain_energy(INITIAL_ENERGY);

            let child = parent.tick(true).child.unwrap();

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
                Heading::East,
                1,
                1,
                INITIAL_ENERGY,
                0,
                Genome::from_instructions(&[Instruction::Split, Instruction::MoveForward]),
            );
            parent.gain_energy(INITIAL_ENERGY);

            let mut child = parent.tick(true).child.unwrap();
            let initial_child_x = child.x();
            child.tick(true); // executes the second instruction (MoveForward)

            assert_eq!(child.x(), initial_child_x + 1);
        }

        #[test]
        fn move_forward_does_not_produce_a_child() {
            let mut critter = Critter::with_genome(
                START_X,
                START_Y,
                Heading::East,
                1,
                1,
                INITIAL_ENERGY,
                0,
                Genome::all(Instruction::MoveForward),
            );

            assert!(critter.tick(true).child.is_none());
        }

        #[test]
        fn turn_left_does_not_produce_a_child() {
            let mut critter = Critter::with_genome(
                START_X,
                START_Y,
                Heading::North,
                1,
                1,
                INITIAL_ENERGY,
                0,
                Genome::all(Instruction::TurnLeft),
            );

            assert!(critter.tick(true).child.is_none());
        }

        #[test]
        fn do_nothing_does_not_produce_a_child() {
            let mut critter = Critter::with_genome(
                START_X,
                START_Y,
                Heading::North,
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
                Heading::North,
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
                Heading::North,
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

            parent.tick(true); // first split
            let second_outcome = parent.tick(true); // repeat → split again

            assert!(second_outcome.child.is_some());
        }
    }

    mod mutation_on_split {
        use super::*;
        use crate::Genome;

        // Build a parent that splits every tick, with a known starting genome
        // and high enough energy to keep splitting. Returns the parent's genome
        // and the genome of one resulting child.
        fn split_once(seed: u64) -> (Genome, Genome) {
            let mut parent = Critter::with_genome(
                10,
                10,
                Heading::North,
                1,
                1,
                10,
                seed,
                Genome::all(Instruction::Split),
            );
            parent.gain_energy(MAX_CRITTER_ENERGY - 10);
            let parent_genome = parent.genome().clone();
            let child = parent.tick(true).child.expect("parent should split");
            (parent_genome, child.genome().clone())
        }

        #[test]
        fn most_children_inherit_the_parents_genome_unchanged() {
            // With MUTATION_CHANCE = 0.1, ~90% of children should be identical
            // to the parent. Asserting that at least 70% match leaves wide
            // statistical headroom while still failing if mutation_chance was
            // accidentally raised to 1.0 or the inheritance broke.
            let mut identical = 0;
            for seed in 0..200 {
                let (parent, child) = split_once(seed);
                if parent == child {
                    identical += 1;
                }
            }
            assert!(
                identical > 140,
                "expected >140/200 children to inherit unchanged, got {identical}"
            );
        }

        #[test]
        fn some_children_have_a_mutated_genome() {
            // With MUTATION_CHANCE = 0.1 and BIT_FLIP_RATE = 0.01, over 200
            // splits we expect roughly 20 children to be mutated. Asserting at
            // least one child differs catches a "mutation never fires" bug.
            let mut any_differ = false;
            for seed in 0..200 {
                let (parent, child) = split_once(seed);
                if parent != child {
                    any_differ = true;
                    break;
                }
            }
            assert!(
                any_differ,
                "expected at least one mutated child in 200 splits"
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
                Heading::North,
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
        fn when_split_is_disallowed_the_parent_still_pays_the_one_energy_cost() {
            let mut parent = ready_to_split(0);
            let energy_before = parent.energy();

            parent.tick(false);

            assert_eq!(parent.energy(), energy_before - 1);
        }

        #[test]
        fn when_split_is_allowed_a_child_is_still_produced() {
            let mut parent = ready_to_split(0);

            let outcome = parent.tick(true);

            assert!(outcome.child.is_some());
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
                Heading::East,
                BASE_TICKS,
                1,
                u32::MAX,
                seed,
                Genome::all(Instruction::MoveForward),
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
        fn the_max_energy_constant_is_five_hundred() {
            assert_eq!(MAX_CRITTER_ENERGY, 500);
        }

        #[test]
        fn gain_energy_caps_the_total_at_max_critter_energy() {
            let mut critter = Critter::with_genome(
                0,
                0,
                Heading::North,
                1,
                1,
                MAX_CRITTER_ENERGY - 10,
                0,
                Genome::all(Instruction::DoNothing),
            );

            critter.gain_energy(1_000);

            assert_eq!(critter.energy(), MAX_CRITTER_ENERGY);
        }

        #[test]
        fn gain_energy_below_the_cap_increases_normally() {
            let mut critter = Critter::with_genome(
                0,
                0,
                Heading::North,
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

    mod gain_energy_saturating {
        use super::*;

        fn idle_critter(energy: u32) -> Critter {
            Critter::with_genome(
                0,
                0,
                Heading::North,
                1,
                1,
                energy,
                0,
                Genome::all(Instruction::DoNothing),
            )
        }

        #[test]
        fn well_below_the_cap_the_full_amount_is_gained() {
            let mut critter = idle_critter(100);

            let gained = critter.gain_energy_saturating(50);

            assert_eq!(gained, 50);
            assert_eq!(critter.energy(), 150);
        }

        #[test]
        fn the_returned_amount_is_only_what_fits_under_the_cap() {
            let mut critter = idle_critter(MAX_CRITTER_ENERGY - 10);

            let gained = critter.gain_energy_saturating(100);

            assert_eq!(gained, 10);
            assert_eq!(critter.energy(), MAX_CRITTER_ENERGY);
        }

        #[test]
        fn at_the_cap_no_energy_is_gained() {
            let mut critter = idle_critter(MAX_CRITTER_ENERGY);

            let gained = critter.gain_energy_saturating(100);

            assert_eq!(gained, 0);
            assert_eq!(critter.energy(), MAX_CRITTER_ENERGY);
        }

        #[test]
        fn requesting_zero_returns_zero_and_leaves_energy_unchanged() {
            let mut critter = idle_critter(100);

            let gained = critter.gain_energy_saturating(0);

            assert_eq!(gained, 0);
            assert_eq!(critter.energy(), 100);
        }
    }
}
