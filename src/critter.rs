use crate::{Heading, Instruction};

#[derive(Clone)]
pub struct Critter {
    x: i32,
    y: i32,
    heading: Heading,
    instructions: Vec<Instruction>,
    next_instruction_index: usize,
    last_executed: Option<Instruction>,
    ticks_per_instruction: u32,
    tick_counter: u32,
    step_size: i32,
    energy: u32,
    initial_energy: u32,
}

impl Critter {
    pub fn new(
        x: i32,
        y: i32,
        heading: Heading,
        instructions: Vec<Instruction>,
        ticks_per_instruction: u32,
        step_size: i32,
        initial_energy: u32,
    ) -> Self {
        Self {
            x,
            y,
            heading,
            instructions,
            next_instruction_index: 0,
            last_executed: None,
            ticks_per_instruction,
            tick_counter: 0,
            step_size,
            energy: initial_energy,
            initial_energy,
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

    pub fn gain_energy(&mut self, amount: u32) {
        self.energy = self.energy.saturating_add(amount).min(self.initial_energy);
    }

    pub fn lose_energy(&mut self, amount: u32) {
        self.energy = self.energy.saturating_sub(amount);
    }

    pub fn wrap_position(&mut self, width: i32, height: i32) {
        self.x = self.x.rem_euclid(width);
        self.y = self.y.rem_euclid(height);
    }

    pub fn tick(&mut self) -> Option<Critter> {
        self.tick_counter += 1;
        if self.tick_counter < self.ticks_per_instruction {
            return None;
        }
        self.tick_counter = 0;

        if self.instructions.is_empty() {
            return None;
        }

        if self.energy == 0 {
            return None;
        }

        let instruction = self.instructions[self.next_instruction_index];
        self.next_instruction_index = (self.next_instruction_index + 1) % self.instructions.len();

        let child = self.execute(instruction);
        self.energy = self.energy.saturating_sub(1);
        child
    }

    fn execute(&mut self, instruction: Instruction) -> Option<Critter> {
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
                None
            }
            Instruction::TurnLeft => {
                self.heading = self.heading.turn_left();
                self.last_executed = Some(Instruction::TurnLeft);
                None
            }
            Instruction::TurnRight => {
                self.heading = self.heading.turn_right();
                self.last_executed = Some(Instruction::TurnRight);
                None
            }
            Instruction::DoNothing => {
                self.last_executed = Some(Instruction::DoNothing);
                None
            }
            Instruction::RepeatPreviousMove => {
                if let Some(previous) = self.last_executed {
                    self.execute(previous)
                } else {
                    None
                }
            }
            Instruction::Split => {
                let child = self.spawn_child();
                self.energy /= 2;
                self.last_executed = Some(Instruction::Split);
                Some(child)
            }
        }
    }

    fn spawn_child(&self) -> Critter {
        let (dx, dy) = self.heading.offset();
        let offset = if self.heading.is_diagonal() {
            ((self.step_size as f32) * std::f32::consts::FRAC_1_SQRT_2).round() as i32
        } else {
            self.step_size
        };
        Critter {
            x: self.x - dx * offset,
            y: self.y - dy * offset,
            heading: self.heading,
            instructions: self.instructions.clone(),
            next_instruction_index: self.next_instruction_index,
            last_executed: None,
            ticks_per_instruction: self.ticks_per_instruction,
            tick_counter: 0,
            step_size: self.step_size,
            energy: self.energy / 2,
            initial_energy: self.initial_energy,
        }
    }
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
            let critter = Critter::new(START_X, START_Y, Heading::North, vec![], 1, 1, u32::MAX);

            assert_eq!((critter.x(), critter.y()), (START_X, START_Y));
        }

        #[test]
        fn it_starts_with_the_given_heading() {
            let critter = Critter::new(START_X, START_Y, Heading::North, vec![], 1, 1, u32::MAX);

            assert_eq!(critter.heading(), Heading::North);
        }
    }

    mod tick {
        use super::*;

        #[test]
        fn it_does_not_execute_an_instruction_before_n_ticks_have_passed() {
            let mut critter = Critter::new(
                START_X,
                START_Y,
                Heading::North,
                vec![Instruction::MoveForward],
                TICKS_PER_INSTRUCTION,
                1,
                u32::MAX,
            );

            for _ in 0..(TICKS_PER_INSTRUCTION - 1) {
                critter.tick();
            }

            assert_eq!(critter.y(), START_Y);
        }

        #[test]
        fn it_executes_an_instruction_on_the_nth_tick() {
            let mut critter = Critter::new(
                START_X,
                START_Y,
                Heading::North,
                vec![Instruction::MoveForward],
                TICKS_PER_INSTRUCTION,
                1,
                u32::MAX,
            );

            for _ in 0..TICKS_PER_INSTRUCTION {
                critter.tick();
            }

            assert_eq!(critter.y(), START_Y - 1);
        }

        #[test]
        fn move_forward_moves_one_pixel_in_the_heading_direction_with_unit_step_size() {
            let mut critter = Critter::new(
                START_X,
                START_Y,
                Heading::East,
                vec![Instruction::MoveForward],
                1,
                1,
                u32::MAX,
            );

            critter.tick();

            assert_eq!((critter.x(), critter.y()), (START_X + 1, START_Y));
        }

        #[test]
        fn move_forward_advances_x_by_the_configured_step_size() {
            const STEP_SIZE: i32 = 25;
            let mut critter = Critter::new(
                START_X,
                START_Y,
                Heading::East,
                vec![Instruction::MoveForward],
                1,
                STEP_SIZE,
                u32::MAX,
            );

            critter.tick();

            assert_eq!(critter.x(), START_X + STEP_SIZE);
        }

        #[test]
        fn move_forward_advances_y_by_the_configured_step_size() {
            const STEP_SIZE: i32 = 25;
            let mut critter = Critter::new(
                START_X,
                START_Y,
                Heading::South,
                vec![Instruction::MoveForward],
                1,
                STEP_SIZE,
                u32::MAX,
            );

            critter.tick();

            assert_eq!(critter.y(), START_Y + STEP_SIZE);
        }

        #[test]
        fn move_forward_on_a_diagonal_scales_the_step_by_root_two_over_two() {
            // step_size 10, scaled by √2/2 ≈ 0.707 and rounded → 7.
            const STEP_SIZE: i32 = 10;
            const DIAGONAL_STEP: i32 = 7;
            let mut critter = Critter::new(
                START_X,
                START_Y,
                Heading::SouthEast,
                vec![Instruction::MoveForward],
                1,
                STEP_SIZE,
                u32::MAX,
            );

            critter.tick();

            assert_eq!(
                (critter.x(), critter.y()),
                (START_X + DIAGONAL_STEP, START_Y + DIAGONAL_STEP)
            );
        }

        #[test]
        fn move_forward_on_a_northwest_diagonal_subtracts_the_scaled_step_from_each_axis() {
            const STEP_SIZE: i32 = 10;
            const DIAGONAL_STEP: i32 = 7;
            let mut critter = Critter::new(
                START_X,
                START_Y,
                Heading::NorthWest,
                vec![Instruction::MoveForward],
                1,
                STEP_SIZE,
                u32::MAX,
            );

            critter.tick();

            assert_eq!(
                (critter.x(), critter.y()),
                (START_X - DIAGONAL_STEP, START_Y - DIAGONAL_STEP)
            );
        }

        #[test]
        fn turn_left_changes_the_heading() {
            let mut critter = Critter::new(
                START_X,
                START_Y,
                Heading::North,
                vec![Instruction::TurnLeft],
                1,
                1,
                u32::MAX,
            );

            critter.tick();

            assert_eq!(critter.heading(), Heading::NorthWest);
        }

        #[test]
        fn turn_left_does_not_change_the_position() {
            let mut critter = Critter::new(
                START_X,
                START_Y,
                Heading::North,
                vec![Instruction::TurnLeft],
                1,
                1,
                u32::MAX,
            );

            critter.tick();

            assert_eq!((critter.x(), critter.y()), (START_X, START_Y));
        }

        #[test]
        fn turn_right_changes_the_heading() {
            let mut critter = Critter::new(
                START_X,
                START_Y,
                Heading::North,
                vec![Instruction::TurnRight],
                1,
                1,
                u32::MAX,
            );

            critter.tick();

            assert_eq!(critter.heading(), Heading::NorthEast);
        }

        #[test]
        fn turn_right_does_not_change_the_position() {
            let mut critter = Critter::new(
                START_X,
                START_Y,
                Heading::North,
                vec![Instruction::TurnRight],
                1,
                1,
                u32::MAX,
            );

            critter.tick();

            assert_eq!((critter.x(), critter.y()), (START_X, START_Y));
        }

        #[test]
        fn do_nothing_leaves_position_unchanged() {
            let mut critter = Critter::new(
                START_X,
                START_Y,
                Heading::North,
                vec![Instruction::DoNothing],
                1,
                1,
                u32::MAX,
            );

            critter.tick();

            assert_eq!((critter.x(), critter.y()), (START_X, START_Y));
        }

        #[test]
        fn do_nothing_leaves_heading_unchanged() {
            let mut critter = Critter::new(
                START_X,
                START_Y,
                Heading::North,
                vec![Instruction::DoNothing],
                1,
                1,
                u32::MAX,
            );

            critter.tick();

            assert_eq!(critter.heading(), Heading::North);
        }

        #[test]
        fn each_tick_consumes_the_next_instruction_in_the_list() {
            let mut critter = Critter::new(
                START_X,
                START_Y,
                Heading::East,
                vec![Instruction::MoveForward, Instruction::TurnRight],
                1,
                1,
                u32::MAX,
            );

            critter.tick();
            critter.tick();

            assert_eq!(critter.x(), START_X + 1);
            assert_eq!(critter.heading(), Heading::SouthEast);
        }

        #[test]
        fn the_instruction_list_loops_when_exhausted() {
            let mut critter = Critter::new(
                START_X,
                START_Y,
                Heading::East,
                vec![Instruction::MoveForward],
                1,
                1,
                u32::MAX,
            );

            critter.tick();
            critter.tick();
            critter.tick();

            assert_eq!(critter.x(), START_X + 3);
        }
    }

    mod repeat_previous_move {
        use super::*;

        #[test]
        fn it_re_executes_the_previously_executed_instruction() {
            let mut critter = Critter::new(
                START_X,
                START_Y,
                Heading::East,
                vec![Instruction::MoveForward, Instruction::RepeatPreviousMove],
                1,
                1,
                u32::MAX,
            );

            critter.tick();
            critter.tick();

            assert_eq!(critter.x(), START_X + 2);
        }

        #[test]
        fn at_start_with_no_previous_move_it_does_not_change_position() {
            let mut critter = Critter::new(
                START_X,
                START_Y,
                Heading::East,
                vec![Instruction::RepeatPreviousMove],
                1,
                1,
                u32::MAX,
            );

            critter.tick();

            assert_eq!((critter.x(), critter.y()), (START_X, START_Y));
        }

        #[test]
        fn at_start_with_no_previous_move_it_does_not_change_heading() {
            let mut critter = Critter::new(
                START_X,
                START_Y,
                Heading::East,
                vec![Instruction::RepeatPreviousMove],
                1,
                1,
                u32::MAX,
            );

            critter.tick();

            assert_eq!(critter.heading(), Heading::East);
        }

        #[test]
        fn repeating_a_repeat_re_executes_the_underlying_move_a_third_time() {
            // After three ticks with [TurnRight, Repeat, Repeat] (each TurnRight = 45°):
            //   tick 1: TurnRight        — East -> SouthEast
            //   tick 2: Repeat -> Right  — SouthEast -> South
            //   tick 3: Repeat -> Right  — South -> SouthWest
            // The third tick must reach back through two repeats to find TurnRight.
            let mut critter = Critter::new(
                START_X,
                START_Y,
                Heading::East,
                vec![
                    Instruction::TurnRight,
                    Instruction::RepeatPreviousMove,
                    Instruction::RepeatPreviousMove,
                ],
                1,
                1,
                u32::MAX,
            );

            critter.tick();
            critter.tick();
            critter.tick();

            assert_eq!(critter.heading(), Heading::SouthWest);
        }
    }

    mod energy {
        use super::*;

        const INITIAL_ENERGY: u32 = 42;

        #[test]
        fn a_new_critter_starts_with_the_given_initial_energy() {
            let critter = Critter::new(
                START_X,
                START_Y,
                Heading::North,
                vec![],
                1,
                1,
                INITIAL_ENERGY,
            );

            assert_eq!(critter.energy(), INITIAL_ENERGY);
        }

        #[test]
        fn initial_energy_reports_the_value_passed_at_construction() {
            let critter = Critter::new(
                START_X,
                START_Y,
                Heading::North,
                vec![],
                1,
                1,
                INITIAL_ENERGY,
            );

            assert_eq!(critter.initial_energy(), INITIAL_ENERGY);
        }

        #[test]
        fn initial_energy_does_not_decrease_when_energy_is_consumed() {
            let mut critter = Critter::new(
                START_X,
                START_Y,
                Heading::North,
                vec![Instruction::DoNothing],
                1,
                1,
                INITIAL_ENERGY,
            );

            critter.tick();

            assert_eq!(critter.initial_energy(), INITIAL_ENERGY);
        }

        #[test]
        fn executing_an_instruction_decrements_energy_by_one() {
            let mut critter = Critter::new(
                START_X,
                START_Y,
                Heading::North,
                vec![Instruction::DoNothing],
                1,
                1,
                INITIAL_ENERGY,
            );

            critter.tick();

            assert_eq!(critter.energy(), INITIAL_ENERGY - 1);
        }

        #[test]
        fn ticks_before_the_threshold_do_not_decrement_energy() {
            let mut critter = Critter::new(
                START_X,
                START_Y,
                Heading::North,
                vec![Instruction::DoNothing],
                TICKS_PER_INSTRUCTION,
                1,
                INITIAL_ENERGY,
            );

            for _ in 0..(TICKS_PER_INSTRUCTION - 1) {
                critter.tick();
            }

            assert_eq!(critter.energy(), INITIAL_ENERGY);
        }

        #[test]
        fn at_zero_energy_move_forward_does_not_move_the_critter() {
            let mut critter = Critter::new(
                START_X,
                START_Y,
                Heading::East,
                vec![Instruction::MoveForward],
                1,
                1,
                0,
            );

            critter.tick();

            assert_eq!((critter.x(), critter.y()), (START_X, START_Y));
        }

        #[test]
        fn ticking_at_zero_energy_keeps_energy_at_zero() {
            let mut critter = Critter::new(
                START_X,
                START_Y,
                Heading::East,
                vec![Instruction::MoveForward],
                1,
                1,
                0,
            );

            for _ in 0..10 {
                critter.tick();
            }

            assert_eq!(critter.energy(), 0);
        }
    }

    mod wrap_position {
        use super::*;

        const WIDTH: i32 = 100;
        const HEIGHT: i32 = 100;

        fn make_critter_at(x: i32, y: i32) -> Critter {
            Critter::new(x, y, Heading::North, vec![], 1, 1, u32::MAX)
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

        const INITIAL_ENERGY: u32 = 60;

        fn splitter() -> Critter {
            Critter::new(
                START_X,
                START_Y,
                Heading::North,
                vec![Instruction::Split],
                1,
                1,
                INITIAL_ENERGY,
            )
        }

        #[test]
        fn ticking_a_split_instruction_returns_a_child_critter() {
            let mut critter = splitter();

            let child = critter.tick();

            assert!(child.is_some());
        }

        #[test]
        fn the_child_receives_half_of_the_parents_pre_split_energy() {
            let mut parent = splitter();

            let child = parent.tick().unwrap();

            assert_eq!(child.energy(), INITIAL_ENERGY / 2);
        }

        #[test]
        fn the_parent_keeps_half_of_its_pre_split_energy_minus_the_instruction_cost() {
            let mut parent = splitter();

            parent.tick();

            assert_eq!(parent.energy(), INITIAL_ENERGY / 2 - 1);
        }

        #[test]
        fn the_child_inherits_the_parents_heading() {
            let mut parent = splitter();

            let child = parent.tick().unwrap();

            assert_eq!(child.heading(), Heading::North);
        }

        #[test]
        fn the_child_inherits_the_parents_initial_energy() {
            let mut parent = splitter();

            let child = parent.tick().unwrap();

            assert_eq!(child.initial_energy(), INITIAL_ENERGY);
        }

        #[test]
        fn the_child_spawns_one_step_behind_the_parent() {
            // Parent facing North at (10, 10) with step_size 1: child should appear
            // one pixel south, at (10, 11) — directly behind.
            let mut parent = splitter();

            let child = parent.tick().unwrap();

            assert_eq!((child.x(), child.y()), (START_X, START_Y + 1));
        }

        #[test]
        fn a_child_spawned_facing_east_appears_one_step_west_of_the_parent() {
            const STEP_SIZE: i32 = 5;
            let mut parent = Critter::new(
                START_X,
                START_Y,
                Heading::East,
                vec![Instruction::Split],
                1,
                STEP_SIZE,
                INITIAL_ENERGY,
            );

            let child = parent.tick().unwrap();

            assert_eq!((child.x(), child.y()), (START_X - STEP_SIZE, START_Y));
        }

        #[test]
        fn a_child_spawned_facing_southeast_uses_the_diagonal_scaled_offset() {
            // step_size 10, scaled by √2/2 ≈ 0.707 and rounded → 7. Child appears
            // northwest of parent (opposite of SouthEast), so at (parent - 7, parent - 7).
            const STEP_SIZE: i32 = 10;
            const DIAGONAL_OFFSET: i32 = 7;
            let mut parent = Critter::new(
                START_X,
                START_Y,
                Heading::SouthEast,
                vec![Instruction::Split],
                1,
                STEP_SIZE,
                INITIAL_ENERGY,
            );

            let child = parent.tick().unwrap();

            assert_eq!(
                (child.x(), child.y()),
                (START_X - DIAGONAL_OFFSET, START_Y - DIAGONAL_OFFSET)
            );
        }

        #[test]
        fn the_child_can_tick_independently_after_being_spawned() {
            // The child should have its own tick_counter starting fresh, and inherit
            // the parent's instruction list — so it can move on its own.
            let mut parent = Critter::new(
                START_X,
                START_Y,
                Heading::East,
                vec![Instruction::Split, Instruction::MoveForward],
                1,
                1,
                INITIAL_ENERGY,
            );

            let mut child = parent.tick().unwrap();
            let initial_child_x = child.x();
            child.tick(); // executes the second instruction (MoveForward)

            assert_eq!(child.x(), initial_child_x + 1);
        }

        #[test]
        fn move_forward_does_not_produce_a_child() {
            let mut critter = Critter::new(
                START_X,
                START_Y,
                Heading::East,
                vec![Instruction::MoveForward],
                1,
                1,
                INITIAL_ENERGY,
            );

            assert!(critter.tick().is_none());
        }

        #[test]
        fn turn_left_does_not_produce_a_child() {
            let mut critter = Critter::new(
                START_X,
                START_Y,
                Heading::North,
                vec![Instruction::TurnLeft],
                1,
                1,
                INITIAL_ENERGY,
            );

            assert!(critter.tick().is_none());
        }

        #[test]
        fn do_nothing_does_not_produce_a_child() {
            let mut critter = Critter::new(
                START_X,
                START_Y,
                Heading::North,
                vec![Instruction::DoNothing],
                1,
                1,
                INITIAL_ENERGY,
            );

            assert!(critter.tick().is_none());
        }

        #[test]
        fn a_tick_before_the_threshold_does_not_produce_a_child() {
            let mut critter = Critter::new(
                START_X,
                START_Y,
                Heading::North,
                vec![Instruction::Split],
                10,
                1,
                INITIAL_ENERGY,
            );

            assert!(critter.tick().is_none());
        }

        #[test]
        fn a_split_at_one_energy_does_not_underflow() {
            // Pre-tick energy = 1: passes the energy>0 guard, Split halves to 0,
            // then the instruction-cost decrement must not underflow.
            let mut parent = Critter::new(
                START_X,
                START_Y,
                Heading::North,
                vec![Instruction::Split],
                1,
                1,
                1,
            );

            parent.tick();

            assert_eq!(parent.energy(), 0);
        }

        #[test]
        fn a_split_when_repeating_a_previous_split_still_returns_a_child() {
            let mut parent = Critter::new(
                START_X,
                START_Y,
                Heading::North,
                vec![Instruction::Split, Instruction::RepeatPreviousMove],
                1,
                1,
                INITIAL_ENERGY,
            );

            parent.tick(); // first split
            let second_child = parent.tick(); // repeat → split again

            assert!(second_child.is_some());
        }
    }
}
