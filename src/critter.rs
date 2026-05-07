use crate::{Heading, Instruction};

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
}

impl Critter {
    pub fn new(
        x: i32,
        y: i32,
        heading: Heading,
        instructions: Vec<Instruction>,
        ticks_per_instruction: u32,
        step_size: i32,
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

    pub fn tick(&mut self) {
        self.tick_counter += 1;
        if self.tick_counter < self.ticks_per_instruction {
            return;
        }
        self.tick_counter = 0;

        if self.instructions.is_empty() {
            return;
        }

        let instruction = self.instructions[self.next_instruction_index];
        self.next_instruction_index =
            (self.next_instruction_index + 1) % self.instructions.len();

        self.execute(instruction);
    }

    fn execute(&mut self, instruction: Instruction) {
        match instruction {
            Instruction::MoveForward => {
                let (dx, dy) = self.heading.offset();
                self.x += dx * self.step_size;
                self.y += dy * self.step_size;
                self.last_executed = Some(Instruction::MoveForward);
            }
            Instruction::TurnLeft => {
                self.heading = self.heading.turn_left();
                self.last_executed = Some(Instruction::TurnLeft);
            }
            Instruction::TurnRight => {
                self.heading = self.heading.turn_right();
                self.last_executed = Some(Instruction::TurnRight);
            }
            Instruction::DoNothing => {
                self.last_executed = Some(Instruction::DoNothing);
            }
            Instruction::RepeatPreviousMove => {
                if let Some(previous) = self.last_executed {
                    self.execute(previous);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Heading, Instruction};

    fn make_critter(instructions: Vec<Instruction>) -> Critter {
        Critter::new(10, 10, Heading::North, instructions, 30, 1)
    }

    mod new {
        use super::*;

        #[test]
        fn it_starts_at_the_given_position() {
            let critter = make_critter(vec![Instruction::DoNothing]);
            assert_eq!(critter.x(), 10);
            assert_eq!(critter.y(), 10);
        }

        #[test]
        fn it_starts_with_the_given_heading() {
            let critter = make_critter(vec![Instruction::DoNothing]);
            assert_eq!(critter.heading(), Heading::North);
        }
    }

    mod tick {
        use super::*;

        #[test]
        fn it_does_not_execute_an_instruction_before_n_ticks_have_passed() {
            let mut critter = make_critter(vec![Instruction::MoveForward]);
            for _ in 0..29 {
                critter.tick();
            }
            assert_eq!(critter.y(), 10);
        }

        #[test]
        fn it_executes_an_instruction_on_the_nth_tick() {
            let mut critter = make_critter(vec![Instruction::MoveForward]);
            for _ in 0..30 {
                critter.tick();
            }
            assert_eq!(critter.y(), 9);
        }

        #[test]
        fn move_forward_moves_in_the_heading_direction() {
            let mut critter = Critter::new(10, 10, Heading::East, vec![Instruction::MoveForward], 1, 1);
            critter.tick();
            assert_eq!((critter.x(), critter.y()), (11, 10));
        }

        #[test]
        fn move_forward_advances_by_the_configured_step_size() {
            let mut critter = Critter::new(10, 10, Heading::East, vec![Instruction::MoveForward], 1, 25);
            critter.tick();
            assert_eq!((critter.x(), critter.y()), (35, 10));
        }

        #[test]
        fn turn_left_changes_the_heading_without_moving() {
            let mut critter = Critter::new(10, 10, Heading::North, vec![Instruction::TurnLeft], 1, 1);
            critter.tick();
            assert_eq!(critter.heading(), Heading::West);
            assert_eq!((critter.x(), critter.y()), (10, 10));
        }

        #[test]
        fn turn_right_changes_the_heading_without_moving() {
            let mut critter = Critter::new(10, 10, Heading::North, vec![Instruction::TurnRight], 1, 1);
            critter.tick();
            assert_eq!(critter.heading(), Heading::East);
            assert_eq!((critter.x(), critter.y()), (10, 10));
        }

        #[test]
        fn do_nothing_leaves_position_and_heading_unchanged() {
            let mut critter = Critter::new(10, 10, Heading::North, vec![Instruction::DoNothing], 1, 1);
            critter.tick();
            assert_eq!((critter.x(), critter.y()), (10, 10));
            assert_eq!(critter.heading(), Heading::North);
        }

        #[test]
        fn it_advances_to_the_next_instruction_each_execution() {
            let mut critter = Critter::new(
                10,
                10,
                Heading::East,
                vec![Instruction::MoveForward, Instruction::TurnRight],
                1,
                1,
            );
            critter.tick();
            critter.tick();
            assert_eq!(critter.x(), 11);
            assert_eq!(critter.heading(), Heading::South);
        }

        #[test]
        fn the_instruction_list_loops_when_exhausted() {
            let mut critter = Critter::new(
                10,
                10,
                Heading::East,
                vec![Instruction::MoveForward],
                1,
                1,
            );
            critter.tick();
            critter.tick();
            critter.tick();
            assert_eq!(critter.x(), 13);
        }
    }

    mod repeat_previous_move {
        use super::*;

        #[test]
        fn it_re_executes_the_previously_executed_instruction() {
            let mut critter = Critter::new(
                10,
                10,
                Heading::East,
                vec![Instruction::MoveForward, Instruction::RepeatPreviousMove],
                1,
                1,
            );
            critter.tick();
            critter.tick();
            assert_eq!(critter.x(), 12);
        }

        #[test]
        fn at_start_with_no_previous_move_it_does_nothing() {
            let mut critter = Critter::new(
                10,
                10,
                Heading::East,
                vec![Instruction::RepeatPreviousMove],
                1,
                1,
            );
            critter.tick();
            assert_eq!((critter.x(), critter.y()), (10, 10));
            assert_eq!(critter.heading(), Heading::East);
        }

        #[test]
        fn repeating_a_repeat_re_executes_the_underlying_move() {
            let mut critter = Critter::new(
                10,
                10,
                Heading::East,
                vec![
                    Instruction::TurnRight,
                    Instruction::RepeatPreviousMove,
                    Instruction::RepeatPreviousMove,
                ],
                1,
                1,
            );
            critter.tick();
            critter.tick();
            critter.tick();
            assert_eq!(critter.heading(), Heading::North);
        }
    }
}
