use rand::Rng;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Instruction {
    MoveForward,
    RepeatPreviousMove,
    DoNothing,
    TurnLeft,
    TurnRight,
}

impl Instruction {
    pub fn random<R: Rng>(rng: &mut R) -> Self {
        match rng.gen_range(0..5) {
            0 => Instruction::MoveForward,
            1 => Instruction::RepeatPreviousMove,
            2 => Instruction::DoNothing,
            3 => Instruction::TurnLeft,
            _ => Instruction::TurnRight,
        }
    }

    pub fn random_list<R: Rng>(rng: &mut R, length: usize) -> Vec<Instruction> {
        (0..length).map(|_| Self::random(rng)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    mod random {
        use super::*;

        #[test]
        fn it_returns_one_of_the_five_variants() {
            let mut rng = StdRng::seed_from_u64(0);
            let instruction = Instruction::random(&mut rng);
            let valid = [
                Instruction::MoveForward,
                Instruction::RepeatPreviousMove,
                Instruction::DoNothing,
                Instruction::TurnLeft,
                Instruction::TurnRight,
            ];
            assert!(valid.contains(&instruction));
        }

        #[test]
        fn over_many_draws_every_variant_appears() {
            let mut rng = StdRng::seed_from_u64(42);
            let mut seen = std::collections::HashSet::new();
            for _ in 0..1000 {
                seen.insert(Instruction::random(&mut rng));
            }
            assert_eq!(seen.len(), 5);
        }
    }

    mod random_list {
        use super::*;

        #[test]
        fn it_returns_a_list_of_the_requested_length() {
            let mut rng = StdRng::seed_from_u64(0);
            let list = Instruction::random_list(&mut rng, 32);
            assert_eq!(list.len(), 32);
        }

        #[test]
        fn an_empty_list_is_returned_when_length_is_zero() {
            let mut rng = StdRng::seed_from_u64(0);
            let list = Instruction::random_list(&mut rng, 0);
            assert!(list.is_empty());
        }
    }
}
