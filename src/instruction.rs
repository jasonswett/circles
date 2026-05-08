use rand::Rng;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Instruction {
    MoveForward,
    RepeatPreviousMove,
    DoNothing,
    TurnLeft,
    TurnRight,
    Split,
}

impl Instruction {
    pub fn random<R: Rng>(rng: &mut R) -> Self {
        // Each common instruction has weight 10; Split has weight 1. Total 51.
        match rng.gen_range(0..51) {
            0..10 => Instruction::MoveForward,
            10..20 => Instruction::RepeatPreviousMove,
            20..30 => Instruction::DoNothing,
            30..40 => Instruction::TurnLeft,
            40..50 => Instruction::TurnRight,
            _ => Instruction::Split,
        }
    }

    pub fn random_list<R: Rng>(rng: &mut R, length: usize) -> Vec<Instruction> {
        (0..length).map(|_| Self::random(rng)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    mod random {
        use super::*;

        #[test]
        fn over_many_draws_every_variant_appears() {
            let mut rng = StdRng::seed_from_u64(42);
            let mut seen = std::collections::HashSet::new();
            for _ in 0..1000 {
                seen.insert(Instruction::random(&mut rng));
            }
            assert_eq!(seen.len(), 6);
        }

        #[test]
        fn split_is_drawn_far_less_often_than_each_other_variant() {
            // Split should be ~1/10 the rate of each common instruction. Asserting
            // that every common instruction appears at least 5x more often than
            // Split tolerates rng noise but fails decisively under a uniform draw.
            let mut rng = StdRng::seed_from_u64(0);
            let mut counts = std::collections::HashMap::new();
            for _ in 0..10_000 {
                *counts.entry(Instruction::random(&mut rng)).or_insert(0) += 1;
            }
            let split_count = counts.get(&Instruction::Split).copied().unwrap_or(0);
            for variant in [
                Instruction::MoveForward,
                Instruction::RepeatPreviousMove,
                Instruction::DoNothing,
                Instruction::TurnLeft,
                Instruction::TurnRight,
            ] {
                let count = counts.get(&variant).copied().unwrap_or(0);
                assert!(
                    count >= split_count * 5,
                    "{variant:?} count {count} should be at least 5× Split count {split_count}"
                );
            }
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
