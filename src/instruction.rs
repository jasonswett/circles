use rand::Rng;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Instruction {
    MoveSlow,
    /// Covers twice the ground of MoveSlow for four times the energy. Speed
    /// costs more than it returns, so hurrying is a gamble on getting
    /// somewhere worth the fare.
    MoveFast,
    RepeatPreviousMove,
    DoNothing,
    /// Turning is several instructions rather than one repeated, because a
    /// manoeuvre held in a single slot survives mutation where the same turn
    /// spread over six consecutive slots does not. Fine turns compose into
    /// anything the sharp ones do not cover.
    TurnLeft15,
    TurnLeft45,
    TurnLeft90,
    TurnRight15,
    TurnRight45,
    TurnRight90,
    /// A reversal. The same turn whichever way it is taken, so there is only
    /// one of it.
    TurnAbout,
    Split,
    Eat,
    /// Moves the playhead forward, skipping the slots between. Gated like any
    /// other instruction, so a critter's inputs decide whether it jumps.
    SkipAhead,
    /// Moves the playhead back toward earlier slots.
    SkipBack,
}

impl Instruction {
    pub fn random<R: Rng>(rng: &mut R) -> Self {
        // Each common instruction has weight 10; Split and Eat have weight 1
        // each. Total 52.
        match rng.gen_range(0..52) {
            0..10 => Instruction::MoveSlow,
            10..20 => Instruction::RepeatPreviousMove,
            20..30 => Instruction::DoNothing,
            30..40 => Instruction::TurnLeft15,
            40..50 => Instruction::TurnRight15,
            50 => Instruction::Split,
            _ => Instruction::Eat,
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
            for _ in 0..2000 {
                seen.insert(Instruction::random(&mut rng));
            }
            assert_eq!(seen.len(), 7);
        }

        #[test]
        fn rare_instructions_are_drawn_far_less_often_than_each_common_variant() {
            // Split and Eat each have weight 1 vs weight 10 for common
            // instructions. Each common instruction should appear at least 5x
            // more often than any rare instruction — tolerating rng noise but
            // failing decisively under a uniform draw.
            let mut rng = StdRng::seed_from_u64(0);
            let mut counts = std::collections::HashMap::new();
            for _ in 0..10_000 {
                *counts.entry(Instruction::random(&mut rng)).or_insert(0) += 1;
            }
            let rare_count = [Instruction::Split, Instruction::Eat]
                .into_iter()
                .map(|v| counts.get(&v).copied().unwrap_or(0))
                .max()
                .unwrap_or(0);
            for variant in [
                Instruction::MoveSlow,
                Instruction::RepeatPreviousMove,
                Instruction::DoNothing,
                Instruction::TurnLeft15,
                Instruction::TurnRight15,
            ] {
                let count = counts.get(&variant).copied().unwrap_or(0);
                assert!(
                    count >= rare_count * 5,
                    "{variant:?} count {count} should be at least 5× rare-instruction count {rare_count}"
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
