use crate::instruction::INSTRUCTION_COUNT;
use crate::Instruction;
use rand::Rng;

const MIN_SOFTNESS: f32 = 10.0;
const MAX_SOFTNESS: f32 = 100.0;

#[derive(Clone, Debug)]
pub struct Genome {
    rules: [Rule; INSTRUCTION_COUNT],
}

#[derive(Clone, Copy, Debug)]
struct Rule {
    threshold: f32,
    softness: f32,
}

impl Genome {
    /// A genome whose `probability_of_acting` is 1.0 for any instruction at any
    /// non-zero energy. Useful for tests that need deterministic execution.
    pub fn always_act() -> Self {
        Self {
            rules: [Rule {
                threshold: f32::NEG_INFINITY,
                softness: MIN_SOFTNESS,
            }; INSTRUCTION_COUNT],
        }
    }

    pub fn random<R: Rng>(rng: &mut R, max_energy: u32) -> Self {
        let max = max_energy as f32;
        let mut rules = [Rule {
            threshold: 0.0,
            softness: MIN_SOFTNESS,
        }; INSTRUCTION_COUNT];
        for rule in &mut rules {
            rule.threshold = rng.gen_range(0.0..=max);
            rule.softness = rng.gen_range(MIN_SOFTNESS..=MAX_SOFTNESS);
        }
        Self { rules }
    }

    pub fn probability_of_acting(&self, instruction: Instruction, energy: u32) -> f32 {
        let rule = &self.rules[instruction.index()];
        let z = (energy as f32 - rule.threshold) / rule.softness;
        sigmoid(z)
    }
}

fn sigmoid(z: f32) -> f32 {
    1.0 / (1.0 + (-z).exp())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::SmallRng;
    use rand::SeedableRng;

    const TEST_MAX_ENERGY: u32 = 500;

    fn random_genome(seed: u64) -> Genome {
        let mut rng = SmallRng::seed_from_u64(seed);
        Genome::random(&mut rng, TEST_MAX_ENERGY)
    }

    #[test]
    fn near_the_threshold_the_probability_of_acting_is_close_to_one_half() {
        // Probing at the integer-truncated threshold leaves at most one unit of
        // input error; with softness >= 10 that's at most ~0.025 of probability.
        let mut rng = SmallRng::seed_from_u64(0);
        let genome = Genome::random(&mut rng, TEST_MAX_ENERGY);

        let threshold = genome.rules[Instruction::Split.index()].threshold;
        let probability = genome.probability_of_acting(Instruction::Split, threshold as u32);

        assert!((probability - 0.5).abs() < 0.05);
    }

    #[test]
    fn far_above_the_threshold_the_probability_approaches_one() {
        let genome = random_genome(0);
        let threshold = genome.rules[Instruction::Split.index()].threshold;
        let softness = genome.rules[Instruction::Split.index()].softness;
        // 10 softness widths above the threshold makes sigmoid effectively 1.
        let energy = (threshold + 10.0 * softness) as u32;

        let probability = genome.probability_of_acting(Instruction::Split, energy);

        assert!(probability > 0.99);
    }

    #[test]
    fn far_below_the_threshold_the_probability_approaches_zero() {
        // Pick a seed whose Split threshold is well above zero so we have room.
        let genome = random_genome(7);
        let threshold = genome.rules[Instruction::Split.index()].threshold;
        let softness = genome.rules[Instruction::Split.index()].softness;
        // Energy 10 softness widths below the threshold (clamped at 0).
        let target = threshold - 10.0 * softness;
        let energy = if target < 0.0 { 0 } else { target as u32 };
        // Only meaningful if we're well below the threshold.
        if (threshold - energy as f32) / softness < 5.0 {
            return;
        }

        let probability = genome.probability_of_acting(Instruction::Split, energy);

        assert!(probability < 0.01);
    }

    #[test]
    fn random_thresholds_lie_within_the_zero_to_max_energy_range() {
        // Sample many seeds — every drawn threshold should be in [0, max].
        for seed in 0..20 {
            let genome = random_genome(seed);
            for rule in &genome.rules {
                assert!(rule.threshold >= 0.0);
                assert!(rule.threshold <= TEST_MAX_ENERGY as f32);
            }
        }
    }

    #[test]
    fn random_softness_values_are_at_least_the_minimum_softness() {
        for seed in 0..20 {
            let genome = random_genome(seed);
            for rule in &genome.rules {
                assert!(rule.softness >= MIN_SOFTNESS);
                assert!(rule.softness <= MAX_SOFTNESS);
            }
        }
    }

    #[test]
    fn different_instructions_have_independent_thresholds() {
        // The genome must look up a different rule for each instruction. With a
        // 6-rule genome drawn from a uniform distribution, different
        // instructions should very rarely collide on a probability — if every
        // instruction shared one rule (an indexing bug), they'd all match.
        let genome = random_genome(0);
        let energy = 250;
        let probabilities = [
            genome.probability_of_acting(Instruction::MoveForward, energy),
            genome.probability_of_acting(Instruction::RepeatPreviousMove, energy),
            genome.probability_of_acting(Instruction::DoNothing, energy),
            genome.probability_of_acting(Instruction::TurnLeft, energy),
            genome.probability_of_acting(Instruction::TurnRight, energy),
            genome.probability_of_acting(Instruction::Split, energy),
        ];
        // At least three distinct probabilities must be observed.
        let mut sorted = probabilities.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        sorted.dedup();
        assert!(sorted.len() >= 3);
    }

    #[test]
    fn the_same_seed_produces_the_same_genome() {
        let a = random_genome(42);
        let b = random_genome(42);

        for i in 0..INSTRUCTION_COUNT {
            assert_eq!(a.rules[i].threshold, b.rules[i].threshold);
            assert_eq!(a.rules[i].softness, b.rules[i].softness);
        }
    }

    #[test]
    fn a_cloned_genome_returns_the_same_probability_as_the_original() {
        let original = random_genome(0);
        let cloned = original.clone();

        for instruction in [
            Instruction::MoveForward,
            Instruction::RepeatPreviousMove,
            Instruction::DoNothing,
            Instruction::TurnLeft,
            Instruction::TurnRight,
            Instruction::Split,
        ] {
            for energy in [0, 100, 250, 400, 500] {
                assert_eq!(
                    original.probability_of_acting(instruction, energy),
                    cloned.probability_of_acting(instruction, energy),
                );
            }
        }
    }
}
