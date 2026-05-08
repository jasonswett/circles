use crate::Instruction;
use rand::Rng;

const GENOME_LENGTH: usize = 64;

/// A binary genome of instruction opcodes. Each entry is a 4-bit code
/// (interpretable as a nibble of a longer bitstring). Decoding a code yields
/// either a known `Instruction` or `DoNothing` for unknown codes.
#[derive(Clone, Debug)]
pub struct Genome {
    codes: Vec<u8>,
}

impl Genome {
    pub fn random<R: Rng>(rng: &mut R) -> Self {
        let codes = (0..GENOME_LENGTH).map(|_| rng.gen_range(0..16)).collect();
        Self { codes }
    }

    /// A genome where every code decodes to the given instruction. Useful for
    /// tests that need a deterministic instruction stream.
    #[cfg(test)]
    pub fn all(instruction: Instruction) -> Self {
        Self {
            codes: vec![encode(instruction); GENOME_LENGTH],
        }
    }

    /// A genome whose decoded sequence matches the given instruction list,
    /// repeated to fill the genome length. Cursor-walking it produces the
    /// instructions in order, then loops.
    #[cfg(test)]
    pub fn from_instructions(instructions: &[Instruction]) -> Self {
        assert!(!instructions.is_empty());
        let codes = instructions
            .iter()
            .copied()
            .cycle()
            .take(GENOME_LENGTH)
            .map(encode)
            .collect();
        Self { codes }
    }

    pub fn len(&self) -> usize {
        self.codes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.codes.is_empty()
    }

    pub fn decode_at(&self, cursor: usize) -> Instruction {
        decode(self.codes[cursor % self.codes.len()])
    }
}

#[cfg(test)]
fn encode(instruction: Instruction) -> u8 {
    match instruction {
        Instruction::TurnRight => 0b0001,
        Instruction::TurnLeft => 0b0010,
        Instruction::MoveForward => 0b0011,
        Instruction::DoNothing => 0b0100,
        Instruction::RepeatPreviousMove => 0b0101,
        Instruction::Split => 0b0110,
    }
}

// The 0b0100 → DoNothing arm is equivalent to the catch-all `_ => DoNothing`.
// Deleting it produces identical behavior, so cargo-mutants can't kill that
// mutation. The arm is kept for symmetry with the other instruction codes.
#[mutants::skip]
fn decode(code: u8) -> Instruction {
    match code {
        0b0001 => Instruction::TurnRight,
        0b0010 => Instruction::TurnLeft,
        0b0011 => Instruction::MoveForward,
        0b0100 => Instruction::DoNothing,
        0b0101 => Instruction::RepeatPreviousMove,
        0b0110 => Instruction::Split,
        _ => Instruction::DoNothing,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::SmallRng;
    use rand::SeedableRng;

    #[test]
    fn an_empty_codes_genome_reports_is_empty_true() {
        let genome = Genome { codes: vec![] };
        assert!(genome.is_empty());
    }

    #[test]
    fn a_random_genome_is_not_empty() {
        let mut rng = SmallRng::seed_from_u64(0);
        let genome = Genome::random(&mut rng);
        assert!(!genome.is_empty());
    }

    #[test]
    fn random_genome_has_the_configured_length() {
        let mut rng = SmallRng::seed_from_u64(0);

        let genome = Genome::random(&mut rng);

        assert_eq!(genome.len(), GENOME_LENGTH);
    }

    #[test]
    fn every_code_in_a_random_genome_is_a_four_bit_value() {
        let mut rng = SmallRng::seed_from_u64(0);
        let genome = Genome::random(&mut rng);

        for code in &genome.codes {
            assert!(*code < 16);
        }
    }

    #[test]
    fn decoding_a_turn_right_code_returns_turn_right() {
        let genome = Genome::all(Instruction::TurnRight);

        assert_eq!(genome.decode_at(0), Instruction::TurnRight);
    }

    #[test]
    fn decoding_a_split_code_returns_split() {
        let genome = Genome::all(Instruction::Split);

        assert_eq!(genome.decode_at(0), Instruction::Split);
    }

    #[test]
    fn unknown_codes_decode_to_do_nothing() {
        let genome = Genome {
            codes: vec![0b1111],
        };

        assert_eq!(genome.decode_at(0), Instruction::DoNothing);
    }

    #[test]
    fn decoding_at_a_cursor_beyond_the_genome_wraps_around() {
        let genome = Genome {
            codes: vec![encode(Instruction::TurnLeft), encode(Instruction::Split)],
        };

        assert_eq!(genome.decode_at(0), Instruction::TurnLeft);
        assert_eq!(genome.decode_at(1), Instruction::Split);
        assert_eq!(genome.decode_at(2), Instruction::TurnLeft);
        assert_eq!(genome.decode_at(3), Instruction::Split);
    }

    #[test]
    fn the_same_seed_produces_the_same_genome() {
        let mut rng_a = SmallRng::seed_from_u64(42);
        let mut rng_b = SmallRng::seed_from_u64(42);

        let a = Genome::random(&mut rng_a);
        let b = Genome::random(&mut rng_b);

        assert_eq!(a.codes, b.codes);
    }

    #[test]
    fn a_cloned_genome_decodes_identically_to_the_original() {
        let mut rng = SmallRng::seed_from_u64(0);
        let original = Genome::random(&mut rng);
        let cloned = original.clone();

        for cursor in 0..GENOME_LENGTH {
            assert_eq!(original.decode_at(cursor), cloned.decode_at(cursor));
        }
    }
}
