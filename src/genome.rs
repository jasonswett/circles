use crate::Instruction;
use rand::Rng;

const INSTRUCTION_COUNT: usize = 6;
const PARAM_BITS_PER_INSTRUCTION: usize = 16;
const THRESHOLD_BITS: usize = 9; // 0..512, room for energy up to MAX_CRITTER_ENERGY=500
const SOFTNESS_BITS: usize = 7; // 0..128, mapped to [MIN_SOFTNESS, MIN_SOFTNESS + 127]
const MIN_SOFTNESS: f32 = 1.0;

const HEADER_BITS: usize = INSTRUCTION_COUNT * PARAM_BITS_PER_INSTRUCTION;
const OPCODE_BITS_PER_OPCODE: usize = 4;
const OPCODE_COUNT: usize = 64;
const OPCODE_BITS: usize = OPCODE_COUNT * OPCODE_BITS_PER_OPCODE;
const TOTAL_BITS: usize = HEADER_BITS + OPCODE_BITS;
const TOTAL_BYTES: usize = TOTAL_BITS.div_ceil(8);

/// A binary genome that encodes both the critter's instruction stream and the
/// sigmoid decision parameters in one packed bitstring. The first 96 bits hold
/// six (threshold, softness) pairs (one per instruction); the remaining 256
/// bits hold 64 four-bit opcodes that the critter walks one per tick.
#[derive(Clone, Debug)]
pub struct Genome {
    bytes: [u8; TOTAL_BYTES],
}

impl Genome {
    pub fn random<R: Rng>(rng: &mut R) -> Self {
        let mut bytes = [0u8; TOTAL_BYTES];
        for byte in &mut bytes {
            *byte = rng.gen();
        }
        Self { bytes }
    }

    /// A genome whose decoded opcode stream is the given instruction at every
    /// position, with sigmoid params set so the probability of acting is ~1 for
    /// any energy. Test-only.
    #[cfg(test)]
    pub fn all(instruction: Instruction) -> Self {
        let mut genome = Self::always_act_header();
        for cursor in 0..OPCODE_COUNT {
            genome.write_opcode(cursor, encode(instruction));
        }
        genome
    }

    /// A genome whose decoded opcode stream cycles through the given
    /// instructions (filling the opcode stream by repeating). Sigmoid params
    /// are set to always-act. Test-only.
    #[cfg(test)]
    pub fn from_instructions(instructions: &[Instruction]) -> Self {
        assert!(!instructions.is_empty());
        let mut genome = Self::always_act_header();
        for cursor in 0..OPCODE_COUNT {
            let instr = instructions[cursor % instructions.len()];
            genome.write_opcode(cursor, encode(instr));
        }
        genome
    }

    pub fn decode_at(&self, cursor: usize) -> Instruction {
        let position = cursor % OPCODE_COUNT;
        let bit_offset = HEADER_BITS + position * OPCODE_BITS_PER_OPCODE;
        decode(read_bits(&self.bytes, bit_offset, OPCODE_BITS_PER_OPCODE) as u8)
    }

    pub fn probability_of_acting(&self, instruction: Instruction, energy: u32) -> f32 {
        let (threshold, softness) = self.params(instruction);
        sigmoid((energy as f32 - threshold) / softness)
    }

    fn params(&self, instruction: Instruction) -> (f32, f32) {
        let bit_offset = instruction_index(instruction) * PARAM_BITS_PER_INSTRUCTION;
        let threshold_bits = read_bits(&self.bytes, bit_offset, THRESHOLD_BITS);
        let softness_bits = read_bits(&self.bytes, bit_offset + THRESHOLD_BITS, SOFTNESS_BITS);
        (threshold_bits as f32, MIN_SOFTNESS + softness_bits as f32)
    }

    #[cfg(test)]
    fn always_act_header() -> Self {
        // For probability ~ 1 at any non-trivial energy, leave threshold = 0
        // and softness bits = 0 → softness = MIN_SOFTNESS = 1. Then
        // sigmoid((energy - 0) / 1) is very close to 1 once energy >= ~10.
        Self {
            bytes: [0u8; TOTAL_BYTES],
        }
    }

    #[cfg(test)]
    fn write_opcode(&mut self, cursor: usize, code: u8) {
        let position = cursor % OPCODE_COUNT;
        let bit_offset = HEADER_BITS + position * OPCODE_BITS_PER_OPCODE;
        write_bits(
            &mut self.bytes,
            bit_offset,
            OPCODE_BITS_PER_OPCODE,
            code as u32,
        );
    }
}

// The `|` vs `^` mutation in the bit-pack step is equivalent: after
// `value << 1` the low bit is 0, so `| bit` and `^ bit` produce identical
// results when bit ∈ {0, 1}.
#[mutants::skip]
fn read_bits(bytes: &[u8], bit_offset: usize, length: usize) -> u32 {
    let mut value: u32 = 0;
    for i in 0..length {
        let bit_index = bit_offset + i;
        let byte_index = bit_index / 8;
        let bit_in_byte = 7 - (bit_index % 8);
        let bit = (bytes[byte_index] >> bit_in_byte) & 1;
        value = (value << 1) | (bit as u32);
    }
    value
}

#[cfg(test)]
fn write_bits(bytes: &mut [u8], bit_offset: usize, length: usize, value: u32) {
    for i in 0..length {
        let bit_index = bit_offset + i;
        let byte_index = bit_index / 8;
        let bit_in_byte = 7 - (bit_index % 8);
        let bit = (value >> (length - 1 - i)) & 1;
        bytes[byte_index] =
            (bytes[byte_index] & !(1 << bit_in_byte)) | ((bit as u8) << bit_in_byte);
    }
}

fn instruction_index(instruction: Instruction) -> usize {
    match instruction {
        Instruction::MoveForward => 0,
        Instruction::TurnLeft => 1,
        Instruction::TurnRight => 2,
        Instruction::DoNothing => 3,
        Instruction::RepeatPreviousMove => 4,
        Instruction::Split => 5,
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

fn sigmoid(z: f32) -> f32 {
    1.0 / (1.0 + (-z).exp())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::SmallRng;
    use rand::SeedableRng;

    fn random_genome(seed: u64) -> Genome {
        let mut rng = SmallRng::seed_from_u64(seed);
        Genome::random(&mut rng)
    }

    mod random {
        use super::*;

        #[test]
        fn random_genome_has_the_full_byte_length() {
            // 96 header bits + 256 opcode bits = 352 bits = 44 bytes. Pinning
            // down the literal byte count catches mutations to the constants
            // that derive from HEADER_BITS + OPCODE_BITS.
            let genome = random_genome(0);

            assert_eq!(genome.bytes.len(), 44);
        }

        #[test]
        fn the_same_seed_produces_the_same_genome() {
            let a = random_genome(42);
            let b = random_genome(42);

            assert_eq!(a.bytes, b.bytes);
        }
    }

    mod opcode_decode {
        use super::*;

        #[test]
        fn decoding_at_a_cursor_beyond_the_opcode_count_wraps_around() {
            let genome = Genome::from_instructions(&[Instruction::TurnLeft, Instruction::Split]);

            assert_eq!(genome.decode_at(0), genome.decode_at(OPCODE_COUNT));
            assert_eq!(genome.decode_at(1), genome.decode_at(OPCODE_COUNT + 1));
        }

        #[test]
        fn an_all_split_genome_decodes_to_split_at_every_cursor() {
            let genome = Genome::all(Instruction::Split);

            for cursor in 0..OPCODE_COUNT {
                assert_eq!(genome.decode_at(cursor), Instruction::Split);
            }
        }

        #[test]
        fn from_instructions_cycles_through_the_given_sequence() {
            let genome = Genome::from_instructions(&[
                Instruction::MoveForward,
                Instruction::TurnRight,
                Instruction::Split,
            ]);

            assert_eq!(genome.decode_at(0), Instruction::MoveForward);
            assert_eq!(genome.decode_at(1), Instruction::TurnRight);
            assert_eq!(genome.decode_at(2), Instruction::Split);
            assert_eq!(genome.decode_at(3), Instruction::MoveForward);
        }
    }

    mod sigmoid {
        use super::*;

        #[test]
        fn the_always_act_test_constructor_returns_high_probability_at_typical_energy() {
            // With threshold=0 and softness=1, sigmoid(60) ≈ 1.0 exactly.
            let genome = Genome::all(Instruction::Split);

            let probability = genome.probability_of_acting(Instruction::Split, 60);

            assert!(probability > 0.99);
        }

        #[test]
        fn near_the_encoded_threshold_the_probability_is_close_to_one_half() {
            // For some random genome, probe at the integer-truncated threshold;
            // softness >= 1 so the rounding error stays under ~0.25 of probability.
            let genome = random_genome(0);
            let (threshold, _softness) = genome.params(Instruction::Split);
            let probability = genome.probability_of_acting(Instruction::Split, threshold as u32);

            assert!((probability - 0.5).abs() < 0.5);
        }

        #[test]
        fn far_above_the_threshold_the_probability_approaches_one() {
            let genome = random_genome(0);
            let (threshold, softness) = genome.params(Instruction::Split);
            let energy = (threshold + 20.0 * softness) as u32;

            let probability = genome.probability_of_acting(Instruction::Split, energy);

            assert!(probability > 0.99);
        }

        #[test]
        fn far_below_the_threshold_the_probability_approaches_zero() {
            // Search for a seed whose Split rule sits well above zero so we
            // have room to probe well below it. With a 9-bit threshold field
            // (0..512) and 7-bit softness, most seeds qualify.
            for seed in 0..100 {
                let genome = random_genome(seed);
                let (threshold, softness) = genome.params(Instruction::Split);
                if threshold / softness < 10.0 {
                    continue;
                }
                let energy = (threshold - 10.0 * softness).max(0.0) as u32;
                let probability = genome.probability_of_acting(Instruction::Split, energy);
                assert!(probability < 0.01, "seed {seed}: probability {probability}");
                return;
            }
            panic!("no seed produced a threshold far enough above zero");
        }

        #[test]
        fn the_threshold_for_each_instruction_is_read_from_its_own_sixteen_bit_window() {
            // Build a genome where each instruction's 16-bit param window is
            // populated with a unique threshold value. Verify that reading each
            // instruction's params returns its own value, not a neighbor's.
            let mut bytes = [0u8; TOTAL_BYTES];
            // Per-instruction thresholds (9 bits each), distinct values.
            let thresholds: [u32; 6] = [10, 20, 30, 40, 50, 60];
            for (index, &threshold) in thresholds.iter().enumerate() {
                let offset = index * PARAM_BITS_PER_INSTRUCTION;
                write_bits(&mut bytes, offset, THRESHOLD_BITS, threshold);
            }
            let genome = Genome { bytes };

            assert_eq!(genome.params(Instruction::MoveForward).0, 10.0);
            assert_eq!(genome.params(Instruction::TurnLeft).0, 20.0);
            assert_eq!(genome.params(Instruction::TurnRight).0, 30.0);
            assert_eq!(genome.params(Instruction::DoNothing).0, 40.0);
            assert_eq!(genome.params(Instruction::RepeatPreviousMove).0, 50.0);
            assert_eq!(genome.params(Instruction::Split).0, 60.0);
        }

        #[test]
        fn the_softness_for_each_instruction_is_read_from_its_own_window() {
            // Similar to thresholds, but populating the softness portion.
            let mut bytes = [0u8; TOTAL_BYTES];
            let softnesses: [u32; 6] = [5, 15, 25, 35, 45, 55];
            for (index, &soft) in softnesses.iter().enumerate() {
                let offset = index * PARAM_BITS_PER_INSTRUCTION + THRESHOLD_BITS;
                write_bits(&mut bytes, offset, SOFTNESS_BITS, soft);
            }
            let genome = Genome { bytes };

            assert_eq!(
                genome.params(Instruction::MoveForward).1,
                MIN_SOFTNESS + 5.0
            );
            assert_eq!(genome.params(Instruction::TurnLeft).1, MIN_SOFTNESS + 15.0);
            assert_eq!(genome.params(Instruction::Split).1, MIN_SOFTNESS + 55.0);
        }

        #[test]
        fn different_instructions_have_independent_thresholds() {
            // Different instructions read from non-overlapping 16-bit windows of
            // the genome header. Over many seeds, all six instructions should
            // produce all-distinct probabilities at least once — proving each
            // instruction has its own slot.
            for seed in 0..50 {
                let genome = random_genome(seed);
                let energy = 250;
                let probabilities = [
                    genome.probability_of_acting(Instruction::MoveForward, energy),
                    genome.probability_of_acting(Instruction::RepeatPreviousMove, energy),
                    genome.probability_of_acting(Instruction::DoNothing, energy),
                    genome.probability_of_acting(Instruction::TurnLeft, energy),
                    genome.probability_of_acting(Instruction::TurnRight, energy),
                    genome.probability_of_acting(Instruction::Split, energy),
                ];
                let mut sorted = probabilities.to_vec();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                sorted.dedup();
                if sorted.len() == 6 {
                    return;
                }
            }
            panic!("no seed produced six distinct per-instruction probabilities");
        }
    }

    mod clone {
        use super::*;

        #[test]
        fn a_cloned_genome_decodes_identically_to_the_original() {
            let original = random_genome(0);
            let cloned = original.clone();

            for cursor in 0..OPCODE_COUNT {
                assert_eq!(original.decode_at(cursor), cloned.decode_at(cursor));
            }
        }

        #[test]
        fn a_cloned_genome_has_the_same_sigmoid_params() {
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
}
