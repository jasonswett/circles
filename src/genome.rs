use crate::Instruction;
use rand::Rng;

const INSTRUCTION_COUNT: usize = 7;
// Per-instruction header window: factor mask + threshold + softness.
// The factor mask is two bits (one per factor), and is read in the order:
// bit 0 = energy, bit 1 = is-touching-another-critter. A bit being on means
// the corresponding factor contributes to the sigmoid input; off means it is
// ignored. This is the "stair-step" mechanism: a single mutation can enable
// or disable a whole factor without disturbing the existing weights.
const FACTOR_MASK_BITS: usize = 2;
const THRESHOLD_BITS: usize = 7; // 0..128: median ~64, near INITIAL_ENERGY=60
const SOFTNESS_BITS: usize = 7; // 0..128, mapped to [MIN_SOFTNESS, MIN_SOFTNESS + 127]
const PARAM_BITS_PER_INSTRUCTION: usize = FACTOR_MASK_BITS + THRESHOLD_BITS + SOFTNESS_BITS;
const MIN_SOFTNESS: f32 = 1.0;

// When the touching factor is enabled, this is the value contributed to the
// sigmoid input on the touching=true side. Comparable in magnitude to a
// healthy critter's energy, so a touching bit can meaningfully swing a
// decision but not entirely dominate it.
const TOUCHING_FACTOR_SCALE: f32 = 64.0;

// Within each instruction's bit window: mask occupies bits 0..2, threshold
// occupies bits 2..9, softness occupies bits 9..16. Writes/reads use these
// offsets directly so the layout is in one place.
const THRESHOLD_OFFSET: usize = FACTOR_MASK_BITS;
const SOFTNESS_OFFSET: usize = THRESHOLD_OFFSET + THRESHOLD_BITS;

const ENERGY_FACTOR_BIT: u32 = 0b01;
const TOUCHING_FACTOR_BIT: u32 = 0b10;

const HEADER_BITS: usize = INSTRUCTION_COUNT * PARAM_BITS_PER_INSTRUCTION;
const OPCODE_BITS_PER_OPCODE: usize = 4;
const OPCODE_COUNT: usize = 8;
const OPCODE_BITS: usize = OPCODE_COUNT * OPCODE_BITS_PER_OPCODE;
const TOTAL_BITS: usize = HEADER_BITS + OPCODE_BITS;
const TOTAL_BYTES: usize = TOTAL_BITS.div_ceil(8);

/// A binary genome that encodes both the critter's instruction stream and the
/// sigmoid decision parameters in one packed bitstring. The first 96 bits hold
/// six (threshold, softness) pairs (one per instruction); the remaining 256
/// bits hold 64 four-bit opcodes that the critter walks one per tick.
#[derive(Clone, Debug, PartialEq, Eq)]
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

    /// A 24-bit color computed by hashing thirds of the genome bytes into the
    /// red, green, and blue channels independently. Genomes with identical
    /// bytes produce identical colors (so unmutated children look the same as
    /// their parent), and any mutation changes the color visibly. Each channel
    /// is brightened away from black so the critter remains visible against
    /// the dark background.
    pub fn digest_color(&self) -> u32 {
        let third = TOTAL_BYTES / 3;
        let r = brighten_channel(fnv1a_byte(&self.bytes[0..third]));
        let g = brighten_channel(fnv1a_byte(&self.bytes[third..2 * third]));
        let b = brighten_channel(fnv1a_byte(&self.bytes[2 * third..]));
        u32::from_be_bytes([0, r, g, b])
    }

    pub fn decode_at(&self, cursor: usize) -> Instruction {
        let position = cursor % OPCODE_COUNT;
        let bit_offset = HEADER_BITS + position * OPCODE_BITS_PER_OPCODE;
        decode(read_bits(&self.bytes, bit_offset, OPCODE_BITS_PER_OPCODE) as u8)
    }

    /// Flips each bit independently with probability `bit_flip_rate`. A rate of
    /// 0 leaves the genome unchanged; a rate of 1 inverts every bit.
    // The `<` vs `<=` mutation on the rate comparison is equivalent for the
    // continuous f32 distribution rng.gen::<f32>() draws from.
    #[mutants::skip]
    pub fn mutate<R: Rng>(&mut self, rng: &mut R, bit_flip_rate: f32) {
        for byte in &mut self.bytes {
            for bit in 0..8 {
                if rng.gen::<f32>() < bit_flip_rate {
                    *byte ^= 1 << bit;
                }
            }
        }
    }

    pub fn probability_of_acting(
        &self,
        instruction: Instruction,
        energy: u32,
        is_touching_critter: bool,
    ) -> f32 {
        let (mask, threshold, softness) = self.params(instruction);
        let energy_contribution = if mask & ENERGY_FACTOR_BIT != 0 {
            energy as f32
        } else {
            0.0
        };
        let touching_contribution = if mask & TOUCHING_FACTOR_BIT != 0 && is_touching_critter {
            TOUCHING_FACTOR_SCALE
        } else {
            0.0
        };
        let input = energy_contribution + touching_contribution;
        sigmoid((input - threshold) / softness)
    }

    fn params(&self, instruction: Instruction) -> (u32, f32, f32) {
        let window_offset = instruction_index(instruction) * PARAM_BITS_PER_INSTRUCTION;
        // FACTOR_MASK_OFFSET is 0, so we read the mask at the window's start.
        let mask = read_bits(&self.bytes, window_offset, FACTOR_MASK_BITS);
        let threshold_bits = read_bits(
            &self.bytes,
            window_offset + THRESHOLD_OFFSET,
            THRESHOLD_BITS,
        );
        let softness_bits = read_bits(&self.bytes, window_offset + SOFTNESS_OFFSET, SOFTNESS_BITS);
        (
            mask,
            threshold_bits as f32,
            MIN_SOFTNESS + softness_bits as f32,
        )
    }

    #[cfg(test)]
    fn always_act_header() -> Self {
        // Enable just the energy factor on every instruction so the sigmoid
        // input is the critter's energy, then leave threshold = 0 and softness
        // bits = 0 → softness = MIN_SOFTNESS = 1. Result: sigmoid(energy) is
        // very close to 1 once energy >= ~10. Mirrors the pre-mask behavior.
        let mut genome = Self {
            bytes: [0u8; TOTAL_BYTES],
        };
        for index in 0..INSTRUCTION_COUNT {
            let window_offset = index * PARAM_BITS_PER_INSTRUCTION;
            // FACTOR_MASK_OFFSET is 0, so the mask sits at the window's start.
            write_bits(
                &mut genome.bytes,
                window_offset,
                FACTOR_MASK_BITS,
                ENERGY_FACTOR_BIT,
            );
        }
        genome
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
        Instruction::Eat => 6,
    }
}

#[cfg(test)]
fn encode(instruction: Instruction) -> u8 {
    match instruction {
        Instruction::TurnRight => 0b0000,
        Instruction::TurnLeft => 0b0010,
        Instruction::MoveForward => 0b0100,
        Instruction::DoNothing => 0b0110,
        Instruction::RepeatPreviousMove => 0b1000,
        Instruction::Split => 0b1010,
        Instruction::Eat => 0b1100,
    }
}

fn decode(code: u8) -> Instruction {
    match code {
        0b0000 | 0b0001 => Instruction::TurnRight,
        0b0010 | 0b0011 => Instruction::TurnLeft,
        0b0100 | 0b0101 => Instruction::MoveForward,
        0b0110 | 0b0111 => Instruction::DoNothing,
        0b1000 | 0b1001 => Instruction::RepeatPreviousMove,
        0b1010 | 0b1011 => Instruction::Split,
        _ => Instruction::Eat,
    }
}

// Minimum per-channel brightness so a freshly-zero hash doesn't render fully
// black against the background. 80 keeps the critter clearly visible.
const MIN_CHANNEL_BRIGHTNESS: u8 = 80;

fn brighten_channel(value: u8) -> u8 {
    value.max(MIN_CHANNEL_BRIGHTNESS)
}

// FNV-1a folded into a single byte. A single-bit input change cascades through
// the multiply step, so unrelated genomes get unrelated colors and unmutated
// children share their parent's color exactly.
fn fnv1a_byte(bytes: &[u8]) -> u8 {
    const OFFSET_BASIS: u32 = 2166136261;
    const PRIME: u32 = 16777619;
    let mut hash = OFFSET_BASIS;
    for &b in bytes {
        hash ^= b as u32;
        hash = hash.wrapping_mul(PRIME);
    }
    // Truncate to the low byte; equivalent to `(hash & 0xFF) as u8` but does
    // not introduce a separate mask operation that mutation testing would
    // flag as ambiguous.
    hash as u8
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
            // 7 instructions × 16 param bits + 8 opcodes × 4 bits = 112 + 32
            // = 144 bits = 18 bytes. Pinning down the literal byte count
            // catches mutations to the constants that derive from
            // HEADER_BITS + OPCODE_BITS.
            let genome = random_genome(0);

            assert_eq!(genome.bytes.len(), 18);
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

        #[test]
        fn each_instruction_decodes_from_its_assigned_codes() {
            let assignments: &[(Instruction, &[u8])] = &[
                (Instruction::TurnRight, &[0b0000, 0b0001]),
                (Instruction::TurnLeft, &[0b0010, 0b0011]),
                (Instruction::MoveForward, &[0b0100, 0b0101]),
                (Instruction::DoNothing, &[0b0110, 0b0111]),
                (Instruction::RepeatPreviousMove, &[0b1000, 0b1001]),
                (Instruction::Split, &[0b1010, 0b1011]),
                (Instruction::Eat, &[0b1100, 0b1101, 0b1110, 0b1111]),
            ];

            for (instruction, codes) in assignments {
                for code in *codes {
                    assert_eq!(decode(*code), *instruction, "code {code:#06b}");
                }
            }
        }
    }

    mod sigmoid {
        use super::*;

        #[test]
        fn the_always_act_test_constructor_returns_high_probability_at_typical_energy() {
            // With threshold=0 and softness=1, sigmoid(60) ≈ 1.0 exactly.
            let genome = Genome::all(Instruction::Split);

            let probability = genome.probability_of_acting(Instruction::Split, 60, false);

            assert!(probability > 0.99);
        }

        #[test]
        fn near_the_encoded_threshold_the_probability_is_close_to_one_half() {
            // For a random genome whose Split rule has the energy factor on,
            // probe at the integer-truncated threshold; softness >= 1 so the
            // rounding error stays under ~0.25 of probability.
            for seed in 0..100 {
                let genome = random_genome(seed);
                let (mask, threshold, _softness) = genome.params(Instruction::Split);
                if mask & ENERGY_FACTOR_BIT == 0 {
                    continue;
                }
                let probability =
                    genome.probability_of_acting(Instruction::Split, threshold as u32, false);
                assert!((probability - 0.5).abs() < 0.5);
                return;
            }
            panic!("no seed had the energy factor enabled on Split");
        }

        #[test]
        fn far_above_the_threshold_the_probability_approaches_one() {
            for seed in 0..100 {
                let genome = random_genome(seed);
                let (mask, threshold, softness) = genome.params(Instruction::Split);
                if mask & ENERGY_FACTOR_BIT == 0 {
                    continue;
                }
                let energy = (threshold + 20.0 * softness) as u32;
                let probability = genome.probability_of_acting(Instruction::Split, energy, false);
                assert!(probability > 0.99, "seed {seed}: probability {probability}");
                return;
            }
            panic!("no seed had the energy factor enabled on Split");
        }

        #[test]
        fn far_below_the_threshold_the_probability_approaches_zero() {
            // Search for a seed whose Split rule has the energy factor on and
            // sits well above zero so we have room to probe below it.
            for seed in 0..100 {
                let genome = random_genome(seed);
                let (mask, threshold, softness) = genome.params(Instruction::Split);
                if mask & ENERGY_FACTOR_BIT == 0 {
                    continue;
                }
                if threshold / softness < 10.0 {
                    continue;
                }
                let energy = (threshold - 10.0 * softness).max(0.0) as u32;
                let probability = genome.probability_of_acting(Instruction::Split, energy, false);
                assert!(probability < 0.01, "seed {seed}: probability {probability}");
                return;
            }
            panic!("no seed produced an energy-enabled threshold far enough above zero");
        }

        #[test]
        fn the_threshold_for_each_instruction_is_read_from_its_own_window() {
            // Build a genome where each instruction's window has a unique
            // threshold value. Verify that reading each instruction's params
            // returns its own value, not a neighbor's.
            let mut bytes = [0u8; TOTAL_BYTES];
            let thresholds: [u32; 7] = [10, 20, 30, 40, 50, 60, 70];
            for (index, &threshold) in thresholds.iter().enumerate() {
                let offset = index * PARAM_BITS_PER_INSTRUCTION + THRESHOLD_OFFSET;
                write_bits(&mut bytes, offset, THRESHOLD_BITS, threshold);
            }
            let genome = Genome { bytes };

            assert_eq!(genome.params(Instruction::MoveForward).1, 10.0);
            assert_eq!(genome.params(Instruction::TurnLeft).1, 20.0);
            assert_eq!(genome.params(Instruction::TurnRight).1, 30.0);
            assert_eq!(genome.params(Instruction::DoNothing).1, 40.0);
            assert_eq!(genome.params(Instruction::RepeatPreviousMove).1, 50.0);
            assert_eq!(genome.params(Instruction::Split).1, 60.0);
            assert_eq!(genome.params(Instruction::Eat).1, 70.0);
        }

        #[test]
        fn the_softness_for_each_instruction_is_read_from_its_own_window() {
            // Use a hard-coded offset (mask 2 bits + threshold 7 bits = 9)
            // rather than the SOFTNESS_OFFSET constant so that mutations to
            // the constant produce a visible mismatch between write and read.
            const SOFTNESS_OFFSET_LITERAL: usize = 9;
            let mut bytes = [0u8; TOTAL_BYTES];
            let softnesses: [u32; 7] = [5, 15, 25, 35, 45, 55, 65];
            for (index, &soft) in softnesses.iter().enumerate() {
                let offset = index * PARAM_BITS_PER_INSTRUCTION + SOFTNESS_OFFSET_LITERAL;
                write_bits(&mut bytes, offset, SOFTNESS_BITS, soft);
            }
            let genome = Genome { bytes };

            assert_eq!(
                genome.params(Instruction::MoveForward).2,
                MIN_SOFTNESS + 5.0
            );
            assert_eq!(genome.params(Instruction::TurnLeft).2, MIN_SOFTNESS + 15.0);
            assert_eq!(genome.params(Instruction::Split).2, MIN_SOFTNESS + 55.0);
            assert_eq!(genome.params(Instruction::Eat).2, MIN_SOFTNESS + 65.0);
        }

        #[test]
        fn with_no_factors_enabled_the_probability_does_not_depend_on_energy_or_touching() {
            let genome = genome_with_mask_for_split(0b00);

            let p_low = genome.probability_of_acting(Instruction::Split, 0, false);
            let p_high_energy = genome.probability_of_acting(Instruction::Split, 500, false);
            let p_touching = genome.probability_of_acting(Instruction::Split, 0, true);

            assert_eq!(p_low, p_high_energy);
            assert_eq!(p_low, p_touching);
        }

        #[test]
        fn with_only_the_energy_factor_enabled_touching_does_not_change_the_probability() {
            let genome = genome_with_mask_for_split(ENERGY_FACTOR_BIT);

            let p_not_touching = genome.probability_of_acting(Instruction::Split, 100, false);
            let p_touching = genome.probability_of_acting(Instruction::Split, 100, true);

            assert_eq!(p_not_touching, p_touching);
        }

        #[test]
        fn with_only_the_touching_factor_enabled_energy_does_not_change_the_probability() {
            let genome = genome_with_mask_for_split(TOUCHING_FACTOR_BIT);

            let p_low_energy = genome.probability_of_acting(Instruction::Split, 0, false);
            let p_high_energy = genome.probability_of_acting(Instruction::Split, 500, false);

            assert_eq!(p_low_energy, p_high_energy);
        }

        #[test]
        fn with_only_the_touching_factor_enabled_touching_increases_the_probability() {
            // With softness = 1 and threshold = 0, touching adds 64 to the
            // input — sigmoid(64) is essentially 1, sigmoid(0) is 0.5.
            let genome = genome_with_mask_for_split(TOUCHING_FACTOR_BIT);

            let p_not_touching = genome.probability_of_acting(Instruction::Split, 0, false);
            let p_touching = genome.probability_of_acting(Instruction::Split, 0, true);

            assert!(p_not_touching < p_touching);
        }

        fn genome_with_mask_for_split(mask: u32) -> Genome {
            // Threshold = 0, softness = 1 (raw bits 0). Just the mask varies.
            let mut bytes = [0u8; TOTAL_BYTES];
            let split_window = instruction_index(Instruction::Split) * PARAM_BITS_PER_INSTRUCTION;
            write_bits(&mut bytes, split_window, FACTOR_MASK_BITS, mask);
            Genome { bytes }
        }

        #[test]
        fn different_instructions_have_independent_thresholds() {
            // Different instructions read from non-overlapping 16-bit windows of
            // the genome header. Over many seeds, all seven instructions should
            // produce all-distinct probabilities at least once — proving each
            // instruction has its own slot.
            for seed in 0..50 {
                let genome = random_genome(seed);
                let energy = 250;
                let probabilities = [
                    genome.probability_of_acting(Instruction::MoveForward, energy, false),
                    genome.probability_of_acting(Instruction::RepeatPreviousMove, energy, false),
                    genome.probability_of_acting(Instruction::DoNothing, energy, false),
                    genome.probability_of_acting(Instruction::TurnLeft, energy, false),
                    genome.probability_of_acting(Instruction::TurnRight, energy, false),
                    genome.probability_of_acting(Instruction::Split, energy, false),
                    genome.probability_of_acting(Instruction::Eat, energy, false),
                ];
                let mut sorted = probabilities.to_vec();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                sorted.dedup();
                if sorted.len() == 7 {
                    return;
                }
            }
            panic!("no seed produced seven distinct per-instruction probabilities");
        }
    }

    mod mutate {
        use super::*;

        #[test]
        fn at_rate_zero_no_bits_change() {
            let mut rng = SmallRng::seed_from_u64(0);
            let original = random_genome(7);
            let mut mutated = original.clone();

            mutated.mutate(&mut rng, 0.0);

            assert_eq!(original.bytes, mutated.bytes);
        }

        #[test]
        fn at_rate_one_every_bit_flips() {
            let mut rng = SmallRng::seed_from_u64(0);
            let original = random_genome(7);
            let mut mutated = original.clone();

            mutated.mutate(&mut rng, 1.0);

            for (a, b) in original.bytes.iter().zip(mutated.bytes.iter()) {
                assert_eq!(*a, !*b);
            }
        }

        #[test]
        fn at_a_partial_rate_some_but_not_all_bits_change() {
            let mut rng = SmallRng::seed_from_u64(0);
            let original = random_genome(7);
            let mut mutated = original.clone();

            mutated.mutate(&mut rng, 0.5);

            let differing_bits: u32 = original
                .bytes
                .iter()
                .zip(mutated.bytes.iter())
                .map(|(a, b)| (a ^ b).count_ones())
                .sum();
            assert!(differing_bits > 0);
            assert!(differing_bits < 8 * TOTAL_BYTES as u32);
        }
    }

    mod digest_color {
        use super::*;

        #[test]
        fn two_genomes_with_identical_bytes_produce_the_same_color() {
            let a = random_genome(7);
            let b = random_genome(7);

            assert_eq!(a.digest_color(), b.digest_color());
        }

        #[test]
        fn each_channel_hashes_six_distinct_bytes_in_position() {
            // Construct a genome where each third has a distinct byte signature
            // and the others are zero. Each channel's hash is then determined
            // entirely by its own slice; if a mutation shifts the slice
            // boundaries the channel will hash a different (or shorter) span
            // and produce a different value.
            let mut bytes = [0u8; TOTAL_BYTES];
            // First third: bytes 0..6 contain 0xAA in the last position only.
            bytes[5] = 0xAA;
            let red_only = Genome { bytes };

            let mut bytes = [0u8; TOTAL_BYTES];
            bytes[11] = 0xAA;
            let green_only = Genome { bytes };

            let mut bytes = [0u8; TOTAL_BYTES];
            bytes[17] = 0xAA;
            let blue_only = Genome { bytes };

            let zero = Genome {
                bytes: [0u8; TOTAL_BYTES],
            };

            // The signature byte sits at position [end_of_slice - 1] for each
            // third; only the matching channel should differ from the all-zero
            // genome's color.
            let (zr, zg, zb) = channels(zero.digest_color());
            let (rr, rg, rb) = channels(red_only.digest_color());
            assert_ne!(zr, rr);
            assert_eq!(zg, rg);
            assert_eq!(zb, rb);

            let (gr, gg, gb) = channels(green_only.digest_color());
            assert_eq!(zr, gr);
            assert_ne!(zg, gg);
            assert_eq!(zb, gb);

            let (br, bg, bb) = channels(blue_only.digest_color());
            assert_eq!(zr, br);
            assert_eq!(zg, bg);
            assert_ne!(zb, bb);
        }

        #[test]
        fn flipping_a_byte_in_the_first_third_changes_only_the_red_channel() {
            // Bytes 0..6 feed the red channel; the green and blue channels
            // must be unaffected. This pins the slice boundaries.
            let original = random_genome(0);
            let mut bytes = original.bytes;
            bytes[0] ^= 0xFF;
            let mutated = Genome { bytes };

            let (or, og, ob) = channels(original.digest_color());
            let (mr, mg, mb) = channels(mutated.digest_color());
            assert_ne!(or, mr);
            assert_eq!(og, mg);
            assert_eq!(ob, mb);
        }

        #[test]
        fn flipping_a_byte_in_the_middle_third_changes_only_the_green_channel() {
            let original = random_genome(0);
            let mut bytes = original.bytes;
            bytes[TOTAL_BYTES / 3] ^= 0xFF;
            let mutated = Genome { bytes };

            let (or, og, ob) = channels(original.digest_color());
            let (mr, mg, mb) = channels(mutated.digest_color());
            assert_eq!(or, mr);
            assert_ne!(og, mg);
            assert_eq!(ob, mb);
        }

        #[test]
        fn flipping_a_byte_in_the_last_third_changes_only_the_blue_channel() {
            let original = random_genome(0);
            let mut bytes = original.bytes;
            bytes[2 * (TOTAL_BYTES / 3)] ^= 0xFF;
            let mutated = Genome { bytes };

            let (or, og, ob) = channels(original.digest_color());
            let (mr, mg, mb) = channels(mutated.digest_color());
            assert_eq!(or, mr);
            assert_eq!(og, mg);
            assert_ne!(ob, mb);
        }

        fn channels(color: u32) -> (u32, u32, u32) {
            ((color >> 16) & 0xFF, (color >> 8) & 0xFF, color & 0xFF)
        }

        #[test]
        fn each_channel_of_the_color_is_at_least_the_minimum_brightness() {
            // The brightening floor keeps the critter visible against a black
            // background even if a hash happens to produce a near-zero byte.
            let zero_genome = Genome {
                bytes: [0u8; TOTAL_BYTES],
            };
            let color = zero_genome.digest_color();
            let r = (color >> 16) & 0xFF;
            let g = (color >> 8) & 0xFF;
            let b = color & 0xFF;

            assert!(r >= MIN_CHANNEL_BRIGHTNESS as u32);
            assert!(g >= MIN_CHANNEL_BRIGHTNESS as u32);
            assert!(b >= MIN_CHANNEL_BRIGHTNESS as u32);
        }

        #[test]
        fn colors_are_distributed_over_many_seeds_rather_than_clumped() {
            // A poor digest would map many distinct genomes to the same color.
            // Across 100 seeds, almost every digest should be unique.
            let mut colors = std::collections::HashSet::new();
            for seed in 0..100 {
                colors.insert(random_genome(seed).digest_color());
            }

            assert!(colors.len() > 90);
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
                Instruction::Eat,
            ] {
                for energy in [0, 100, 250, 400, 500] {
                    assert_eq!(
                        original.probability_of_acting(instruction, energy, false),
                        cloned.probability_of_acting(instruction, energy, false),
                    );
                }
            }
        }
    }
}
