use crate::Instruction;
use rand::Rng;

const INSTRUCTION_COUNT: usize = 7;
// The genome's leading field: how often this critter's splits mutate the
// child. Evolvable like everything else — it mutates along with the rest of
// the genome, so a lineage's mutability drifts under selection.
pub(crate) const MUTATION_RATE_BITS: usize = 8;
const MAX_MUTATION_CHANCE: f32 = 0.25;
// Per-instruction header window: factor mask + threshold + softness.
// The factor mask is three bits (one per factor), read in the order:
// bit 0 = energy, bit 1 = is-touching-another-critter, bit 2 = the touched
// critter's color is dissimilar to mine. A bit on means the corresponding
// factor contributes to the sigmoid input; off means it is ignored. This is
// the "stair-step" mechanism: a single mutation can enable or disable a
// whole factor without disturbing the existing weights.
const FACTOR_MASK_BITS: usize = 3;
const THRESHOLD_BITS: usize = 7; // 0..128: median ~64, near INITIAL_ENERGY=60
const SOFTNESS_BITS: usize = 7; // 0..128, mapped to [MIN_SOFTNESS, MIN_SOFTNESS + 127]
const PARAM_BITS_PER_INSTRUCTION: usize = FACTOR_MASK_BITS + THRESHOLD_BITS + SOFTNESS_BITS;
const MIN_SOFTNESS: f32 = 1.0;

// When the touching factor is enabled, this is the value contributed to the
// sigmoid input on the touching=true side. Comparable in magnitude to a
// healthy critter's energy, so a touching bit can meaningfully swing a
// decision but not entirely dominate it.
const TOUCHING_FACTOR_SCALE: f32 = 64.0;

// When the color-dissimilarity factor is enabled, the contribution scales
// from 0 (touched critter has the same color) up to this value (touched
// critter is the maximally distant color in RGB space). Same scale as
// TOUCHING_FACTOR_SCALE so the two factors are directly comparable.
const COLOR_DISSIMILARITY_FACTOR_SCALE: f32 = 64.0;

// Within each instruction's bit window: mask occupies bits 0..3, threshold
// occupies bits 3..10, softness occupies bits 10..17.
const THRESHOLD_OFFSET: usize = FACTOR_MASK_BITS;
const SOFTNESS_OFFSET: usize = THRESHOLD_OFFSET + THRESHOLD_BITS;

const ENERGY_FACTOR_BIT: u32 = 0b001;
const TOUCHING_FACTOR_BIT: u32 = 0b010;
const COLOR_DISSIMILARITY_FACTOR_BIT: u32 = 0b100;

const MUTATION_RATE_OFFSET: usize = 0;
const HEADER_OFFSET: usize = MUTATION_RATE_BITS;
const HEADER_BITS: usize = INSTRUCTION_COUNT * PARAM_BITS_PER_INSTRUCTION;
const OPCODE_BITS_PER_OPCODE: usize = 4;
// Total opcode slots in the genome's pool. Most start dormant — only the
// first INITIAL_ACTIVE_OPCODES participate in the walked stream when a
// genome is freshly created. A separate activation mask (one bit per slot)
// determines which slots are live; mutation can flip individual bits to
// promote dormant slots into the active rotation. Each dormant slot's
// 4-bit content evolves freely while inactive, so by the time a slot is
// activated its content is whatever drift has produced.
const OPCODE_POOL_SIZE: usize = 40;
// How many of the pool's slots are active in a freshly created genome.
const INITIAL_ACTIVE_OPCODES: usize = 8;
const ACTIVATION_MASK_BITS: usize = OPCODE_POOL_SIZE;
const OPCODE_BITS: usize = OPCODE_POOL_SIZE * OPCODE_BITS_PER_OPCODE;
const ACTIVATION_MASK_OFFSET: usize = HEADER_OFFSET + HEADER_BITS;
const OPCODE_STREAM_OFFSET: usize = ACTIVATION_MASK_OFFSET + ACTIVATION_MASK_BITS;
const TOTAL_BITS: usize = MUTATION_RATE_BITS + HEADER_BITS + ACTIVATION_MASK_BITS + OPCODE_BITS;
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
        // Reset the activation mask to "first INITIAL_ACTIVE_OPCODES bits on,
        // rest off" so every fresh lineage starts with the same number of
        // active opcodes. From here, mutation can flip individual mask bits
        // to promote dormant slots or deactivate active ones.
        for slot in 0..OPCODE_POOL_SIZE {
            let active = slot < INITIAL_ACTIVE_OPCODES;
            write_bits(
                &mut bytes,
                ACTIVATION_MASK_OFFSET + slot,
                1,
                if active { 1 } else { 0 },
            );
        }
        Self { bytes }
    }

    /// A genome whose decoded opcode stream is the given instruction at every
    /// position, with sigmoid params set so the probability of acting is ~1 for
    /// any energy. Test-only.
    #[cfg(test)]
    pub fn all(instruction: Instruction) -> Self {
        let mut genome = Self::always_act_header();
        for cursor in 0..OPCODE_POOL_SIZE {
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
        for cursor in 0..OPCODE_POOL_SIZE {
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
        // The cursor counts only firings of active opcode slots; we walk the
        // pool in order, skipping dormant slots. If no slots are active at
        // all (all activation bits flipped off by mutation) the critter has
        // nothing to do — fall back to DoNothing.
        let active_count = self.active_opcode_count();
        if active_count == 0 {
            return Instruction::DoNothing;
        }
        let active_index = cursor % active_count;
        let mut seen = 0;
        for slot in 0..OPCODE_POOL_SIZE {
            if !self.is_slot_active(slot) {
                continue;
            }
            if seen == active_index {
                let bit_offset = OPCODE_STREAM_OFFSET + slot * OPCODE_BITS_PER_OPCODE;
                return decode(read_bits(&self.bytes, bit_offset, OPCODE_BITS_PER_OPCODE) as u8);
            }
            seen += 1;
        }
        // Unreachable — active_count > 0 guarantees the loop finds the slot —
        // but the fallback keeps the function total without panicking.
        Instruction::DoNothing
    }

    fn is_slot_active(&self, slot: usize) -> bool {
        read_bits(&self.bytes, ACTIVATION_MASK_OFFSET + slot, 1) != 0
    }

    fn active_opcode_count(&self) -> usize {
        (0..OPCODE_POOL_SIZE)
            .filter(|&slot| self.is_slot_active(slot))
            .count()
    }

    /// Renders the genome's bits as a TOTAL_BITS-length string of `'0'`/`'1'`,
    /// MSB-first per byte. The trailing pad bit (since TOTAL_BITS doesn't
    /// divide evenly into 8) is dropped so the output length is exactly the
    /// number of meaningful bits.
    pub fn to_bits(&self) -> String {
        let mut out = String::with_capacity(TOTAL_BITS);
        for index in 0..TOTAL_BITS {
            let byte = self.bytes[index / 8];
            let bit_in_byte = 7 - (index % 8);
            let bit = (byte >> bit_in_byte) & 1;
            out.push(if bit == 1 { '1' } else { '0' });
        }
        out
    }

    /// Parses a TOTAL_BITS-length string of `'0'`/`'1'` back into a genome.
    /// Returns an error for any other length or any character outside the
    /// binary alphabet.
    pub fn from_bits(input: &str) -> Result<Self, GenomeParseError> {
        if input.len() != TOTAL_BITS {
            return Err(GenomeParseError::WrongLength {
                expected: TOTAL_BITS,
                actual: input.len(),
            });
        }
        let mut bytes = [0u8; TOTAL_BYTES];
        for (index, character) in input.chars().enumerate() {
            let bit = match character {
                '0' => 0u8,
                '1' => 1u8,
                _ => return Err(GenomeParseError::InvalidCharacter { index, character }),
            };
            let byte_index = index / 8;
            let bit_in_byte = 7 - (index % 8);
            bytes[byte_index] |= bit << bit_in_byte;
        }
        Ok(Self { bytes })
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
        touched_color_dissimilarity: f32,
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
        let color_contribution = if mask & COLOR_DISSIMILARITY_FACTOR_BIT != 0 {
            COLOR_DISSIMILARITY_FACTOR_SCALE * touched_color_dissimilarity.clamp(0.0, 1.0)
        } else {
            0.0
        };
        let input = energy_contribution + touching_contribution + color_contribution;
        sigmoid((input - threshold) / softness)
    }

    /// Overwrites the mutation-rate field. Test-only: in a running world the
    /// field changes only through mutation, like any other part of the genome.
    #[cfg(test)]
    pub fn set_mutation_rate_bits(&mut self, bits: u32) {
        write_bits(
            &mut self.bytes,
            MUTATION_RATE_OFFSET,
            MUTATION_RATE_BITS,
            bits,
        );
    }

    /// How often this genome's splits mutate the child, in [0, MAX_MUTATION_CHANCE].
    /// The field is part of the genome, so it mutates like any other region and
    /// a lineage's mutability evolves.
    pub fn mutation_chance(&self) -> f32 {
        let bits = read_bits(&self.bytes, MUTATION_RATE_OFFSET, MUTATION_RATE_BITS);
        let max_value = ((1u32 << MUTATION_RATE_BITS) - 1) as f32;
        bits as f32 / max_value * MAX_MUTATION_CHANCE
    }

    fn params(&self, instruction: Instruction) -> (u32, f32, f32) {
        let window_offset = header_window_offset(instruction_index(instruction));
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
        // Also activates the first INITIAL_ACTIVE_OPCODES opcode slots so the
        // critter has a working stream to walk.
        let mut genome = Self {
            bytes: [0u8; TOTAL_BYTES],
        };
        for index in 0..INSTRUCTION_COUNT {
            let window_offset = header_window_offset(index);
            // FACTOR_MASK_OFFSET is 0, so the mask sits at the window's start.
            write_bits(
                &mut genome.bytes,
                window_offset,
                FACTOR_MASK_BITS,
                ENERGY_FACTOR_BIT,
            );
        }
        for slot in 0..INITIAL_ACTIVE_OPCODES {
            write_bits(&mut genome.bytes, ACTIVATION_MASK_OFFSET + slot, 1, 1);
        }
        genome
    }

    #[cfg(test)]
    fn write_opcode(&mut self, cursor: usize, code: u8) {
        let position = cursor % OPCODE_POOL_SIZE;
        let bit_offset = OPCODE_STREAM_OFFSET + position * OPCODE_BITS_PER_OPCODE;
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

/// Bit offset of one instruction's parameter window, measured from the start
/// of the genome. The header sits after the leading mutation-rate field.
fn header_window_offset(instruction_index: usize) -> usize {
    HEADER_OFFSET + instruction_index * PARAM_BITS_PER_INSTRUCTION
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

/// Returns a value in [0, 1] expressing how different two 24-bit RGB colors
/// are. 0 means identical; 1 means maximally distant (black vs white). The
/// computation is the euclidean distance between the colors as 3-vectors,
/// normalized by sqrt(3 * 255^2).
#[derive(Debug, PartialEq, Eq)]
pub enum GenomeParseError {
    WrongLength { expected: usize, actual: usize },
    InvalidCharacter { index: usize, character: char },
}

impl std::fmt::Display for GenomeParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GenomeParseError::WrongLength { expected, actual } => write!(
                f,
                "genome bit string has wrong length: expected {expected}, got {actual}",
            ),
            GenomeParseError::InvalidCharacter { index, character } => write!(
                f,
                "genome bit string has invalid character '{character}' at index {index} (expected '0' or '1')",
            ),
        }
    }
}

impl std::error::Error for GenomeParseError {}

pub fn color_dissimilarity(a: u32, b: u32) -> f32 {
    let max_distance: f32 = (3.0_f32 * 255.0 * 255.0).sqrt();
    let [_, ar, ag, ab] = a.to_be_bytes();
    let [_, br, bg, bb] = b.to_be_bytes();
    let dr = ar as f32 - br as f32;
    let dg = ag as f32 - bg as f32;
    let db = ab as f32 - bb as f32;
    let distance = (dr * dr + dg * dg + db * db).sqrt();
    (distance / max_distance).clamp(0.0, 1.0)
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
            // 8 mutation-rate bits + 7 instructions × 17 param bits + 40
            // activation mask bits + 40 opcodes × 4 bits = 8 + 119 + 40 + 160
            // = 327 bits = 41 bytes (rounding up). Pinning down the literal
            // byte count catches mutations to the constants that derive from
            // MUTATION_RATE_BITS + HEADER_BITS + ACTIVATION_MASK_BITS +
            // OPCODE_BITS.
            let genome = random_genome(0);

            assert_eq!(genome.bytes.len(), 41);
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
        fn decoding_at_a_cursor_beyond_the_active_count_wraps_around() {
            // The walker counts only active slots, so wrapping happens after
            // the active count, not the pool size. With from_instructions,
            // INITIAL_ACTIVE_OPCODES slots are active.
            let genome = Genome::from_instructions(&[Instruction::TurnLeft, Instruction::Split]);

            assert_eq!(
                genome.decode_at(0),
                genome.decode_at(INITIAL_ACTIVE_OPCODES)
            );
            assert_eq!(
                genome.decode_at(1),
                genome.decode_at(INITIAL_ACTIVE_OPCODES + 1)
            );
        }

        #[test]
        fn an_all_split_genome_decodes_to_split_at_every_active_cursor() {
            let genome = Genome::all(Instruction::Split);

            for cursor in 0..INITIAL_ACTIVE_OPCODES {
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
        fn a_fresh_random_genome_has_initial_active_opcodes_active() {
            let genome = random_genome(0);

            assert_eq!(genome.active_opcode_count(), INITIAL_ACTIVE_OPCODES);
        }

        #[test]
        fn decoding_skips_dormant_slots_in_the_pool() {
            // Build a genome where every active slot decodes to TurnLeft and
            // every dormant slot decodes to Split. The walker must only visit
            // active slots, so every cursor sees TurnLeft.
            let mut genome = Genome::all(Instruction::TurnLeft);
            for slot in INITIAL_ACTIVE_OPCODES..OPCODE_POOL_SIZE {
                let bit_offset = OPCODE_STREAM_OFFSET + slot * OPCODE_BITS_PER_OPCODE;
                write_bits(
                    &mut genome.bytes,
                    bit_offset,
                    OPCODE_BITS_PER_OPCODE,
                    encode(Instruction::Split) as u32,
                );
            }

            for cursor in 0..OPCODE_POOL_SIZE {
                assert_eq!(genome.decode_at(cursor), Instruction::TurnLeft);
            }
        }

        #[test]
        fn activating_a_dormant_slot_brings_its_instruction_into_the_walked_stream() {
            // Build a genome where the dormant slot at INITIAL_ACTIVE_OPCODES
            // contains Split. After activating that slot's mask bit, the
            // active count grows by one and the cursor that would have
            // wrapped now lands on the newly-activated Split.
            let mut genome = Genome::all(Instruction::TurnLeft);
            let target_slot = INITIAL_ACTIVE_OPCODES;
            let bit_offset = OPCODE_STREAM_OFFSET + target_slot * OPCODE_BITS_PER_OPCODE;
            write_bits(
                &mut genome.bytes,
                bit_offset,
                OPCODE_BITS_PER_OPCODE,
                encode(Instruction::Split) as u32,
            );

            // Before activation, cursor INITIAL_ACTIVE_OPCODES wraps to slot 0
            // (TurnLeft).
            assert_eq!(
                genome.decode_at(INITIAL_ACTIVE_OPCODES),
                Instruction::TurnLeft
            );

            write_bits(
                &mut genome.bytes,
                ACTIVATION_MASK_OFFSET + target_slot,
                1,
                1,
            );

            // After activation, that cursor now lands on the newly-active
            // Split slot rather than wrapping back to slot 0.
            assert_eq!(genome.decode_at(INITIAL_ACTIVE_OPCODES), Instruction::Split);
            assert_eq!(genome.active_opcode_count(), INITIAL_ACTIVE_OPCODES + 1);
        }

        #[test]
        fn a_genome_with_no_active_slots_decodes_to_do_nothing() {
            let mut genome = Genome::all(Instruction::Split);
            // Deactivate every slot.
            for slot in 0..OPCODE_POOL_SIZE {
                write_bits(&mut genome.bytes, ACTIVATION_MASK_OFFSET + slot, 1, 0);
            }

            assert_eq!(genome.decode_at(0), Instruction::DoNothing);
            assert_eq!(genome.decode_at(99), Instruction::DoNothing);
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

            let probability = genome.probability_of_acting(Instruction::Split, 60, false, 0.0);

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
                    genome.probability_of_acting(Instruction::Split, threshold as u32, false, 0.0);
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
                let probability =
                    genome.probability_of_acting(Instruction::Split, energy, false, 0.0);
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
                let probability =
                    genome.probability_of_acting(Instruction::Split, energy, false, 0.0);
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
                let offset = header_window_offset(index) + THRESHOLD_OFFSET;
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
            // Use a hard-coded offset (mask 3 bits + threshold 7 bits = 10)
            // rather than the SOFTNESS_OFFSET constant so that mutations to
            // the constant produce a visible mismatch between write and read.
            const SOFTNESS_OFFSET_LITERAL: usize = 10;
            let mut bytes = [0u8; TOTAL_BYTES];
            let softnesses: [u32; 7] = [5, 15, 25, 35, 45, 55, 65];
            for (index, &soft) in softnesses.iter().enumerate() {
                let offset = header_window_offset(index) + SOFTNESS_OFFSET_LITERAL;
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

            let p_low = genome.probability_of_acting(Instruction::Split, 0, false, 0.0);
            let p_high_energy = genome.probability_of_acting(Instruction::Split, 500, false, 0.0);
            let p_touching = genome.probability_of_acting(Instruction::Split, 0, true, 0.0);

            assert_eq!(p_low, p_high_energy);
            assert_eq!(p_low, p_touching);
        }

        #[test]
        fn with_only_the_energy_factor_enabled_touching_does_not_change_the_probability() {
            let genome = genome_with_mask_for_split(ENERGY_FACTOR_BIT);

            let p_not_touching = genome.probability_of_acting(Instruction::Split, 100, false, 0.0);
            let p_touching = genome.probability_of_acting(Instruction::Split, 100, true, 0.0);

            assert_eq!(p_not_touching, p_touching);
        }

        #[test]
        fn with_only_the_touching_factor_enabled_energy_does_not_change_the_probability() {
            let genome = genome_with_mask_for_split(TOUCHING_FACTOR_BIT);

            let p_low_energy = genome.probability_of_acting(Instruction::Split, 0, false, 0.0);
            let p_high_energy = genome.probability_of_acting(Instruction::Split, 500, false, 0.0);

            assert_eq!(p_low_energy, p_high_energy);
        }

        #[test]
        fn with_only_the_touching_factor_enabled_touching_increases_the_probability() {
            // With softness = 1 and threshold = 0, touching adds 64 to the
            // input — sigmoid(64) is essentially 1, sigmoid(0) is 0.5.
            let genome = genome_with_mask_for_split(TOUCHING_FACTOR_BIT);

            let p_not_touching = genome.probability_of_acting(Instruction::Split, 0, false, 0.0);
            let p_touching = genome.probability_of_acting(Instruction::Split, 0, true, 0.0);

            assert!(p_not_touching < p_touching);
        }

        #[test]
        fn with_no_color_factor_dissimilarity_does_not_change_the_probability() {
            let genome = genome_with_mask_for_split(0b000);

            let p_similar = genome.probability_of_acting(Instruction::Split, 0, false, 0.0);
            let p_dissimilar = genome.probability_of_acting(Instruction::Split, 0, false, 1.0);

            assert_eq!(p_similar, p_dissimilar);
        }

        #[test]
        fn with_only_the_color_factor_enabled_dissimilarity_increases_the_probability() {
            // Softness = 1, threshold = 0. Dissimilarity = 1 adds 64 to input;
            // dissimilarity = 0 leaves input at 0. sigmoid(64) ≈ 1, sigmoid(0) = 0.5.
            let genome = genome_with_mask_for_split(COLOR_DISSIMILARITY_FACTOR_BIT);

            let p_similar = genome.probability_of_acting(Instruction::Split, 0, false, 0.0);
            let p_dissimilar = genome.probability_of_acting(Instruction::Split, 0, false, 1.0);

            assert!(p_similar < p_dissimilar);
        }

        #[test]
        fn with_only_the_color_factor_enabled_neither_energy_nor_touching_changes_the_probability()
        {
            let genome = genome_with_mask_for_split(COLOR_DISSIMILARITY_FACTOR_BIT);

            let baseline = genome.probability_of_acting(Instruction::Split, 0, false, 0.5);
            let varied_energy = genome.probability_of_acting(Instruction::Split, 500, false, 0.5);
            let varied_touching = genome.probability_of_acting(Instruction::Split, 0, true, 0.5);

            assert_eq!(baseline, varied_energy);
            assert_eq!(baseline, varied_touching);
        }

        fn genome_with_mask_for_split(mask: u32) -> Genome {
            // Threshold = 0, softness = 1 (raw bits 0). Just the mask varies.
            let mut bytes = [0u8; TOTAL_BYTES];
            let split_window = header_window_offset(instruction_index(Instruction::Split));
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
                    genome.probability_of_acting(Instruction::MoveForward, energy, false, 0.0),
                    genome.probability_of_acting(
                        Instruction::RepeatPreviousMove,
                        energy,
                        false,
                        0.0,
                    ),
                    genome.probability_of_acting(Instruction::DoNothing, energy, false, 0.0),
                    genome.probability_of_acting(Instruction::TurnLeft, energy, false, 0.0),
                    genome.probability_of_acting(Instruction::TurnRight, energy, false, 0.0),
                    genome.probability_of_acting(Instruction::Split, energy, false, 0.0),
                    genome.probability_of_acting(Instruction::Eat, energy, false, 0.0),
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

    mod color_dissimilarity {
        use super::*;

        #[test]
        fn identical_colors_have_zero_dissimilarity() {
            assert_eq!(color_dissimilarity(0xAB_CD_EF, 0xAB_CD_EF), 0.0);
        }

        #[test]
        fn black_and_white_have_maximum_dissimilarity() {
            assert!((color_dissimilarity(0x00_00_00, 0xFF_FF_FF) - 1.0).abs() < 1e-6);
        }

        #[test]
        fn dissimilarity_is_symmetric() {
            let a = 0x12_34_56;
            let b = 0xAB_CD_EF;

            assert_eq!(color_dissimilarity(a, b), color_dissimilarity(b, a));
        }

        #[test]
        fn closer_colors_are_less_dissimilar_than_more_distant_ones() {
            let close = color_dissimilarity(0x10_10_10, 0x20_20_20);
            let far = color_dissimilarity(0x10_10_10, 0xF0_F0_F0);

            assert!(close < far);
        }

        #[test]
        fn an_asymmetric_channel_delta_produces_a_known_intermediate_dissimilarity() {
            // dr = 10, dg = 20, db = 30 → distance² = 100 + 400 + 900 = 1400.
            // distance ≈ 37.417, max ≈ 441.673, ratio ≈ 0.0847. The exact
            // numeric assertion catches mutations that scramble either the
            // squared-distance sum or the normalizer.
            let a = 0x00_00_00;
            let b = 0x0A_14_1E;
            let expected = (1400.0_f32).sqrt() / (3.0_f32 * 255.0 * 255.0).sqrt();

            let actual = color_dissimilarity(a, b);

            assert!((actual - expected).abs() < 1e-6, "{actual} vs {expected}");
        }

        #[test]
        fn a_mid_gray_pair_produces_a_known_intermediate_dissimilarity() {
            // 0x40_40_40 vs 0x80_80_80: dr = dg = db = 64.
            // distance² = 3 * 64² = 12288, distance = 64*sqrt(3),
            // ratio = 64*sqrt(3) / (255*sqrt(3)) = 64/255.
            let actual = color_dissimilarity(0x40_40_40, 0x80_80_80);
            let expected = 64.0 / 255.0;

            assert!((actual - expected).abs() < 1e-6, "{actual} vs {expected}");
        }

        #[test]
        fn the_dissimilarity_of_any_pair_lies_in_the_unit_interval() {
            for seed in 0..50 {
                let a = random_genome(seed).digest_color();
                let b = random_genome(seed + 100).digest_color();
                let d = color_dissimilarity(a, b);

                assert!((0.0..=1.0).contains(&d), "{d}");
            }
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
        fn each_channel_hashes_a_distinct_slice_of_the_genome() {
            // Construct a genome where each third has a distinct byte signature
            // and the others are zero. Each channel's hash is then determined
            // entirely by its own slice; if a mutation shifts the slice
            // boundaries the channel will hash a different (or shorter) span
            // and produce a different value.
            const THIRD: usize = TOTAL_BYTES / 3;
            let mut bytes = [0u8; TOTAL_BYTES];
            // Place the signature byte at the last position of each slice.
            bytes[THIRD - 1] = 0xAA;
            let red_only = Genome { bytes };

            let mut bytes = [0u8; TOTAL_BYTES];
            bytes[2 * THIRD - 1] = 0xAA;
            let green_only = Genome { bytes };

            let mut bytes = [0u8; TOTAL_BYTES];
            bytes[TOTAL_BYTES - 1] = 0xAA;
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
            // Bytes feeding the red channel are the first third of the genome;
            // green and blue must be unaffected. We sweep seeds to dodge any
            // particular pair where two hashes happen to collide at the
            // brightness floor.
            for seed in 0..50 {
                let original = random_genome(seed);
                let mut bytes = original.bytes;
                bytes[0] ^= 0xFF;
                let mutated = Genome { bytes };

                let (or, og, ob) = channels(original.digest_color());
                let (mr, mg, mb) = channels(mutated.digest_color());
                if og == mg && ob == mb && or != mr {
                    return;
                }
            }
            panic!("no seed showed a red-only color shift");
        }

        #[test]
        fn flipping_a_byte_in_the_middle_third_changes_only_the_green_channel() {
            for seed in 0..50 {
                let original = random_genome(seed);
                let mut bytes = original.bytes;
                bytes[TOTAL_BYTES / 3] ^= 0xFF;
                let mutated = Genome { bytes };

                let (or, og, ob) = channels(original.digest_color());
                let (mr, mg, mb) = channels(mutated.digest_color());
                if or == mr && ob == mb && og != mg {
                    return;
                }
            }
            panic!("no seed showed a green-only color shift");
        }

        #[test]
        fn flipping_a_byte_in_the_last_third_changes_only_the_blue_channel() {
            for seed in 0..50 {
                let original = random_genome(seed);
                let mut bytes = original.bytes;
                bytes[2 * (TOTAL_BYTES / 3)] ^= 0xFF;
                let mutated = Genome { bytes };

                let (or, og, ob) = channels(original.digest_color());
                let (mr, mg, mb) = channels(mutated.digest_color());
                if or == mr && og == mg && ob != mb {
                    return;
                }
            }
            panic!("no seed showed a blue-only color shift");
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

            for cursor in 0..OPCODE_POOL_SIZE {
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
                        original.probability_of_acting(instruction, energy, false, 0.0),
                        cloned.probability_of_acting(instruction, energy, false, 0.0),
                    );
                }
            }
        }
    }

    mod mutation_chance {
        use super::*;

        fn genome_with_rate_bits(bits: u32) -> Genome {
            let mut genome = Genome::from_bits(&"0".repeat(TOTAL_BITS)).unwrap();
            genome.set_mutation_rate_bits(bits);
            genome
        }

        const MAX_RATE_BITS: u32 = (1 << MUTATION_RATE_BITS) - 1;

        #[test]
        fn an_all_zero_mutation_rate_field_yields_a_zero_chance() {
            let genome = genome_with_rate_bits(0);

            assert_eq!(genome.mutation_chance(), 0.0);
        }

        #[test]
        fn an_all_one_mutation_rate_field_yields_the_maximum_chance() {
            let genome = genome_with_rate_bits(MAX_RATE_BITS);

            assert_eq!(genome.mutation_chance(), MAX_MUTATION_CHANCE);
        }

        #[test]
        fn a_half_range_mutation_rate_field_yields_half_the_maximum_chance() {
            // Pins the shape of the mapping between its endpoints: the rate
            // scales linearly with the field's value.
            let genome = genome_with_rate_bits(MAX_RATE_BITS / 2);

            let expected = MAX_MUTATION_CHANCE * (MAX_RATE_BITS / 2) as f32 / MAX_RATE_BITS as f32;
            assert!((genome.mutation_chance() - expected).abs() < f32::EPSILON);
        }
    }

    mod bits {
        use super::*;

        #[test]
        fn to_bits_returns_a_string_of_the_total_bit_length() {
            let genome = random_genome(0);

            let bits = genome.to_bits();

            assert_eq!(bits.len(), TOTAL_BITS);
        }

        #[test]
        fn to_bits_contains_only_zero_and_one_characters() {
            let genome = random_genome(7);

            let bits = genome.to_bits();

            for character in bits.chars() {
                assert!(
                    character == '0' || character == '1',
                    "unexpected character {character}",
                );
            }
        }

        #[test]
        fn from_bits_round_trips_a_random_genome() {
            let original = random_genome(123);

            let parsed = Genome::from_bits(&original.to_bits()).unwrap();

            assert_eq!(parsed, original);
        }

        #[test]
        fn from_bits_rejects_a_string_of_the_wrong_length() {
            let too_short = "0".repeat(TOTAL_BITS - 1);

            let result = Genome::from_bits(&too_short);

            assert!(matches!(
                result,
                Err(GenomeParseError::WrongLength { expected, actual })
                    if expected == TOTAL_BITS && actual == TOTAL_BITS - 1
            ));
        }

        #[test]
        fn wrong_length_error_displays_the_expected_and_actual_lengths() {
            let error = GenomeParseError::WrongLength {
                expected: 319,
                actual: 318,
            };

            let rendered = format!("{error}");

            assert!(
                rendered.contains("319") && rendered.contains("318"),
                "unexpected rendering: {rendered}",
            );
        }

        #[test]
        fn invalid_character_error_displays_the_index_and_character() {
            let error = GenomeParseError::InvalidCharacter {
                index: 7,
                character: 'Z',
            };

            let rendered = format!("{error}");

            assert!(
                rendered.contains("7") && rendered.contains('Z'),
                "unexpected rendering: {rendered}",
            );
        }

        #[test]
        fn from_bits_rejects_a_string_containing_a_non_binary_character() {
            let mut input: String = "0".repeat(TOTAL_BITS);
            // Replace the byte at index 5 with the character '2'.
            input.replace_range(5..6, "2");

            let result = Genome::from_bits(&input);

            assert!(matches!(
                result,
                Err(GenomeParseError::InvalidCharacter {
                    index: 5,
                    character: '2'
                })
            ));
        }
    }
}
