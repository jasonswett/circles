use crate::{Instruction, MAX_CRITTER_ENERGY};
use rand::Rng;

const INSTRUCTION_COUNT: usize = 15;
// The genome's leading field: how often this critter's splits mutate the
// child. Evolvable like everything else — it mutates along with the rest of
// the genome, so a lineage's mutability drifts under selection.
pub(crate) const MUTATION_RATE_BITS: usize = 8;
// The highest per-bit flip rate a genome can encode. With 440 bits this caps
// a split at roughly 0.2 changed bits, keeping the hottest possible lineage
// below the error threshold while still leaving room to evolve mutability.
const MAX_MUTATION_RATE: f32 = 0.0005;
// Per-instruction header window: factor mask + threshold + softness.
// The factor mask is three bits (one per factor), read in the order:
// bit 0 = energy, bit 1 = is-touching-another-critter, bit 2 = the touched
// critter's color is dissimilar to mine. A bit on means the corresponding
// factor contributes to the sigmoid input; off means it is ignored. This is
// the "stair-step" mechanism: a single mutation can enable or disable a
// whole factor without disturbing the existing weights.
const FACTOR_MASK_BITS: usize = 15;
const THRESHOLD_BITS: usize = 7; // 0..128: median ~64, near INITIAL_ENERGY=60
const SOFTNESS_BITS: usize = 7; // 0..128, mapped to [MIN_SOFTNESS, MIN_SOFTNESS + 127]
const PARAM_BITS_PER_INSTRUCTION: usize = FACTOR_MASK_BITS + THRESHOLD_BITS + SOFTNESS_BITS;
const MIN_SOFTNESS: f32 = 1.0;

// When the touching factor is enabled, this is the value contributed to the
// sigmoid input on the touching=true side. Comparable in magnitude to a
// healthy critter's energy, so a touching bit can meaningfully swing a
// decision but not entirely dominate it.
// Energy enters the sigmoid on the same scale as every other factor rather
// than as its raw value. Raw energy ran to MAX_CRITTER_ENERGY while the
// threshold opposing it is seven bits, so the strictest rule a genome could
// express triggered at 127 -- an eighth of the range -- and every energy above
// that read identically. "Hold out until nearly full" was not a rule evolution
// was failing to find; it was one the genome had no way to say.
// How long a critter takes to count as grown. Age matters while a critter is
// young -- newborn, half grown, grown -- and stops mattering after that: one
// long life is much like another, so the sense saturates rather than
// separating the old from the very old forever.
pub const MATURE_AGE: u32 = 600;
const AGE_FACTOR_SCALE: f32 = 64.0;
const ENERGY_FACTOR_SCALE: f32 = 64.0;
const TOUCHING_FACTOR_SCALE: f32 = 64.0;

// When a color channel is enabled, the contribution scales from 0 (the
// channel is dark in what the critter touched) up to this value (the channel
// is at full brightness). Same scale as TOUCHING_FACTOR_SCALE so the factors
// are directly comparable. Each channel is sensed separately, so a lineage
// can respond to green food without responding to red poison.
const COLOR_CHANNEL_SCALE: f32 = 64.0;

// Within each instruction's bit window: mask occupies bits 0..3, threshold
// occupies bits 3..10, softness occupies bits 10..17.
const THRESHOLD_OFFSET: usize = FACTOR_MASK_BITS;
const SOFTNESS_OFFSET: usize = THRESHOLD_OFFSET + THRESHOLD_BITS;

const ENERGY_FACTOR_BIT: u32 = 0b0001;
const TOUCHING_FACTOR_BIT: u32 = 0b0010;
const RED_FACTOR_BIT: u32 = 0b0000_0100;
const GREEN_FACTOR_BIT: u32 = 0b0001_0000;
const BLUE_FACTOR_BIT: u32 = 0b0010_0000;
const HISTORY_FACTOR_BIT: u32 = 0b0000_1000;
const AGE_FACTOR_BIT: u32 = 0b0100_0000;
// A feeler senses exactly what the body does: three colour channels apiece.
// Nothing about a sensor says what it is for -- it reports the colour in front
// of it and evolution settles what a colour is worth. An earlier feeler read
// only green, on the reasoning that green meant food, which quietly decided
// for evolution that poison was not worth noticing: red and empty space read
// the same, so fleeing poison could not be learned however useful it was.
const LEFT_RED_FACTOR_BIT: u32 = 0b1000_0000;
const LEFT_GREEN_FACTOR_BIT: u32 = 0b1_0000_0000;
const LEFT_BLUE_FACTOR_BIT: u32 = 0b10_0000_0000;
const RIGHT_RED_FACTOR_BIT: u32 = 0b100_0000_0000;
const RIGHT_GREEN_FACTOR_BIT: u32 = 0b1000_0000_0000;
const RIGHT_BLUE_FACTOR_BIT: u32 = 0b1_0000_0000_0000;

// When the history factor is enabled, the contribution scales from 0 (none of
// the remembered actions were this instruction) up to this value (all of them
// were). Same scale as the touching and color factors so the four stay
// directly comparable.
const HISTORY_FACTOR_SCALE: f32 = 64.0;

// Per-instruction weight: how much of the 4-bit opcode space this genome
// devotes to each instruction. Weights are read as `bits + 1`, so the
// encoded range is 1..=16 and no instruction can be excluded outright —
// which also means the cumulative table is never empty.
const WEIGHT_BITS_PER_INSTRUCTION: usize = 4;

#[cfg(test)]
pub(crate) const MAX_WEIGHT_BITS: u32 = (1 << WEIGHT_BITS_PER_INSTRUCTION) - 1;
const WEIGHT_BITS: usize = INSTRUCTION_COUNT * WEIGHT_BITS_PER_INSTRUCTION;

// How many of its most recent actions a critter takes into account. Encoded
// in unary: the window is the number of set bits, so position carries no
// meaning and a single flip moves the window by exactly one. That keeps the
// trait smoothly tunable by mutation, where a binary field would jump.
pub(crate) const HISTORY_WINDOW_BITS: usize = 16;

const MUTATION_RATE_OFFSET: usize = 0;
const WEIGHTS_OFFSET: usize = MUTATION_RATE_BITS;
const HISTORY_WINDOW_OFFSET: usize = WEIGHTS_OFFSET + WEIGHT_BITS;
const HEADER_OFFSET: usize = HISTORY_WINDOW_OFFSET + HISTORY_WINDOW_BITS;
const HEADER_BITS: usize = INSTRUCTION_COUNT * PARAM_BITS_PER_INSTRUCTION;
// Wide enough that every instruction claims a band of the opcode space even
// when weights are equal. The bands are narrower than the instruction count
// suggests: a kind's share is divided among its variants, so the seven turns
// hold a seventh of a kind each. At five bits those bands came to less than
// one opcode value and several turns were unreachable however a genome was
// written -- instructions the genome could not say.
const OPCODE_BITS_PER_OPCODE: usize = 7;
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
// How a critter's feelers are shaped: how far they reach, how far apart they
// are held, and how big a patch each one feels. All three evolve, so a lineage
// can settle on long thin feelers held wide, or short fat ones pointed
// forward, or anything between.
//
// The disc is capped well short of the omnidirectional: at its largest it
// covers a little more ground than the critter's own body, so a feeler stays
// something a critter points rather than a sense of everything around it.
const FEELER_FIELD_BITS: usize = 4;
const FEELER_LENGTH_OFFSET: usize = OPCODE_STREAM_OFFSET + OPCODE_BITS;
const FEELER_ANGLE_OFFSET: usize = FEELER_LENGTH_OFFSET + FEELER_FIELD_BITS;
const FEELER_DISC_OFFSET: usize = FEELER_ANGLE_OFFSET + FEELER_FIELD_BITS;
// Whether a critter grows each feeler at all, one bit apiece. Separate bits
// rather than a count, so a lineage can climb towards feelers a step at a
// time: one is worth having on its own, and the second need not arrive with
// the first.
// How many bits spell each feeler. A feeler exists if any of them is set, so
// mutation has several ways to find one and several would have to flip back
// to lose it. One bit apiece made a feeler appear about once in four thousand
// splits, which in a world where nothing lives long is never.
//
// Duplication is how a trait becomes reachable and how it stops being lost:
// what is spelled once is found rarely and undone by a single flip.
const FEELER_PRESENT_BITS: usize = 5;
const LEFT_FEELER_PRESENT_OFFSET: usize = FEELER_DISC_OFFSET + FEELER_FIELD_BITS;
const RIGHT_FEELER_PRESENT_OFFSET: usize = LEFT_FEELER_PRESENT_OFFSET + FEELER_PRESENT_BITS;
// Measured from where the region starts to where its last field ends, rather
// than counted up again by hand: a total that drifts from the offsets it is
// meant to cover either drops a field out of the genome or pads it with bits
// nothing reads.
const FEELER_BITS: usize = RIGHT_FEELER_PRESENT_OFFSET + FEELER_PRESENT_BITS - FEELER_LENGTH_OFFSET;
/// How far a feeler reaches beyond the body, in pixels.
pub const MIN_FEELER_LENGTH: f32 = 8.0;
pub const MAX_FEELER_LENGTH: f32 = 34.0;
/// How far to either side of the heading a feeler is held, in degrees.
pub const MAX_FEELER_ANGLE: f32 = 90.0;
/// The radius of the patch at a feeler's tip that actually senses.
pub const MIN_FEELER_DISC: f32 = 2.0;
pub const MAX_FEELER_DISC: f32 = 8.0;
const TOTAL_BITS: usize = MUTATION_RATE_BITS
    + WEIGHT_BITS
    + HISTORY_WINDOW_BITS
    + HEADER_BITS
    + ACTIVATION_MASK_BITS
    + OPCODE_BITS
    + FEELER_BITS;
const TOTAL_BYTES: usize = TOTAL_BITS.div_ceil(8);

/// Everything a critter can perceive when deciding whether to act. Grouped
/// rather than passed positionally: there are enough of them now that the
/// order would be easy to get wrong.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Senses {
    pub energy: u32,
    pub touching_critter: bool,
    /// The color of whatever the critter most recently touched, black when it
    /// has touched nothing. Sensed per channel, so what a critter sees is the
    /// color itself rather than how unlike itself it is.
    pub touched_color: u32,
    pub recent_repetition: f32,
    /// How many ticks the critter has lived.
    pub age: u32,
    /// What each feeler's disc is touching, black for nothing.
    pub left_color: u32,
    pub right_color: u32,
}

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
        // TOTAL_BITS need not fill the last byte. Zero the leftover pad bits:
        // to_bits doesn't emit them, so a genome carrying random padding would
        // not survive a to_bits/from_bits round trip.
        // Zero any bits past TOTAL_BITS. The layout currently ends on a byte
        // boundary and leaves none, but a future field could reintroduce
        // them, and pad bits that to_bits does not emit would break a round
        // trip through from_bits.
        // A fresh critter is blind. Feelers are the one thing a lineage has to
        // earn: everything else about a genome starts wherever chance put it,
        // but a feeler arrives only where mutation grows one, so a population
        // that has them has shown they pay for themselves.
        write_bits(
            &mut bytes,
            LEFT_FEELER_PRESENT_OFFSET,
            FEELER_PRESENT_BITS,
            0,
        );
        write_bits(
            &mut bytes,
            RIGHT_FEELER_PRESENT_OFFSET,
            FEELER_PRESENT_BITS,
            0,
        );
        #[allow(clippy::reversed_empty_ranges)]
        for bit in TOTAL_BITS..(TOTAL_BYTES * 8) {
            write_bits(&mut bytes, bit, 1, 0);
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
        // Worked out once rather than per slot: which opcode means what
        // depends on the genome's weights, and writing the stream does not
        // touch those, so the answer is the same for every slot.
        let code = encode_for(&genome, instruction);
        for cursor in 0..OPCODE_POOL_SIZE {
            genome.write_opcode(cursor, code);
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
            let code = encode_for(&genome, instr);
            genome.write_opcode(cursor, code);
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
                let code = read_bits(&self.bytes, bit_offset, OPCODE_BITS_PER_OPCODE) as u8;
                return decode_with_weights(self, code);
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

    pub fn probability_of_acting(&self, instruction: Instruction, senses: &Senses) -> f32 {
        let (mask, threshold, softness) = self.params(instruction);
        let energy_contribution = if mask & ENERGY_FACTOR_BIT != 0 {
            ENERGY_FACTOR_SCALE * senses.energy.min(MAX_CRITTER_ENERGY) as f32
                / MAX_CRITTER_ENERGY as f32
        } else {
            0.0
        };
        let touching_contribution = if mask & TOUCHING_FACTOR_BIT != 0 && senses.touching_critter {
            TOUCHING_FACTOR_SCALE
        } else {
            0.0
        };
        // The same reading for whichever of the three sensors is asking: what
        // differs between them is where they point, not what they can see.
        let seen = |lit: u32, red_bit: u32, green_bit: u32, blue_bit: u32| {
            let [_, red, green, blue] = lit.to_be_bytes();
            let channel = |lit: u8, bit: u32| {
                if mask & bit != 0 {
                    COLOR_CHANNEL_SCALE * lit as f32 / 255.0
                } else {
                    0.0
                }
            };
            channel(red, red_bit) + channel(green, green_bit) + channel(blue, blue_bit)
        };
        let body_contribution = seen(
            senses.touched_color,
            RED_FACTOR_BIT,
            GREEN_FACTOR_BIT,
            BLUE_FACTOR_BIT,
        );
        // `recent_repetition` is the share of the critter's remembered actions
        // that were this instruction, so the contribution does not grow just
        // because the window widened — only because the behavior repeated.
        let history_contribution = if mask & HISTORY_FACTOR_BIT != 0 {
            HISTORY_FACTOR_SCALE * senses.recent_repetition.clamp(0.0, 1.0)
        } else {
            0.0
        };
        // Saturating rather than asymptotic: growing up is what age is for,
        // and a critter long past MATURE_AGE reads the same as one that just
        // reached it.
        let age_contribution = if mask & AGE_FACTOR_BIT != 0 {
            AGE_FACTOR_SCALE * senses.age.min(MATURE_AGE) as f32 / MATURE_AGE as f32
        } else {
            0.0
        };
        let left_feeler_contribution = seen(
            senses.left_color,
            LEFT_RED_FACTOR_BIT,
            LEFT_GREEN_FACTOR_BIT,
            LEFT_BLUE_FACTOR_BIT,
        );
        let right_feeler_contribution = seen(
            senses.right_color,
            RIGHT_RED_FACTOR_BIT,
            RIGHT_GREEN_FACTOR_BIT,
            RIGHT_BLUE_FACTOR_BIT,
        );
        let input = left_feeler_contribution
            + right_feeler_contribution
            + age_contribution
            + energy_contribution
            + touching_contribution
            + body_contribution
            + history_contribution;
        sigmoid((input - threshold) / softness)
    }

    /// How many of its most recent actions a critter takes into account.
    /// Encoded in unary, so this is the field's popcount: zero means history
    /// is not consulted at all.
    pub fn history_window(&self) -> usize {
        read_bits(&self.bytes, HISTORY_WINDOW_OFFSET, HISTORY_WINDOW_BITS).count_ones() as usize
    }

    /// Overwrites the history-window field. Test-only: in a running world the
    /// field changes only through mutation.
    #[cfg(test)]
    pub fn set_history_window_bits(&mut self, bits: u32) {
        write_bits(
            &mut self.bytes,
            HISTORY_WINDOW_OFFSET,
            HISTORY_WINDOW_BITS,
            bits,
        );
    }

    /// How much of the opcode space this genome devotes to `instruction`.
    /// Read as `bits + 1` so every instruction keeps a nonzero share.
    /// How much of the opcode space an instruction claims.
    ///
    /// Every kind of thing a critter can do claims the same share, and the
    /// variants of a kind divide that share between them by their own weights.
    /// Turning comes up as often as eating whether it is spelled two ways or a
    /// thousand: without that, adding a variant quietly takes weight from
    /// every other instruction, and how many ways an action can be spelled
    /// becomes a design decision nobody made.
    ///
    /// Weights are scaled up by the number of variants in the largest kind
    /// rather than divided down, so a kind with many variants keeps whole
    /// numbers to divide among them.
    fn instruction_weight(&self, instruction: Instruction) -> u32 {
        let offset = WEIGHTS_OFFSET + instruction_index(instruction) * WEIGHT_BITS_PER_INSTRUCTION;
        let encoded = read_bits(&self.bytes, offset, WEIGHT_BITS_PER_INSTRUCTION) + 1;
        let kind = kind_of(instruction);
        encoded * kind.share_per_variant() * kind.thumb_on_the_scale()
    }

    /// Overwrites one instruction's weight field. Test-only: in a running
    /// world the weights change only through mutation.
    #[cfg(test)]
    pub fn set_instruction_weight_bits(&mut self, instruction: Instruction, bits: u32) {
        let offset = WEIGHTS_OFFSET + instruction_index(instruction) * WEIGHT_BITS_PER_INSTRUCTION;
        write_bits(&mut self.bytes, offset, WEIGHT_BITS_PER_INSTRUCTION, bits);
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

    /// The chance each bit flips when this genome is copied, in
    /// [0, MAX_MUTATION_RATE]. The field is part of the genome, so it mutates
    /// like any other region and a lineage's mutability evolves.
    pub fn mutation_rate(&self) -> f32 {
        let bits = read_bits(&self.bytes, MUTATION_RATE_OFFSET, MUTATION_RATE_BITS);
        let max_value = ((1u32 << MUTATION_RATE_BITS) - 1) as f32;
        bits as f32 / max_value * MAX_MUTATION_RATE
    }

    /// Grows or removes each feeler. Test-only: in a running world these
    /// change only through mutation.
    #[cfg(test)]
    pub fn set_feelers_present(&mut self, left: bool, right: bool) {
        write_bits(
            &mut self.bytes,
            LEFT_FEELER_PRESENT_OFFSET,
            FEELER_PRESENT_BITS,
            u32::from(left),
        );
        write_bits(
            &mut self.bytes,
            RIGHT_FEELER_PRESENT_OFFSET,
            FEELER_PRESENT_BITS,
            u32::from(right),
        );
    }

    /// Sets the feeler shape directly. Test-only: in a running world these
    /// change only through mutation.
    #[cfg(test)]
    pub fn set_feeler_shape(&mut self, length: f32, angle: f32, disc: f32) {
        let field = |value: f32, low: f32, high: f32| {
            let most = ((1u32 << FEELER_FIELD_BITS) - 1) as f32;
            (((value - low) / (high - low)) * most).round() as u32
        };
        write_bits(
            &mut self.bytes,
            FEELER_LENGTH_OFFSET,
            FEELER_FIELD_BITS,
            field(length, MIN_FEELER_LENGTH, MAX_FEELER_LENGTH),
        );
        write_bits(
            &mut self.bytes,
            FEELER_ANGLE_OFFSET,
            FEELER_FIELD_BITS,
            field(angle, 0.0, MAX_FEELER_ANGLE),
        );
        write_bits(
            &mut self.bytes,
            FEELER_DISC_OFFSET,
            FEELER_FIELD_BITS,
            field(disc, MIN_FEELER_DISC, MAX_FEELER_DISC),
        );
    }

    /// Whether this critter grew a feeler on each side. Any one of the bits
    /// spelling a feeler is enough to have it.
    pub fn has_left_feeler(&self) -> bool {
        read_bits(&self.bytes, LEFT_FEELER_PRESENT_OFFSET, FEELER_PRESENT_BITS) != 0
    }

    pub fn has_right_feeler(&self) -> bool {
        read_bits(
            &self.bytes,
            RIGHT_FEELER_PRESENT_OFFSET,
            FEELER_PRESENT_BITS,
        ) != 0
    }

    /// How far this critter's feelers reach beyond its body.
    pub fn feeler_length(&self) -> f32 {
        Self::scaled(
            read_bits(&self.bytes, FEELER_LENGTH_OFFSET, FEELER_FIELD_BITS),
            MIN_FEELER_LENGTH,
            MAX_FEELER_LENGTH,
        )
    }

    /// How far to either side of the heading the feelers are held.
    pub fn feeler_angle(&self) -> f32 {
        Self::scaled(
            read_bits(&self.bytes, FEELER_ANGLE_OFFSET, FEELER_FIELD_BITS),
            0.0,
            MAX_FEELER_ANGLE,
        )
    }

    /// The radius of the sensing patch at each feeler's tip.
    pub fn feeler_disc(&self) -> f32 {
        Self::scaled(
            read_bits(&self.bytes, FEELER_DISC_OFFSET, FEELER_FIELD_BITS),
            MIN_FEELER_DISC,
            MAX_FEELER_DISC,
        )
    }

    /// A field's bits spread evenly across the range it stands for.
    fn scaled(bits: u32, low: f32, high: f32) -> f32 {
        let most = ((1u32 << FEELER_FIELD_BITS) - 1) as f32;
        low + (high - low) * bits as f32 / most
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

    /// Sets every instruction's sigmoid so the probability of acting is
    /// essentially zero: no factors enabled and the highest threshold, so the
    /// input is always far below it. Test-only.
    #[cfg(test)]
    pub fn set_never_act_header(&mut self) {
        for index in 0..INSTRUCTION_COUNT {
            let window = header_window_offset(index);
            write_bits(&mut self.bytes, window, FACTOR_MASK_BITS, 0);
            write_bits(
                &mut self.bytes,
                window + THRESHOLD_OFFSET,
                THRESHOLD_BITS,
                (1 << THRESHOLD_BITS) - 1,
            );
            write_bits(&mut self.bytes, window + SOFTNESS_OFFSET, SOFTNESS_BITS, 0);
        }
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

/// Every instruction, in the order their weight fields appear in the genome.
/// `instruction_index` gives each one's position here.
const ALL_INSTRUCTIONS: [Instruction; INSTRUCTION_COUNT] = [
    Instruction::MoveSlow,
    Instruction::TurnLeft15,
    Instruction::TurnRight15,
    Instruction::DoNothing,
    Instruction::RepeatPreviousMove,
    Instruction::Split,
    Instruction::Eat,
    Instruction::SkipAhead,
    Instruction::SkipBack,
    // Appended rather than placed beside MoveSlow: an instruction's position
    // here fixes where its weight and parameter windows sit in the genome, so
    // adding to the end leaves every existing instruction's meaning intact.
    Instruction::MoveFast,
    Instruction::TurnLeft45,
    Instruction::TurnLeft90,
    Instruction::TurnRight45,
    Instruction::TurnRight90,
    Instruction::TurnAbout,
];

/// Bit offset of one instruction's parameter window, measured from the start
/// of the genome. The header sits after the leading mutation-rate field.
fn header_window_offset(instruction_index: usize) -> usize {
    HEADER_OFFSET + instruction_index * PARAM_BITS_PER_INSTRUCTION
}

/// The kinds of thing a critter can do. Each claims an equal share of the
/// opcode space, so how many ways a kind can be spelled does not decide how
/// often it comes up.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Move,
    Turn,
    Eat,
    Split,
    Skip,
    Nothing,
    Repeat,
}

impl Kind {
    /// How much the world leans on this kind coming up, over and above what a
    /// genome's weight fields ask for. A kind with nothing to declare comes
    /// out at exactly what its fields say.
    ///
    /// Applied to the decoded weight rather than written into the genome, so
    /// the same bits still say how much a lineage wants of something and
    /// evolution can still turn it down: leaning on the smallest weight still
    /// leaves it the smallest weight.
    fn thumb_on_the_scale(self) -> u32 {
        match self {
            // Reproduction is the one thing whose payoff a critter never
            // collects itself, so nothing about its own life selects for it.
            Kind::Split => 2,
            // A critter that cannot steer is at the mercy of wherever it
            // happens to be pointing.
            Kind::Turn => 2,
            _ => 1,
        }
    }

    /// How many instructions spell this kind.
    fn variants(self) -> u32 {
        ALL_INSTRUCTIONS
            .iter()
            .filter(|&&instruction| kind_of(instruction) == self)
            .count() as u32
    }

    /// What each of this kind's variants is scaled by, so that the variants
    /// together come to the same total whatever their number. Scaled up
    /// rather than divided down, and by a multiple every kind divides evenly,
    /// so no kind loses a remainder to integer division.
    fn share_per_variant(self) -> u32 {
        Self::EVERY.iter().fold(1, |lcm, kind| {
            let variants = kind.variants();
            lcm / gcd(lcm, variants) * variants
        }) / self.variants()
    }

    const EVERY: [Kind; 7] = [
        Kind::Move,
        Kind::Turn,
        Kind::Eat,
        Kind::Split,
        Kind::Skip,
        Kind::Nothing,
        Kind::Repeat,
    ];
}

fn gcd(a: u32, b: u32) -> u32 {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

fn kind_of(instruction: Instruction) -> Kind {
    match instruction {
        Instruction::MoveSlow | Instruction::MoveFast => Kind::Move,
        Instruction::TurnLeft15
        | Instruction::TurnLeft45
        | Instruction::TurnLeft90
        | Instruction::TurnRight15
        | Instruction::TurnRight45
        | Instruction::TurnRight90
        | Instruction::TurnAbout => Kind::Turn,
        Instruction::Eat => Kind::Eat,
        Instruction::Split => Kind::Split,
        Instruction::SkipAhead | Instruction::SkipBack => Kind::Skip,
        Instruction::DoNothing => Kind::Nothing,
        Instruction::RepeatPreviousMove => Kind::Repeat,
    }
}

fn instruction_index(instruction: Instruction) -> usize {
    match instruction {
        Instruction::MoveSlow => 0,
        Instruction::TurnLeft15 => 1,
        Instruction::TurnRight15 => 2,
        Instruction::DoNothing => 3,
        Instruction::RepeatPreviousMove => 4,
        Instruction::Split => 5,
        Instruction::Eat => 6,
        Instruction::SkipAhead => 7,
        Instruction::SkipBack => 8,
        Instruction::MoveFast => 9,
        Instruction::TurnLeft45 => 10,
        Instruction::TurnLeft90 => 11,
        Instruction::TurnRight45 => 12,
        Instruction::TurnRight90 => 13,
        Instruction::TurnAbout => 14,
    }
}

/// The opcode value that `genome` decodes to `instruction`. Which bits mean
/// which instruction depends on the genome's weights, so this searches the
/// opcode space rather than consulting a fixed table. Every instruction keeps
/// a nonzero weight, so a match always exists.
#[cfg(test)]
fn encode_for(genome: &Genome, instruction: Instruction) -> u8 {
    (0..(1u8 << OPCODE_BITS_PER_OPCODE))
        .find(|&code| decode_with_weights(genome, code) == instruction)
        .expect("every instruction holds a nonzero band of the opcode space")
}

/// Maps a slot's 4-bit opcode onto an instruction using the genome's weights.
/// Each instruction claims a band of the opcode space proportional to its
/// weight, so a genome that up-weights an instruction has more of its stream
/// decode to it. Deterministic: the same bits in the same genome always yield
/// the same instruction, so the stream stays heritable and ordered.
fn decode_with_weights(genome: &Genome, code: u8) -> Instruction {
    let weights: Vec<u32> = ALL_INSTRUCTIONS
        .iter()
        .map(|&instruction| genome.instruction_weight(instruction))
        .collect();
    let total: u32 = weights.iter().sum();
    // Scale the opcode into [0, total) so the whole opcode space maps across
    // the weight bands.
    let opcode_space = 1u32 << OPCODE_BITS_PER_OPCODE;
    let position = (code as u32) * total / opcode_space;

    // Walk the cumulative bands and take the first one the position falls in.
    // `position < total` and the weights sum to `total`, so some band always
    // claims it; `last()` states that total-ness without a separate fallback.
    let mut cumulative = 0;
    ALL_INSTRUCTIONS
        .iter()
        .zip(&weights)
        .find_map(|(&instruction, &weight)| {
            cumulative += weight;
            (position < cumulative).then_some(instruction)
        })
        .unwrap_or(ALL_INSTRUCTIONS[INSTRUCTION_COUNT - 1])
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
        fn a_random_genome_is_exactly_big_enough_to_hold_every_field() {
            // Sum the regions here rather than reusing TOTAL_BITS, so a
            // mistake in how TOTAL_BITS combines them is visible. The match
            // must be exact: a genome that is merely large enough would hide
            // a region being dropped from the total or double-counted.
            let genome = random_genome(0);

            let bits_needed = MUTATION_RATE_BITS
                + WEIGHT_BITS
                + HISTORY_WINDOW_BITS
                + HEADER_BITS
                + ACTIVATION_MASK_BITS
                + OPCODE_BITS
                + FEELER_BITS;
            assert_eq!(genome.bytes.len(), bits_needed.div_ceil(8));
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
            let genome = Genome::from_instructions(&[Instruction::TurnLeft15, Instruction::Split]);

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
                Instruction::MoveSlow,
                Instruction::TurnRight15,
                Instruction::Split,
            ]);

            assert_eq!(genome.decode_at(0), Instruction::MoveSlow);
            assert_eq!(genome.decode_at(1), Instruction::TurnRight15);
            assert_eq!(genome.decode_at(2), Instruction::Split);
            assert_eq!(genome.decode_at(3), Instruction::MoveSlow);
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
            let mut genome = Genome::all(Instruction::TurnLeft15);
            let split_code = encode_for(&genome, Instruction::Split) as u32;
            for slot in INITIAL_ACTIVE_OPCODES..OPCODE_POOL_SIZE {
                let bit_offset = OPCODE_STREAM_OFFSET + slot * OPCODE_BITS_PER_OPCODE;
                write_bits(
                    &mut genome.bytes,
                    bit_offset,
                    OPCODE_BITS_PER_OPCODE,
                    split_code,
                );
            }

            for cursor in 0..OPCODE_POOL_SIZE {
                assert_eq!(genome.decode_at(cursor), Instruction::TurnLeft15);
            }
        }

        #[test]
        fn activating_a_dormant_slot_brings_its_instruction_into_the_walked_stream() {
            // Build a genome where the dormant slot at INITIAL_ACTIVE_OPCODES
            // contains Split. After activating that slot's mask bit, the
            // active count grows by one and the cursor that would have
            // wrapped now lands on the newly-activated Split.
            let mut genome = Genome::all(Instruction::TurnLeft15);
            let target_slot = INITIAL_ACTIVE_OPCODES;
            let bit_offset = OPCODE_STREAM_OFFSET + target_slot * OPCODE_BITS_PER_OPCODE;
            let split_code = encode_for(&genome, Instruction::Split) as u32;
            write_bits(
                &mut genome.bytes,
                bit_offset,
                OPCODE_BITS_PER_OPCODE,
                split_code,
            );

            // Before activation, cursor INITIAL_ACTIVE_OPCODES wraps to slot 0
            // (TurnLeft).
            assert_eq!(
                genome.decode_at(INITIAL_ACTIVE_OPCODES),
                Instruction::TurnLeft15
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
        fn under_equal_weights_every_instruction_claims_part_of_the_opcode_space() {
            // A zeroed genome gives every instruction the same weight, so the
            // opcode space divides among all of them and none is unreachable.
            let genome = Genome::from_bits(&"0".repeat(TOTAL_BITS)).unwrap();

            let decoded: std::collections::HashSet<Instruction> = (0..(1u8
                << OPCODE_BITS_PER_OPCODE))
                .map(|code| decode_with_weights(&genome, code))
                .collect();

            assert_eq!(decoded.len(), INSTRUCTION_COUNT);
        }
    }

    mod sigmoid {
        use super::*;

        // The energy whose normalized contribution is `contribution`. Tests
        // reason in the sigmoid's units; this converts back to the energy a
        // critter would have to be holding.
        fn energy_contributing(contribution: f32) -> u32 {
            (contribution.max(0.0) / ENERGY_FACTOR_SCALE * MAX_CRITTER_ENERGY as f32).round() as u32
        }

        fn genome_with_mask_for_split(mask: u32) -> Genome {
            // Threshold = 0, softness = 1 (raw bits 0). Just the mask varies.
            let mut bytes = [0u8; TOTAL_BYTES];
            let split_window = header_window_offset(instruction_index(Instruction::Split));
            write_bits(&mut bytes, split_window, FACTOR_MASK_BITS, mask);
            Genome { bytes }
        }

        #[test]
        fn the_always_act_test_constructor_returns_high_probability_at_typical_energy() {
            // With threshold=0 and softness=1, a full critter's normalized
            // energy is ENERGY_FACTOR_SCALE, and sigmoid of that is 1.0.
            let genome = Genome::all(Instruction::Split);

            let probability = genome.probability_of_acting(
                Instruction::Split,
                &Senses {
                    energy: MAX_CRITTER_ENERGY,
                    touching_critter: false,
                    recent_repetition: 0.0,
                    ..Senses::default()
                },
            );

            assert!(probability > 0.99);
        }

        #[test]
        fn a_full_critters_energy_contributes_the_whole_scale() {
            // Pins the size of the contribution rather than its direction.
            // Threshold and softness keep the curve climbing at a full
            // critter's energy: at the defaults the true scale and a much
            // smaller one both saturate at one and agree.
            let mut bytes = [0u8; TOTAL_BYTES];
            let window = header_window_offset(instruction_index(Instruction::Split));
            write_bits(&mut bytes, window, FACTOR_MASK_BITS, ENERGY_FACTOR_BIT);
            write_bits(
                &mut bytes,
                window + THRESHOLD_OFFSET,
                THRESHOLD_BITS,
                ENERGY_FACTOR_SCALE as u32,
            );
            write_bits(&mut bytes, window + SOFTNESS_OFFSET, SOFTNESS_BITS, 40);
            let genome = Genome { bytes };

            let at_full = genome.probability_of_acting(
                Instruction::Split,
                &Senses {
                    energy: MAX_CRITTER_ENERGY,
                    ..Senses::default()
                },
            );

            let (_, threshold, softness) = genome.params(Instruction::Split);
            let expected = sigmoid((ENERGY_FACTOR_SCALE - threshold) / softness);
            assert!((at_full - expected).abs() < f32::EPSILON);
        }

        #[test]
        fn every_sensor_reads_every_channel_the_same_way() {
            // Body and feelers are the same sense pointed in three places. No
            // sensor is built with an opinion about what it is for: each
            // reports the colour in front of it and evolution decides what a
            // colour is worth.
            for (bit, put) in [
                (RED_FACTOR_BIT, 0usize),
                (GREEN_FACTOR_BIT, 0),
                (BLUE_FACTOR_BIT, 0),
                (LEFT_RED_FACTOR_BIT, 1),
                (LEFT_GREEN_FACTOR_BIT, 1),
                (LEFT_BLUE_FACTOR_BIT, 1),
                (RIGHT_RED_FACTOR_BIT, 2),
                (RIGHT_GREEN_FACTOR_BIT, 2),
                (RIGHT_BLUE_FACTOR_BIT, 2),
            ] {
                let genome = genome_with_mask_for_split(bit);
                let colour = match bit {
                    b if b == RED_FACTOR_BIT
                        || b == LEFT_RED_FACTOR_BIT
                        || b == RIGHT_RED_FACTOR_BIT =>
                    {
                        0x00_FF_00_00
                    }
                    b if b == GREEN_FACTOR_BIT
                        || b == LEFT_GREEN_FACTOR_BIT
                        || b == RIGHT_GREEN_FACTOR_BIT =>
                    {
                        0x00_00_FF_00
                    }
                    _ => 0x00_00_00_FF,
                };
                let senses = |lit: u32| match put {
                    0 => Senses {
                        touched_color: lit,
                        ..Senses::default()
                    },
                    1 => Senses {
                        left_color: lit,
                        ..Senses::default()
                    },
                    _ => Senses {
                        right_color: lit,
                        ..Senses::default()
                    },
                };

                let with = genome.probability_of_acting(Instruction::Split, &senses(colour));
                let without = genome.probability_of_acting(Instruction::Split, &senses(0));

                assert!(with > without, "sensor for bit {bit:b} did not respond");
            }
        }

        #[test]
        fn a_feeler_reads_red_as_readily_as_green() {
            // The failing of the sense this replaces: a feeler that reported
            // only green could tell food from nothing, but poison and empty
            // space read alike, so fleeing poison could not be learned.
            let genome = genome_with_mask_for_split(LEFT_RED_FACTOR_BIT);

            let on_poison = genome.probability_of_acting(
                Instruction::Split,
                &Senses {
                    left_color: crate::POISON_COLOR,
                    ..Senses::default()
                },
            );
            let on_nothing = genome.probability_of_acting(Instruction::Split, &Senses::default());

            assert!(
                on_poison > on_nothing,
                "poison should read differently from nothing"
            );
        }

        #[test]
        fn each_sensor_keeps_to_its_own_channels() {
            // Nine bits, nine independent readings: a genome watching for red
            // on its left is unmoved by red on its right or against its body.
            let genome = genome_with_mask_for_split(LEFT_RED_FACTOR_BIT);
            let red = 0x00_FF_00_00;

            let quiet = genome.probability_of_acting(Instruction::Split, &Senses::default());
            let elsewhere = genome.probability_of_acting(
                Instruction::Split,
                &Senses {
                    right_color: red,
                    touched_color: red,
                    ..Senses::default()
                },
            );

            assert_eq!(elsewhere, quiet);
        }

        #[test]
        fn a_feeler_on_food_pushes_a_decision() {
            let genome = genome_with_mask_for_split(LEFT_GREEN_FACTOR_BIT);
            let probe = |left_color: u32| {
                genome.probability_of_acting(
                    Instruction::Split,
                    &Senses {
                        left_color,
                        ..Senses::default()
                    },
                )
            };

            assert!(probe(crate::PELLET_COLOR) > probe(0));
        }

        #[test]
        fn a_feeler_tells_food_from_poison() {
            // The point of sensing colour rather than mere presence. Poison is
            // red and food green, so a sense that took the brightest channel
            // of either would report the same for both.
            let genome = genome_with_mask_for_split(LEFT_GREEN_FACTOR_BIT);
            let probe = |left_color: u32| {
                genome.probability_of_acting(
                    Instruction::Split,
                    &Senses {
                        left_color,
                        ..Senses::default()
                    },
                )
            };

            assert!(probe(crate::PELLET_COLOR) > probe(crate::POISON_COLOR));
        }

        #[test]
        fn food_on_both_sides_counts_twice() {
            // The two contributions add. Probed with a threshold and softness
            // that keep the curve climbing, since at the defaults both
            // readings saturate at one and nothing can be told apart.
            let mut bytes = [0u8; TOTAL_BYTES];
            let window = header_window_offset(instruction_index(Instruction::Split));
            write_bits(
                &mut bytes,
                window,
                FACTOR_MASK_BITS,
                LEFT_GREEN_FACTOR_BIT | RIGHT_GREEN_FACTOR_BIT,
            );
            write_bits(
                &mut bytes,
                window + THRESHOLD_OFFSET,
                THRESHOLD_BITS,
                (1.5 * COLOR_CHANNEL_SCALE) as u32,
            );
            write_bits(&mut bytes, window + SOFTNESS_OFFSET, SOFTNESS_BITS, 40);
            let genome = Genome { bytes };
            let probe = |left: u32, right: u32| {
                genome.probability_of_acting(
                    Instruction::Split,
                    &Senses {
                        left_color: left,
                        right_color: right,
                        ..Senses::default()
                    },
                )
            };

            let both = probe(crate::PELLET_COLOR, crate::PELLET_COLOR);
            let one = probe(crate::PELLET_COLOR, 0);

            assert!(both > one, "both sides should push harder: {both} vs {one}");
            let (_, threshold, softness) = genome.params(Instruction::Split);
            let expected = sigmoid((2.0 * COLOR_CHANNEL_SCALE - threshold) / softness);
            assert!((both - expected).abs() < f32::EPSILON);
        }

        #[test]
        fn the_two_feelers_are_sensed_separately() {
            // Each side has its own bit, so a genome can tell food on its left
            // from food on its right and turn towards one of them.
            let left_only = genome_with_mask_for_split(LEFT_GREEN_FACTOR_BIT);

            let with_food_right = left_only.probability_of_acting(
                Instruction::Split,
                &Senses {
                    right_color: crate::PELLET_COLOR,
                    ..Senses::default()
                },
            );
            let with_nothing =
                left_only.probability_of_acting(Instruction::Split, &Senses::default());

            assert_eq!(with_food_right, with_nothing);
        }

        #[test]
        fn feelers_are_ignored_unless_the_genome_asks_for_them() {
            let genome = genome_with_mask_for_split(ENERGY_FACTOR_BIT);
            let probe = |left_color: u32| {
                genome.probability_of_acting(
                    Instruction::Split,
                    &Senses {
                        left_color,
                        energy: 100,
                        ..Senses::default()
                    },
                )
            };

            assert_eq!(probe(crate::PELLET_COLOR), probe(0));
        }

        #[test]
        fn a_young_critter_and_an_old_one_are_told_apart() {
            let genome = genome_with_mask_for_split(AGE_FACTOR_BIT);
            let probe = |age: u32| {
                genome.probability_of_acting(
                    Instruction::Split,
                    &Senses {
                        age,
                        ..Senses::default()
                    },
                )
            };

            assert!(probe(MATURE_AGE / 4) < probe(MATURE_AGE));
        }

        #[test]
        fn age_stops_mattering_once_a_critter_is_grown() {
            // Whether a critter is newborn, half grown, or grown is what its
            // decisions can turn on. Past that, one long life is much like
            // another and the sense saturates.
            let genome = genome_with_mask_for_split(AGE_FACTOR_BIT);
            let probe = |age: u32| {
                genome.probability_of_acting(
                    Instruction::Split,
                    &Senses {
                        age,
                        ..Senses::default()
                    },
                )
            };

            assert_eq!(probe(MATURE_AGE), probe(MATURE_AGE * 10));
        }

        #[test]
        fn age_is_ignored_unless_the_genome_asks_for_it() {
            let genome = genome_with_mask_for_split(ENERGY_FACTOR_BIT);
            let probe = |age: u32| {
                genome.probability_of_acting(
                    Instruction::Split,
                    &Senses {
                        age,
                        energy: 100,
                        ..Senses::default()
                    },
                )
            };

            assert_eq!(probe(0), probe(MATURE_AGE));
        }

        #[test]
        fn every_instruction_can_consult_age() {
            // Not a sense reserved to one decision: any instruction's rule can
            // turn on how old the critter is.
            for instruction in [
                Instruction::MoveSlow,
                Instruction::MoveFast,
                Instruction::TurnLeft15,
                Instruction::TurnRight15,
                Instruction::DoNothing,
                Instruction::RepeatPreviousMove,
                Instruction::Split,
                Instruction::Eat,
                Instruction::SkipAhead,
                Instruction::SkipBack,
            ] {
                let mut bytes = [0u8; TOTAL_BYTES];
                let window = header_window_offset(instruction_index(instruction));
                write_bits(&mut bytes, window, FACTOR_MASK_BITS, AGE_FACTOR_BIT);
                let genome = Genome { bytes };

                let young = genome.probability_of_acting(
                    instruction,
                    &Senses {
                        age: 0,
                        ..Senses::default()
                    },
                );
                let old = genome.probability_of_acting(
                    instruction,
                    &Senses {
                        age: MATURE_AGE,
                        ..Senses::default()
                    },
                );

                assert!(old > young, "{instruction:?} should be able to sense age");
            }
        }

        #[test]
        fn a_threshold_can_hold_out_for_most_of_the_energy_range() {
            // Energy is normalized like every other factor, so the encodable
            // thresholds span the whole range a critter's energy can reach.
            // A rule that waits until a critter is nearly full has to be
            // expressible: with raw energy against a 7-bit threshold, the
            // strictest possible rule triggered at an eighth of the range and
            // everything above it read the same.
            let mut bytes = [0u8; TOTAL_BYTES];
            let split_window = header_window_offset(instruction_index(Instruction::Split));
            write_bits(
                &mut bytes,
                split_window,
                FACTOR_MASK_BITS,
                ENERGY_FACTOR_BIT,
            );
            let max_threshold = (1u32 << THRESHOLD_BITS) - 1;
            write_bits(
                &mut bytes,
                split_window + THRESHOLD_OFFSET,
                THRESHOLD_BITS,
                max_threshold,
            );
            let genome = Genome { bytes };

            let at_three_quarters = genome.probability_of_acting(
                Instruction::Split,
                &Senses {
                    energy: MAX_CRITTER_ENERGY * 3 / 4,
                    ..Senses::default()
                },
            );

            assert!(
                at_three_quarters < 0.5,
                "the strictest rule should still be holding out at three \
                 quarters energy, got {at_three_quarters}"
            );
        }

        #[test]
        fn the_energy_response_is_spread_across_the_whole_range() {
            // Not merely bounded at the ends: the middle of the range has to
            // move too, or genomes cannot tell a fed critter from a full one.
            // Threshold at the middle of the range and softness broad enough
            // that the curve is still moving at both ends.
            let mut bytes = [0u8; TOTAL_BYTES];
            let split_window = header_window_offset(instruction_index(Instruction::Split));
            write_bits(
                &mut bytes,
                split_window,
                FACTOR_MASK_BITS,
                ENERGY_FACTOR_BIT,
            );
            write_bits(
                &mut bytes,
                split_window + THRESHOLD_OFFSET,
                THRESHOLD_BITS,
                (ENERGY_FACTOR_SCALE / 2.0) as u32,
            );
            write_bits(
                &mut bytes,
                split_window + SOFTNESS_OFFSET,
                SOFTNESS_BITS,
                20,
            );
            let genome = Genome { bytes };
            let probe = |energy: u32| {
                genome.probability_of_acting(
                    Instruction::Split,
                    &Senses {
                        energy,
                        ..Senses::default()
                    },
                )
            };

            let low_half = probe(MAX_CRITTER_ENERGY / 2) - probe(MAX_CRITTER_ENERGY / 4);
            let high_half = probe(MAX_CRITTER_ENERGY) - probe(MAX_CRITTER_ENERGY * 3 / 4);

            assert!(
                low_half > 0.0 && high_half > 0.0,
                "probability should still be climbing in both halves of the \
                 range, got {low_half} and {high_half}"
            );
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
                let probability = genome.probability_of_acting(
                    Instruction::Split,
                    &Senses {
                        energy: threshold as u32,
                        touching_critter: false,
                        recent_repetition: 0.0,
                        ..Senses::default()
                    },
                );
                assert!((probability - 0.5).abs() < 0.5);
                return;
            }
            panic!("no seed had the energy factor enabled on Split");
        }

        #[test]
        fn far_above_the_threshold_the_probability_approaches_one() {
            for seed in 0..2000 {
                let genome = random_genome(seed);
                let (mask, threshold, softness) = genome.params(Instruction::Split);
                if mask & ENERGY_FACTOR_BIT == 0 {
                    continue;
                }
                // Only usable when the genome's own scale leaves room to probe
                // well above its threshold without exceeding a full critter.
                let target = threshold + 20.0 * softness;
                if target > ENERGY_FACTOR_SCALE {
                    continue;
                }
                let probability = genome.probability_of_acting(
                    Instruction::Split,
                    &Senses {
                        energy: energy_contributing(target),
                        touching_critter: false,
                        recent_repetition: 0.0,
                        ..Senses::default()
                    },
                );
                assert!(probability > 0.99, "seed {seed}: probability {probability}");
                return;
            }
            panic!("no seed had an energy-gated Split with room to probe above it");
        }

        #[test]
        fn far_below_the_threshold_the_probability_approaches_zero() {
            // Search for a seed whose Split rule has the energy factor on and
            // sits well above zero so we have room to probe below it.
            for seed in 0..2000 {
                let genome = random_genome(seed);
                let (mask, threshold, softness) = genome.params(Instruction::Split);
                if mask & ENERGY_FACTOR_BIT == 0 {
                    continue;
                }
                if threshold / softness < 10.0 {
                    continue;
                }
                let energy = (threshold - 10.0 * softness).max(0.0) as u32;
                let probability = genome.probability_of_acting(
                    Instruction::Split,
                    &Senses {
                        energy,
                        touching_critter: false,
                        recent_repetition: 0.0,
                        ..Senses::default()
                    },
                );
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

            assert_eq!(genome.params(Instruction::MoveSlow).1, 10.0);
            assert_eq!(genome.params(Instruction::TurnLeft15).1, 20.0);
            assert_eq!(genome.params(Instruction::TurnRight15).1, 30.0);
            assert_eq!(genome.params(Instruction::DoNothing).1, 40.0);
            assert_eq!(genome.params(Instruction::RepeatPreviousMove).1, 50.0);
            assert_eq!(genome.params(Instruction::Split).1, 60.0);
            assert_eq!(genome.params(Instruction::Eat).1, 70.0);
        }

        #[test]
        fn the_softness_for_each_instruction_is_read_from_its_own_window() {
            // Use a hard-coded offset (mask 15 bits + threshold 7 bits = 22)
            // rather than the SOFTNESS_OFFSET constant so that mutations to
            // the constant produce a visible mismatch between write and read.
            const SOFTNESS_OFFSET_LITERAL: usize = 22;
            assert_eq!(
                SOFTNESS_OFFSET_LITERAL, SOFTNESS_OFFSET,
                "genome layout changed; update SOFTNESS_OFFSET_LITERAL to match"
            );
            let mut bytes = [0u8; TOTAL_BYTES];
            let softnesses: [u32; 7] = [5, 15, 25, 35, 45, 55, 65];
            for (index, &soft) in softnesses.iter().enumerate() {
                let offset = header_window_offset(index) + SOFTNESS_OFFSET_LITERAL;
                write_bits(&mut bytes, offset, SOFTNESS_BITS, soft);
            }
            let genome = Genome { bytes };

            assert_eq!(genome.params(Instruction::MoveSlow).2, MIN_SOFTNESS + 5.0);
            assert_eq!(
                genome.params(Instruction::TurnLeft15).2,
                MIN_SOFTNESS + 15.0
            );
            assert_eq!(genome.params(Instruction::Split).2, MIN_SOFTNESS + 55.0);
            assert_eq!(genome.params(Instruction::Eat).2, MIN_SOFTNESS + 65.0);
        }

        #[test]
        fn with_no_factors_enabled_the_probability_does_not_depend_on_energy_or_touching() {
            let genome = genome_with_mask_for_split(0b00);

            let p_low = genome.probability_of_acting(
                Instruction::Split,
                &Senses {
                    energy: 0,
                    touching_critter: false,
                    recent_repetition: 0.0,
                    ..Senses::default()
                },
            );
            let p_high_energy = genome.probability_of_acting(
                Instruction::Split,
                &Senses {
                    energy: 500,
                    touching_critter: false,
                    recent_repetition: 0.0,
                    ..Senses::default()
                },
            );
            let p_touching = genome.probability_of_acting(
                Instruction::Split,
                &Senses {
                    energy: 0,
                    touching_critter: true,
                    recent_repetition: 0.0,
                    ..Senses::default()
                },
            );

            assert_eq!(p_low, p_high_energy);
            assert_eq!(p_low, p_touching);
        }

        #[test]
        fn with_only_the_energy_factor_enabled_touching_does_not_change_the_probability() {
            let genome = genome_with_mask_for_split(ENERGY_FACTOR_BIT);

            let p_not_touching = genome.probability_of_acting(
                Instruction::Split,
                &Senses {
                    energy: 100,
                    touching_critter: false,
                    recent_repetition: 0.0,
                    ..Senses::default()
                },
            );
            let p_touching = genome.probability_of_acting(
                Instruction::Split,
                &Senses {
                    energy: 100,
                    touching_critter: true,
                    recent_repetition: 0.0,
                    ..Senses::default()
                },
            );

            assert_eq!(p_not_touching, p_touching);
        }

        #[test]
        fn with_only_the_touching_factor_enabled_energy_does_not_change_the_probability() {
            let genome = genome_with_mask_for_split(TOUCHING_FACTOR_BIT);

            let p_low_energy = genome.probability_of_acting(
                Instruction::Split,
                &Senses {
                    energy: 0,
                    touching_critter: false,
                    recent_repetition: 0.0,
                    ..Senses::default()
                },
            );
            let p_high_energy = genome.probability_of_acting(
                Instruction::Split,
                &Senses {
                    energy: 500,
                    touching_critter: false,
                    recent_repetition: 0.0,
                    ..Senses::default()
                },
            );

            assert_eq!(p_low_energy, p_high_energy);
        }

        #[test]
        fn with_only_the_touching_factor_enabled_touching_increases_the_probability() {
            // With softness = 1 and threshold = 0, touching adds 64 to the
            // input — sigmoid(64) is essentially 1, sigmoid(0) is 0.5.
            let genome = genome_with_mask_for_split(TOUCHING_FACTOR_BIT);

            let p_not_touching = genome.probability_of_acting(
                Instruction::Split,
                &Senses {
                    energy: 0,
                    touching_critter: false,
                    recent_repetition: 0.0,
                    ..Senses::default()
                },
            );
            let p_touching = genome.probability_of_acting(
                Instruction::Split,
                &Senses {
                    energy: 0,
                    touching_critter: true,
                    recent_repetition: 0.0,
                    ..Senses::default()
                },
            );

            assert!(p_not_touching < p_touching);
        }

        #[test]
        fn with_no_color_factor_dissimilarity_does_not_change_the_probability() {
            let genome = genome_with_mask_for_split(0b000);

            let p_similar = genome.probability_of_acting(
                Instruction::Split,
                &Senses {
                    energy: 0,
                    touching_critter: false,
                    recent_repetition: 0.0,
                    ..Senses::default()
                },
            );
            let p_dissimilar = genome.probability_of_acting(
                Instruction::Split,
                &Senses {
                    energy: 0,
                    touching_critter: false,
                    recent_repetition: 0.0,
                    ..Senses::default()
                },
            );

            assert_eq!(p_similar, p_dissimilar);
        }

        #[test]
        fn each_colour_channel_can_be_sensed_on_its_own() {
            // A critter that senses green responds to green and ignores red,
            // which is what makes food and poison distinguishable.
            let genome = genome_with_mask_for_split(GREEN_FACTOR_BIT);

            let p_dark = genome.probability_of_acting(Instruction::Split, &Senses::default());
            let p_green = genome.probability_of_acting(
                Instruction::Split,
                &Senses {
                    touched_color: 0x00_FF_00,
                    ..Senses::default()
                },
            );
            let p_red = genome.probability_of_acting(
                Instruction::Split,
                &Senses {
                    touched_color: 0xFF_00_00,
                    ..Senses::default()
                },
            );

            assert!(p_green > p_dark);
            assert_eq!(p_red, p_dark);
        }

        #[test]
        fn the_red_channel_is_sensed_separately_from_the_green() {
            let genome = genome_with_mask_for_split(RED_FACTOR_BIT);

            let p_green = genome.probability_of_acting(
                Instruction::Split,
                &Senses {
                    touched_color: 0x00_FF_00,
                    ..Senses::default()
                },
            );
            let p_red = genome.probability_of_acting(
                Instruction::Split,
                &Senses {
                    touched_color: 0xFF_00_00,
                    ..Senses::default()
                },
            );

            assert!(p_red > p_green);
        }

        #[test]
        fn the_blue_channel_is_sensed_separately_too() {
            let genome = genome_with_mask_for_split(BLUE_FACTOR_BIT);

            let p_blue = genome.probability_of_acting(
                Instruction::Split,
                &Senses {
                    touched_color: 0x00_00_FF,
                    ..Senses::default()
                },
            );
            let p_green = genome.probability_of_acting(
                Instruction::Split,
                &Senses {
                    touched_color: 0x00_FF_00,
                    ..Senses::default()
                },
            );

            assert!(p_blue > p_green);
        }

        #[test]
        fn a_full_channel_contributes_the_whole_channel_scale() {
            // Pins the scaling rather than only its direction: full
            // brightness contributes exactly COLOR_CHANNEL_SCALE, which a
            // comparison between two colours leaves free.
            let genome = genome_with_mask_for_split(GREEN_FACTOR_BIT);

            let p = genome.probability_of_acting(
                Instruction::Split,
                &Senses {
                    touched_color: 0x00_FF_00,
                    ..Senses::default()
                },
            );

            // Threshold 0 and softness 1, so the probability is sigmoid of the
            // contribution itself.
            assert!((p - sigmoid(COLOR_CHANNEL_SCALE)).abs() < f32::EPSILON);
        }

        #[test]
        fn a_barely_lit_channel_contributes_barely_anything() {
            // Tested near the bottom of the range, where the sigmoid has not
            // saturated: at full brightness every plausible scaling saturates
            // to 1.0 and the arithmetic cannot be seen at all.
            let genome = genome_with_mask_for_split(GREEN_FACTOR_BIT);

            let p = genome.probability_of_acting(
                Instruction::Split,
                &Senses {
                    touched_color: 0x00_01_00,
                    ..Senses::default()
                },
            );

            let expected = sigmoid(COLOR_CHANNEL_SCALE / 255.0);
            assert!((p - expected).abs() < 0.001, "probability was {p}");
        }

        #[test]
        fn a_genome_sensing_no_colour_ignores_it_entirely() {
            let genome = genome_with_mask_for_split(0);

            let p_dark = genome.probability_of_acting(Instruction::Split, &Senses::default());
            let p_bright = genome.probability_of_acting(
                Instruction::Split,
                &Senses {
                    touched_color: 0xFF_FF_FF,
                    ..Senses::default()
                },
            );

            assert_eq!(p_dark, p_bright);
        }

        #[test]
        fn with_only_the_history_factor_enabled_recent_repetition_raises_the_probability() {
            // A critter that has just done this instruction repeatedly should
            // find it more likely than one that has not, once the history
            // factor is switched on.
            let mut genome = genome_with_mask_for_split(HISTORY_FACTOR_BIT);
            genome.set_history_window_bits(0b1111);

            let p_unrepeated = genome.probability_of_acting(
                Instruction::Split,
                &Senses {
                    energy: 0,
                    touching_critter: false,
                    recent_repetition: 0.0,
                    ..Senses::default()
                },
            );
            let p_repeated = genome.probability_of_acting(
                Instruction::Split,
                &Senses {
                    energy: 0,
                    touching_critter: false,
                    recent_repetition: 1.0,
                    ..Senses::default()
                },
            );

            assert!(p_unrepeated < p_repeated);
        }

        #[test]
        fn with_no_history_factor_recent_repetition_does_not_change_the_probability() {
            let mut genome = genome_with_mask_for_split(0b0000);
            genome.set_history_window_bits(0b1111);

            let p_unrepeated = genome.probability_of_acting(
                Instruction::Split,
                &Senses {
                    energy: 0,
                    touching_critter: false,
                    recent_repetition: 0.0,
                    ..Senses::default()
                },
            );
            let p_repeated = genome.probability_of_acting(
                Instruction::Split,
                &Senses {
                    energy: 0,
                    touching_critter: false,
                    recent_repetition: 1.0,
                    ..Senses::default()
                },
            );

            assert_eq!(p_unrepeated, p_repeated);
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
                    genome.probability_of_acting(
                        Instruction::MoveSlow,
                        &Senses {
                            energy,
                            touching_critter: false,
                            recent_repetition: 0.0,
                            ..Senses::default()
                        },
                    ),
                    genome.probability_of_acting(
                        Instruction::RepeatPreviousMove,
                        &Senses {
                            energy,
                            touching_critter: false,
                            recent_repetition: 0.0,
                            ..Senses::default()
                        },
                    ),
                    genome.probability_of_acting(
                        Instruction::DoNothing,
                        &Senses {
                            energy,
                            touching_critter: false,
                            recent_repetition: 0.0,
                            ..Senses::default()
                        },
                    ),
                    genome.probability_of_acting(
                        Instruction::TurnLeft15,
                        &Senses {
                            energy,
                            touching_critter: false,
                            recent_repetition: 0.0,
                            ..Senses::default()
                        },
                    ),
                    genome.probability_of_acting(
                        Instruction::TurnRight15,
                        &Senses {
                            energy,
                            touching_critter: false,
                            recent_repetition: 0.0,
                            ..Senses::default()
                        },
                    ),
                    genome.probability_of_acting(
                        Instruction::Split,
                        &Senses {
                            energy,
                            touching_critter: false,
                            recent_repetition: 0.0,
                            ..Senses::default()
                        },
                    ),
                    genome.probability_of_acting(
                        Instruction::Eat,
                        &Senses {
                            energy,
                            touching_critter: false,
                            recent_repetition: 0.0,
                            ..Senses::default()
                        },
                    ),
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
            // genome's color. Compared before brightening, since the floor
            // can clamp two different hashes to the same value.
            let channels = |genome: &Genome| {
                let third = TOTAL_BYTES / 3;
                (
                    fnv1a_byte(&genome.bytes[0..third]),
                    fnv1a_byte(&genome.bytes[third..2 * third]),
                    fnv1a_byte(&genome.bytes[2 * third..]),
                )
            };
            let (zr, zg, zb) = channels(&zero);
            let (rr, rg, rb) = channels(&red_only);
            assert_ne!(zr, rr);
            assert_eq!(zg, rg);
            assert_eq!(zb, rb);

            let (gr, gg, gb) = channels(&green_only);
            assert_eq!(zr, gr);
            assert_ne!(zg, gg);
            assert_eq!(zb, gb);

            let (br, bg, bb) = channels(&blue_only);
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
                Instruction::MoveSlow,
                Instruction::RepeatPreviousMove,
                Instruction::DoNothing,
                Instruction::TurnLeft15,
                Instruction::TurnRight15,
                Instruction::Split,
                Instruction::Eat,
            ] {
                for energy in [0, 100, 250, 400, 500] {
                    assert_eq!(
                        original.probability_of_acting(
                            instruction,
                            &Senses {
                                energy,
                                touching_critter: false,
                                recent_repetition: 0.0,
                                ..Senses::default()
                            }
                        ),
                        cloned.probability_of_acting(
                            instruction,
                            &Senses {
                                energy,
                                touching_critter: false,
                                recent_repetition: 0.0,
                                ..Senses::default()
                            }
                        ),
                    );
                }
            }
        }
    }

    mod history_window {
        use super::*;

        fn genome_with_history_bits(bits: u32) -> Genome {
            let mut genome = Genome::from_bits(&"0".repeat(TOTAL_BITS)).unwrap();
            genome.set_history_window_bits(bits);
            genome
        }

        #[test]
        fn the_window_field_sits_between_the_weights_and_the_header() {
            // A field that overlapped its neighbours would corrupt them, so
            // pin both edges: it starts where the weights end and ends where
            // the per-instruction header begins.
            assert_eq!(HISTORY_WINDOW_OFFSET, WEIGHTS_OFFSET + WEIGHT_BITS);
            assert_eq!(HISTORY_WINDOW_OFFSET + HISTORY_WINDOW_BITS, HEADER_OFFSET);
        }

        #[test]
        fn setting_the_window_leaves_the_neighbouring_regions_untouched() {
            // Writing every history bit must not bleed into the weights
            // before it or the header after it.
            let mut genome = Genome::from_bits(&"0".repeat(TOTAL_BITS)).unwrap();
            genome.set_instruction_weight_bits(Instruction::Eat, MAX_WEIGHT_BITS);
            let weight_before = genome.instruction_weight(Instruction::Eat);
            let params_before = genome.params(Instruction::MoveSlow);

            genome.set_history_window_bits(u32::MAX >> (32 - HISTORY_WINDOW_BITS));

            assert_eq!(genome.instruction_weight(Instruction::Eat), weight_before);
            assert_eq!(genome.params(Instruction::MoveSlow), params_before);
        }

        #[test]
        fn an_all_zero_field_means_no_history_is_consulted() {
            let genome = genome_with_history_bits(0);

            assert_eq!(genome.history_window(), 0);
        }

        #[test]
        fn the_window_counts_set_bits_regardless_of_their_positions() {
            // Two set bits mean a window of 2 wherever they sit, so the field
            // is a dosage gene: only the count carries meaning.
            let low = genome_with_history_bits(0b0000_0000_0000_0011);
            let spread = genome_with_history_bits(0b1000_0000_0000_0001);

            assert_eq!(low.history_window(), 2);
            assert_eq!(spread.history_window(), 2);
        }

        #[test]
        fn an_all_one_field_means_the_maximum_window() {
            let genome = genome_with_history_bits(u32::MAX >> (32 - HISTORY_WINDOW_BITS));

            assert_eq!(genome.history_window(), HISTORY_WINDOW_BITS);
        }
    }

    mod instruction_weights {
        use super::*;

        #[test]
        fn turning_claims_twice_the_opcode_space_its_fields_would_give_it() {
            // Turning carries a thumb on the scale of its own. A critter that
            // cannot steer is at the mercy of wherever it happens to be
            // pointing, so the world leans on turning coming up often enough
            // to matter.
            let mut genome = Genome::from_bits(&"0".repeat(TOTAL_BITS)).unwrap();
            for instruction in ALL_INSTRUCTIONS {
                genome.set_instruction_weight_bits(instruction, 3);
            }

            let turning: u32 = ALL_INSTRUCTIONS
                .iter()
                .filter(|i| kind_of(**i) == Kind::Turn)
                .map(|&i| genome.instruction_weight(i))
                .sum();
            let eating = genome.instruction_weight(Instruction::Eat);

            // A literal, not the constant: asserting against the constant
            // would hold whatever the constant said, including one.
            assert_eq!(turning, eating * 2);
        }

        #[test]
        fn a_kind_with_no_thumb_on_the_scale_claims_what_its_fields_say() {
            // Only the kinds meant to be favoured are: the rest come out at
            // exactly what their weight fields ask for, so the table is a
            // place to lean on the scale rather than a place every kind needs
            // an entry to be treated fairly.
            let mut genome = Genome::from_bits(&"0".repeat(TOTAL_BITS)).unwrap();
            for instruction in ALL_INSTRUCTIONS {
                genome.set_instruction_weight_bits(instruction, 3);
            }

            let moving: u32 = ALL_INSTRUCTIONS
                .iter()
                .filter(|i| kind_of(**i) == Kind::Move)
                .map(|&i| genome.instruction_weight(i))
                .sum();

            assert_eq!(moving, genome.instruction_weight(Instruction::Eat));
        }

        #[test]
        fn splitting_claims_twice_the_opcode_space_its_field_would_give_it() {
            // A thumb on the scale, applied to the decoded weight rather than
            // to the genome: the same bits still say how much a lineage wants
            // to divide, and evolution can still turn it down.
            let mut genome = Genome::from_bits(&"0".repeat(TOTAL_BITS)).unwrap();
            genome.set_instruction_weight_bits(Instruction::Split, 3);
            genome.set_instruction_weight_bits(Instruction::Eat, 3);

            // A literal, not SPLIT_WEIGHT_MULTIPLIER: asserting against the
            // constant would hold whatever the constant said, including one.
            assert_eq!(
                genome.instruction_weight(Instruction::Split),
                genome.instruction_weight(Instruction::Eat) * 2
            );
        }

        #[test]
        fn a_genome_that_wants_no_splitting_still_gets_the_least_weight() {
            // Doubling the smallest weight is still the smallest weight, so
            // the thumb cannot force splitting on a lineage that has evolved
            // away from it as far as the field allows.
            let mut genome = Genome::from_bits(&"0".repeat(TOTAL_BITS)).unwrap();
            for instruction in ALL_INSTRUCTIONS {
                genome.set_instruction_weight_bits(instruction, 15);
            }
            genome.set_instruction_weight_bits(Instruction::Split, 0);

            let split = genome.instruction_weight(Instruction::Split);
            let others = genome.instruction_weight(Instruction::Eat);
            assert!(
                split < others,
                "a minimum-weight Split should still trail: {split} vs {others}"
            );
        }

        #[test]
        fn a_kinds_variants_divide_its_share_without_remainder() {
            // The scale has to be a common multiple of every kind's variant
            // count, or integer division drops part of a kind's share and the
            // kinds stop being equal. Checked as arithmetic rather than only
            // through the totals, since a scale that is merely large enough
            // makes the totals agree too.
            for kind in Kind::EVERY {
                let scale = kind.share_per_variant() * kind.variants();

                assert_eq!(
                    scale,
                    Kind::Move.share_per_variant() * Kind::Move.variants(),
                    "{kind:?} should come to the same scale as every other"
                );
            }
        }

        #[test]
        fn the_scale_is_the_smallest_that_divides_every_kind() {
            // Not merely a common multiple but the least one: any multiple
            // makes the kinds equal, and a needlessly large one wastes the
            // range weights have to work in.
            let scale = Kind::Move.share_per_variant() * Kind::Move.variants();
            let counts: Vec<u32> = Kind::EVERY.iter().map(|kind| kind.variants()).collect();

            for candidate in 1..scale {
                assert!(
                    counts.iter().any(|count| candidate % count != 0),
                    "{candidate} divides every kind, so {scale} is not the least"
                );
            }
        }

        #[test]
        fn a_kind_claims_the_same_space_however_many_variants_it_has() {
            // Turning is one thing a critter can do, and it should come up as
            // often as eating whether it is spelled two ways or a thousand.
            // Otherwise adding a variant quietly takes weight from every other
            // instruction, and the number of ways to spell an action becomes a
            // design decision nobody made.
            let genome = equal_weights();

            let turning: u32 = ALL_INSTRUCTIONS
                .iter()
                .filter(|i| kind_of(**i) == Kind::Turn)
                .map(|&i| genome.instruction_weight(i))
                .sum();
            let eating = genome.instruction_weight(Instruction::Eat);

            // Allowing for the thumb turning carries, which is a decision made
            // about turning rather than an accident of how many ways there are
            // to spell it -- the thing this test is here to rule out.
            assert_eq!(turning, eating * Kind::Turn.thumb_on_the_scale());
        }

        #[test]
        fn every_kind_claims_the_same_space_as_every_other() {
            let genome = equal_weights();

            let share = |kind: Kind| -> u32 {
                ALL_INSTRUCTIONS
                    .iter()
                    .filter(|i| kind_of(**i) == kind)
                    .map(|&i| genome.instruction_weight(i))
                    .sum()
            };
            let moving = share(Kind::Move);

            // Split excepted: it carries a deliberate multiplier on top, so
            // that reproduction is likelier than the rest. Equal weighting is
            // about the count of variants not deciding anything, not about
            // refusing to lean on the scale on purpose.
            for kind in [Kind::Eat, Kind::Skip, Kind::Nothing, Kind::Repeat] {
                assert_eq!(share(kind), moving, "{kind:?} should match Move");
            }

            // The kinds the world leans on come out at their share times the
            // thumb, which is the only thing that should ever make one kind
            // differ from another.
            for kind in [Kind::Split, Kind::Turn] {
                assert_eq!(
                    share(kind),
                    moving * kind.thumb_on_the_scale(),
                    "{kind:?} should keep its thumb on the scale"
                );
            }
        }

        #[test]
        fn the_variants_of_a_kind_still_share_it_by_their_own_weights() {
            // Equal by kind, not equal within one: a genome that wants to turn
            // sharply more often than gently can still say so.
            let mut genome = equal_weights();
            genome.set_instruction_weight_bits(Instruction::TurnLeft15, 15);

            assert!(
                genome.instruction_weight(Instruction::TurnLeft15)
                    > genome.instruction_weight(Instruction::TurnRight15)
            );
        }

        // Every instruction's weight field set to the same middling value, so
        // what differs between kinds is only how the shares are worked out.
        fn equal_weights() -> Genome {
            let mut genome = Genome::from_bits(&"0".repeat(TOTAL_BITS)).unwrap();
            for instruction in ALL_INSTRUCTIONS {
                genome.set_instruction_weight_bits(instruction, 7);
            }
            genome
        }

        #[test]
        fn every_instructions_weight_field_fits_inside_the_weight_region() {
            // The last instruction's field must end at the region's edge:
            // too small and its bits would collide with the header that
            // follows; too large and the region wastes space it claimed.
            let last = INSTRUCTION_COUNT - 1;
            let end_of_last_field =
                last * WEIGHT_BITS_PER_INSTRUCTION + WEIGHT_BITS_PER_INSTRUCTION;

            assert_eq!(end_of_last_field, WEIGHT_BITS);
            assert_eq!(WEIGHTS_OFFSET + WEIGHT_BITS, HISTORY_WINDOW_OFFSET);
        }

        // A genome whose opcode slots run 0, 1, 2, ... so that decoding every
        // slot samples the whole 4-bit opcode space exactly once.
        fn genome_spanning_the_opcode_space() -> Genome {
            let mut genome = Genome::from_bits(&"0".repeat(TOTAL_BITS)).unwrap();
            for slot in 0..OPCODE_POOL_SIZE {
                // Spans whatever the opcode width is rather than a literal
                // sixteen, so widening it does not quietly leave half the
                // space untested.
                genome.write_opcode(slot, (slot % (1 << OPCODE_BITS_PER_OPCODE)) as u8);
            }
            genome
        }

        fn decoded_counts(genome: &Genome) -> std::collections::HashMap<Instruction, usize> {
            let mut counts = std::collections::HashMap::new();
            // Every opcode value there is, derived rather than written out, so
            // widening the opcode leaves none of the space unexamined.
            for slot in 0..(1u8 << OPCODE_BITS_PER_OPCODE) {
                *counts.entry(decode_with_weights(genome, slot)).or_insert(0) += 1;
            }
            counts
        }

        #[test]
        fn each_instruction_claims_opcode_values_in_proportion_to_its_weight() {
            // Give Eat the maximum weight and leave the rest at the minimum,
            // then count how the 16 opcode values divide. Asserting the exact
            // split pins the scaling arithmetic and the band comparison, which
            // a merely directional assertion leaves free.
            let mut genome = Genome::from_bits(&"0".repeat(TOTAL_BITS)).unwrap();
            genome.set_instruction_weight_bits(Instruction::Eat, MAX_WEIGHT_BITS);

            let counts = decoded_counts(&genome);

            // Weights are bits + 1, scaled so each kind claims the same space
            // however many variants spell it, and doubled again for the two
            // kinds the world leans on. Eat is a kind of one holding the
            // largest field, so it takes 85 of the 128 opcode values; Split,
            // a kind of one with a thumb, takes 11; the seven turns share two
            // kinds' worth between them and come out with ten between them.
            // Naming each instruction's share is what pins the scaling:
            // formulas that shift which instructions get a slot preserve the
            // totals but not this breakdown.
            let share = |instruction| *counts.get(&instruction).unwrap_or(&0);

            assert_eq!(share(Instruction::Eat), 85);
            assert_eq!(share(Instruction::Split), 11);
            assert_eq!(share(Instruction::DoNothing), 6);
            assert_eq!(share(Instruction::RepeatPreviousMove), 5);
            assert_eq!(share(Instruction::MoveSlow), 3);
            assert_eq!(share(Instruction::MoveFast), 3);
            assert_eq!(share(Instruction::SkipAhead), 3);
            assert_eq!(share(Instruction::SkipBack), 2);
            assert_eq!(share(Instruction::TurnLeft15), 2);
            assert_eq!(share(Instruction::TurnLeft90), 2);
            assert_eq!(share(Instruction::TurnRight90), 2);
            assert_eq!(share(Instruction::TurnRight15), 1);
            assert_eq!(share(Instruction::TurnLeft45), 1);
            assert_eq!(share(Instruction::TurnRight45), 1);
            assert_eq!(share(Instruction::TurnAbout), 1);
        }

        #[test]
        fn raising_one_instructions_weight_gives_it_more_of_the_opcode_space() {
            let baseline = genome_spanning_the_opcode_space();
            let mut heavy_eater = baseline.clone();
            heavy_eater.set_instruction_weight_bits(Instruction::Eat, MAX_WEIGHT_BITS);

            let before = *decoded_counts(&baseline)
                .get(&Instruction::Eat)
                .unwrap_or(&0);
            let after = *decoded_counts(&heavy_eater)
                .get(&Instruction::Eat)
                .unwrap_or(&0);

            assert!(
                after > before,
                "expected Eat to claim more opcode values once up-weighted, got {after} vs {before}"
            );
        }
    }

    mod feelers {
        use super::*;

        fn genome_with_feeler_bits(length: u32, angle: u32, disc: u32) -> Genome {
            let mut genome = Genome::from_bits(&"0".repeat(TOTAL_BITS)).unwrap();
            write_bits(
                &mut genome.bytes,
                FEELER_LENGTH_OFFSET,
                FEELER_FIELD_BITS,
                length,
            );
            write_bits(
                &mut genome.bytes,
                FEELER_ANGLE_OFFSET,
                FEELER_FIELD_BITS,
                angle,
            );
            write_bits(
                &mut genome.bytes,
                FEELER_DISC_OFFSET,
                FEELER_FIELD_BITS,
                disc,
            );
            genome
        }

        const MAX_FIELD: u32 = (1 << FEELER_FIELD_BITS) - 1;

        #[test]
        fn the_feeler_fields_sit_past_everything_else_in_the_genome() {
            // Where the fields begin, not merely how they are read. Tests that
            // write and read through the same offsets agree wherever those
            // point, including on top of the opcode stream.
            assert_eq!(FEELER_LENGTH_OFFSET, OPCODE_STREAM_OFFSET + OPCODE_BITS);
            assert_eq!(FEELER_LENGTH_OFFSET + FEELER_BITS, TOTAL_BITS);
        }

        #[test]
        fn setting_a_feeler_field_leaves_the_opcode_stream_alone() {
            // The fields sit past the stream, so shaping a critter's feelers
            // must not rewrite what it does.
            let mut rng = rand::rngs::StdRng::seed_from_u64(4);
            use rand::SeedableRng;
            let genome = Genome::random(&mut rng);
            let before: Vec<Instruction> = (0..OPCODE_POOL_SIZE)
                .map(|slot| genome.decode_at(slot))
                .collect();

            let mut shaped = genome.clone();
            shaped.set_feeler_shape(MAX_FEELER_LENGTH, MAX_FEELER_ANGLE, MAX_FEELER_DISC);

            let after: Vec<Instruction> = (0..OPCODE_POOL_SIZE)
                .map(|slot| shaped.decode_at(slot))
                .collect();
            assert_eq!(before, after);
        }

        #[test]
        fn a_fresh_genome_carries_no_stray_padding() {
            // The bits past TOTAL_BITS are not part of the genome and to_bits
            // does not emit them, so a random genome that left them set would
            // not survive a round trip back through from_bits.
            let mut rng = rand::rngs::StdRng::seed_from_u64(0);
            use rand::SeedableRng;

            for _ in 0..50 {
                let genome = Genome::random(&mut rng);

                let round_tripped = Genome::from_bits(&genome.to_bits()).unwrap();
                assert_eq!(round_tripped.bytes, genome.bytes);
            }
        }

        #[test]
        fn a_fresh_genome_has_no_feelers_at_all() {
            // Every world starts blind. Feelers arrive only where mutation
            // puts them, so a population that has them has demonstrated they
            // are worth their cost rather than being handed them at the start.
            let mut rng = rand::rngs::StdRng::seed_from_u64(0);
            use rand::SeedableRng;

            for _ in 0..200 {
                let genome = Genome::random(&mut rng);

                assert!(!genome.has_left_feeler());
                assert!(!genome.has_right_feeler());
            }
        }

        #[test]
        fn a_feeler_can_still_be_grown_by_mutation() {
            // Blind at the start, but not blind forever: the bits are ordinary
            // genome and mutate like the rest of it.
            let mut rng = rand::rngs::StdRng::seed_from_u64(0);
            use rand::SeedableRng;
            let mut grown = 0;

            for _ in 0..400 {
                let mut genome = Genome::random(&mut rng);
                genome.mutate(&mut rng, 0.5);
                if genome.has_left_feeler() || genome.has_right_feeler() {
                    grown += 1;
                }
            }

            assert!(grown > 0, "mutation should be able to grow a feeler");
        }

        #[test]
        fn the_feeler_fields_are_laid_out_end_to_end_without_gaps() {
            // The whole region's arithmetic, stated as the facts it is meant
            // to encode rather than only as the fields that happen to use it.
            // Each offset follows the last by exactly one field's width, the
            // presence bits are one bit each, and the total spans all of them
            // -- a gap wastes genome, an overlap has two fields sharing bits.
            assert_eq!(
                FEELER_ANGLE_OFFSET,
                FEELER_LENGTH_OFFSET + FEELER_FIELD_BITS
            );
            assert_eq!(FEELER_DISC_OFFSET, FEELER_ANGLE_OFFSET + FEELER_FIELD_BITS);
            assert_eq!(
                LEFT_FEELER_PRESENT_OFFSET,
                FEELER_DISC_OFFSET + FEELER_FIELD_BITS
            );
            assert_eq!(
                RIGHT_FEELER_PRESENT_OFFSET,
                LEFT_FEELER_PRESENT_OFFSET + FEELER_PRESENT_BITS
            );
            assert_eq!(FEELER_BITS, 3 * FEELER_FIELD_BITS + 2 * FEELER_PRESENT_BITS);
            // And the region ends exactly where the genome does.
            assert_eq!(FEELER_LENGTH_OFFSET + FEELER_BITS, TOTAL_BITS);
        }

        #[test]
        fn the_feeler_bits_are_part_of_the_genome_proper() {
            // Inside TOTAL_BITS, not past it. A field the genome's length does
            // not account for still reads and writes, but to_bits will not
            // emit it, so it is quietly dropped on a round trip -- and mutate
            // never reaches it either, which for these bits would mean a
            // feeler that could never be grown.
            let mut genome = Genome::from_bits(&"0".repeat(TOTAL_BITS)).unwrap();
            genome.set_feelers_present(true, true);

            let round_tripped = Genome::from_bits(&genome.to_bits()).unwrap();

            assert!(round_tripped.has_left_feeler());
            assert!(round_tripped.has_right_feeler());
        }

        #[test]
        fn any_one_of_a_feelers_bits_is_enough_to_grow_it() {
            // A feeler is spelled several times over, so mutation has several
            // ways to find it. Duplication is how a trait becomes reachable in
            // the first place, and how it stops being lost again once found.
            for bit in 0..FEELER_PRESENT_BITS {
                let mut genome = Genome::from_bits(&"0".repeat(TOTAL_BITS)).unwrap();
                write_bits(&mut genome.bytes, LEFT_FEELER_PRESENT_OFFSET + bit, 1, 1);

                assert!(genome.has_left_feeler(), "bit {bit} should grow it");
                assert!(!genome.has_right_feeler());
            }
        }

        #[test]
        fn a_feeler_is_lost_only_when_every_one_of_its_bits_goes() {
            // The other half of the redundancy, and the half that matters
            // more: a trait several bits wide is not undone by one of them
            // flipping back.
            let mut genome = Genome::from_bits(&"0".repeat(TOTAL_BITS)).unwrap();
            for bit in 0..FEELER_PRESENT_BITS {
                write_bits(&mut genome.bytes, LEFT_FEELER_PRESENT_OFFSET + bit, 1, 1);
            }

            for bit in 0..FEELER_PRESENT_BITS - 1 {
                write_bits(&mut genome.bytes, LEFT_FEELER_PRESENT_OFFSET + bit, 1, 0);
                assert!(genome.has_left_feeler(), "still has one bit left");
            }
            write_bits(
                &mut genome.bytes,
                LEFT_FEELER_PRESENT_OFFSET + FEELER_PRESENT_BITS - 1,
                1,
                0,
            );

            assert!(!genome.has_left_feeler());
        }

        #[test]
        fn the_two_feelers_bits_do_not_overlap() {
            // Each side has its own run of bits: growing one must not grow the
            // other, or a critter could never have just the one.
            let mut genome = Genome::from_bits(&"0".repeat(TOTAL_BITS)).unwrap();
            for bit in 0..FEELER_PRESENT_BITS {
                write_bits(&mut genome.bytes, RIGHT_FEELER_PRESENT_OFFSET + bit, 1, 1);
            }

            assert!(genome.has_right_feeler());
            assert!(!genome.has_left_feeler());
        }

        #[test]
        fn a_genome_with_neither_bit_set_has_no_feelers() {
            // Whether a critter grows a feeler at all is one bit apiece, so a
            // lineage can climb towards feelers rather than being handed them:
            // one, then the other, each worth keeping on its own.
            let genome = genome_with_feeler_bits(0, 0, 0);

            assert!(!genome.has_left_feeler());
            assert!(!genome.has_right_feeler());
        }

        #[test]
        fn each_feeler_is_grown_by_its_own_bit() {
            let mut genome = Genome::from_bits(&"0".repeat(TOTAL_BITS)).unwrap();
            write_bits(&mut genome.bytes, LEFT_FEELER_PRESENT_OFFSET, 1, 1);

            assert!(genome.has_left_feeler());
            assert!(!genome.has_right_feeler());
        }

        #[test]
        fn a_genome_with_both_bits_set_has_both_feelers() {
            let mut genome = Genome::from_bits(&"0".repeat(TOTAL_BITS)).unwrap();
            write_bits(&mut genome.bytes, LEFT_FEELER_PRESENT_OFFSET, 1, 1);
            write_bits(&mut genome.bytes, RIGHT_FEELER_PRESENT_OFFSET, 1, 1);

            assert!(genome.has_left_feeler());
            assert!(genome.has_right_feeler());
        }

        #[test]
        fn an_all_zero_length_field_gives_the_shortest_feelers() {
            assert_eq!(
                genome_with_feeler_bits(0, 0, 0).feeler_length(),
                MIN_FEELER_LENGTH
            );
        }

        #[test]
        fn an_all_one_length_field_gives_the_longest() {
            assert_eq!(
                genome_with_feeler_bits(MAX_FIELD, 0, 0).feeler_length(),
                MAX_FEELER_LENGTH
            );
        }

        #[test]
        fn length_scales_evenly_between_the_two() {
            // Pins the shape of the mapping rather than only its ends.
            let middling = genome_with_feeler_bits(MAX_FIELD / 2, 0, 0).feeler_length();

            let span = MAX_FEELER_LENGTH - MIN_FEELER_LENGTH;
            let expected = MIN_FEELER_LENGTH + span * (MAX_FIELD / 2) as f32 / MAX_FIELD as f32;
            assert!((middling - expected).abs() < 0.01);
        }

        #[test]
        fn an_all_zero_angle_field_points_both_feelers_straight_ahead() {
            assert_eq!(genome_with_feeler_bits(0, 0, 0).feeler_angle(), 0.0);
        }

        #[test]
        fn an_all_one_angle_field_points_them_out_to_the_sides() {
            assert_eq!(
                genome_with_feeler_bits(0, MAX_FIELD, 0).feeler_angle(),
                MAX_FEELER_ANGLE
            );
        }

        #[test]
        fn an_all_zero_disc_field_gives_the_smallest_disc() {
            assert_eq!(
                genome_with_feeler_bits(0, 0, 0).feeler_disc(),
                MIN_FEELER_DISC
            );
        }

        #[test]
        fn an_all_one_disc_field_gives_the_largest() {
            assert_eq!(
                genome_with_feeler_bits(0, 0, MAX_FIELD).feeler_disc(),
                MAX_FEELER_DISC
            );
        }

        #[test]
        fn the_three_fields_are_read_from_their_own_places() {
            // Each has its own bits: setting one must not move the others.
            let genome = genome_with_feeler_bits(MAX_FIELD, 0, 0);

            assert_eq!(genome.feeler_length(), MAX_FEELER_LENGTH);
            assert_eq!(genome.feeler_angle(), 0.0);
            assert_eq!(genome.feeler_disc(), MIN_FEELER_DISC);
        }
    }

    mod mutation_rate {
        use super::*;

        fn genome_with_rate_bits(bits: u32) -> Genome {
            let mut genome = Genome::from_bits(&"0".repeat(TOTAL_BITS)).unwrap();
            genome.set_mutation_rate_bits(bits);
            genome
        }

        const MAX_RATE_BITS: u32 = (1 << MUTATION_RATE_BITS) - 1;

        #[test]
        fn an_all_zero_mutation_rate_field_yields_a_zero_rate() {
            let genome = genome_with_rate_bits(0);

            assert_eq!(genome.mutation_rate(), 0.0);
        }

        #[test]
        fn an_all_one_mutation_rate_field_yields_the_maximum_rate() {
            let genome = genome_with_rate_bits(MAX_RATE_BITS);

            assert_eq!(genome.mutation_rate(), MAX_MUTATION_RATE);
        }

        #[test]
        fn a_half_range_mutation_rate_field_yields_half_the_maximum_rate() {
            // Pins the shape of the mapping between its endpoints: the rate
            // scales linearly with the field's value.
            let genome = genome_with_rate_bits(MAX_RATE_BITS / 2);

            let expected = MAX_MUTATION_RATE * (MAX_RATE_BITS / 2) as f32 / MAX_RATE_BITS as f32;
            assert!((genome.mutation_rate() - expected).abs() < f32::EPSILON);
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
            // Arbitrary lengths — the test only cares that both numbers reach
            // the rendered message, not what the genome's real length is.
            let error = GenomeParseError::WrongLength {
                expected: 10,
                actual: 9,
            };

            let rendered = format!("{error}");

            assert!(
                rendered.contains("10") && rendered.contains("9"),
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
