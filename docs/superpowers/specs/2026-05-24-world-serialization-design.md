# World serialization (dominant-genome snapshot)

## Purpose

Give the simulation a minimal "save / load" capability so an interesting lineage
can be captured and revived in a future run. Scope is deliberately small: we
serialize only the *most populous genome*, not the full world.

## What gets serialized

The 40-byte (319-bit) `Genome` of the most populous lineage in the live world,
rendered as a 319-character string of `'0'` and `'1'`. The genome is already a
packed bitstring; we expose its bits verbatim as the transport format. No
positions, energies, pellets, or generation counter are saved.

"Most populous" = the genome with the largest number of exact-byte-match
critters. Tie-break: first-encountered (lowest index in `World::critters()`).

## Output

Every 30 wall-clock seconds, the running app prints a block to stdout in this
shape (with literal blank lines between the bit string and the next block):

```
<RFC 3339 timestamp>
<319-char bit string>

```

That is: timestamp line, newline, bit string line, newline, blank line,
newline. So each snapshot occupies four physical lines including the trailing
separator. Stdout only — no file I/O.

If the world has no critters at the 30-second tick, nothing is printed for
that tick.

## Hydration

A new CLI flag at startup:

```
circles --genome <319-char-bit-string>
```

When present, the simulation is initialized as today (1000 critters at random
positions/headings, default pellets) **except** every critter's genome is the
parsed seed genome instead of `Genome::random(...)`. Positions, headings,
energies, and pellet placements remain randomized. Mutation, replenishment,
reset rules — all unchanged.

If `--genome` is absent: behavior is identical to today.

Parse errors (wrong length, non-`0`/`1` characters) print a short error to
stderr and exit with a non-zero status. No silent fallback.

## Architecture

Three additions, each small and self-contained:

### 1. `Genome::to_bits(&self) -> String`

Renders the underlying 40 bytes as a 319-char string of `'0'`/`'1'`, MSB-first
per byte. Length is exactly `TOTAL_BITS` — we drop the 1-bit padding that
exists only because `TOTAL_BITS` doesn't divide evenly into 8.

### 2. `Genome::from_bits(s: &str) -> Result<Genome, GenomeParseError>`

The inverse. Validates length (`TOTAL_BITS` chars) and alphabet (`0` or `1`
only). Returns an error otherwise.

`GenomeParseError` is a small enum with `WrongLength { expected, actual }` and
`InvalidCharacter { index, character }`. `Display` produces a single-line
human-readable message suitable for stderr.

### 3. `World::dominant_genome(&self) -> Option<&Genome>`

Returns the `Genome` of the most populous lineage, or `None` if the population
is empty. Implementation: walk `critters` once, group by genome bytes
(`HashMap<&Genome, usize>` or equivalent — `Genome` already derives `Eq`/`Hash`
is `Eq`/`PartialEq`, so add `Hash`), track the first-seen order, return the
genome with the highest count (first-seen wins ties).

### 4. `World::with_seed_genome(...)`

A new constructor parallel to `World::new` that takes a `Genome` and uses it
for every critter instead of generating one per critter. Positions, headings,
pellets — same randomization as `new`.

### 5. `main.rs` wiring

- Parse `std::env::args()` for `--genome <bits>` once at startup. Use
  `from_bits` to produce a `Genome` and call `World::with_seed_genome` instead
  of `World::new` when present.
- Track an `Instant` for the last snapshot print. Each loop iteration, check
  `last_snapshot.elapsed() >= Duration::from_secs(30)`; if so, ask the world
  for its dominant genome, print the block (if any), reset the timer.
- Timestamp uses `SystemTime::now()` formatted in RFC 3339. Adding `chrono` is
  overkill; we'll format manually from `UNIX_EPOCH` seconds to keep the dep
  list as-is.

## Testing

Test surface, each TDD'd:

- `Genome::to_bits` round-trips through `from_bits` for an `all`/`from_instructions`
  genome and for a `random` genome.
- `to_bits` produces exactly `TOTAL_BITS` characters, all `0` or `1`.
- `from_bits` rejects wrong length with `WrongLength`.
- `from_bits` rejects non-`0`/`1` characters with `InvalidCharacter` carrying
  the index and character.
- `World::dominant_genome` returns `None` for an empty world.
- `World::dominant_genome` returns the most populous genome when one is
  strictly dominant.
- `World::dominant_genome` breaks ties by first-seen order.
- `World::with_seed_genome` produces a world whose every critter has the given
  genome bytes.

`main.rs`'s wall-clock loop and CLI parsing are not unit-tested — they're
exercised by manual run-through and the existing smoke test continues to pass.

## Out of scope

- Saving anything other than the dominant genome (positions, energies, pellet
  layout, generation, RNG state).
- Multi-genome snapshots (top-N, full population, etc.).
- File output, append logs, or rolling files.
- Live keypress-driven save/load inside the running app.
- Reproducible RNG seeding from the snapshot.

If any of these become interesting later, the current shape — bit-string
transport, a small accessor on `World`, an alt constructor — extends
naturally to a `WorldSnapshot` struct without breaking the existing API.
