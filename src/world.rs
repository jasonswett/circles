use crate::{
    Critter, Genome, Heading, Pellet, CRITTER_RADIUS, MAX_CRITTER_ENERGY, PELLETS_PER_POISON,
    PELLET_MAX_DRIFT, PELLET_MIN_DRIFT, PELLET_RADIUS, POISON_DAMAGE_PERCENT,
};
use rand::Rng;

// What share of its prey's energy a predator takes in one bite. Partial, so
// predation wears prey down rather than executing it: what the predator does
// not take, the prey keeps, and only a critter with nothing left to give dies
// of being eaten.
const PREDATION_SHARE_PERCENT: u32 = 25;
// How far the eruption site travels each frame. Small, so the source drifts
// visibly across the world rather than jumping between places.
const ERUPTION_SITE_DRIFT: f32 = 1.5;
// How old a world is when its eruption site reaches full speed, and stops
// gathering more. Where food comes from is somewhere in particular at first
// and less and less so as the world runs on: a site that wanders from the
// outset never lets anywhere be worth being.
//
// The climb is slow enough to be hard to notice while it happens -- a minute
// either side of any moment looks much the same.
const ERUPTION_DRIFT_RAMP_TICKS: u32 = 6 * 60 * 60;
// The population a world needs before its food will move at all. Food that
// keeps wandering away from a world already in trouble finishes it, so below
// this the site stops where it stands and its clock goes back to nothing: a
// world that has fallen this far starts its food over and has to hold a
// population again before it moves.
const DRIFT_POPULATION_FLOOR: usize = 100;
// How sharply the site's heading can turn each frame, in radians. Low enough
// that the source wanders on a curve rather than jittering in place.
const ERUPTION_SITE_TURN: f32 = 0.08;
// How often a predator does not survive its own attack. The bite always
// lands; this is the chance it costs the predator its life.
//
// The risk scales with what the victim has left to fight with: the base
// applies against prey with nothing, and a victim at full energy adds the
// whole of the scaled part on top. Since the bite is a fixed share of the
// victim's energy, the fattest target is also the most rewarding one --
// scaling the danger the same way puts reward and risk on one axis, so
// attacking indiscriminately is punished while attacking the depleted is
// nearly free. It is also how predation works: healthy prey fights back, and
// wolves take the elk that cannot run.
// What a predator spends to attack, charged whether or not the bite is worth
// having. Eating a pellet is free: the charge falls on cannibalism alone.
//
// Firing Eat costs nothing, so a genome of nothing but Eat was close to
// optimal -- it takes a pellet when one is there and bites a neighbor
// otherwise, at no cost either way. Cannibalism was not a strategy so much as
// the default, and nothing selected against it. Attacking now has to pay for
// itself: the share taken from a depleted victim does not cover this, so
// preying on the weak loses energy, while preying on the strong is profitable
// but carries the death roll above.
//
// At a quarter share this puts break-even at a victim holding 160 -- well
// above the energy a critter is seeded or born with, so attacking anything
// but a conspicuously well-fed neighbor is a losing move.
pub const PREDATION_ATTACK_COST: u32 = 100;
const PREDATION_BASE_DEATH_PERCENT: u32 = 5;
const PREDATION_ENERGY_DEATH_PERCENT: u32 = 30;

// The population a world grows toward. It does not start there: spawning
// thousands of critters at once stalls the first seconds of a run, so a world
// begins at SEED_POPULATION and is topped up gradually while the frame rate
// holds.
const NUM_CRITTERS: usize = 4000;
// How many critters a world starts with. Above MIN_POPULATION so a freshly
// seeded world is not immediately judged too small and reset. Small, because
// the opening moments are the leanest: no food has been delivered yet, so
// every critter alive is spending energy it has no way to replace.
const SEED_POPULATION: usize = 50;
// How much food a world's energy budget is sized for. Not a ceiling on
// pellets: nothing caps the larder, which settles wherever consumption meets
// the feed rate.
const NUM_PELLETS: usize = 4000;
// How many pellets a single replenishment call may add. Small, because the
// call happens often: food seeps into the world a few at a time rather than
// a whole deficit's worth arriving at once.
pub const PELLET_BATCH_SIZE: usize = 10;
// Feelers are read on the same schedule as poison, and for the same reason:
// asking every pellet about every critter is the costliest thing the world
// can do, and what a feeler touched a few ticks ago is close enough when a
// critter moves a few pixels a tick.
const FEELER_INTERVAL_TICKS: u32 = 10;
// How wide a cell of the pellet index is. Only has to cover a feeler's disc,
// not its length: a tip's position is worked out before the index is asked, so
// what is looked up is a small patch and not the whole reach.
const FEELER_CELL_SIZE: i32 = 24;
// Scanning every pellet for every critter is the most expensive thing the
// world does, so poison is checked periodically rather than every tick. A
// critter can cross poison between checks and survive; poison is a hazard,
// not a guarantee, and the frame budget matters more than the precision.
const POISON_CHECK_INTERVAL_TICKS: u32 = 10;
// How often a replenishment begins. At ~60 FPS this is ten seconds: a
// world starts feeding on this cadence and keeps at it until its energy is
pub const MIN_POPULATION: usize = 20;
const INITIAL_ENERGY: u32 = 60;
const TICKS_PER_INSTRUCTION: u32 = 5;
const STEP_SIZE: i32 = 5;
// How many critters to refresh per overlap-detection call. The detector cycles
// through the population round-robin: critters not reached on a given call
// keep their previous flag until they come up again.
const OVERLAP_DETECTION_BUDGET: usize = 20;
// How many ticks a critter stays marked as overlapping after a confirmed
// detection. Smooths out the visual flicker caused by the round-robin budget.
const OVERLAP_INDICATOR_LINGER_TICKS: u32 = 30;
// How many ticks a victim stays marked as "being eaten" after it is killed.
// Brief — the kill is supposed to look like a flash.
const EATEN_INDICATOR_LINGER_TICKS: u32 = 10;

pub struct World {
    width: usize,
    height: usize,
    critters: Vec<Critter>,
    pellets: Vec<Pellet>,
    original_total_energy: u32,
    generation: u32,
    overlap_detection_cursor: usize,
    /// How many pellets this world has emitted, so every PELLETS_PER_POISON
    /// of them can be poison.
    pellets_emitted: usize,
    ticks: u32,
    /// How long this world has held a population worth moving its food for.
    /// Separate from `ticks` because it stops and restarts with the
    /// population, where the world's own clock never does.
    drift_age: u32,
    /// Frames until the next replenishment, or zero while feeding.
    /// Where food is currently erupting from, and the heading it is
    /// wandering along. Both drift every frame, so the source travels.
    eruption_site: (f32, f32),
    eruption_heading: f32,
    /// The genome new critters are seeded with, if this world was started
    /// from one. None means each newcomer gets a fresh random genome.
    seed_genome: Option<Genome>,
}

impl World {
    pub fn new<R: Rng>(width: usize, height: usize, rng: &mut R) -> Self {
        let critters: Vec<Critter> = (0..SEED_POPULATION)
            .map(|_| spawn_critter(width, height, rng))
            .collect();
        // Budget for what the world is growing toward, not what it starts
        // with: both the population and the larder fill up over time, and a
        // budget sized to the opening moment would never top either up.
        let original_total_energy = full_population_energy() + full_larder_energy();
        Self {
            width,
            height,
            critters,
            pellets: Vec::new(),
            original_total_energy,
            generation: 1,
            overlap_detection_cursor: 0,
            pellets_emitted: 0,
            ticks: 0,
            drift_age: 0,
            eruption_site: (width as f32 / 2.0, height as f32 / 2.0),
            eruption_heading: rng.gen_range(0.0..std::f32::consts::TAU),
            seed_genome: None,
        }
    }

    /// Build a world where every critter starts with the given seed genome
    /// instead of a freshly randomized one. Positions, headings, and pellets
    /// are randomized as in `new`.
    pub fn with_seed_genome<R: Rng>(
        width: usize,
        height: usize,
        seed_genome: Genome,
        rng: &mut R,
    ) -> Self {
        let critters: Vec<Critter> = (0..SEED_POPULATION)
            .map(|_| spawn_critter_with_genome(width, height, seed_genome.clone(), rng))
            .collect();
        let original_total_energy = full_population_energy() + full_larder_energy();
        Self {
            width,
            height,
            critters,
            pellets: Vec::new(),
            original_total_energy,
            generation: 1,
            overlap_detection_cursor: 0,
            pellets_emitted: 0,
            ticks: 0,
            drift_age: 0,
            eruption_site: (width as f32 / 2.0, height as f32 / 2.0),
            eruption_heading: rng.gen_range(0.0..std::f32::consts::TAU),
            seed_genome: Some(seed_genome),
        }
    }

    /// Returns the genome with the most copies among the current critters.
    /// Ties are broken by first-seen order in `critters()`. Returns `None` if
    /// there are no critters.
    pub fn dominant_genome(&self) -> Option<&Genome> {
        // Vec<(&Genome, usize)> preserves first-seen order naturally; we walk
        // the critters once, incrementing the count for any matching entry or
        // appending a new one. Linear in the number of unique genomes — fine
        // for this scale.
        let mut tally: Vec<(&Genome, usize)> = Vec::new();
        for critter in &self.critters {
            let genome = critter.genome();
            if let Some(entry) = tally.iter_mut().find(|(g, _)| *g == genome) {
                entry.1 += 1;
            } else {
                tally.push((genome, 1));
            }
        }
        // Manual fold instead of `max_by_key`: the standard iterator method
        // keeps the *last* maximum on ties, but we want the first-seen
        // winner. Walk left-to-right, replacing only on a strict increase.
        let mut best: Option<(&Genome, usize)> = None;
        for (genome, count) in tally {
            match best {
                Some((_, best_count)) if count <= best_count => {}
                _ => best = Some((genome, count)),
            }
        }
        best.map(|(genome, _)| genome)
    }

    #[cfg(test)]
    pub fn with_critters_and_pellets(
        width: usize,
        height: usize,
        critters: Vec<Critter>,
        pellets: Vec<Pellet>,
    ) -> Self {
        let original_total_energy = critter_total_energy(&critters) + pellet_total_energy(&pellets);
        Self {
            width,
            height,
            critters,
            pellets,
            original_total_energy,
            generation: 1,
            overlap_detection_cursor: 0,
            pellets_emitted: 0,
            ticks: 0,
            drift_age: 0,
            // Test-only constructor: a fixed site, since it has no rng.
            eruption_site: (width as f32 / 2.0, height as f32 / 2.0),
            eruption_heading: 0.0,
            seed_genome: None,
        }
    }

    pub fn original_total_energy(&self) -> u32 {
        self.original_total_energy
    }

    /// Takes on food for this frame. The world's energy budget is what limits
    /// feeding: replenish_pellets stops as soon as energy is restored, so
    /// deliveries track consumption directly rather than arriving in rounds.
    pub fn feed<R: Rng>(&mut self, rng: &mut R) {
        self.drift_eruption_site(rng);
        self.replenish_pellets(PELLET_BATCH_SIZE, rng);
    }

    /// Where food is currently erupting from.
    pub fn eruption_site(&self) -> (f32, f32) {
        self.eruption_site
    }

    /// Nudges the eruption site along its heading, turning it slightly, so
    /// the source wanders continuously instead of sitting still.
    //
    // The `+=` on the heading is an equivalent mutant: the turn is drawn
    // symmetrically about zero, so subtracting it yields the same
    // distribution of headings as adding it.
    #[mutants::skip]
    fn drift_eruption_site<R: Rng>(&mut self, rng: &mut R) {
        if self.critters.len() < DRIFT_POPULATION_FLOOR {
            self.drift_age = 0;
            return;
        }
        self.drift_age += 1;
        self.eruption_heading += rng.gen_range(-ERUPTION_SITE_TURN..=ERUPTION_SITE_TURN);
        let speed = self.eruption_drift_speed();
        let (width, height) = (self.width as f32, self.height as f32);
        self.eruption_site = (
            (self.eruption_site.0 + self.eruption_heading.cos() * speed).rem_euclid(width),
            (self.eruption_site.1 + self.eruption_heading.sin() * speed).rem_euclid(height),
        );
    }

    /// How far the eruption site travels each tick, which grows with the age
    /// of the world and goes on growing.
    fn eruption_drift_speed(&self) -> f32 {
        let ramped = self.drift_age.min(ERUPTION_DRIFT_RAMP_TICKS) as f32;
        ERUPTION_SITE_DRIFT * ramped / ERUPTION_DRIFT_RAMP_TICKS as f32
    }

    /// Whether the world holds less energy than its budget allows. Counts
    /// energy inside critters as well as on the ground, so a well-fed
    /// population slows feeding by itself -- the budget is a thermostat on
    /// the whole world rather than a ceiling on pellets.
    pub fn needs_more_food(&self) -> bool {
        self.total_energy() < self.original_total_energy
    }

    /// Adds up to `count` pellets, stopping once the world's energy budget is
    /// met. What limits food is the budget, the length of a feeding run, and
    /// what the critters eat.
    pub fn replenish_pellets<R: Rng>(&mut self, count: usize, rng: &mut R) {
        for _ in 0..count {
            if !self.needs_more_food() {
                return;
            }
            self.pellets_emitted += 1;
            let poisonous = self.pellets_emitted.is_multiple_of(PELLETS_PER_POISON);
            self.pellets
                .push(spawn_pellet_of_kind(self.eruption_site, poisonous, rng));
        }
    }

    /// Adds a pellet to the world.
    pub fn add_pellet(&mut self, pellet: Pellet) {
        self.pellets.push(pellet);
    }

    /// Adds a critter to the world.
    pub fn add_critter(&mut self, critter: Critter) {
        self.critters.push(critter);
    }

    /// Fills in what each critter's feelers are touching: the colour of a
    /// pellet under the disc at either tip, or black for nothing. A critter's
    /// only sense of anything it is not already standing on.
    pub fn sense_feelers(&mut self) {
        // A world whose critters have not grown any feelers has nothing to
        // sense with, and indexing the larder for them would be work done for
        // nobody. Worth asking, since every world starts out this way and some
        // never leave it.
        if !self
            .critters
            .iter()
            .any(|critter| critter.has_left_feeler() || critter.has_right_feeler())
        {
            return;
        }
        let (width, height) = (self.width as i32, self.height as i32);
        // Bucketed by row band first. Asking every pellet about every critter
        // is tens of millions of distance checks a tick at a full world, which
        // costs more than everything else the world does put together.
        let grid = PelletGrid::build(&self.pellets, width, height);

        for critter in &mut self.critters {
            let (left, right) = critter.feeler_tips();
            let disc = critter.feeler_disc();
            let disc_squared = (disc * disc) as i32;
            let under = |(tx, ty): (i32, i32)| {
                grid.near(tx, ty, &self.pellets, width, height, |pellet| {
                    let dx = toroidal_delta(tx, pellet.x.round() as i32, width);
                    let dy = toroidal_delta(ty, pellet.y.round() as i32, height);
                    dx * dx + dy * dy <= disc_squared
                })
                .map_or(0, |pellet| pellet.color())
            };
            // A feeler a critter never grew senses nothing, whatever is where
            // its tip would have been.
            let l = if critter.has_left_feeler() {
                under(left)
            } else {
                0
            };
            let r = if critter.has_right_feeler() {
                under(right)
            } else {
                0
            };
            critter.set_feeler_colors(l, r);
        }
    }

    /// Kills any critter touching poison, and consumes the poison with it.
    /// Contact alone is fatal: a critter need not be trying to eat, and
    /// nothing in its sensorium warns it.
    fn resolve_poison(&mut self) {
        let (width, height) = (self.width as i32, self.height as i32);
        let mut spent: Vec<usize> = Vec::new();

        for critter in &mut self.critters {
            if critter.energy() == 0 {
                continue;
            }
            let reach = critter.radius() + PELLET_RADIUS;
            let touch_distance_squared = reach * reach;
            let touched = self.pellets.iter().position(|pellet| {
                if !pellet.poisonous {
                    return false;
                }
                let dx = toroidal_delta(critter.x(), pellet.x.round() as i32, width);
                let dy = toroidal_delta(critter.y(), pellet.y.round() as i32, height);
                dx * dx + dy * dy < touch_distance_squared
            });
            if let Some(index) = touched {
                let toll = critter.energy() * POISON_DAMAGE_PERCENT / 100;
                critter.lose_energy(toll.max(1));
                if !spent.contains(&index) {
                    spent.push(index);
                }
            }
        }

        spent.sort_unstable_by(|a, b| b.cmp(a));
        for index in spent {
            self.pellets.swap_remove(index);
        }
    }

    pub fn critters(&self) -> &[Critter] {
        &self.critters
    }

    pub fn pellets(&self) -> &[Pellet] {
        &self.pellets
    }

    pub fn total_energy(&self) -> u32 {
        let critter_energy: u32 = self.critters.iter().map(|c| c.energy()).sum();
        let pellet_energy = self.pellets.len() as u32 * crate::PELLET_ENERGY;
        critter_energy + pellet_energy
    }

    pub fn tick(&mut self, allow_split: bool) {
        let mut children = Vec::new();
        let mut eater_indices: Vec<usize> = Vec::new();
        for (index, critter) in self.critters.iter_mut().enumerate() {
            critter.age_overlap_indicator();
            critter.age_being_eaten_indicator();
            let outcome = critter.tick(allow_split);
            if let Some(mut child) = outcome.child {
                child.wrap_position(self.width as i32, self.height as i32);
                children.push(child);
            }
            if outcome.attempted_eat {
                eater_indices.push(index);
            }
            critter.wrap_position(self.width as i32, self.height as i32);
        }
        self.critters.extend(children);
        let (width, height) = (self.width as f32, self.height as f32);
        for pellet in &mut self.pellets {
            pellet.drift(width, height);
        }
        self.pellets.retain(|pellet| !pellet.is_expired());
        if self.ticks.is_multiple_of(POISON_CHECK_INTERVAL_TICKS) {
            self.resolve_poison();
        }
        if self.ticks.is_multiple_of(FEELER_INTERVAL_TICKS) {
            self.sense_feelers();
        }
        self.ticks = self.ticks.wrapping_add(1);
        self.resolve_eats(&eater_indices);
        self.detect_critter_overlaps();
    }

    fn resolve_eats(&mut self, eater_indices: &[usize]) {
        let count = self.critters.len();
        let width = self.width as i32;
        let height = self.height as i32;

        for &eater_index in eater_indices {
            if eater_index >= count {
                continue;
            }
            if self.critters[eater_index].energy() == 0 {
                continue;
            }
            let eater_x = self.critters[eater_index].x();
            let eater_y = self.critters[eater_index].y();
            // Reach is the eater's own radius, so a well-fed critter covers
            // more ground than a starving one.
            let eater_radius = self.critters[eater_index].radius();
            let pellet_reach = eater_radius + PELLET_RADIUS;
            let pellet_eat_distance_squared = pellet_reach * pellet_reach;

            // Look for an overlapping pellet first; eat it if found.
            let pellet_position = self.pellets.iter().position(|pellet| {
                let dx = toroidal_delta(eater_x, pellet.x.round() as i32, width);
                let dy = toroidal_delta(eater_y, pellet.y.round() as i32, height);
                dx * dx + dy * dy < pellet_eat_distance_squared
            });
            if let Some(position) = pellet_position {
                self.pellets.swap_remove(position);
                self.critters[eater_index].gain_energy(crate::PELLET_ENERGY);
                continue;
            }

            // No pellet in range — take a bite out of the first overlapping
            // critter, if any. Predation is partial: the predator takes a
            // share and the prey keeps the rest, so a well-fed critter
            // survives being bitten and a depleted one does not.
            let victim_index = (0..count).find(|&victim_index| {
                if victim_index == eater_index {
                    return false;
                }
                if self.critters[victim_index].energy() == 0 {
                    return false;
                }
                let dx = toroidal_delta(eater_x, self.critters[victim_index].x(), width);
                let dy = toroidal_delta(eater_y, self.critters[victim_index].y(), height);
                // Both bring their own radius to the contact, so a fat victim
                // is easier to catch as well as more worth catching.
                let reach = eater_radius + self.critters[victim_index].radius();
                dx * dx + dy * dy < reach * reach
            });
            if let Some(victim_index) = victim_index {
                // Paid up front, out of what the predator actually has. A
                // critter that cannot cover the attack spends everything
                // trying and dies of the effort, taking nothing: netting the
                // cost against the meal instead would let the bite refund it,
                // so no cost however large could ever be fatal.
                if self.critters[eater_index].energy() <= PREDATION_ATTACK_COST {
                    self.critters[eater_index].die();
                    continue;
                }
                let victim_energy = self.critters[victim_index].energy();
                let bite = victim_energy * PREDATION_SHARE_PERCENT / 100;
                self.critters[eater_index].lose_energy(PREDATION_ATTACK_COST);
                self.critters[eater_index].gain_energy(bite);
                self.critters[victim_index].lose_energy(bite.max(1));
                self.critters[victim_index].mark_being_eaten_for(EATEN_INDICATOR_LINGER_TICKS);
                // The bite lands either way; the predator may not survive it.
                let risk = predation_death_percent(victim_energy);
                if self.critters[eater_index].roll_predation_death(risk) {
                    self.critters[eater_index].die();
                }
            }
        }
    }

    pub fn detect_critter_overlaps(&mut self) {
        let count = self.critters.len();
        if count < 2 {
            return;
        }
        let width = self.width as i32;
        let height = self.height as i32;
        let budget = OVERLAP_DETECTION_BUDGET.min(count);

        for offset in 0..budget {
            let i = (self.overlap_detection_cursor + offset) % count;
            if self.critters[i].energy() == 0 {
                continue;
            }
            for j in 0..count {
                if i == j {
                    continue;
                }
                if self.critters[j].energy() == 0 {
                    continue;
                }
                let dx = toroidal_delta(self.critters[i].x(), self.critters[j].x(), width);
                let dy = toroidal_delta(self.critters[i].y(), self.critters[j].y(), height);
                let reach = self.critters[i].radius() + self.critters[j].radius();
                if dx * dx + dy * dy < reach * reach {
                    let i_color = self.critters[i].genome_color();
                    let j_color = self.critters[j].genome_color();
                    self.critters[i].mark_overlapping_critter_for(OVERLAP_INDICATOR_LINGER_TICKS);
                    self.critters[i].record_overlap_color(j_color);
                    self.critters[j].mark_overlapping_critter_for(OVERLAP_INDICATOR_LINGER_TICKS);
                    self.critters[j].record_overlap_color(i_color);
                }
            }
        }

        // The inner `(cursor + offset) % count` indexing normalizes the cursor
        // on every read, so we don't need to keep the cursor itself bounded;
        // wrapping_add makes the eventual overflow explicit and harmless.
        self.overlap_detection_cursor = self.overlap_detection_cursor.wrapping_add(budget);
    }

    pub fn reap_dead_critters(&mut self) {
        self.critters.retain(|c| c.energy() > 0);
    }

    /// Adds up to `count` more critters, stopping at the target population.
    /// The caller decides when the world can afford to grow. Critters carry
    /// the world's seed genome when it has one, so a `--genome` run stays a
    /// monoculture as it ramps rather than diluting with random newcomers.
    pub fn seed_more_critters<R: Rng>(&mut self, count: usize, rng: &mut R) {
        let room = NUM_CRITTERS.saturating_sub(self.critters.len());
        for _ in 0..count.min(room) {
            let critter = match &self.seed_genome {
                Some(genome) => {
                    spawn_critter_with_genome(self.width, self.height, genome.clone(), rng)
                }
                None => spawn_critter(self.width, self.height, rng),
            };
            self.critters.push(critter);
        }
    }

    /// Whether the world has finished ramping up to its target population.
    pub fn is_fully_seeded(&self) -> bool {
        self.critters.len() >= NUM_CRITTERS
    }

    pub fn population_too_low(&self) -> bool {
        self.critters.len() < MIN_POPULATION
    }

    pub fn generation(&self) -> u32 {
        self.generation
    }

    pub fn reset<R: Rng>(&mut self, rng: &mut R) {
        self.critters = (0..SEED_POPULATION)
            .map(|_| spawn_critter(self.width, self.height, rng))
            .collect();
        self.pellets.clear();
        // A reset is a new world: its food starts in the middle and holds
        // still there, rather than carrying on from wherever the last one had
        // wandered to at whatever speed it had worked up.
        self.eruption_site = (self.width as f32 / 2.0, self.height as f32 / 2.0);
        self.eruption_heading = rng.gen_range(0.0..std::f32::consts::TAU);
        self.ticks = 0;
        self.drift_age = 0;
        self.generation += 1;
    }
}

#[cfg(test)]
fn critter_total_energy(critters: &[Critter]) -> u32 {
    critters.iter().map(|c| c.energy()).sum()
}

#[cfg(test)]
fn pellet_total_energy(pellets: &[Pellet]) -> u32 {
    pellets.len() as u32 * crate::PELLET_ENERGY
}

fn toroidal_delta(a: i32, b: i32, size: i32) -> i32 {
    let raw = (a - b).rem_euclid(size);
    if raw <= size / 2 {
        raw
    } else {
        raw - size
    }
}

/// The energy a full target population would hold. Used to size a world's
/// budget up front, so replenishment aims at the population being grown
/// toward rather than whatever happens to be alive at construction.
/// The energy a world's food stocks hold when full. Like the population
/// budget, this is what the world is filling toward rather than what it
/// holds, so a world that starts empty still knows to feed itself.
fn full_larder_energy() -> u32 {
    NUM_PELLETS as u32 * crate::PELLET_ENERGY
}

fn full_population_energy() -> u32 {
    NUM_CRITTERS as u32 * INITIAL_ENERGY
}

fn spawn_critter<R: Rng>(width: usize, height: usize, rng: &mut R) -> Critter {
    let x = rng.gen_range(CRITTER_RADIUS..(width as i32 - CRITTER_RADIUS));
    let y = rng.gen_range(CRITTER_RADIUS..(height as i32 - CRITTER_RADIUS));
    Critter::new(
        x,
        y,
        Heading::random(rng),
        TICKS_PER_INSTRUCTION,
        STEP_SIZE,
        INITIAL_ENERGY,
        rng.gen(),
    )
}

fn spawn_critter_with_genome<R: Rng>(
    width: usize,
    height: usize,
    genome: Genome,
    rng: &mut R,
) -> Critter {
    let x = rng.gen_range(CRITTER_RADIUS..(width as i32 - CRITTER_RADIUS));
    let y = rng.gen_range(CRITTER_RADIUS..(height as i32 - CRITTER_RADIUS));
    Critter::with_genome(
        x,
        y,
        Heading::random(rng),
        TICKS_PER_INSTRUCTION,
        STEP_SIZE,
        INITIAL_ENERGY,
        rng.gen(),
        genome,
    )
}

/// Emits a pellet from the world's center on a random heading. Food streams
/// outward from one source rather than appearing evenly across the world, so
/// where a critter forages matters.
#[cfg(test)]
fn spawn_pellet<R: Rng>(width: usize, height: usize, rng: &mut R) -> Pellet {
    spawn_pellet_of_kind((width as f32 / 2.0, height as f32 / 2.0), false, rng)
}

/// How likely an attack is to kill the predator, given what the victim has
/// left to resist with. Ranges from the base risk against spent prey up to
/// the base plus the full scaled part against prey at full energy.
fn predation_death_percent(victim_energy: u32) -> u32 {
    let scaled =
        PREDATION_ENERGY_DEATH_PERCENT * victim_energy.min(MAX_CRITTER_ENERGY) / MAX_CRITTER_ENERGY;
    PREDATION_BASE_DEATH_PERCENT + scaled
}

/// Pellets bucketed into a grid of cells, so a feeler looks only at what its
/// disc could possibly touch. Bands by row alone left a critter checking every
/// pellet across the whole width of its three rows -- some hundreds of them --
/// to find the two or three within a disc's reach. Purely a filter: what it
/// offers is still checked by true distance, so a cell too generous costs time
/// and never correctness.
struct PelletGrid {
    cells: Vec<Vec<u32>>,
    columns: i32,
    rows: i32,
}

impl PelletGrid {
    // The `/` is an equivalent mutant in both of these: multiplying instead
    // over-allocates cells rather than under-allocating them, and cell_of
    // indexes within either, so the filter still answers correctly and only
    // wastes room.
    #[mutants::skip]
    fn columns(width: i32) -> i32 {
        (width / FEELER_CELL_SIZE).max(1) + 1
    }

    #[mutants::skip]
    fn rows(height: i32) -> i32 {
        (height / FEELER_CELL_SIZE).max(1) + 1
    }

    fn cell_of(x: i32, y: i32, width: i32, height: i32) -> (i32, i32) {
        (
            x.rem_euclid(width) / FEELER_CELL_SIZE,
            y.rem_euclid(height) / FEELER_CELL_SIZE,
        )
    }

    fn build(pellets: &[Pellet], width: i32, height: i32) -> Self {
        let (columns, rows) = (Self::columns(width), Self::rows(height));
        let mut cells = vec![Vec::new(); (columns * rows) as usize];
        for (index, pellet) in pellets.iter().enumerate() {
            let (column, row) = Self::cell_of(
                pellet.x.round() as i32,
                pellet.y.round() as i32,
                width,
                height,
            );
            cells[(row * columns + column) as usize].push(index as u32);
        }
        Self {
            cells,
            columns,
            rows,
        }
    }

    /// The first pellet in the cells around (x, y) that `wanted` accepts.
    //
    // The `+` on the offsets is an equivalent mutant: they run symmetrically
    // about zero, so subtracting them visits the same nine cells in another
    // order.
    #[mutants::skip]
    fn near<'a, F>(
        &self,
        x: i32,
        y: i32,
        pellets: &'a [Pellet],
        width: i32,
        height: i32,
        wanted: F,
    ) -> Option<&'a Pellet>
    where
        F: Fn(&Pellet) -> bool,
    {
        let (column, row) = Self::cell_of(x, y, width, height);
        (-1..=1)
            .flat_map(move |row_offset| {
                (-1..=1).map(move |column_offset| (row_offset, column_offset))
            })
            .flat_map(|(row_offset, column_offset)| {
                let r = (row + row_offset).rem_euclid(self.rows);
                let c = (column + column_offset).rem_euclid(self.columns);
                self.cells[(r * self.columns + c) as usize].iter()
            })
            .map(|&index| &pellets[index as usize])
            .find(|pellet| wanted(pellet))
    }
}

fn spawn_pellet_of_kind<R: Rng>(site: (f32, f32), poisonous: bool, rng: &mut R) -> Pellet {
    let angle = rng.gen_range(0.0..std::f32::consts::TAU);
    let speed = rng.gen_range(PELLET_MIN_DRIFT..=PELLET_MAX_DRIFT);
    Pellet {
        x: site.0,
        y: site.1,
        dx: angle.cos() * speed,
        dy: angle.sin() * speed,
        poisonous,
        age: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    // Feeds a world the way the main loop does. Bounded so a broken feed
    // fails the test rather than hanging it.
    fn feed_a_while<R: Rng>(world: &mut World, rng: &mut R) {
        for _ in 0..(NUM_PELLETS / PELLET_BATCH_SIZE + 2) {
            world.feed(rng);
        }
    }

    const TEST_WIDTH: usize = 200;
    const TEST_HEIGHT: usize = 200;

    mod new {
        use super::*;

        #[test]
        fn each_critter_starts_with_full_energy() {
            let mut rng = StdRng::seed_from_u64(0);

            let world = World::new(TEST_WIDTH, TEST_HEIGHT, &mut rng);

            assert!(world
                .critters()
                .iter()
                .all(|c| c.energy() == c.initial_energy()));
        }

        #[test]
        fn every_critter_spawns_fully_inside_the_world_bounds() {
            // Use multiple seeds so we exhaust the rng range and reliably catch
            // boundary-mutation bugs in the spawn rectangle.
            for seed in 0..50 {
                let mut rng = StdRng::seed_from_u64(seed);
                let world = World::new(TEST_WIDTH, TEST_HEIGHT, &mut rng);
                for critter in world.critters() {
                    assert!(critter.x() >= CRITTER_RADIUS);
                    assert!(critter.x() < TEST_WIDTH as i32 - CRITTER_RADIUS);
                    assert!(critter.y() >= CRITTER_RADIUS);
                    assert!(critter.y() < TEST_HEIGHT as i32 - CRITTER_RADIUS);
                }
            }
        }

        #[test]
        fn critters_spread_across_the_full_world_width_and_height() {
            // Across many spawn batches, at least one critter must land in the
            // right half and at least one in the bottom half — proving the spawn
            // range is the full canvas, not a clipped corner.
            let mut any_right = false;
            let mut any_bottom = false;
            let half_width = (TEST_WIDTH as i32) / 2;
            let half_height = (TEST_HEIGHT as i32) / 2;
            for seed in 0..50 {
                let mut rng = StdRng::seed_from_u64(seed);
                let world = World::new(TEST_WIDTH, TEST_HEIGHT, &mut rng);
                for critter in world.critters() {
                    if critter.x() >= half_width {
                        any_right = true;
                    }
                    if critter.y() >= half_height {
                        any_bottom = true;
                    }
                }
            }
            assert!(any_right);
            assert!(any_bottom);
        }
    }

    mod tick {
        use super::*;
        use crate::{Critter, Genome, Instruction, NORTH};

        #[test]
        fn ticking_a_world_decreases_total_critter_energy() {
            // Construct with no pellets so eating doesn't interfere with the
            // per-instruction energy decrement. Some critters may stall on a
            // Split they can't afford, but the total across all critters must
            // still drop because at least some will execute non-Split instructions.
            let mut rng = StdRng::seed_from_u64(0);
            let world_with_critters = World::new(TEST_WIDTH, TEST_HEIGHT, &mut rng);
            let critters: Vec<Critter> = world_with_critters.critters().to_vec();
            let mut world =
                World::with_critters_and_pellets(TEST_WIDTH, TEST_HEIGHT, critters.clone(), vec![]);
            let initial_total: u32 = critters.iter().map(|c| c.energy()).sum();

            for _ in 0..TICKS_PER_INSTRUCTION {
                world.tick(true);
            }

            let final_total: u32 = world.critters().iter().map(|c| c.energy()).sum();
            assert!(final_total < initial_total);
        }

        #[test]
        fn the_overlap_indicator_decays_to_off_over_enough_ticks_with_no_overlap() {
            // A lone critter has nothing to overlap with. Mark it for a few
            // linger ticks; after enough world ticks, the indicator should be
            // gone — proving World::tick ages the indicator.
            let mut critter = Critter::with_genome(
                100,
                100,
                NORTH,
                u32::MAX,
                1,
                100,
                0,
                Genome::all(Instruction::DoNothing),
            );
            critter.mark_overlapping_critter_for(3);
            let mut world =
                World::with_critters_and_pellets(TEST_WIDTH, TEST_HEIGHT, vec![critter], vec![]);

            for _ in 0..3 {
                world.tick(true);
            }

            assert!(!world.critters()[0].is_overlapping_critter());
        }

        #[test]
        fn the_being_eaten_indicator_decays_to_off_over_enough_ticks() {
            // A lone critter cannot be eaten. Mark it for a few linger
            // ticks; after enough world ticks the indicator should be gone —
            // proving World::tick ages this indicator too.
            let mut critter = Critter::with_genome(
                100,
                100,
                NORTH,
                u32::MAX,
                1,
                100,
                0,
                Genome::all(Instruction::DoNothing),
            );
            critter.mark_being_eaten_for(3);
            let mut world =
                World::with_critters_and_pellets(TEST_WIDTH, TEST_HEIGHT, vec![critter], vec![]);

            for _ in 0..3 {
                world.tick(true);
            }

            assert!(!world.critters()[0].is_being_eaten());
        }
    }

    mod reset {
        use super::*;

        #[test]
        fn after_reset_every_critter_is_at_full_energy() {
            let mut rng = StdRng::seed_from_u64(0);
            let mut world =
                World::with_critters_and_pellets(TEST_WIDTH, TEST_HEIGHT, vec![], vec![]);

            world.reset(&mut rng);

            assert!(world
                .critters()
                .iter()
                .all(|c| c.energy() == INITIAL_ENERGY));
        }

        #[test]
        fn after_reset_the_critter_positions_change() {
            let mut rng = StdRng::seed_from_u64(0);
            let mut world = World::new(TEST_WIDTH, TEST_HEIGHT, &mut rng);
            let original_positions: Vec<(i32, i32)> =
                world.critters().iter().map(|c| (c.x(), c.y())).collect();

            world.reset(&mut rng);

            let new_positions: Vec<(i32, i32)> =
                world.critters().iter().map(|c| (c.x(), c.y())).collect();
            assert_ne!(new_positions, original_positions);
        }
    }

    mod toroidal_delta_tests {
        use super::super::toroidal_delta;

        #[test]
        fn equal_values_have_zero_delta() {
            assert_eq!(toroidal_delta(50, 50, 200), 0);
        }

        #[test]
        fn small_positive_delta_is_returned_as_is() {
            assert_eq!(toroidal_delta(50, 30, 200), 20);
        }

        #[test]
        fn small_negative_delta_is_returned_as_negative() {
            assert_eq!(toroidal_delta(30, 50, 200), -20);
        }

        #[test]
        fn delta_larger_than_half_size_wraps_to_a_smaller_negative_value() {
            // Half-size = 100. raw = (10 - 190).rem_euclid(200) = 20.
            // 20 < 100, so the result is 20 — but that's the unwrapped path.
            // For 150: raw = (190 - 40).rem_euclid(200) = 150. 150 > 100, returns 150 - 200 = -50.
            assert_eq!(toroidal_delta(190, 40, 200), -50);
        }

        #[test]
        fn delta_just_above_half_size_wraps_to_negative() {
            // (5 - 100).rem_euclid(200) = 105. > 100 → returns 105 - 200 = -95.
            assert_eq!(toroidal_delta(5, 100, 200), -95);
        }
    }

    mod splitting {
        use super::*;
        use crate::{Critter, Genome, Instruction, NORTH};

        #[test]
        fn a_critter_that_splits_appears_twice_once_its_division_finishes() {
            // Well fed: a division costs the attempt plus one energy per
            // tick of its duration, so a critter needs reserves to see one
            // through.
            let splitter = Critter::with_genome(
                100,
                100,
                NORTH,
                1,
                1,
                crate::MAX_CRITTER_ENERGY,
                0,
                Genome::all(Instruction::Split),
            );
            let mut world =
                World::with_critters_and_pellets(TEST_WIDTH, TEST_HEIGHT, vec![splitter], vec![]);

            // Division takes time, so the child arrives some ticks later.
            for _ in 0..=crate::SPLIT_DURATION_TICKS {
                world.tick(true);
            }

            assert_eq!(world.critters().len(), 2);
        }
    }

    mod wrapping {
        use super::*;
        use crate::{Critter, Genome, Instruction, EAST, NORTH};

        #[test]
        fn a_critter_that_walks_past_the_right_edge_wraps_to_the_left() {
            let critter = Critter::with_genome(
                TEST_WIDTH as i32 - 1,
                50,
                EAST,
                1,
                1,
                u32::MAX,
                0,
                Genome::all(Instruction::MoveSlow),
            );
            let mut world =
                World::with_critters_and_pellets(TEST_WIDTH, TEST_HEIGHT, vec![critter], vec![]);

            world.tick(true);

            assert_eq!(world.critters()[0].x(), 0);
        }

        #[test]
        fn a_critter_that_walks_past_the_top_edge_wraps_to_the_bottom() {
            let critter = Critter::with_genome(
                50,
                0,
                NORTH,
                1,
                1,
                u32::MAX,
                0,
                Genome::all(Instruction::MoveSlow),
            );
            let mut world =
                World::with_critters_and_pellets(TEST_WIDTH, TEST_HEIGHT, vec![critter], vec![]);

            world.tick(true);

            assert_eq!(world.critters()[0].y(), TEST_HEIGHT as i32 - 1);
        }
    }

    mod eating {
        use super::*;
        use crate::{Critter, Genome, Instruction, Pellet, NORTH, PELLET_ENERGY};

        const HUNGRY_INITIAL: u32 = 200;
        const STARTING_ENERGY: u32 = 10;
        // Energy after a single Eat firing tick where no pellet was found —
        // just the base 1-energy tick cost is paid since Eat itself is free.
        const STARTING_AFTER_FAILED_EAT: u32 =
            STARTING_ENERGY - Critter::upkeep_for(STARTING_ENERGY);

        // A critter whose genome decodes to Eat at every cursor and which fires
        // an instruction every tick. Energy is set just above zero so that
        // gaining a pellet is observable without hitting the cap.
        fn eating_critter(x: i32, y: i32) -> Critter {
            let mut critter = Critter::with_genome(
                x,
                y,
                NORTH,
                1, // fire every tick
                1,
                HUNGRY_INITIAL,
                0,
                Genome::all(Instruction::Eat),
            );
            critter.lose_energy(HUNGRY_INITIAL - STARTING_ENERGY);
            critter
        }

        // A critter whose genome decodes to DoNothing at every cursor — used
        // to confirm that critters which never execute Eat do not consume
        // pellets, even when overlapping one.
        fn idle_critter(x: i32, y: i32) -> Critter {
            let mut critter = Critter::with_genome(
                x,
                y,
                NORTH,
                1,
                1,
                HUNGRY_INITIAL,
                0,
                Genome::all(Instruction::DoNothing),
            );
            critter.lose_energy(HUNGRY_INITIAL - STARTING_ENERGY);
            critter
        }

        fn world_with(critter: Critter, pellet: Pellet) -> World {
            World::with_critters_and_pellets(TEST_WIDTH, TEST_HEIGHT, vec![critter], vec![pellet])
        }

        #[test]
        fn a_critter_executing_eat_while_overlapping_a_pellet_consumes_it() {
            let mut world = world_with(eating_critter(100, 100), Pellet::at(100, 100));

            world.tick(true);

            assert_eq!(world.pellets().len(), 0);
        }

        #[test]
        fn eating_a_pellet_increases_energy_by_the_pellet_energy_amount() {
            // Start, minus eat-attempt cost, minus the base tick cost, plus
            // the pellet's energy.
            let mut world = world_with(eating_critter(100, 100), Pellet::at(100, 100));

            world.tick(true);

            assert_eq!(
                world.critters()[0].energy(),
                STARTING_AFTER_FAILED_EAT + PELLET_ENERGY
            );
        }

        #[test]
        fn a_critter_that_does_not_execute_eat_leaves_an_overlapping_pellet_alone() {
            let mut world = world_with(idle_critter(100, 100), Pellet::at(100, 100));

            world.tick(true);

            assert_eq!(world.pellets().len(), 1);
            assert_eq!(
                world.critters()[0].energy(),
                STARTING_ENERGY - Critter::upkeep_for(STARTING_ENERGY)
            );
        }

        #[test]
        fn an_eating_critter_that_does_not_overlap_a_pellet_leaves_it_alone() {
            let mut world = world_with(eating_critter(50, 100), Pellet::at(100, 100));

            world.tick(true);

            assert_eq!(world.pellets().len(), 1);
            assert_eq!(world.critters()[0].energy(), STARTING_AFTER_FAILED_EAT);
        }

        #[test]
        fn a_pellet_just_inside_the_eating_distance_is_consumed() {
            let eater = eating_critter(100, 100);
            let pellet = Pellet::at(100 + (eater.radius() + PELLET_RADIUS - 1), 100);
            let mut world = world_with(eater, pellet);

            world.tick(true);

            assert_eq!(world.pellets().len(), 0);
        }

        #[test]
        fn a_pellet_outside_the_eating_distance_along_a_dominant_axis_is_not_consumed() {
            // dx=2, dy=15: dx² + dy² = 229 > (CRITTER_RADIUS+PELLET_RADIUS)² = 196.
            let mut world = world_with(eating_critter(100, 100), Pellet::at(102, 115));

            world.tick(true);

            assert_eq!(world.pellets().len(), 1);
        }

        #[test]
        fn a_pellet_at_exactly_the_eating_distance_is_not_consumed() {
            let eater = eating_critter(100, 100);
            let pellet = Pellet::at(100 + eater.radius() + PELLET_RADIUS, 100);
            let mut world = world_with(eater, pellet);

            world.tick(true);

            assert_eq!(world.pellets().len(), 1);
        }

        #[test]
        fn eating_can_push_energy_past_initial_energy() {
            // A critter at full initial energy that successfully eats ends up
            // above its initial — there is no cap at initial_energy, only at
            // MAX_CRITTER_ENERGY.
            let critter = Critter::with_genome(
                100,
                100,
                NORTH,
                1,
                1,
                HUNGRY_INITIAL,
                0,
                Genome::all(Instruction::Eat),
            );
            let mut world = world_with(critter, Pellet::at(100, 100));

            world.tick(true);

            assert_eq!(
                world.critters()[0].energy(),
                HUNGRY_INITIAL - Critter::upkeep_for(HUNGRY_INITIAL) + PELLET_ENERGY
            );
        }

        #[test]
        fn a_critter_near_the_left_edge_can_eat_a_pellet_near_the_right_edge_via_wrap() {
            // Placed within the eater's own reach of the far edge, so what
            // is being tested is the wrap rather than the reach.
            let eater = eating_critter(1, 100);
            let pellet = Pellet::at(TEST_WIDTH as i32 - 1, 100);
            let mut world = world_with(eater, pellet);

            world.tick(true);

            assert_eq!(world.pellets().len(), 0);
        }

        #[test]
        fn a_critter_near_the_top_edge_can_eat_a_pellet_near_the_bottom_edge_via_wrap() {
            let eater = eating_critter(100, 1);
            let pellet = Pellet::at(100, TEST_HEIGHT as i32 - 1);
            let mut world = world_with(eater, pellet);

            world.tick(true);

            assert_eq!(world.pellets().len(), 0);
        }
    }

    mod pellet_expiry {
        use super::*;
        use crate::PELLET_LIFESPAN_TICKS;

        fn world_with_one_pellet(pellet: Pellet) -> World {
            World::with_critters_and_pellets(TEST_WIDTH, TEST_HEIGHT, Vec::new(), vec![pellet])
        }

        #[test]
        fn a_pellet_short_of_its_lifespan_remains() {
            let mut world = world_with_one_pellet(Pellet::at(50, 50));

            for _ in 0..PELLET_LIFESPAN_TICKS - 1 {
                world.tick(true);
            }

            assert_eq!(world.pellets().len(), 1);
        }

        #[test]
        fn a_pellet_that_reaches_its_lifespan_is_gone() {
            let mut world = world_with_one_pellet(Pellet::at(50, 50));

            for _ in 0..PELLET_LIFESPAN_TICKS {
                world.tick(true);
            }

            assert!(world.pellets().is_empty());
        }

        #[test]
        fn poison_expires_on_the_same_schedule_as_food() {
            // Poison is food that kills; nothing about it makes it keep.
            let mut world = world_with_one_pellet(Pellet::poison_at(50, 50));

            for _ in 0..PELLET_LIFESPAN_TICKS {
                world.tick(true);
            }

            assert!(world.pellets().is_empty());
        }

        #[test]
        fn a_younger_pellet_outlives_an_older_one() {
            // Pellets age individually rather than being cleared in batches.
            let mut world = world_with_one_pellet(Pellet::at(50, 50));
            for _ in 0..PELLET_LIFESPAN_TICKS / 2 {
                world.tick(true);
            }
            world.add_pellet(Pellet::at(80, 80));

            for _ in 0..PELLET_LIFESPAN_TICKS / 2 {
                world.tick(true);
            }

            assert_eq!(world.pellets().len(), 1);
            assert_eq!(world.pellets()[0].x, 80.0);
        }
    }

    mod pellets {
        use super::*;

        #[test]
        fn the_world_feeds_itself_from_empty() {
            let mut rng = StdRng::seed_from_u64(0);
            let mut world = World::new(TEST_WIDTH, TEST_HEIGHT, &mut rng);

            feed_a_while(&mut world, &mut rng);

            // Deliveries recur, so repeated runs do fill the larder.
            assert!(!world.pellets().is_empty());
        }

        #[test]
        fn every_pellet_stays_inside_the_world_as_it_drifts() {
            // Pellets are emitted at the center and wrap at the edges, so no
            // amount of drifting should carry one outside the world.
            for seed in 0..20 {
                let mut rng = StdRng::seed_from_u64(seed);
                let mut world = World::new(TEST_WIDTH, TEST_HEIGHT, &mut rng);
                for _ in 0..200 {
                    world.tick(true);
                }
                for pellet in world.pellets() {
                    assert!(pellet.x >= 0.0 && pellet.x < TEST_WIDTH as f32);
                    assert!(pellet.y >= 0.0 && pellet.y < TEST_HEIGHT as f32);
                }
            }
        }

        #[test]
        fn drifting_pellets_spread_away_from_the_emitter_in_every_direction() {
            // Given time to travel, the batch should reach all four quadrants
            // rather than streaming off in one direction.
            let mut rng = StdRng::seed_from_u64(0);
            let mut world = World::new(TEST_WIDTH, TEST_HEIGHT, &mut rng);
            feed_a_while(&mut world, &mut rng);
            let (cx, cy) = (TEST_WIDTH as f32 / 2.0, TEST_HEIGHT as f32 / 2.0);

            for _ in 0..100 {
                world.tick(true);
            }

            let quadrants: std::collections::HashSet<(bool, bool)> = world
                .pellets()
                .iter()
                .map(|pellet| (pellet.x > cx, pellet.y > cy))
                .collect();
            assert_eq!(quadrants.len(), 4);
        }

        #[test]
        fn reset_empties_the_larder_and_feeds_it_afresh() {
            let mut rng = StdRng::seed_from_u64(0);
            let mut world = World::new(TEST_WIDTH, TEST_HEIGHT, &mut rng);
            feed_a_while(&mut world, &mut rng);
            let original: Vec<_> = world.pellets().to_vec();

            world.reset(&mut rng);
            assert!(world.pellets().is_empty());

            feed_a_while(&mut world, &mut rng);
            assert!(!world.pellets().is_empty());
            assert_ne!(world.pellets(), original.as_slice());
        }
    }

    mod total_energy {
        use super::*;
        use crate::{Critter, Genome, Instruction, Pellet, NORTH, PELLET_ENERGY};

        #[test]
        fn an_empty_world_has_zero_total_energy() {
            let world = World::with_critters_and_pellets(TEST_WIDTH, TEST_HEIGHT, vec![], vec![]);

            assert_eq!(world.total_energy(), 0);
        }

        #[test]
        fn total_energy_sums_each_critters_current_energy() {
            let critter_a = Critter::with_genome(
                50,
                50,
                NORTH,
                1,
                1,
                30,
                0,
                Genome::all(Instruction::DoNothing),
            );
            let critter_b = Critter::with_genome(
                70,
                70,
                NORTH,
                1,
                1,
                25,
                0,
                Genome::all(Instruction::DoNothing),
            );
            let world = World::with_critters_and_pellets(
                TEST_WIDTH,
                TEST_HEIGHT,
                vec![critter_a, critter_b],
                vec![],
            );

            assert_eq!(world.total_energy(), 55);
        }

        #[test]
        fn total_energy_counts_each_pellet_at_pellet_energy_value() {
            let pellets = vec![Pellet::at(10, 10), Pellet::at(20, 20), Pellet::at(30, 30)];
            let world = World::with_critters_and_pellets(TEST_WIDTH, TEST_HEIGHT, vec![], pellets);

            assert_eq!(world.total_energy(), 3 * PELLET_ENERGY);
        }

        #[test]
        fn total_energy_combines_critters_and_pellets() {
            let critter = Critter::with_genome(
                50,
                50,
                NORTH,
                1,
                1,
                40,
                0,
                Genome::all(Instruction::DoNothing),
            );
            let pellet = Pellet::at(20, 20);
            let world = World::with_critters_and_pellets(
                TEST_WIDTH,
                TEST_HEIGHT,
                vec![critter],
                vec![pellet],
            );

            assert_eq!(world.total_energy(), 40 + PELLET_ENERGY);
        }
    }

    mod reap_dead_critters {
        use super::*;
        use crate::{Critter, Genome, Instruction, NORTH};

        fn critter_with_energy(x: i32, y: i32, energy: u32) -> Critter {
            let mut critter = Critter::with_genome(
                x,
                y,
                NORTH,
                u32::MAX,
                1,
                100,
                0,
                Genome::all(Instruction::DoNothing),
            );
            critter.lose_energy(100 - energy);
            critter
        }

        #[test]
        fn it_removes_critters_whose_energy_is_zero() {
            let dead = critter_with_energy(50, 50, 0);
            let mut world =
                World::with_critters_and_pellets(TEST_WIDTH, TEST_HEIGHT, vec![dead], vec![]);

            world.reap_dead_critters();

            assert!(world.critters().is_empty());
        }

        #[test]
        fn it_leaves_critters_with_energy_above_zero() {
            let alive = critter_with_energy(50, 50, 1);
            let mut world =
                World::with_critters_and_pellets(TEST_WIDTH, TEST_HEIGHT, vec![alive], vec![]);

            world.reap_dead_critters();

            assert_eq!(world.critters().len(), 1);
        }

        #[test]
        fn it_removes_only_the_dead_critters_in_a_mixed_population() {
            let alive = critter_with_energy(50, 50, 5);
            let dead = critter_with_energy(60, 60, 0);
            let also_alive = critter_with_energy(70, 70, 10);
            let mut world = World::with_critters_and_pellets(
                TEST_WIDTH,
                TEST_HEIGHT,
                vec![alive, dead, also_alive],
                vec![],
            );

            world.reap_dead_critters();

            let energies: Vec<u32> = world.critters().iter().map(|c| c.energy()).collect();
            assert_eq!(energies, vec![5, 10]);
        }

        #[test]
        fn it_does_not_remove_pellets() {
            let dead = critter_with_energy(50, 50, 0);
            let pellet = Pellet::at(30, 30);
            let mut world =
                World::with_critters_and_pellets(TEST_WIDTH, TEST_HEIGHT, vec![dead], vec![pellet]);

            world.reap_dead_critters();

            assert_eq!(world.pellets().len(), 1);
        }
    }

    mod original_total_energy {
        use super::*;

        #[test]
        fn it_exceeds_the_energy_present_in_a_freshly_seeded_world() {
            // The budget is sized for the target population, so a world that
            // has only just started seeding holds less than its budget — the
            // gap is the room it has to grow into.
            let mut rng = StdRng::seed_from_u64(0);

            let world = World::new(TEST_WIDTH, TEST_HEIGHT, &mut rng);

            assert!(world.original_total_energy() > world.total_energy());
        }

        #[test]
        fn it_does_not_change_after_critters_lose_energy() {
            let mut rng = StdRng::seed_from_u64(0);
            let mut world = World::new(TEST_WIDTH, TEST_HEIGHT, &mut rng);
            let original = world.original_total_energy();

            for _ in 0..100 {
                world.tick(true);
            }

            assert_eq!(world.original_total_energy(), original);
            assert!(world.total_energy() < original);
        }
    }

    mod replenish_pellets {
        use super::*;
        use crate::{Critter, Genome, Instruction, MAX_CRITTER_ENERGY, NORTH, PELLET_ENERGY};

        // Empty of critters and food, but budgeted for a full world, so it
        // is hungry. The test-only constructor sizes its budget from actual
        // contents, which for an empty world would be zero.
        fn empty_world() -> World {
            let mut world =
                World::with_critters_and_pellets(TEST_WIDTH, TEST_HEIGHT, vec![], vec![]);
            world.original_total_energy = full_population_energy() + full_larder_energy();
            world
        }

        // Empty of food but peopled enough that its eruption site is allowed
        // to move, and up to full speed, for the tests about where the site
        // goes rather than about what feeding does.
        fn drifting_world() -> World {
            let mut rng = StdRng::seed_from_u64(11);
            let mut world = empty_world();
            for _ in 0..DRIFT_POPULATION_FLOOR {
                world
                    .critters
                    .push(spawn_critter(TEST_WIDTH, TEST_HEIGHT, &mut rng));
            }
            world.drift_age = ERUPTION_DRIFT_RAMP_TICKS;
            world
        }

        #[test]
        fn a_new_world_starts_empty_of_food_and_feeds_like_any_other() {
            // The opening larder is filled by the same metered spells as
            // every later one, not handed over whole at construction.
            let mut rng = StdRng::seed_from_u64(0);

            let world = World::new(TEST_WIDTH, TEST_HEIGHT, &mut rng);

            assert!(world.pellets().is_empty());
        }

        #[test]
        fn a_reset_world_starts_empty_of_food_too() {
            let mut rng = StdRng::seed_from_u64(0);
            let mut world = World::new(TEST_WIDTH, TEST_HEIGHT, &mut rng);
            for _ in 0..500 {
                world.feed(&mut rng);
            }

            world.reset(&mut rng);

            assert!(world.pellets().is_empty());
        }

        #[test]
        fn a_new_worlds_budget_covers_a_full_larder() {
            // Sized for the food the world is filling toward rather than what
            // it holds, or an empty world would consider itself stocked.
            let mut rng = StdRng::seed_from_u64(0);

            let world = World::new(TEST_WIDTH, TEST_HEIGHT, &mut rng);

            assert_eq!(
                world.original_total_energy(),
                full_population_energy() + full_larder_energy()
            );
        }

        #[test]
        fn the_site_drifts_every_frame() {
            // On a world old enough to have any speed at all: a fresh one
            // holds its site still on purpose.
            let mut world = drifting_world();
            let mut rng = StdRng::seed_from_u64(0);
            let before = world.eruption_site();

            world.feed(&mut rng);

            assert_ne!(world.eruption_site(), before);
        }

        #[test]
        fn each_step_is_the_configured_drift_in_some_direction() {
            // Pins the step's size exactly: the site moves at whatever speed
            // its age has earned, decomposed across the axes by its heading.
            // Measured away from the edges so no wrap folds the comparison,
            // and on a world old enough to be up to full pace.
            let mut world = drifting_world();
            let mut rng = StdRng::seed_from_u64(0);
            world.eruption_site = (TEST_WIDTH as f32 / 2.0, TEST_HEIGHT as f32 / 2.0);

            for _ in 0..50 {
                let before = world.eruption_site();
                world.feed(&mut rng);
                let after = world.eruption_site();
                let (dx, dy) = (after.0 - before.0, after.1 - before.1);
                let step = (dx * dx + dy * dy).sqrt();

                assert!(
                    (step - ERUPTION_SITE_DRIFT).abs() < 0.001,
                    "step was {step}, expected {ERUPTION_SITE_DRIFT}"
                );
                world.eruption_site = (TEST_WIDTH as f32 / 2.0, TEST_HEIGHT as f32 / 2.0);
            }
        }

        #[test]
        fn the_site_travels_along_its_heading_not_against_it() {
            // Pins direction as well as distance: the site moves the way its
            // heading points, which a step-size check alone cannot tell.
            let mut world = drifting_world();
            let mut rng = StdRng::seed_from_u64(0);
            world.eruption_site = (TEST_WIDTH as f32 / 2.0, TEST_HEIGHT as f32 / 2.0);
            world.eruption_heading = 0.0; // due east

            let before = world.eruption_site();
            world.feed(&mut rng);
            let after = world.eruption_site();

            // The heading turns a little first, but never past a right angle,
            // so an eastward heading must still carry the site east.
            assert!(
                after.0 > before.0,
                "site moved west ({} to {}) on an eastward heading",
                before.0,
                after.0
            );
        }

        #[test]
        fn the_site_travels_north_on_a_northward_heading() {
            let mut world = drifting_world();
            let mut rng = StdRng::seed_from_u64(0);
            world.eruption_site = (TEST_WIDTH as f32 / 2.0, TEST_HEIGHT as f32 / 2.0);
            world.eruption_heading = std::f32::consts::FRAC_PI_2; // due south in screen terms

            let before = world.eruption_site();
            world.feed(&mut rng);
            let after = world.eruption_site();

            assert!(after.1 > before.1);
        }

        #[test]
        fn the_heading_turns_both_ways() {
            // A heading that could only turn one way would spiral rather than
            // wander, so the turn spans zero in both directions.
            let mut world = drifting_world();
            let mut rng = StdRng::seed_from_u64(0);
            let (mut left, mut right) = (false, false);

            for _ in 0..200 {
                let before = world.eruption_heading;
                world.feed(&mut rng);
                if world.eruption_heading < before {
                    left = true;
                }
                if world.eruption_heading > before {
                    right = true;
                }
            }

            assert!(left && right, "heading turned only one way");
        }

        #[test]
        fn the_heading_turns_by_no_more_than_its_limit() {
            let mut world = empty_world();
            let mut rng = StdRng::seed_from_u64(0);

            for _ in 0..200 {
                let before = world.eruption_heading;
                world.feed(&mut rng);
                let turn = (world.eruption_heading - before).abs();

                assert!(turn <= ERUPTION_SITE_TURN + 0.001, "heading turned {turn}");
            }
        }

        #[test]
        fn the_site_drifts_slowly_rather_than_jumping() {
            // Continuous drift: a frame moves the site a little, so a viewer
            // can see it travel rather than teleport.
            let mut world = empty_world();
            let mut rng = StdRng::seed_from_u64(0);

            for _ in 0..200 {
                let before = world.eruption_site();
                world.feed(&mut rng);
                let after = world.eruption_site();
                let dx = toroidal_delta(
                    after.0.round() as i32,
                    before.0.round() as i32,
                    TEST_WIDTH as i32,
                ) as f32;
                let dy = toroidal_delta(
                    after.1.round() as i32,
                    before.1.round() as i32,
                    TEST_HEIGHT as i32,
                ) as f32;
                let step = (dx * dx + dy * dy).sqrt();
                assert!(
                    step <= ERUPTION_SITE_DRIFT + 1.5,
                    "site moved {step} in one frame"
                );
            }
        }

        #[test]
        fn the_site_wanders_across_the_world_over_time() {
            let mut world = drifting_world();
            let mut rng = StdRng::seed_from_u64(0);
            let start = world.eruption_site();

            for _ in 0..2000 {
                world.feed(&mut rng);
            }

            let (dx, dy) = (
                world.eruption_site().0 - start.0,
                world.eruption_site().1 - start.1,
            );
            assert!((dx * dx + dy * dy).sqrt() > ERUPTION_SITE_DRIFT * 10.0);
        }

        #[test]
        fn the_site_stays_inside_the_world() {
            let mut world = empty_world();
            let mut rng = StdRng::seed_from_u64(0);

            for _ in 0..5000 {
                world.feed(&mut rng);
                let (x, y) = world.eruption_site();
                assert!((0.0..TEST_WIDTH as f32).contains(&x));
                assert!((0.0..TEST_HEIGHT as f32).contains(&y));
            }
        }

        #[test]
        fn a_hungry_world_keeps_feeding_until_its_energy_is_restored() {
            // Nothing bounds the run but the budget: it lasts exactly as long
            // as the world is short of energy.
            let mut world = empty_world();
            let target = 4 * PELLET_BATCH_SIZE;
            world.original_total_energy = target as u32 * PELLET_ENERGY;
            let mut rng = StdRng::seed_from_u64(0);

            for _ in 0..4 {
                world.feed(&mut rng);
            }

            assert_eq!(world.pellets().len(), target);
        }

        #[test]
        fn a_restored_world_takes_on_no_more_food() {
            let mut world = empty_world();
            world.original_total_energy = 2 * PELLET_ENERGY;
            let mut rng = StdRng::seed_from_u64(0);
            world.feed(&mut rng);
            let after_run = world.pellets().len();

            world.feed(&mut rng);

            assert_eq!(after_run, 2);
            assert_eq!(world.pellets().len(), 2);
        }

        #[test]
        fn a_world_drained_of_food_refills_at_once() {
            // Feeding tracks consumption rather than arriving in rounds, so
            // there is no interval during which a starving world is refused.
            let mut world = empty_world();
            world.original_total_energy = 2 * PELLET_ENERGY;
            let mut rng = StdRng::seed_from_u64(0);
            world.feed(&mut rng);
            world.pellets.clear();

            world.feed(&mut rng);

            assert!(!world.pellets().is_empty());
        }

        #[test]
        fn a_world_below_its_energy_budget_takes_on_food() {
            let world = empty_world();

            assert!(world.needs_more_food());
        }

        #[test]
        fn a_world_at_its_energy_budget_takes_on_none() {
            let mut world = empty_world();
            world.original_total_energy = 0;

            assert!(!world.needs_more_food());
        }

        #[test]
        fn energy_inside_critters_counts_toward_the_budget() {
            // This is what makes the budget a thermostat rather than a
            // ceiling on pellets: a world whose critters are fat stops
            // feeding even with bare ground.
            let mut world = empty_world();
            world.original_total_energy = MAX_CRITTER_ENERGY;
            assert!(world.needs_more_food());

            world.critters.push(Critter::with_genome(
                10,
                10,
                NORTH,
                u32::MAX,
                1,
                MAX_CRITTER_ENERGY,
                0,
                Genome::all(Instruction::DoNothing),
            ));

            assert!(world.pellets().is_empty());
            assert!(!world.needs_more_food());
        }

        #[test]
        fn replenishing_stops_at_the_energy_budget() {
            let mut world = empty_world();
            world.original_total_energy = 3 * PELLET_ENERGY;
            let mut rng = StdRng::seed_from_u64(0);

            world.replenish_pellets(100, &mut rng);

            assert_eq!(world.pellets().len(), 3);
        }

        #[test]
        fn replenishing_adds_only_a_small_batch_at_a_time() {
            // Food trickles in the way critters do, rather than a whole
            // deficit's worth landing in one frame.
            let mut world = empty_world();
            world.original_total_energy = 500 * PELLET_ENERGY;
            let mut rng = StdRng::seed_from_u64(0);

            world.replenish_pellets(PELLET_BATCH_SIZE, &mut rng);

            assert_eq!(world.pellets().len(), PELLET_BATCH_SIZE);
        }

        #[test]
        fn repeated_small_batches_accumulate_without_bound() {
            // Nothing caps the larder: what limits food in a running world is
            // the length of a feeding run and what the critters eat.
            let mut world = empty_world();
            let mut rng = StdRng::seed_from_u64(0);
            let batches = NUM_PELLETS / PELLET_BATCH_SIZE + 2;

            for _ in 0..batches {
                world.replenish_pellets(PELLET_BATCH_SIZE, &mut rng);
            }

            assert_eq!(world.pellets().len(), batches * PELLET_BATCH_SIZE);
        }

        // Fills a world the way the main loop does: many small batches.
        // Bounded rather than looping on a condition, so a broken
        // condition fails the test instead of hanging it.
        fn fill_to_target<R: Rng>(world: &mut World, rng: &mut R) {
            for _ in 0..(NUM_PELLETS / PELLET_BATCH_SIZE + 2) {
                world.replenish_pellets(PELLET_BATCH_SIZE, rng);
            }
        }

        fn world_with_target(target: u32) -> World {
            // Construct an "empty" world and then make the target deterministic
            // by directly using with_critters_and_pellets with a known
            // composition that achieves total_energy == target.
            let mut world = empty_world();
            let pellets_needed = target / PELLET_ENERGY;
            let mut rng = StdRng::seed_from_u64(0);
            for _ in 0..pellets_needed {
                world
                    .pellets
                    .push(super::spawn_pellet(TEST_WIDTH, TEST_HEIGHT, &mut rng));
            }
            world.original_total_energy = world.total_energy();
            world
        }

        #[test]
        fn it_adds_enough_pellets_to_bring_total_energy_back_to_the_original() {
            let target = 10 * PELLET_ENERGY;
            let mut world = world_with_target(target);
            // Drain energy by removing pellets directly.
            world.pellets.clear();
            let mut rng = StdRng::seed_from_u64(1);

            fill_to_target(&mut world, &mut rng);

            assert!(world.total_energy() >= target);
        }

        #[test]
        fn replenished_pellets_emerge_from_the_emitter() {
            // Replenishment uses the same emitter as the initial batch, so new
            // food arrives from the center rather than appearing underfoot.
            let mut world = empty_world();
            world.original_total_energy = 50 * PELLET_ENERGY;
            for seed in 0..20 {
                let mut rng = StdRng::seed_from_u64(seed);
                world.pellets.clear();
                world.replenish_pellets(PELLET_BATCH_SIZE, &mut rng);
                for pellet in world.pellets() {
                    assert_eq!(pellet.x, TEST_WIDTH as f32 / 2.0);
                    assert_eq!(pellet.y, TEST_HEIGHT as f32 / 2.0);
                }
            }
        }

        #[test]
        fn replenishment_accounts_for_critter_energy_too() {
            // The original target = critter_energy + pellet_energy. If a critter
            // exists with substantial energy, fewer pellets are needed to top up.
            let mut world = empty_world();
            let critter = Critter::with_genome(
                50,
                50,
                NORTH,
                u32::MAX,
                1,
                100,
                0,
                Genome::all(Instruction::DoNothing),
            );
            world.critters.push(critter);
            // Target = 100 (critter) + 5 pellets * 100 (pellet energy hypothetical) — let's be concrete:
            world.original_total_energy = 100 + 5 * PELLET_ENERGY;
            let mut rng = StdRng::seed_from_u64(0);

            fill_to_target(&mut world, &mut rng);

            assert!(world.total_energy() >= world.original_total_energy);
        }
    }

    mod detect_critter_overlaps {
        use super::*;
        use crate::{Critter, Genome, Instruction, NORTH};

        fn idle_critter_at(x: i32, y: i32) -> Critter {
            Critter::with_genome(
                x,
                y,
                NORTH,
                u32::MAX,
                1,
                100,
                0,
                Genome::all(Instruction::DoNothing),
            )
        }

        #[test]
        fn two_overlapping_critters_both_get_their_flag_set() {
            let a = idle_critter_at(100, 100);
            let b = idle_critter_at(105, 100);
            let mut world =
                World::with_critters_and_pellets(TEST_WIDTH, TEST_HEIGHT, vec![a, b], vec![]);

            world.detect_critter_overlaps();

            assert!(world.critters()[0].is_overlapping_critter());
            assert!(world.critters()[1].is_overlapping_critter());
        }

        #[test]
        fn each_critter_records_the_genome_color_of_the_critter_it_overlaps() {
            // The two idle critters share the same DoNothing genome and so
            // share the same genome color. Each ends up recording that color.
            let a = idle_critter_at(100, 100);
            let b = idle_critter_at(105, 100);
            let expected_color = a.genome_color();
            let mut world =
                World::with_critters_and_pellets(TEST_WIDTH, TEST_HEIGHT, vec![a, b], vec![]);

            world.detect_critter_overlaps();

            assert_eq!(
                world.critters()[0].most_recent_overlap_color(),
                Some(expected_color)
            );
            assert_eq!(
                world.critters()[1].most_recent_overlap_color(),
                Some(expected_color)
            );
        }

        #[test]
        fn two_non_overlapping_critters_keep_their_flags_clear() {
            // Distance 50 between centers, well outside 2 * CRITTER_RADIUS.
            let a = idle_critter_at(50, 100);
            let b = idle_critter_at(100, 100);
            let mut world =
                World::with_critters_and_pellets(TEST_WIDTH, TEST_HEIGHT, vec![a, b], vec![]);

            world.detect_critter_overlaps();

            assert!(!world.critters()[0].is_overlapping_critter());
            assert!(!world.critters()[1].is_overlapping_critter());
        }

        #[test]
        fn critters_that_touch_at_the_overlap_threshold_are_not_marked_overlapping() {
            // Distance equals exactly the two radii together: tangent, not
            // overlapping. The strict `<` comparison must reject this pair.
            let a = idle_critter_at(100, 100);
            let touching = a.radius() * 2;
            let b = idle_critter_at(100 + touching, 100);
            let mut world =
                World::with_critters_and_pellets(TEST_WIDTH, TEST_HEIGHT, vec![a, b], vec![]);

            world.detect_critter_overlaps();

            assert!(!world.critters()[0].is_overlapping_critter());
            assert!(!world.critters()[1].is_overlapping_critter());
        }

        #[test]
        fn an_overlap_with_a_zero_energy_critter_marks_neither_critter() {
            let a = idle_critter_at(100, 100);
            let mut b = idle_critter_at(105, 100);
            b.lose_energy(b.energy());
            assert_eq!(b.energy(), 0);
            let mut world =
                World::with_critters_and_pellets(TEST_WIDTH, TEST_HEIGHT, vec![a, b], vec![]);

            world.detect_critter_overlaps();

            assert!(!world.critters()[0].is_overlapping_critter());
            assert!(!world.critters()[1].is_overlapping_critter());
        }

        #[test]
        fn two_critters_just_inside_the_overlap_threshold_are_both_marked() {
            // Centers one pixel closer than the two radii together, so the
            // pair counts as overlapping. Sitting just inside the boundary
            // kills threshold mutations whose squared cutoff drops below this
            // separation.
            let a = idle_critter_at(100, 100);
            let gap = a.radius() + idle_critter_at(0, 0).radius() - 1;
            let b = idle_critter_at(100 + gap, 100);
            let mut world =
                World::with_critters_and_pellets(TEST_WIDTH, TEST_HEIGHT, vec![a, b], vec![]);

            world.detect_critter_overlaps();

            assert!(world.critters()[0].is_overlapping_critter());
            assert!(world.critters()[1].is_overlapping_critter());
        }

        #[test]
        fn an_asymmetric_pair_outside_the_threshold_along_one_axis_is_not_marked() {
            // dx = R, dy = 2R - 1, so distance² = R² + (2R-1)², which clears
            // the (2R)² threshold by a hair for any radius. Asymmetry between
            // the axes ensures the squared-distance computation must square
            // each component separately rather than sum, subtract, or
            // otherwise collapse them into one value.
            let a = idle_critter_at(100, 100);
            let r = a.radius();
            let b = idle_critter_at(100 + r, 100 + 2 * r - 1);
            let mut world =
                World::with_critters_and_pellets(TEST_WIDTH, TEST_HEIGHT, vec![a, b], vec![]);

            world.detect_critter_overlaps();

            assert!(!world.critters()[0].is_overlapping_critter());
            assert!(!world.critters()[1].is_overlapping_critter());
        }

        #[test]
        fn an_overlapping_pair_at_indices_well_past_zero_is_still_marked() {
            // The detector's outer cursor steps i across the population. A
            // mutation that collapses i to a constant (e.g. `(cursor + offset)
            // / count` instead of `% count`) would only ever look at critter
            // zero. Hide the overlapping pair near the end of the list so the
            // cursor must actually walk through the indices to find it.
            let pair_first_index = OVERLAP_DETECTION_BUDGET - 2;
            let stride: i32 = 40;
            let per_row: usize = (TEST_WIDTH as i32 / stride) as usize;
            let mut critters: Vec<Critter> = (0..OVERLAP_DETECTION_BUDGET)
                .map(|i| {
                    let col = (i % per_row) as i32;
                    let row = (i / per_row) as i32;
                    idle_critter_at(20 + col * stride, 20 + row * stride)
                })
                .collect();
            // Place the last two critters on top of each other so they overlap.
            let overlap_x = 100;
            let overlap_y = 100;
            critters[pair_first_index] = idle_critter_at(overlap_x, overlap_y);
            critters[pair_first_index + 1] = idle_critter_at(overlap_x + 5, overlap_y);
            let mut world =
                World::with_critters_and_pellets(TEST_WIDTH, TEST_HEIGHT, critters, vec![]);

            world.detect_critter_overlaps();

            assert!(world.critters()[pair_first_index].is_overlapping_critter());
            assert!(world.critters()[pair_first_index + 1].is_overlapping_critter());
        }

        #[test]
        fn the_first_detection_call_does_not_visit_critters_beyond_its_budget() {
            // With a population larger than the budget, the first detect call
            // can only sweep the first OVERLAP_DETECTION_BUDGET critters. An
            // overlapping pair placed beyond that should remain unmarked.
            let world = world_with_overlapping_pair_at(OVERLAP_DETECTION_BUDGET);

            let world = run_detection_once(world);

            assert!(!world.critters()[OVERLAP_DETECTION_BUDGET].is_overlapping_critter());
            assert!(!world.critters()[OVERLAP_DETECTION_BUDGET + 1].is_overlapping_critter());
        }

        #[test]
        fn the_second_detection_call_visits_critters_the_first_call_did_not_reach() {
            // After the first call the cursor advances past the initial sweep,
            // so a second call covers the next slice — including a pair placed
            // beyond the first budget window.
            let world = world_with_overlapping_pair_at(OVERLAP_DETECTION_BUDGET);

            let world = run_detection_twice(world);

            assert!(world.critters()[OVERLAP_DETECTION_BUDGET].is_overlapping_critter());
            assert!(world.critters()[OVERLAP_DETECTION_BUDGET + 1].is_overlapping_critter());
        }

        #[test]
        fn two_zero_energy_critters_overlapping_each_other_mark_neither() {
            let mut a = idle_critter_at(100, 100);
            a.lose_energy(a.energy());
            let mut b = idle_critter_at(105, 100);
            b.lose_energy(b.energy());
            let mut world =
                World::with_critters_and_pellets(TEST_WIDTH, TEST_HEIGHT, vec![a, b], vec![]);

            world.detect_critter_overlaps();

            assert!(!world.critters()[0].is_overlapping_critter());
            assert!(!world.critters()[1].is_overlapping_critter());
        }

        fn world_with_overlapping_pair_at(pair_first_index: usize) -> World {
            // Build a population large enough that the pair sits beyond the
            // first detection budget, with all other critters spaced far apart
            // on a grid so they never overlap. The overlap pair sits at an
            // off-grid position so no grid critter accidentally overlaps it.
            let critter_count = pair_first_index + 2;
            let stride: i32 = 40;
            let per_row: usize = (TEST_WIDTH as i32 / stride) as usize;
            let mut critters: Vec<Critter> = (0..critter_count)
                .map(|i| {
                    let col = (i % per_row) as i32;
                    let row = (i / per_row) as i32;
                    idle_critter_at(20 + col * stride, 20 + row * stride)
                })
                .collect();
            // Off-grid coordinates: midpoint between four grid critters, so
            // no grid neighbor sits within the overlap radius.
            let overlap_x = 40;
            let overlap_y = 40;
            critters[pair_first_index] = idle_critter_at(overlap_x, overlap_y);
            critters[pair_first_index + 1] = idle_critter_at(overlap_x + 5, overlap_y);
            World::with_critters_and_pellets(TEST_WIDTH, TEST_HEIGHT, critters, vec![])
        }

        fn run_detection_once(mut world: World) -> World {
            world.detect_critter_overlaps();
            world
        }

        fn run_detection_twice(mut world: World) -> World {
            world.detect_critter_overlaps();
            world.detect_critter_overlaps();
            world
        }
    }

    mod physical_size {
        use super::*;
        use crate::{Critter, Genome, Instruction, NORTH, REFERENCE_ENERGY};

        fn eater_with_energy(x: i32, y: i32, energy: u32) -> Critter {
            Critter::with_genome(x, y, NORTH, 1, 1, energy, 0, Genome::all(Instruction::Eat))
        }

        fn idle_with_energy(x: i32, y: i32, energy: u32) -> Critter {
            Critter::with_genome(
                x,
                y,
                NORTH,
                u32::MAX,
                1,
                energy,
                0,
                Genome::all(Instruction::DoNothing),
            )
        }

        #[test]
        fn a_large_critter_reaches_a_pellet_a_small_one_cannot() {
            // Reach is the critter's own radius, so being well fed extends it.
            let small = eater_with_energy(100, 100, REFERENCE_ENERGY / 4);
            let large = eater_with_energy(100, 100, MAX_CRITTER_ENERGY);
            let gap = large.radius() + PELLET_RADIUS - 1;
            assert!(gap > small.radius() + PELLET_RADIUS);

            let mut small_world = World::with_critters_and_pellets(
                TEST_WIDTH,
                TEST_HEIGHT,
                vec![small],
                vec![Pellet::at(100 + gap, 100)],
            );
            let mut large_world = World::with_critters_and_pellets(
                TEST_WIDTH,
                TEST_HEIGHT,
                vec![large],
                vec![Pellet::at(100 + gap, 100)],
            );

            small_world.tick(true);
            large_world.tick(true);

            assert_eq!(
                small_world.pellets().len(),
                1,
                "small critter should not reach"
            );
            assert_eq!(large_world.pellets().len(), 0, "large critter should reach");
        }

        #[test]
        fn two_critters_touch_when_the_gap_is_less_than_their_radii_together() {
            // Each critter brings its own radius to the contact, so a fat one
            // is easier to reach and easier to be reached by.
            let eater = eater_with_energy(100, 100, MAX_CRITTER_ENERGY);
            let victim = idle_with_energy(100, 100, MAX_CRITTER_ENERGY);
            let gap = eater.radius() + victim.radius() - 1;
            let victim_energy = victim.energy();
            let mut world = World::with_critters_and_pellets(
                TEST_WIDTH,
                TEST_HEIGHT,
                vec![eater, idle_with_energy(100 + gap, 100, victim_energy)],
                vec![],
            );

            world.tick(true);

            assert!(
                world.critters()[1].energy() < victim_energy,
                "a victim within the two radii should have been bitten"
            );
        }

        #[test]
        fn a_small_victim_at_the_same_gap_is_out_of_reach() {
            // The same predator, the same distance, a smaller victim: now the
            // radii do not meet.
            let eater = eater_with_energy(100, 100, MAX_CRITTER_ENERGY);
            let fat = idle_with_energy(0, 0, MAX_CRITTER_ENERGY);
            let gap = eater.radius() + fat.radius() - 1;
            let lean_energy = REFERENCE_ENERGY / 16;
            let lean = idle_with_energy(100 + gap, 100, lean_energy);
            assert!(eater.radius() + lean.radius() <= gap);

            let mut world = World::with_critters_and_pellets(
                TEST_WIDTH,
                TEST_HEIGHT,
                vec![eater, lean],
                vec![],
            );

            world.tick(true);

            assert_eq!(world.critters()[1].energy(), lean_energy);
        }

        #[test]
        fn a_large_critter_reaches_exactly_as_far_as_its_own_radius() {
            // The boundary for a critter whose radius is not the reference
            // one: a pellet one pixel inside is eaten, one exactly at the
            // edge is not.
            let eater = eater_with_energy(100, 100, MAX_CRITTER_ENERGY);
            let reach = eater.radius() + PELLET_RADIUS;

            let mut inside = World::with_critters_and_pellets(
                TEST_WIDTH,
                TEST_HEIGHT,
                vec![eater_with_energy(100, 100, MAX_CRITTER_ENERGY)],
                vec![Pellet::at(100 + reach - 1, 100)],
            );
            let mut at_edge = World::with_critters_and_pellets(
                TEST_WIDTH,
                TEST_HEIGHT,
                vec![eater_with_energy(100, 100, MAX_CRITTER_ENERGY)],
                vec![Pellet::at(100 + reach, 100)],
            );

            inside.tick(true);
            at_edge.tick(true);

            assert_eq!(inside.pellets().len(), 0, "one inside should be eaten");
            assert_eq!(at_edge.pellets().len(), 1, "one at the edge should not");
        }

        #[test]
        fn a_large_critter_touches_poison_at_exactly_its_own_radius() {
            let probe = idle_with_energy(0, 0, MAX_CRITTER_ENERGY);
            let reach = probe.radius() + PELLET_RADIUS;

            let mut inside = World::with_critters_and_pellets(
                TEST_WIDTH,
                TEST_HEIGHT,
                vec![idle_with_energy(100, 100, MAX_CRITTER_ENERGY)],
                vec![Pellet::poison_at(100 + reach - 1, 100)],
            );
            let mut at_edge = World::with_critters_and_pellets(
                TEST_WIDTH,
                TEST_HEIGHT,
                vec![idle_with_energy(100, 100, MAX_CRITTER_ENERGY)],
                vec![Pellet::poison_at(100 + reach, 100)],
            );

            inside.tick(true);
            at_edge.tick(true);

            // What is being pinned is the reach, so this asks whether contact
            // happened at all rather than what the contact cost.
            assert!(
                inside.critters()[0].energy() < MAX_CRITTER_ENERGY,
                "one inside should be touched"
            );
            assert_eq!(
                at_edge.critters()[0].energy(),
                MAX_CRITTER_ENERGY,
                "one at the edge should not"
            );
        }

        #[test]
        fn a_large_critters_poison_reach_is_measured_by_true_distance() {
            // Offset on both axes, just outside the reach. Measuring the axes
            // by anything but the true diagonal would put this inside.
            let probe = idle_with_energy(0, 0, MAX_CRITTER_ENERGY);
            let reach = probe.radius() + PELLET_RADIUS;
            let offset = reach - 4;
            assert!(offset * offset * 2 > reach * reach, "must sit outside");

            let mut world = World::with_critters_and_pellets(
                TEST_WIDTH,
                TEST_HEIGHT,
                vec![idle_with_energy(100, 100, MAX_CRITTER_ENERGY)],
                vec![Pellet::poison_at(100 + offset, 100 + offset)],
            );

            world.tick(true);

            assert!(world.critters()[0].energy() > 0);
        }

        #[test]
        fn two_large_critters_overlap_at_exactly_their_radii_together() {
            let probe = idle_with_energy(0, 0, MAX_CRITTER_ENERGY);
            let reach = probe.radius() * 2;

            let mut inside = World::with_critters_and_pellets(
                TEST_WIDTH,
                TEST_HEIGHT,
                vec![
                    idle_with_energy(100, 100, MAX_CRITTER_ENERGY),
                    idle_with_energy(100 + reach - 1, 100, MAX_CRITTER_ENERGY),
                ],
                vec![],
            );
            let mut at_edge = World::with_critters_and_pellets(
                TEST_WIDTH,
                TEST_HEIGHT,
                vec![
                    idle_with_energy(100, 100, MAX_CRITTER_ENERGY),
                    idle_with_energy(100 + reach, 100, MAX_CRITTER_ENERGY),
                ],
                vec![],
            );

            inside.detect_critter_overlaps();
            at_edge.detect_critter_overlaps();

            assert!(inside.critters()[0].is_overlapping_critter());
            assert!(!at_edge.critters()[0].is_overlapping_critter());
        }

        #[test]
        fn a_large_predator_bites_at_exactly_the_two_radii() {
            let probe = idle_with_energy(0, 0, MAX_CRITTER_ENERGY);
            let reach = probe.radius() * 2;
            let victim_energy = MAX_CRITTER_ENERGY;

            let mut inside = World::with_critters_and_pellets(
                TEST_WIDTH,
                TEST_HEIGHT,
                vec![
                    eater_with_energy(100, 100, MAX_CRITTER_ENERGY),
                    idle_with_energy(100 + reach - 1, 100, victim_energy),
                ],
                vec![],
            );
            let mut at_edge = World::with_critters_and_pellets(
                TEST_WIDTH,
                TEST_HEIGHT,
                vec![
                    eater_with_energy(100, 100, MAX_CRITTER_ENERGY),
                    idle_with_energy(100 + reach, 100, victim_energy),
                ],
                vec![],
            );

            inside.tick(true);
            at_edge.tick(true);

            assert!(inside.critters()[1].energy() < victim_energy);
            assert_eq!(at_edge.critters()[1].energy(), victim_energy);
        }

        #[test]
        fn a_spent_critter_still_takes_up_space() {
            // The free allowance means a corpse is a full-sized body, not a
            // speck: it is still something a neighbor can run into.
            let critter = idle_with_energy(0, 0, 0);

            assert_eq!(critter.radius(), CRITTER_RADIUS);
        }
    }

    mod eating_critters {
        use super::*;
        use crate::{Critter, Genome, Instruction, Pellet, MAX_CRITTER_ENERGY, NORTH};

        const HUNGRY_INITIAL: u32 = 200;
        // Comfortably more than PREDATION_ATTACK_COST, so a predator in these
        // tests can afford to attack.
        const SOLVENT_PREDATOR_ENERGY: u32 = 300;
        // Energy after a single Eat firing tick where no transfer happened —
        // just the base 1-energy tick cost, since firing Eat is free and
        // nothing was attacked.
        const STARTING_AFTER_FAILED_EAT: u32 =
            SOLVENT_PREDATOR_ENERGY - Critter::upkeep_for(SOLVENT_PREDATOR_ENERGY);

        fn eating_critter_with_energy(x: i32, y: i32, energy: u32) -> Critter {
            Critter::with_genome(x, y, NORTH, 1, 1, energy, 0, Genome::all(Instruction::Eat))
        }

        // A predator with energy enough to cover the attack cost several
        // times over, so tests of what a bite does to the victim are not
        // derailed by the predator going broke.
        fn eating_critter(x: i32, y: i32) -> Critter {
            eating_critter_with_energy(x, y, SOLVENT_PREDATOR_ENERGY)
        }

        // A passive critter that does not execute Eat itself. Its energy is
        // set explicitly so it can serve as a victim of nearby eaters.
        fn idle_critter_with_energy(x: i32, y: i32, energy: u32) -> Critter {
            Critter::with_genome(
                x,
                y,
                NORTH,
                u32::MAX,
                1,
                energy,
                0,
                Genome::all(Instruction::DoNothing),
            )
        }

        #[test]
        fn predation_sometimes_kills_the_predator() {
            // Attacking carries a risk: some share of the time the predator
            // does not survive the meal.
            let deaths = predator_deaths_against_victim_energy(80, 200);

            // Near the rate the victim's energy implies, rather than merely
            // nonzero: a wide band, since 200 trials vary considerably.
            let expected = 200 * predation_death_percent(80) as usize / 100;
            assert!(
                deaths > expected / 2 && deaths < expected * 2,
                "{deaths} of 200 predators died, expected near {expected}"
            );
        }

        // How many of `trials` predators died biting a victim of this energy.
        fn predator_deaths_against_victim_energy(victim_energy: u32, trials: u64) -> usize {
            (0..trials)
                .filter(|&seed| {
                    let eater = Critter::with_genome(
                        100,
                        100,
                        NORTH,
                        1,
                        1,
                        HUNGRY_INITIAL,
                        seed,
                        Genome::all(Instruction::Eat),
                    );
                    let victim = idle_critter_with_energy(105, 100, victim_energy);
                    let mut world = World::with_critters_and_pellets(
                        TEST_WIDTH,
                        TEST_HEIGHT,
                        vec![eater, victim],
                        vec![],
                    );
                    world.tick(true);
                    world.critters()[0].energy() == 0
                })
                .count()
        }

        #[test]
        fn biting_a_strong_victim_kills_more_predators_than_biting_a_weak_one() {
            // Healthy prey fights back. The risk of attacking scales with what
            // the victim has left, so the fattest target is also the most
            // dangerous one.
            let against_weak = predator_deaths_against_victim_energy(1, 300);
            let against_strong = predator_deaths_against_victim_energy(MAX_CRITTER_ENERGY, 300);

            assert!(
                against_strong > against_weak * 2,
                "expected biting a strong victim to be far deadlier, \
                 got {against_strong} vs {against_weak} of 300"
            );
        }

        #[test]
        fn biting_a_spent_victim_is_near_the_base_risk() {
            // A victim with nothing left offers no resistance beyond the
            // baseline danger of attacking at all.
            let deaths = predator_deaths_against_victim_energy(1, 300);

            let expected = 300 * PREDATION_BASE_DEATH_PERCENT as usize / 100;
            assert!(
                deaths > expected / 2 && deaths < expected * 2 + 5,
                "{deaths} of 300 predators died against a spent victim, \
                 expected near {expected}"
            );
        }

        #[test]
        fn a_predator_that_dies_still_lands_its_bite() {
            // The meal happens; the predator merely does not survive it.
            let mut bitten_despite_death = false;
            for seed in 0..200u64 {
                let eater = Critter::with_genome(
                    100,
                    100,
                    NORTH,
                    1,
                    1,
                    HUNGRY_INITIAL,
                    seed,
                    Genome::all(Instruction::Eat),
                );
                let victim = idle_critter_with_energy(105, 100, 80);
                let mut world = World::with_critters_and_pellets(
                    TEST_WIDTH,
                    TEST_HEIGHT,
                    vec![eater, victim],
                    vec![],
                );
                world.tick(true);
                if world.critters()[0].energy() == 0 && world.critters()[1].energy() < 80 {
                    bitten_despite_death = true;
                    break;
                }
            }

            assert!(bitten_despite_death);
        }

        #[test]
        fn eating_a_pellet_carries_no_such_risk() {
            // Only predation is dangerous; foraging is not.
            for seed in 0..50u64 {
                let eater = Critter::with_genome(
                    100,
                    100,
                    NORTH,
                    1,
                    1,
                    HUNGRY_INITIAL,
                    seed,
                    Genome::all(Instruction::Eat),
                );
                let mut world = World::with_critters_and_pellets(
                    TEST_WIDTH,
                    TEST_HEIGHT,
                    vec![eater],
                    vec![Pellet::at(100, 100)],
                );

                world.tick(true);

                assert!(world.critters()[0].energy() > 0);
            }
        }

        #[test]
        fn attacking_a_critter_costs_the_predator_energy() {
            // A bite is work: the predator pays for the attempt out of its own
            // reserves, so a meal has to be worth more than the effort.
            let eater = eating_critter_with_energy(100, 100, 300);
            let victim = idle_critter_with_energy(105, 100, 80);
            let mut world = World::with_critters_and_pellets(
                TEST_WIDTH,
                TEST_HEIGHT,
                vec![eater, victim],
                vec![],
            );

            world.tick(true);

            let taken = 80 * PREDATION_SHARE_PERCENT / 100;
            assert_eq!(
                world.critters()[0].energy(),
                300 - Critter::upkeep_for(300) - PREDATION_ATTACK_COST + taken
            );
        }

        #[test]
        fn biting_spent_prey_leaves_the_predator_worse_off() {
            // The share taken from a depleted victim does not cover the cost
            // of attacking it, so preying on the weak is a losing move.
            let eater = eating_critter_with_energy(100, 100, 300);
            let victim = idle_critter_with_energy(105, 100, 4);
            let mut world = World::with_critters_and_pellets(
                TEST_WIDTH,
                TEST_HEIGHT,
                vec![eater, victim],
                vec![],
            );

            world.tick(true);

            assert!(
                world.critters()[0].energy() < 300 - Critter::upkeep_for(300),
                "expected the predator to end up down on the exchange"
            );
        }

        #[test]
        fn eating_a_pellet_carries_no_attack_cost() {
            // The cost is on attacking a critter, not on firing Eat: foraging
            // stays free, so the charge falls on cannibalism alone.
            let eater = eating_critter_with_energy(100, 100, 300);
            let mut world = World::with_critters_and_pellets(
                TEST_WIDTH,
                TEST_HEIGHT,
                vec![eater],
                vec![Pellet::at(100, 100)],
            );

            world.tick(true);

            assert_eq!(
                world.critters()[0].energy(),
                300 - Critter::upkeep_for(300) + crate::PELLET_ENERGY
            );
        }

        #[test]
        fn a_fruitless_eat_costs_nothing_beyond_the_tick() {
            // Nothing in range means no attack, so no attack cost.
            let eater = eating_critter_with_energy(100, 100, 300);
            let mut world =
                World::with_critters_and_pellets(TEST_WIDTH, TEST_HEIGHT, vec![eater], vec![]);

            world.tick(true);

            assert_eq!(world.critters()[0].energy(), 300 - Critter::upkeep_for(300));
        }

        #[test]
        fn a_predator_that_cannot_afford_the_attack_dies_of_it() {
            // The cost is paid before the meal, not netted against it. A
            // critter without the energy to attack spends itself doing it,
            // however rich the victim: otherwise the bite refunds the cost
            // and attacking can never be fatal, no matter how dear it is.
            let eater = eating_critter_with_energy(100, 100, PREDATION_ATTACK_COST);
            let victim = idle_critter_with_energy(105, 100, MAX_CRITTER_ENERGY);
            let mut world = World::with_critters_and_pellets(
                TEST_WIDTH,
                TEST_HEIGHT,
                vec![eater, victim],
                vec![],
            );

            world.tick(true);

            assert_eq!(world.critters()[0].energy(), 0);
        }

        #[test]
        fn a_predator_too_poor_to_attack_takes_no_bite() {
            // It cannot pay, so it does not eat: the victim keeps everything.
            let eater = eating_critter_with_energy(100, 100, 5);
            let victim = idle_critter_with_energy(105, 100, 200);
            let mut world = World::with_critters_and_pellets(
                TEST_WIDTH,
                TEST_HEIGHT,
                vec![eater, victim],
                vec![],
            );

            world.tick(true);

            assert_eq!(world.critters()[1].energy(), 200);
        }

        #[test]
        fn a_predator_takes_only_a_share_of_its_prey() {
            // Energy enough to cover the attack cost, so what is measured here
            // is the share taken rather than what the attempt cost.
            let eater = eating_critter_with_energy(100, 100, 300);
            let victim = idle_critter_with_energy(105, 100, 80);
            let mut world = World::with_critters_and_pellets(
                TEST_WIDTH,
                TEST_HEIGHT,
                vec![eater, victim],
                vec![],
            );

            world.tick(true);

            let taken = 80 * PREDATION_SHARE_PERCENT / 100;
            assert_eq!(
                world.critters()[0].energy(),
                300 - Critter::upkeep_for(300) - PREDATION_ATTACK_COST + taken
            );
        }

        #[test]
        fn prey_survives_a_bite_it_can_afford() {
            // Predation is a bite rather than an execution: what the predator
            // does not take, the prey keeps.
            let eater = eating_critter(100, 100);
            let victim = idle_critter_with_energy(105, 100, 80);
            let mut world = World::with_critters_and_pellets(
                TEST_WIDTH,
                TEST_HEIGHT,
                vec![eater, victim],
                vec![],
            );

            world.tick(true);

            let taken = 80 * PREDATION_SHARE_PERCENT / 100;
            assert_eq!(world.critters()[1].energy(), 80 - taken);
            assert!(world.critters()[1].energy() > 0);
        }

        #[test]
        fn prey_bitten_down_to_nothing_dies() {
            // A critter with too little left to give up its share dies of it.
            let eater = eating_critter(100, 100);
            let victim = idle_critter_with_energy(105, 100, 1);
            let mut world = World::with_critters_and_pellets(
                TEST_WIDTH,
                TEST_HEIGHT,
                vec![eater, victim],
                vec![],
            );

            world.tick(true);

            assert_eq!(world.critters()[1].energy(), 0);
        }

        #[test]
        fn repeated_bites_wear_prey_down() {
            let eater = eating_critter_with_energy(100, 100, MAX_CRITTER_ENERGY);
            let victim = idle_critter_with_energy(105, 100, 200);
            let mut world = World::with_critters_and_pellets(
                TEST_WIDTH,
                TEST_HEIGHT,
                vec![eater, victim],
                vec![],
            );

            let mut previous = world.critters()[1].energy();
            for _ in 0..5 {
                world.tick(true);
                let now = world.critters()[1].energy();
                assert!(
                    now < previous,
                    "prey did not lose energy: {now} vs {previous}"
                );
                previous = now;
            }
        }

        #[test]
        fn a_well_fed_eater_keeps_taking_on_energy() {
            // Nothing caps what a critter can hold, so a bite lands in full
            // however much the eater already has.
            let start = MAX_CRITTER_ENERGY - 5;
            let eater = eating_critter_with_energy(100, 100, start);
            let victim = idle_critter_with_energy(105, 100, MAX_CRITTER_ENERGY);
            let mut world = World::with_critters_and_pellets(
                TEST_WIDTH,
                TEST_HEIGHT,
                vec![eater, victim],
                vec![],
            );

            world.tick(true);

            let bite = MAX_CRITTER_ENERGY * PREDATION_SHARE_PERCENT / 100;
            assert_eq!(
                world.critters()[0].energy(),
                start - Critter::upkeep_for(start) - PREDATION_ATTACK_COST + bite
            );
            assert!(world.critters()[1].energy() > 0);
        }

        #[test]
        fn an_eaten_victim_is_marked_as_being_eaten() {
            let eater = eating_critter(100, 100);
            let victim = idle_critter_with_energy(105, 100, 80);
            let mut world = World::with_critters_and_pellets(
                TEST_WIDTH,
                TEST_HEIGHT,
                vec![eater, victim],
                vec![],
            );

            world.tick(true);

            assert!(world.critters()[1].is_being_eaten());
        }

        #[test]
        fn an_eater_prefers_a_pellet_over_a_touching_critter_when_both_are_in_range() {
            let eater = eating_critter(100, 100);
            let victim = idle_critter_with_energy(105, 100, 80);
            let pellet = Pellet::at(100, 100);
            let mut world = World::with_critters_and_pellets(
                TEST_WIDTH,
                TEST_HEIGHT,
                vec![eater, victim],
                vec![pellet],
            );

            world.tick(true);

            assert_eq!(world.pellets().len(), 0);
            // Victim untouched because the pellet was eaten instead.
            assert_eq!(world.critters()[1].energy(), 80);
        }

        #[test]
        fn an_eater_does_not_drain_a_critter_outside_the_overlap_radius() {
            let eater = eating_critter(100, 100);
            // 50 px away — well outside 2 * CRITTER_RADIUS.
            let bystander = idle_critter_with_energy(150, 100, 80);
            let mut world = World::with_critters_and_pellets(
                TEST_WIDTH,
                TEST_HEIGHT,
                vec![eater, bystander],
                vec![],
            );

            world.tick(true);

            assert_eq!(world.critters()[0].energy(), STARTING_AFTER_FAILED_EAT);
            assert_eq!(world.critters()[1].energy(), 80);
        }

        #[test]
        fn an_eater_does_not_drain_a_zero_energy_victim() {
            let eater = eating_critter(100, 100);
            let mut victim = idle_critter_with_energy(105, 100, 1);
            victim.lose_energy(1);
            let mut world = World::with_critters_and_pellets(
                TEST_WIDTH,
                TEST_HEIGHT,
                vec![eater, victim],
                vec![],
            );

            world.tick(true);

            assert_eq!(world.critters()[0].energy(), STARTING_AFTER_FAILED_EAT);
        }

        #[test]
        fn a_lone_eater_does_not_drain_itself() {
            let eater = eating_critter(100, 100);
            let mut world =
                World::with_critters_and_pellets(TEST_WIDTH, TEST_HEIGHT, vec![eater], vec![]);

            world.tick(true);

            assert_eq!(world.critters()[0].energy(), STARTING_AFTER_FAILED_EAT);
        }

        #[test]
        fn a_victim_just_inside_the_critter_eat_radius_is_bitten() {
            // Centers one pixel closer than the two radii together, which is
            // where the eat threshold falls. Mutations that shrink the
            // threshold below this separation would mistakenly classify the
            // pair as out of range.
            let eater = eating_critter(100, 100);
            let victim_probe = idle_critter_with_energy(0, 0, 80);
            let gap = eater.radius() + victim_probe.radius() - 1;
            let victim = idle_critter_with_energy(100 + gap, 100, 80);
            let mut world = World::with_critters_and_pellets(
                TEST_WIDTH,
                TEST_HEIGHT,
                vec![eater, victim],
                vec![],
            );

            world.tick(true);

            assert!(world.critters()[1].energy() < 80);
        }

        #[test]
        fn a_critter_at_exactly_the_eat_distance_is_not_drained() {
            // Distance equals exactly the two radii together — circles
            // tangent, not overlapping. The strict `<` comparison must reject
            // this pair.
            let eater = eating_critter(100, 100);
            let touching = eater.radius() + idle_critter_with_energy(0, 0, 80).radius();
            let victim = idle_critter_with_energy(100 + touching, 100, 80);
            let mut world = World::with_critters_and_pellets(
                TEST_WIDTH,
                TEST_HEIGHT,
                vec![eater, victim],
                vec![],
            );

            world.tick(true);

            assert_eq!(world.critters()[1].energy(), 80);
        }

        #[test]
        fn an_asymmetric_pair_outside_the_critter_eat_radius_is_not_drained() {
            // dx = R, dy = 2R - 1, so distance² = R² + (2R-1)², just past the
            // (2R)² eat threshold for any radius. Asymmetry between the axes
            // proves the squared-distance computation must square each
            // component independently rather than summing or subtracting them.
            let eater = eating_critter(100, 100);
            let r = (eater.radius() + idle_critter_with_energy(0, 0, 80).radius()) / 2;
            let victim = idle_critter_with_energy(100 + r, 100 + 2 * r - 1, 80);
            let mut world = World::with_critters_and_pellets(
                TEST_WIDTH,
                TEST_HEIGHT,
                vec![eater, victim],
                vec![],
            );

            world.tick(true);

            assert_eq!(world.critters()[1].energy(), 80);
        }

        #[test]
        fn biting_a_critter_works_across_the_toroidal_wrap() {
            let eater = eating_critter(2, 100);
            let victim = idle_critter_with_energy(TEST_WIDTH as i32 - 2, 100, 80);
            let mut world = World::with_critters_and_pellets(
                TEST_WIDTH,
                TEST_HEIGHT,
                vec![eater, victim],
                vec![],
            );

            world.tick(true);

            assert!(world.critters()[1].energy() < 80);
        }
    }

    mod feelers {
        use super::*;
        use crate::{
            Critter, Genome, Instruction, MAX_FEELER_ANGLE, MAX_FEELER_DISC, MAX_FEELER_LENGTH,
            MIN_FEELER_DISC, MIN_FEELER_LENGTH, NORTH,
        };

        // A critter whose feelers are held at `angle` degrees either side, of
        // the given length and disc size, facing north.
        // Grown both feelers, since these tests are about what a feeler does
        // rather than whether a critter has one.
        fn feeler_critter(x: i32, y: i32, length: f32, angle: f32, disc: f32) -> Critter {
            let mut genome = Genome::all(Instruction::DoNothing);
            genome.set_feeler_shape(length, angle, disc);
            genome.set_feelers_present(true, true);
            Critter::with_genome(x, y, NORTH, u32::MAX, 1, 60, 0, genome)
        }

        // Where a critter's left feeler tip sits, by the genome's shape.
        fn left_tip(critter: &Critter) -> (i32, i32) {
            critter.feeler_tips().0
        }

        #[test]
        fn a_feeler_feels_food_under_its_own_tip() {
            let critter = feeler_critter(100, 100, 30.0, 45.0, 4.0);
            let (tx, ty) = left_tip(&critter);
            let mut world = World::with_critters_and_pellets(
                TEST_WIDTH,
                TEST_HEIGHT,
                vec![critter],
                vec![Pellet::at(tx, ty)],
            );

            world.sense_feelers();

            assert_eq!(world.critters()[0].left_color(), crate::PELLET_COLOR);
        }

        #[test]
        fn a_feeler_feels_nothing_along_its_own_length() {
            // The disc at the tip is what senses, not the whole line: food
            // halfway along a feeler is not touched by it.
            let critter = feeler_critter(100, 100, 60.0, 45.0, 4.0);
            let (tx, ty) = left_tip(&critter);
            let halfway = ((100 + tx) / 2, (100 + ty) / 2);
            let mut world = World::with_critters_and_pellets(
                TEST_WIDTH,
                TEST_HEIGHT,
                vec![critter],
                vec![Pellet::at(halfway.0, halfway.1)],
            );

            world.sense_feelers();

            assert_eq!(world.critters()[0].left_color(), 0);
        }

        #[test]
        fn the_two_feelers_sense_their_own_sides() {
            let critter = feeler_critter(100, 100, 30.0, 45.0, 4.0);
            let (tx, ty) = left_tip(&critter);
            let mut world = World::with_critters_and_pellets(
                TEST_WIDTH,
                TEST_HEIGHT,
                vec![critter],
                vec![Pellet::at(tx, ty)],
            );

            world.sense_feelers();

            assert_eq!(world.critters()[0].left_color(), crate::PELLET_COLOR);
            assert_eq!(world.critters()[0].right_color(), 0);
        }

        #[test]
        fn the_grid_narrows_the_search_to_what_is_nearby() {
            // The filter's whole purpose, and the only thing about it a test
            // can see: a pellet far away vertically is never offered for a
            // distance check. Without this the band arithmetic could put every
            // pellet in one band, stay correct, and cost what it was written
            // to save.
            let (width, height) = (TEST_WIDTH as i32, TEST_HEIGHT as i32);
            let pellets: Vec<Pellet> = (0..height / 4).map(|r| Pellet::at(100, r * 4)).collect();
            let total = pellets.len();
            let grid = PelletGrid::build(&pellets, width, height);

            let offered = std::cell::Cell::new(0usize);
            grid.near(100, 0, &pellets, width, height, |_| {
                offered.set(offered.get() + 1);
                false
            });

            assert!(
                offered.get() * 2 < total,
                "should have looked at a small share of {total}, looked at {}",
                offered.get()
            );
        }

        #[test]
        fn the_grid_covers_a_full_sized_world() {
            // A cell for every part of a real field. Too few and a pellet near
            // an edge indexes past the end of them.
            let (width, height) = (1920, 1080);
            let pellets: Vec<Pellet> = (0..height / 8)
                .map(|r| Pellet::at(width - 3, r * 8))
                .collect();

            let grid = PelletGrid::build(&pellets, width, height);

            let filed: usize = grid.cells.iter().map(|cell| cell.len()).sum();
            assert_eq!(filed, pellets.len());
        }

        #[test]
        fn a_feeler_feels_food_far_down_the_field() {
            // Away from the origin on the axis the bands divide, so the band
            // arithmetic has to be right and not merely small.
            let y = TEST_HEIGHT as i32 - 60;
            let critter = feeler_critter(100, y, 30.0, 45.0, 4.0);
            let (tx, ty) = left_tip(&critter);
            let mut world = World::with_critters_and_pellets(
                TEST_WIDTH,
                TEST_HEIGHT,
                vec![critter],
                vec![Pellet::at(tx, ty)],
            );

            world.sense_feelers();

            assert_eq!(world.critters()[0].left_color(), crate::PELLET_COLOR);
        }

        #[test]
        fn a_discs_reach_is_measured_by_true_distance() {
            // Offset on both axes, inside the disc on each one alone but
            // outside it diagonally. Squaring and summing both is what tells
            // them apart.
            let disc = MAX_FEELER_DISC;
            let critter = feeler_critter(100, 100, 30.0, 45.0, disc);
            let (tx, ty) = left_tip(&critter);
            let offset = disc as i32 - 1;
            assert!(
                offset * offset * 2 > (disc * disc) as i32,
                "must sit outside"
            );
            let mut world = World::with_critters_and_pellets(
                TEST_WIDTH,
                TEST_HEIGHT,
                vec![critter],
                vec![Pellet::at(tx + offset, ty + offset)],
            );

            world.sense_feelers();

            assert_eq!(world.critters()[0].left_color(), 0);
        }

        #[test]
        fn a_disc_feels_food_just_inside_its_edge_on_both_axes() {
            // The other side of the same boundary, so the comparison cannot
            // simply be too strict.
            let disc = MAX_FEELER_DISC;
            let critter = feeler_critter(100, 100, 30.0, 45.0, disc);
            let (tx, ty) = left_tip(&critter);
            // Far enough out that the disc's radius has to be squared to
            // admit it: a radius merely doubled would put this outside.
            let offset = 5;
            assert!(offset * offset < (disc * disc) as i32, "must sit inside");
            assert!(
                offset * offset > (disc + disc) as i32,
                "must sit outside a radius that was doubled rather than squared"
            );
            let mut world = World::with_critters_and_pellets(
                TEST_WIDTH,
                TEST_HEIGHT,
                vec![critter],
                vec![Pellet::at(tx + offset, ty + offset)],
            );

            world.sense_feelers();

            assert_eq!(world.critters()[0].left_color(), crate::PELLET_COLOR);
        }

        #[test]
        fn the_right_feeler_reports_what_it_touches() {
            // Its own test rather than only appearing as the side that felt
            // nothing: a reader stuck at black would satisfy every test that
            // merely checks the other side.
            let critter = feeler_critter(100, 100, 30.0, 45.0, 4.0);
            let (tx, ty) = critter.feeler_tips().1;
            let mut world = World::with_critters_and_pellets(
                TEST_WIDTH,
                TEST_HEIGHT,
                vec![critter],
                vec![Pellet::at(tx, ty)],
            );

            world.sense_feelers();

            assert_eq!(world.critters()[0].right_color(), crate::PELLET_COLOR);
            assert_eq!(world.critters()[0].left_color(), 0);
        }

        #[test]
        fn a_critter_without_feelers_senses_nothing_through_them() {
            // A feeler a critter never grew reports darkness however much food
            // is sitting where its tip would have been.
            let mut genome = Genome::all(Instruction::DoNothing);
            genome.set_feeler_shape(30.0, 45.0, 4.0);
            let critter = Critter::with_genome(100, 100, NORTH, u32::MAX, 1, 60, 0, genome);
            let (tx, ty) = critter.feeler_tips().0;
            let (rx, ry) = critter.feeler_tips().1;
            let mut world = World::with_critters_and_pellets(
                TEST_WIDTH,
                TEST_HEIGHT,
                vec![critter],
                vec![Pellet::at(tx, ty), Pellet::at(rx, ry)],
            );

            world.sense_feelers();

            assert_eq!(world.critters()[0].left_color(), 0);
            assert_eq!(world.critters()[0].right_color(), 0);
        }

        #[test]
        fn a_critter_with_one_feeler_senses_only_through_that_one() {
            let mut genome = Genome::all(Instruction::DoNothing);
            genome.set_feeler_shape(30.0, 45.0, 4.0);
            genome.set_feelers_present(true, false);
            let critter = Critter::with_genome(100, 100, NORTH, u32::MAX, 1, 60, 0, genome);
            let (lx, ly) = critter.feeler_tips().0;
            let (rx, ry) = critter.feeler_tips().1;
            let mut world = World::with_critters_and_pellets(
                TEST_WIDTH,
                TEST_HEIGHT,
                vec![critter],
                vec![Pellet::at(lx, ly), Pellet::at(rx, ry)],
            );

            world.sense_feelers();

            assert_eq!(world.critters()[0].left_color(), crate::PELLET_COLOR);
            assert_eq!(world.critters()[0].right_color(), 0);
        }

        #[test]
        fn a_feeler_reaching_nothing_reports_darkness() {
            let mut world = World::with_critters_and_pellets(
                TEST_WIDTH,
                TEST_HEIGHT,
                vec![feeler_critter(100, 100, 30.0, 45.0, 4.0)],
                vec![],
            );

            world.sense_feelers();

            assert_eq!(world.critters()[0].left_color(), 0);
            assert_eq!(world.critters()[0].right_color(), 0);
        }

        #[test]
        fn a_feeler_tells_poison_from_food() {
            let critter = feeler_critter(100, 100, 30.0, 45.0, 4.0);
            let (tx, ty) = left_tip(&critter);
            let mut world = World::with_critters_and_pellets(
                TEST_WIDTH,
                TEST_HEIGHT,
                vec![critter],
                vec![Pellet::poison_at(tx, ty)],
            );

            world.sense_feelers();

            assert_eq!(world.critters()[0].left_color(), crate::POISON_COLOR);
        }

        #[test]
        fn a_bigger_disc_feels_what_a_smaller_one_misses() {
            // The disc's size is the critter's own, so a lineage can trade
            // precision for the chance of touching anything at all.
            let narrow = feeler_critter(100, 100, 30.0, 45.0, MIN_FEELER_DISC);
            let wide = feeler_critter(100, 100, 30.0, 45.0, MAX_FEELER_DISC);
            let (tx, ty) = left_tip(&narrow);
            // Just outside the small disc, inside the large one.
            let offset = MIN_FEELER_DISC as i32 + 2;
            let pellet = Pellet::at(tx + offset, ty);

            let mut narrow_world = World::with_critters_and_pellets(
                TEST_WIDTH,
                TEST_HEIGHT,
                vec![narrow],
                vec![pellet],
            );
            let mut wide_world =
                World::with_critters_and_pellets(TEST_WIDTH, TEST_HEIGHT, vec![wide], vec![pellet]);

            narrow_world.sense_feelers();
            wide_world.sense_feelers();

            assert_eq!(narrow_world.critters()[0].left_color(), 0);
            assert_eq!(wide_world.critters()[0].left_color(), crate::PELLET_COLOR);
        }

        #[test]
        fn longer_feelers_reach_further_out() {
            let short = feeler_critter(100, 100, MIN_FEELER_LENGTH, 45.0, 4.0);
            let long = feeler_critter(100, 100, MAX_FEELER_LENGTH, 45.0, 4.0);

            let (sx, sy) = left_tip(&short);
            let (lx, ly) = left_tip(&long);

            let short_reach = (sx - 100).pow(2) + (sy - 100).pow(2);
            let long_reach = (lx - 100).pow(2) + (ly - 100).pow(2);
            assert!(long_reach > short_reach);
        }

        #[test]
        fn a_wider_angle_holds_the_feelers_further_apart() {
            let narrow = feeler_critter(100, 100, 30.0, 0.0, 4.0);
            let wide = feeler_critter(100, 100, 30.0, MAX_FEELER_ANGLE, 4.0);

            let (nlx, nly) = narrow.feeler_tips().0;
            let (nrx, nry) = narrow.feeler_tips().1;
            let (wlx, wly) = wide.feeler_tips().0;
            let (wrx, wry) = wide.feeler_tips().1;

            let narrow_gap = (nlx - nrx).pow(2) + (nly - nry).pow(2);
            let wide_gap = (wlx - wrx).pow(2) + (wly - wry).pow(2);
            assert!(
                wide_gap > narrow_gap,
                "wide {wide_gap} should exceed narrow {narrow_gap}"
            );
        }
    }

    mod poison {
        use super::*;
        use crate::{Critter, Genome, Instruction, NORTH, PELLETS_PER_POISON};
        use rand::rngs::StdRng;
        use rand::SeedableRng;

        // What critter_at starts with, so tests can say what poison cost it.
        const POISONED_START: u32 = 60;

        fn critter_with_energy(x: i32, y: i32, energy: u32) -> Critter {
            Critter::with_genome(
                x,
                y,
                NORTH,
                u32::MAX, // never fires, so any harm is from contact alone
                1,
                energy,
                0,
                Genome::all(Instruction::DoNothing),
            )
        }

        fn critter_at(x: i32, y: i32) -> Critter {
            Critter::with_genome(
                x,
                y,
                NORTH,
                u32::MAX, // never fires, so any harm is from contact alone
                1,
                POISONED_START,
                0,
                Genome::all(Instruction::DoNothing),
            )
        }

        #[test]
        fn a_fixed_share_of_emitted_pellets_is_poison() {
            let mut world =
                World::with_critters_and_pellets(TEST_WIDTH, TEST_HEIGHT, vec![], vec![]);
            world.original_total_energy = full_population_energy() + full_larder_energy();
            let mut rng = StdRng::seed_from_u64(0);

            world.replenish_pellets(PELLETS_PER_POISON * 3, &mut rng);

            let poison = world.pellets().iter().filter(|p| p.poisonous).count();
            assert_eq!(poison, 3);
        }

        #[test]
        fn a_critter_touching_poison_is_harmed_by_it() {
            // No eating involved: the critter never fires an instruction.
            let mut world = World::with_critters_and_pellets(
                TEST_WIDTH,
                TEST_HEIGHT,
                vec![critter_at(100, 100)],
                vec![Pellet::poison_at(100, 100)],
            );

            world.tick(true);

            assert!(world.critters()[0].energy() < POISONED_START);
        }

        #[test]
        fn touching_poison_consumes_it() {
            let mut world = World::with_critters_and_pellets(
                TEST_WIDTH,
                TEST_HEIGHT,
                vec![critter_at(100, 100)],
                vec![Pellet::poison_at(100, 100)],
            );

            world.tick(true);

            assert!(world.pellets().is_empty());
        }

        #[test]
        fn poison_just_inside_the_touch_radius_harms() {
            // One pixel closer than the boundary. Pins where the radius
            // falls, which a poison pellet sitting on the critter does not.
            let touch = critter_at(0, 0).radius() + PELLET_RADIUS;
            let mut world = World::with_critters_and_pellets(
                TEST_WIDTH,
                TEST_HEIGHT,
                vec![critter_at(100, 100)],
                vec![Pellet::poison_at(100 + touch - 1, 100)],
            );

            world.tick(true);

            assert!(world.critters()[0].energy() < POISONED_START);
        }

        #[test]
        fn poison_at_exactly_the_touch_radius_does_not_kill() {
            // Tangent, not overlapping: the comparison is strict.
            let touch = critter_at(0, 0).radius() + PELLET_RADIUS;
            let mut world = World::with_critters_and_pellets(
                TEST_WIDTH,
                TEST_HEIGHT,
                vec![critter_at(100, 100)],
                vec![Pellet::poison_at(100 + touch, 100)],
            );

            world.tick(true);

            assert!(world.critters()[0].energy() > 0);
        }

        #[test]
        fn poison_offset_on_both_axes_is_measured_by_true_distance() {
            // dx = dy = touch - 2 puts the poison well beyond the radius
            // diagonally even though each axis alone is inside it. Squaring
            // and summing both axes is what tells them apart: any other way
            // of combining the axes reads this as a hit.
            let touch = critter_at(0, 0).radius() + PELLET_RADIUS;
            let offset = touch - 2;
            assert!(offset * offset * 2 > touch * touch, "must sit outside");
            let mut world = World::with_critters_and_pellets(
                TEST_WIDTH,
                TEST_HEIGHT,
                vec![critter_at(100, 100)],
                vec![Pellet::poison_at(100 + offset, 100 + offset)],
            );

            world.tick(true);

            // Untouched, not merely alive: halving never reaches zero, so
            // "still has energy" would hold whether or not the poison landed.
            assert_eq!(world.critters()[0].energy(), POISONED_START);
        }

        #[test]
        fn poison_reaches_across_the_toroidal_wrap() {
            // The critter sits against the left edge, the poison against the
            // right: they touch the short way round.
            let mut world = World::with_critters_and_pellets(
                TEST_WIDTH,
                TEST_HEIGHT,
                vec![critter_at(1, 100)],
                vec![Pellet::poison_at(TEST_WIDTH as i32 - 2, 100)],
            );

            world.tick(true);

            assert!(world.critters()[0].energy() < POISONED_START);
        }

        #[test]
        fn poison_is_not_checked_on_every_tick() {
            // Scanning poison is the most expensive thing the world does, so
            // it runs periodically rather than every tick. A critter can pass
            // through poison between checks and live.
            let mut world = World::with_critters_and_pellets(
                TEST_WIDTH,
                TEST_HEIGHT,
                vec![critter_at(100, 100)],
                vec![Pellet::poison_at(100, 100)],
            );
            world.tick(true); // the world's first tick checks, consuming it
            world.add_pellet(Pellet::poison_at(200, 200));
            world.add_critter(critter_at(200, 200));

            world.tick(true);

            assert!(world.critters()[1].energy() > 0);
        }

        #[test]
        fn poison_is_checked_again_once_the_interval_comes_round() {
            let mut world = World::with_critters_and_pellets(
                TEST_WIDTH,
                TEST_HEIGHT,
                vec![critter_at(100, 100)],
                vec![Pellet::poison_at(100, 100)],
            );
            world.tick(true);
            world.add_pellet(Pellet::poison_at(200, 200));
            world.add_critter(critter_at(200, 200));

            for _ in 0..POISON_CHECK_INTERVAL_TICKS - 1 {
                world.tick(true);
            }
            assert_eq!(
                world.critters()[1].energy(),
                POISONED_START,
                "should have been untouched until the interval came round"
            );

            world.tick(true);

            assert!(world.critters()[1].energy() < POISONED_START);
        }

        #[test]
        fn poison_takes_the_share_of_a_critter_the_pellet_says() {
            // No living cost here: this critter never fires an instruction,
            // so what it lost is the poison alone.
            let start = 800;
            let mut world = World::with_critters_and_pellets(
                TEST_WIDTH,
                TEST_HEIGHT,
                vec![critter_with_energy(100, 100, start)],
                vec![Pellet::poison_at(100, 100)],
            );

            world.tick(true);

            let taken = start * POISON_DAMAGE_PERCENT / 100;
            assert_eq!(world.critters()[0].energy(), start - taken);
        }

        #[test]
        fn poison_costs_the_rich_more_than_the_poor() {
            // Proportional rather than flat, so poison stays worth avoiding
            // however much a critter has banked: no reserve makes it trivial.
            let mut rich = World::with_critters_and_pellets(
                TEST_WIDTH,
                TEST_HEIGHT,
                vec![critter_with_energy(100, 100, 4000)],
                vec![Pellet::poison_at(100, 100)],
            );
            let mut poor = World::with_critters_and_pellets(
                TEST_WIDTH,
                TEST_HEIGHT,
                vec![critter_with_energy(100, 100, 400)],
                vec![Pellet::poison_at(100, 100)],
            );

            rich.tick(true);
            poor.tick(true);

            let rich_loss = 4000 - rich.critters()[0].energy();
            let poor_loss = 400 - poor.critters()[0].energy();
            assert!(
                rich_loss > poor_loss,
                "the richer critter should lose more: {rich_loss} vs {poor_loss}"
            );
        }

        #[test]
        fn poison_leaves_a_critter_alive_to_recover() {
            // Halving always leaves something behind, so poison sets a critter
            // back rather than ending it outright.
            let mut world = World::with_critters_and_pellets(
                TEST_WIDTH,
                TEST_HEIGHT,
                vec![critter_with_energy(100, 100, 400)],
                vec![Pellet::poison_at(100, 100)],
            );

            world.tick(true);

            assert!(world.critters()[0].energy() > 0);
        }

        #[test]
        fn poison_finishes_a_critter_with_almost_nothing_left() {
            // Half of one is nothing: a critter already down to its last is
            // still killed by what it touches.
            let mut world = World::with_critters_and_pellets(
                TEST_WIDTH,
                TEST_HEIGHT,
                vec![critter_with_energy(100, 100, 1)],
                vec![Pellet::poison_at(100, 100)],
            );

            world.tick(true);

            assert_eq!(world.critters()[0].energy(), 0);
        }

        #[test]
        fn surviving_poison_still_consumes_it() {
            // The pellet is spent on the critter that took it, whether or not
            // it proved fatal, so one poison cannot harm a crowd.
            let mut world = World::with_critters_and_pellets(
                TEST_WIDTH,
                TEST_HEIGHT,
                vec![critter_with_energy(100, 100, 600)],
                vec![Pellet::poison_at(100, 100)],
            );

            world.tick(true);

            assert!(world.critters()[0].energy() > 0);
            assert!(world.pellets().is_empty());
        }

        #[test]
        fn a_critter_clear_of_poison_lives() {
            let mut world = World::with_critters_and_pellets(
                TEST_WIDTH,
                TEST_HEIGHT,
                vec![critter_at(100, 100)],
                vec![Pellet::poison_at(160, 100)],
            );

            world.tick(true);

            assert!(world.critters()[0].energy() > 0);
            assert_eq!(world.pellets().len(), 1);
        }

        #[test]
        fn ordinary_food_is_harmless_to_touch() {
            let mut world = World::with_critters_and_pellets(
                TEST_WIDTH,
                TEST_HEIGHT,
                vec![critter_at(100, 100)],
                vec![Pellet::at(100, 100)],
            );

            world.tick(true);

            assert!(world.critters()[0].energy() > 0);
            assert_eq!(world.pellets().len(), 1);
        }
    }

    mod eruption_drift {
        use super::*;
        use rand::rngs::StdRng;
        use rand::SeedableRng;

        // How far the site travelled over `ticks`, starting from a world of
        // the given age.
        fn distance_covered(from_age: u32, ticks: u32) -> f32 {
            let mut rng = StdRng::seed_from_u64(4);
            let mut world = World::new(TEST_WIDTH, TEST_HEIGHT, &mut rng);
            // Populous enough that its food is allowed to move at all, and
            // aged on the drift's own clock rather than the world's.
            while world.critters.len() < DRIFT_POPULATION_FLOOR {
                world
                    .critters
                    .push(spawn_critter(TEST_WIDTH, TEST_HEIGHT, &mut rng));
            }
            world.drift_age = from_age;
            let start = world.eruption_site();
            let mut travelled = 0.0;
            let mut previous = start;
            for _ in 0..ticks {
                world.drift_eruption_site(&mut rng);
                let now = world.eruption_site();
                let dx = toroidal_delta(previous.0 as i32, now.0 as i32, TEST_WIDTH as i32) as f32;
                let dy = toroidal_delta(previous.1 as i32, now.1 as i32, TEST_HEIGHT as i32) as f32;
                travelled += (dx * dx + dy * dy).sqrt();
                previous = now;
            }
            travelled
        }

        // A world with a given number of critters in it, aged so its site is
        // up to full speed.
        fn world_of(population: usize) -> World {
            let mut rng = StdRng::seed_from_u64(4);
            let mut world = World::new(TEST_WIDTH, TEST_HEIGHT, &mut rng);
            world.drift_age = ERUPTION_DRIFT_RAMP_TICKS;
            world.critters.clear();
            for _ in 0..population {
                world
                    .critters
                    .push(spawn_critter(TEST_WIDTH, TEST_HEIGHT, &mut rng));
            }
            world
        }

        #[test]
        fn a_thin_population_stops_the_site_where_it_stands() {
            // Food that keeps moving away from a world already in trouble
            // finishes it. Below the mark the site holds still and waits.
            let mut world = world_of(DRIFT_POPULATION_FLOOR - 1);
            let mut rng = StdRng::seed_from_u64(0);
            let before = world.eruption_site();

            for _ in 0..600 {
                world.drift_eruption_site(&mut rng);
            }

            assert_eq!(world.eruption_site(), before);
        }

        #[test]
        fn a_healthy_population_lets_the_site_move() {
            let mut world = world_of(DRIFT_POPULATION_FLOOR);
            let mut rng = StdRng::seed_from_u64(0);
            let before = world.eruption_site();

            world.drift_eruption_site(&mut rng);

            assert_ne!(world.eruption_site(), before);
        }

        #[test]
        fn a_world_that_thins_out_loses_the_speed_it_had_earned() {
            // Back to a standstill rather than merely paused: a world that has
            // fallen that far starts its food over, and has to hold a
            // population again before it moves at all.
            let mut world = world_of(DRIFT_POPULATION_FLOOR - 1);
            let mut rng = StdRng::seed_from_u64(0);
            assert!(world.drift_age > 0);

            world.drift_eruption_site(&mut rng);

            assert_eq!(world.drift_age, 0);
        }

        #[test]
        fn a_recovered_world_climbs_again_from_nothing() {
            // And climbs from the beginning, so the minutes after a collapse
            // are as gentle as the minutes after a fresh start.
            let mut world = world_of(DRIFT_POPULATION_FLOOR - 1);
            let mut rng = StdRng::seed_from_u64(0);
            world.drift_eruption_site(&mut rng);

            let mut world = World {
                critters: (0..DRIFT_POPULATION_FLOOR)
                    .map(|_| spawn_critter(TEST_WIDTH, TEST_HEIGHT, &mut rng))
                    .collect(),
                ..world
            };
            for _ in 0..600 {
                world.drift_eruption_site(&mut rng);
            }

            assert_eq!(world.drift_age, 600);
        }

        #[test]
        fn a_recovered_world_goes_on_gathering_speed() {
            let mut world = world_of(DRIFT_POPULATION_FLOOR);
            let mut rng = StdRng::seed_from_u64(0);
            world.drift_age = 0;

            for _ in 0..600 {
                world.drift_eruption_site(&mut rng);
            }

            assert_eq!(world.drift_age, 600);
        }

        #[test]
        fn a_new_world_erupts_from_its_middle() {
            // Somewhere in particular, and the same somewhere every time: a
            // world that starts its food in a random corner starts some runs
            // with the larder against an edge and others with it in the open,
            // which is a difference between worlds that nothing chose.
            let mut rng = StdRng::seed_from_u64(0);

            for _ in 0..20 {
                let world = World::new(TEST_WIDTH, TEST_HEIGHT, &mut rng);

                assert_eq!(
                    world.eruption_site(),
                    (TEST_WIDTH as f32 / 2.0, TEST_HEIGHT as f32 / 2.0)
                );
            }
        }

        #[test]
        fn a_world_hydrated_from_a_genome_erupts_from_its_middle_too() {
            // The other way a world is made. Both constructors place the site,
            // so both have to place it in the same spot.
            let mut rng = StdRng::seed_from_u64(0);
            let genome = Genome::random(&mut rng);

            let world = World::with_seed_genome(TEST_WIDTH, TEST_HEIGHT, genome, &mut rng);

            assert_eq!(
                world.eruption_site(),
                (TEST_WIDTH as f32 / 2.0, TEST_HEIGHT as f32 / 2.0)
            );
        }

        #[test]
        fn a_reset_world_erupts_from_its_middle_again() {
            // A reset is a new world, so it begins where a new world begins
            // rather than wherever the last one had wandered to.
            let mut rng = StdRng::seed_from_u64(0);
            let mut world = world_of(DRIFT_POPULATION_FLOOR);
            for _ in 0..600 {
                world.drift_eruption_site(&mut rng);
            }
            assert_ne!(
                world.eruption_site(),
                (TEST_WIDTH as f32 / 2.0, TEST_HEIGHT as f32 / 2.0)
            );

            world.reset(&mut rng);

            assert_eq!(
                world.eruption_site(),
                (TEST_WIDTH as f32 / 2.0, TEST_HEIGHT as f32 / 2.0)
            );
        }

        #[test]
        fn a_reset_world_starts_its_site_still_again() {
            // The speed is earned by a world's own age, so a fresh one has
            // none of it however long the last one ran.
            let mut rng = StdRng::seed_from_u64(0);
            let mut world = World::new(TEST_WIDTH, TEST_HEIGHT, &mut rng);
            world.ticks = ERUPTION_DRIFT_RAMP_TICKS;

            world.reset(&mut rng);

            let before = world.eruption_site();
            world.drift_eruption_site(&mut rng);
            assert_eq!(world.eruption_site(), before);
        }

        #[test]
        fn the_site_takes_six_minutes_to_reach_full_speed() {
            // Stated in the time it means rather than only in ticks: every
            // other test here reads the constant, so any value would satisfy
            // them, and how long the climb takes is the whole point of it.
            const TICKS_PER_SECOND: u32 = 60;

            assert_eq!(ERUPTION_DRIFT_RAMP_TICKS / TICKS_PER_SECOND / 60, 6);
        }

        #[test]
        fn a_new_world_barely_moves_its_eruption_site() {
            // Where food comes from should be somewhere, at first. A site that
            // wanders from the outset never gives a place time to be worth
            // being in.
            let travelled = distance_covered(0, 60);

            // Not nothing at all: the helper measures through whole pixels, so
            // a site standing still still shows a little rounding.
            assert!(
                travelled < 4.0,
                "a fresh site should hardly move, went {travelled}"
            );
        }

        #[test]
        fn an_older_world_moves_its_site_faster() {
            let young = distance_covered(0, 600);
            let old = distance_covered(ERUPTION_DRIFT_RAMP_TICKS, 600);

            assert!(
                old > young * 10.0,
                "an old world should drift far faster: {old} against {young}"
            );
        }

        #[test]
        fn the_site_stops_gathering_speed_once_it_is_up_to_pace() {
            // The climb has an end: past the ramp a world drifts at a settled
            // speed rather than winding up without limit.
            let ramped = distance_covered(ERUPTION_DRIFT_RAMP_TICKS, 600);
            let much_later = distance_covered(ERUPTION_DRIFT_RAMP_TICKS * 4, 600);

            assert!(
                (much_later - ramped).abs() < ramped * 0.05,
                "speed should have settled: {much_later} against {ramped}"
            );
        }

        #[test]
        fn the_speed_climbs_gently_rather_than_in_a_jump() {
            // The acceleration itself should be hard to notice: a minute on
            // either side of any moment looks much the same.
            let before = distance_covered(ERUPTION_DRIFT_RAMP_TICKS, 600);
            let after = distance_covered(ERUPTION_DRIFT_RAMP_TICKS + 3600, 600);

            assert!(
                after < before * 2.0,
                "a minute should not double the pace: {after} against {before}"
            );
        }
    }

    mod emanating_pellets {
        use super::*;

        use rand::rngs::StdRng;
        use rand::SeedableRng;

        #[test]
        fn a_spawned_pellet_starts_at_the_center_of_the_world() {
            let mut rng = StdRng::seed_from_u64(0);

            let pellet = spawn_pellet(TEST_WIDTH, TEST_HEIGHT, &mut rng);

            assert_eq!(pellet.x.round() as i32, TEST_WIDTH as i32 / 2);
            assert_eq!(pellet.y.round() as i32, TEST_HEIGHT as i32 / 2);
        }

        #[test]
        fn spawned_pellets_head_off_in_a_range_of_directions() {
            // Every pellet leaves the emitter on its own heading, so a batch
            // spreads rather than travelling as one clump.
            let mut rng = StdRng::seed_from_u64(0);

            let headings: Vec<(i32, i32)> = (0..20)
                .map(|_| {
                    let pellet = spawn_pellet(TEST_WIDTH, TEST_HEIGHT, &mut rng);
                    // Scale up so distinct slow velocities do not round together.
                    ((pellet.dx * 1000.0) as i32, (pellet.dy * 1000.0) as i32)
                })
                .collect();

            let distinct: std::collections::HashSet<(i32, i32)> =
                headings.iter().copied().collect();
            assert!(
                distinct.len() > 10,
                "expected varied headings, got {} distinct out of 20",
                distinct.len()
            );
        }

        #[test]
        fn a_spawned_pellet_drifts_slowly() {
            // Slow enough that pellets linger near the emitter rather than
            // shooting to the edge in a moment.
            let mut rng = StdRng::seed_from_u64(0);

            for _ in 0..20 {
                let pellet = spawn_pellet(TEST_WIDTH, TEST_HEIGHT, &mut rng);

                let speed = (pellet.dx * pellet.dx + pellet.dy * pellet.dy).sqrt();
                assert!(
                    speed > 0.0 && speed <= PELLET_MAX_DRIFT,
                    "speed {speed} outside the expected drift range"
                );
            }
        }

        #[test]
        fn ticking_moves_a_pellet_along_its_heading() {
            let mut rng = StdRng::seed_from_u64(0);
            let mut world = World::new(TEST_WIDTH, TEST_HEIGHT, &mut rng);
            world.feed(&mut rng);
            let before = world.pellets()[0];

            world.tick(true);

            let after = world.pellets()[0];
            assert_eq!(after.x, before.x + before.dx);
            assert_eq!(after.y, before.y + before.dy);
        }

        #[test]
        fn a_drifting_pellet_wraps_around_the_world_edge() {
            let mut rng = StdRng::seed_from_u64(0);
            let mut world = World::new(TEST_WIDTH, TEST_HEIGHT, &mut rng);
            world.feed(&mut rng);
            world.pellets[0] = Pellet {
                x: TEST_WIDTH as f32 - 0.5,
                y: 10.0,
                dx: 1.0,
                dy: 0.0,
                poisonous: false,
                age: 0,
            };

            world.tick(true);

            assert!(world.pellets()[0].x < 1.0);
        }
    }

    mod seeding {
        use super::*;
        use crate::Instruction;
        use rand::rngs::StdRng;
        use rand::SeedableRng;

        #[test]
        fn a_new_world_starts_with_a_seed_population_rather_than_the_full_target() {
            let mut rng = StdRng::seed_from_u64(0);

            let world = World::new(TEST_WIDTH, TEST_HEIGHT, &mut rng);

            assert_eq!(world.critters().len(), SEED_POPULATION);
        }

        #[test]
        fn a_new_worlds_energy_budget_covers_the_full_target_population() {
            // Replenishment tops the world back up to its original budget, so
            // that budget must reflect the population the world is growing
            // toward — not the handful it starts with, which would starve it.
            let mut rng = StdRng::seed_from_u64(0);

            let world = World::new(TEST_WIDTH, TEST_HEIGHT, &mut rng);

            let full_population_energy = NUM_CRITTERS as u32 * INITIAL_ENERGY;
            assert!(world.original_total_energy >= full_population_energy);
        }

        #[test]
        fn seeding_more_critters_adds_them_to_the_population() {
            let mut rng = StdRng::seed_from_u64(0);
            let mut world = World::new(TEST_WIDTH, TEST_HEIGHT, &mut rng);
            let before = world.critters().len();

            world.seed_more_critters(10, &mut rng);

            assert_eq!(world.critters().len(), before + 10);
        }

        #[test]
        fn seeding_stops_at_the_target_population() {
            let mut rng = StdRng::seed_from_u64(0);
            let mut world = World::new(TEST_WIDTH, TEST_HEIGHT, &mut rng);

            world.seed_more_critters(NUM_CRITTERS * 2, &mut rng);

            assert_eq!(world.critters().len(), NUM_CRITTERS);
        }

        #[test]
        fn a_seed_genome_world_seeds_newcomers_with_that_same_genome() {
            // A --genome run should stay a monoculture as it ramps, rather
            // than diluting itself with randomly generated newcomers.
            let mut rng = StdRng::seed_from_u64(0);
            let seed = Genome::all(Instruction::Split);
            let mut world =
                World::with_seed_genome(TEST_WIDTH, TEST_HEIGHT, seed.clone(), &mut rng);

            world.seed_more_critters(10, &mut rng);

            assert!(world
                .critters()
                .iter()
                .all(|critter| *critter.genome() == seed));
        }

        #[test]
        fn a_world_without_a_seed_genome_seeds_newcomers_randomly() {
            let mut rng = StdRng::seed_from_u64(0);
            let mut world = World::new(TEST_WIDTH, TEST_HEIGHT, &mut rng);

            world.seed_more_critters(10, &mut rng);

            let first = world.critters()[0].genome().clone();
            assert!(world
                .critters()
                .iter()
                .any(|critter| *critter.genome() != first));
        }

        #[test]
        fn a_world_at_its_target_population_is_done_seeding() {
            let mut rng = StdRng::seed_from_u64(0);
            let mut world = World::new(TEST_WIDTH, TEST_HEIGHT, &mut rng);
            assert!(!world.is_fully_seeded());

            world.seed_more_critters(NUM_CRITTERS, &mut rng);

            assert!(world.is_fully_seeded());
        }
    }

    mod population_too_low {
        use super::*;
        use crate::{Critter, Genome, Instruction, NORTH};

        fn world_with_critter_count(count: usize) -> World {
            let critters = (0..count)
                .map(|i| {
                    Critter::with_genome(
                        i as i32,
                        0,
                        NORTH,
                        1,
                        1,
                        100,
                        i as u64,
                        Genome::all(Instruction::DoNothing),
                    )
                })
                .collect();
            World::with_critters_and_pellets(TEST_WIDTH, TEST_HEIGHT, critters, vec![])
        }

        #[test]
        fn it_returns_false_when_population_equals_the_minimum() {
            let world = world_with_critter_count(MIN_POPULATION);

            assert!(!world.population_too_low());
        }

        #[test]
        fn it_returns_true_when_population_is_just_below_the_minimum() {
            let world = world_with_critter_count(MIN_POPULATION - 1);

            assert!(world.population_too_low());
        }

        #[test]
        fn it_returns_false_when_population_is_well_above_the_minimum() {
            let world = world_with_critter_count(MIN_POPULATION + 30);

            assert!(!world.population_too_low());
        }
    }

    mod generation {
        use super::*;

        #[test]
        fn a_new_world_starts_at_generation_one() {
            let mut rng = StdRng::seed_from_u64(0);

            let world = World::new(TEST_WIDTH, TEST_HEIGHT, &mut rng);

            assert_eq!(world.generation(), 1);
        }

        #[test]
        fn each_reset_increments_the_generation() {
            let mut rng = StdRng::seed_from_u64(0);
            let mut world = World::new(TEST_WIDTH, TEST_HEIGHT, &mut rng);

            world.reset(&mut rng);

            assert_eq!(world.generation(), 2);
        }

        #[test]
        fn the_generation_count_persists_across_multiple_resets() {
            let mut rng = StdRng::seed_from_u64(0);
            let mut world = World::new(TEST_WIDTH, TEST_HEIGHT, &mut rng);

            world.reset(&mut rng);
            world.reset(&mut rng);
            world.reset(&mut rng);

            assert_eq!(world.generation(), 4);
        }
    }

    mod dominant_genome {
        use super::*;
        use crate::{Critter, Genome, Instruction, EAST};

        fn critter_with(x: i32, genome: Genome) -> Critter {
            Critter::with_genome(x, 0, EAST, 1, 1, 10, 0, genome)
        }

        #[test]
        fn it_returns_none_when_the_world_has_no_critters() {
            let world = World::with_critters_and_pellets(100, 100, Vec::new(), Vec::new());

            assert!(world.dominant_genome().is_none());
        }

        #[test]
        fn it_returns_the_genome_shared_by_the_majority() {
            let majority = Genome::all(Instruction::TurnLeft15);
            let minority = Genome::all(Instruction::TurnRight15);
            let critters = vec![
                critter_with(0, majority.clone()),
                critter_with(1, minority.clone()),
                critter_with(2, majority.clone()),
                critter_with(3, majority.clone()),
            ];
            let world = World::with_critters_and_pellets(100, 100, critters, Vec::new());

            assert_eq!(world.dominant_genome(), Some(&majority));
        }

        #[test]
        fn it_picks_the_genome_with_the_higher_count_even_when_seen_later() {
            // The minority appears first; the majority appears later. The
            // accessor must follow the count, not the encounter order.
            let later_majority = Genome::all(Instruction::TurnLeft15);
            let early_minority = Genome::all(Instruction::TurnRight15);
            let critters = vec![
                critter_with(0, early_minority.clone()),
                critter_with(1, later_majority.clone()),
                critter_with(2, later_majority.clone()),
                critter_with(3, later_majority.clone()),
            ];
            let world = World::with_critters_and_pellets(100, 100, critters, Vec::new());

            assert_eq!(world.dominant_genome(), Some(&later_majority));
        }

        #[test]
        fn it_counts_more_than_two_copies_correctly() {
            // Three of one genome beats two of another. Catches mutations
            // that turn the count increment into a no-op (all counts stay
            // at 1) or that mishandle the equality check (every critter
            // becomes its own entry with count 1).
            let three_copy = Genome::all(Instruction::TurnLeft15);
            let two_copy = Genome::all(Instruction::TurnRight15);
            let critters = vec![
                critter_with(0, two_copy.clone()),
                critter_with(1, three_copy.clone()),
                critter_with(2, two_copy.clone()),
                critter_with(3, three_copy.clone()),
                critter_with(4, three_copy.clone()),
            ];
            let world = World::with_critters_and_pellets(100, 100, critters, Vec::new());

            assert_eq!(world.dominant_genome(), Some(&three_copy));
        }

        #[test]
        fn ties_are_broken_by_first_seen_in_the_critter_list() {
            let first = Genome::all(Instruction::TurnLeft15);
            let second = Genome::all(Instruction::TurnRight15);
            let critters = vec![
                critter_with(0, first.clone()),
                critter_with(1, second.clone()),
                critter_with(2, first.clone()),
                critter_with(3, second.clone()),
            ];
            let world = World::with_critters_and_pellets(100, 100, critters, Vec::new());

            assert_eq!(world.dominant_genome(), Some(&first));
        }
    }

    mod with_seed_genome {
        use super::*;
        use crate::{Genome, Instruction};

        #[test]
        fn every_critter_in_the_world_starts_with_the_given_genome() {
            let seed = Genome::all(Instruction::Split);
            let mut rng = StdRng::seed_from_u64(0);

            let world = World::with_seed_genome(TEST_WIDTH, TEST_HEIGHT, seed.clone(), &mut rng);

            assert!(!world.critters().is_empty());
            for critter in world.critters() {
                assert_eq!(critter.genome(), &seed);
            }
        }

        #[test]
        fn it_feeds_itself_from_empty_like_any_other_world() {
            let seed = Genome::all(Instruction::Split);
            let mut rng = StdRng::seed_from_u64(0);
            let mut world = World::with_seed_genome(TEST_WIDTH, TEST_HEIGHT, seed, &mut rng);

            feed_a_while(&mut world, &mut rng);

            assert!(!world.pellets().is_empty());
        }

        #[test]
        fn every_critter_spawns_fully_inside_the_world_bounds() {
            // Same invariant the default-spawn path enforces: positions are
            // drawn from [CRITTER_RADIUS, width - CRITTER_RADIUS) so a
            // critter's circle never crosses the edge.
            let seed = Genome::all(Instruction::Split);
            let mut rng = StdRng::seed_from_u64(0);

            let world = World::with_seed_genome(TEST_WIDTH, TEST_HEIGHT, seed, &mut rng);

            for critter in world.critters() {
                assert!(critter.x() >= CRITTER_RADIUS);
                assert!(critter.x() < TEST_WIDTH as i32 - CRITTER_RADIUS);
                assert!(critter.y() >= CRITTER_RADIUS);
                assert!(critter.y() < TEST_HEIGHT as i32 - CRITTER_RADIUS);
            }
        }

        #[test]
        fn critters_spread_across_the_full_world_width_and_height() {
            // Mirrors the same invariant the default-spawn path is tested
            // against: across many spawns, at least one critter lands in the
            // right half and at least one in the bottom half. Pins the spawn
            // range to the full canvas rather than a clipped corner.
            let seed = Genome::all(Instruction::Split);
            let mut any_right = false;
            let mut any_bottom = false;
            let half_width = (TEST_WIDTH as i32) / 2;
            let half_height = (TEST_HEIGHT as i32) / 2;
            for rng_seed in 0..50 {
                let mut rng = StdRng::seed_from_u64(rng_seed);
                let world =
                    World::with_seed_genome(TEST_WIDTH, TEST_HEIGHT, seed.clone(), &mut rng);
                for critter in world.critters() {
                    if critter.x() >= half_width {
                        any_right = true;
                    }
                    if critter.y() >= half_height {
                        any_bottom = true;
                    }
                }
            }
            assert!(any_right);
            assert!(any_bottom);
        }
    }
}
