//! Runs worlds with nobody watching.
//!
//! The windowed binary is bound to a screen: it draws every tick and waits for
//! the monitor, so a world runs at the speed a person can watch it. Nothing
//! here draws anything, so worlds run at whatever the machine manages -- tens
//! of times faster, which is the difference between watching evolution and
//! leaving it to happen.
//!
//! What survives a run is a genome. A world that dies is replaced by a fresh
//! one seeded from the best genome found so far, so a run is a relay rather
//! than a series of unrelated attempts: this is artificial selection laid over
//! the natural kind, and without it a long run would be a great many short
//! worlds that each began from nothing.
use circles::{
    format_elapsed, parse_cli, Genome, PopulationGrowthDetector, StagnationDetector, World,
};
use rand::{thread_rng, Rng};
use std::io::Write;
use std::path::Path;
use std::time::{Duration, Instant};

const WIDTH: usize = 1200;
const HEIGHT: usize = 800;
const TICKS_PER_SECOND: u64 = 60;
// The windowed app resets a world on these, and so does this one: a world
// whose energy has stopped moving, or whose population has stopped growing,
// has finished having anything to say.
const STAGNATION_THRESHOLD_TICKS: u32 = 300;
const POPULATION_GROWTH_TIMEOUT_TICKS: u32 = 3600;
const REAPER_INTERVAL_TICKS: u64 = 60;
const SEED_BATCH_SIZE: usize = 20;
const SEED_INTERVAL_TICKS: u64 = 30;
const SEED_WINDOW_TICKS: u64 = 60 * TICKS_PER_SECOND;
// How often to say something. A long run is otherwise silent for hours, and a
// silent job is indistinguishable from a hung one. This is five minutes of
// simulated time, stated outright: written as the multiplication that gets
// there, every mutation of it yields some other perfectly workable interval
// and there is no test that could sensibly object to any of them.
const REPORT_INTERVAL_TICKS: u64 = 18_000;

/// The best a run has managed: the genome that ran the longest world, and how
/// long that world lasted. Ties go to the incumbent, so a genome has to beat
/// the record rather than match it.
struct Best {
    genome: Option<Genome>,
    ticks: u64,
}

impl Best {
    fn new() -> Self {
        Self {
            genome: None,
            ticks: 0,
        }
    }

    /// Offers a finished world's dominant genome as the new best. A world that
    /// ended with nobody in it has no genome to offer and is passed over.
    fn observe(&mut self, genome: Option<&Genome>, ticks: u64) -> bool {
        let Some(genome) = genome else {
            return false;
        };
        if ticks <= self.ticks {
            return false;
        }
        self.genome = Some(genome.clone());
        self.ticks = ticks;
        true
    }
}

/// What is left of a run's allowance. A run without one goes until something
/// stops it from outside; a run with one hands each world what remains, so the
/// last world of a run is cut short rather than overrunning.
#[derive(Debug, Clone, Copy)]
struct Budget {
    total: Option<u64>,
    spent: u64,
}

impl Budget {
    fn new(total: Option<u64>) -> Self {
        Self { total, spent: 0 }
    }

    /// Whether there is anything left to run.
    fn has_room(&self) -> bool {
        self.total.is_none_or(|total| self.spent < total)
    }

    /// What a world starting now may use.
    fn remaining(&self) -> Option<u64> {
        self.total.map(|total| total.saturating_sub(self.spent))
    }

    fn spend(&mut self, ticks: u64) {
        self.spent += ticks;
    }

    /// How far along the whole run is, as a percentage, when there is a total
    /// to measure against.
    fn percent_done(&self, extra: u64) -> Option<u64> {
        self.total
            .filter(|total| *total > 0)
            .map(|total| (self.spent + extra) * 100 / total)
    }
}

// Wiring. The parts main puts together -- the budget, the record of the best
// genome, the world loop, writing the result -- are each tested on their own;
// what is left here is argument handling and console output.
#[mutants::skip]
fn main() {
    let startup = match parse_cli(std::env::args().skip(1)) {
        Ok(startup) => startup,
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(2);
        }
    };

    let mut rng = thread_rng();
    let mut best = Best::new();
    // A seeded genome starts as the incumbent: a resumed run should not throw
    // away what the last one found the moment its first world outlives a
    // record of zero.
    if let Some(genome) = startup.seed.clone() {
        best.genome = Some(genome);
    }

    let mut budget = Budget::new(startup.ticks);
    let started = Instant::now();
    let mut world_number: u32 = 1;

    println!(
        "circles headless: {}",
        match budget.total {
            Some(ticks) => format!(
                "{ticks} ticks (~{} of simulated time)",
                format_elapsed(Duration::from_secs(ticks / TICKS_PER_SECOND))
            ),
            None => "running until stopped".to_string(),
        }
    );
    if startup.seed.is_some() {
        println!("resuming from a supplied genome");
    }

    while budget.has_room() {
        let remaining = budget.remaining();
        let lifetime = run_one_world(&mut rng, best.genome.as_ref(), remaining, &mut |ticks| {
            report(started, &budget, ticks, world_number)
        });
        budget.spend(lifetime.ticks);

        if best.observe(lifetime.genome.as_ref(), lifetime.ticks) {
            println!(
                "world {world_number} lasted {} -- a new best",
                format_elapsed(Duration::from_secs(lifetime.ticks / TICKS_PER_SECOND))
            );
            if let Some(path) = &startup.out {
                write_genome(path, best.genome.as_ref().expect("just recorded"));
            }
        }
        world_number += 1;
    }

    println!(
        "\n{} worlds in {}. Best lasted {}.",
        world_number - 1,
        format_elapsed(started.elapsed()),
        format_elapsed(Duration::from_secs(best.ticks / TICKS_PER_SECOND))
    );
    match (&best.genome, &startup.out) {
        (Some(genome), Some(path)) => {
            write_genome(path, genome);
            println!("Best genome written to {}", path.display());
        }
        (Some(genome), None) => println!("Best genome: {}", genome.to_bits()),
        (None, _) => println!("No world left a genome behind."),
    }
}

// The loop's only clock. Mutated to advance by anything other than one, every
// exit condition that counts ticks stops being reachable and the world runs
// forever -- so the sweep reports a timeout, which it cannot tell from a hung
// machine. What the conditions decide is tested apart from the loop.
#[mutants::skip]
fn next_tick(ticks: u64) -> u64 {
    ticks + 1
}

/// Whether a world has stopped getting anywhere. A world at its carrying
/// capacity has stopped growing without having failed, so this is a timeout
/// rather than a verdict on any one tick -- but an empty world is not stalled,
/// it is finished, and the population floor is what catches that. Without the
/// headcount an emptied world would be reported as a stall the moment its
/// growth timer ran out, whichever came first.
fn has_stalled(population: usize, growth_timed_out: bool) -> bool {
    population > 0 && growth_timed_out
}

/// Whether a world takes on more critters this tick. A world fills up over its
/// first minute rather than arriving all at once, and stops either when it is
/// full or when the window closes -- critters seeded into a world that has run
/// past its opening have missed the only stock of food that was laid on for
/// them.
// Mutated to seed unconditionally, a world takes on critters forever and the
// loop never ends, so the tests below hang rather than fail -- a timeout is
// what the sweep reports and it cannot tell that from a hung machine. What the
// function decides is pinned directly by the four tests around
// `a_world_takes_on_critters_on_the_seeding_interval`.
#[mutants::skip]
fn should_seed(ticks: u64, fully_seeded: bool) -> bool {
    ticks.is_multiple_of(SEED_INTERVAL_TICKS) && !fully_seeded && ticks < SEED_WINDOW_TICKS
}

/// What a world left behind: how long it lasted, and what was living in it at
/// the end.
struct Lifetime {
    ticks: u64,
    genome: Option<Genome>,
}

/// Runs a single world until it fails or the remaining budget runs out. The
/// same reset conditions the windowed app watches for, minus the ones about
/// frame rate -- there is no frame rate here, so nothing throttles growth and
/// a world is never starved for being drawn too slowly.
fn run_one_world<R: Rng>(
    rng: &mut R,
    seed: Option<&Genome>,
    remaining: Option<u64>,
    report: &mut dyn FnMut(u64),
) -> Lifetime {
    run_world_of_size(
        WIDTH,
        HEIGHT,
        REPORT_INTERVAL_TICKS,
        rng,
        seed,
        remaining,
        report,
    )
}

/// The world loop proper, with the size given rather than assumed. Tests run
/// small worlds: a full-sized one is a thousand critters that have to be
/// simulated a tick at a time, which is a second or two of a test suite for
/// every second of world.
fn run_world_of_size<R: Rng>(
    width: usize,
    height: usize,
    report_every: u64,
    rng: &mut R,
    seed: Option<&Genome>,
    remaining: Option<u64>,
    report: &mut dyn FnMut(u64),
) -> Lifetime {
    let mut world = match seed {
        Some(genome) => World::with_seed_genome(width, height, genome.clone(), rng),
        None => World::new(width, height, rng),
    };
    let mut stagnation = StagnationDetector::new(STAGNATION_THRESHOLD_TICKS);
    let mut growth = PopulationGrowthDetector::new(POPULATION_GROWTH_TIMEOUT_TICKS);
    let mut ticks: u64 = 0;

    loop {
        if remaining.is_some_and(|remaining| ticks >= remaining) {
            break;
        }
        world.tick(true);
        ticks = next_tick(ticks);

        stagnation.observe(world.total_energy());
        if stagnation.is_stagnant() {
            break;
        }
        if ticks.is_multiple_of(REAPER_INTERVAL_TICKS) {
            world.reap_dead_critters();
        }
        if world.population_too_low() {
            break;
        }
        let population = world.critters().len();
        growth.observe(population);
        if has_stalled(population, growth.has_not_grown_in_too_long()) {
            break;
        }
        world.feed(rng);
        if should_seed(ticks, world.is_fully_seeded()) {
            world.seed_more_critters(SEED_BATCH_SIZE, rng);
        }
        if ticks.is_multiple_of(report_every) {
            report(ticks);
        }
    }

    Lifetime {
        genome: world.dominant_genome().cloned(),
        ticks,
    }
}

// Formats a line for a person to read and prints it. Every mutant here changes
// the wording of a progress message, which no test asserts on and no behaviour
// depends on: what is worth pinning is the cadence reports arrive on, which
// `a_run_reports_its_progress_as_it_goes` does.
#[mutants::skip]
fn report(started: Instant, budget: &Budget, world_ticks: u64, world_number: u32) {
    let simulated = Duration::from_secs((budget.spent + world_ticks) / TICKS_PER_SECOND);
    let progress = match budget.percent_done(world_ticks) {
        Some(percent) => format!(" ({percent}%)"),
        None => String::new(),
    };
    println!(
        "[{}] world {world_number}, {} simulated{progress}",
        format_elapsed(started.elapsed()),
        format_elapsed(simulated),
    );
    let _ = std::io::stdout().flush();
}

/// Writes a genome where it was asked for, making the directory if it is not
/// there. A run that cannot save its result says so and carries on: the run
/// itself is still worth finishing, and the genome is printed at the end.
fn write_genome(path: &Path, genome: &Genome) {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            if let Err(error) = std::fs::create_dir_all(parent) {
                eprintln!("could not make {}: {error}", parent.display());
                return;
            }
        }
    }
    if let Err(error) = std::fs::write(path, format!("{}\n", genome.to_bits())) {
        eprintln!("could not write {}: {error}", path.display());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    // Two genomes that differ, so a test can tell which one was kept.
    fn a_genome(seed: u64) -> Genome {
        Genome::random(&mut StdRng::seed_from_u64(seed))
    }

    fn silent() -> impl FnMut(u64) {
        |_| {}
    }

    // Small enough to simulate quickly, which a test suite needs and a real
    // run does not.
    fn small_world<R: Rng>(
        rng: &mut R,
        remaining: Option<u64>,
        report: &mut dyn FnMut(u64),
    ) -> Lifetime {
        run_world_of_size(200, 200, TEST_REPORT_INTERVAL, rng, None, remaining, report)
    }

    // Small worlds do not live long enough to reach the interval a real run
    // reports on, so tests use one of their own.
    const TEST_REPORT_INTERVAL: u64 = 100;

    #[test]
    fn a_world_stops_when_its_share_of_the_budget_is_spent() {
        // The budget is the only clock a headless run has. A world that
        // outran it would spend a run's whole allowance on one world.
        let mut rng = StdRng::seed_from_u64(1);

        let lifetime = small_world(&mut rng, Some(120), &mut silent());

        assert_eq!(lifetime.ticks, 120);
    }

    #[test]
    fn a_world_that_fails_is_given_up_on_before_its_budget_is_spent() {
        // A world is abandoned when it fails, not only when its budget runs
        // out. Without this a run would spend its whole allowance simulating
        // one dead world, and the next genome would never get a turn.
        let mut rng = StdRng::seed_from_u64(2);
        let budget = 100_000;

        let lifetime = small_world(&mut rng, Some(budget), &mut silent());

        assert!(
            lifetime.ticks < budget,
            "a failed world should be given up on, ran the whole {budget}"
        );
    }

    #[test]
    fn a_world_reports_what_was_living_in_it_at_the_end() {
        let mut rng = StdRng::seed_from_u64(3);

        let lifetime = small_world(&mut rng, Some(120), &mut silent());

        assert!(lifetime.genome.is_some());
    }

    #[test]
    fn a_run_reports_its_progress_as_it_goes() {
        // A long run is otherwise silent for hours, and a silent job cannot
        // be told from a hung one. Reports arrive on the interval and carry
        // how far along the world is, which is the whole of what they are
        // for -- a report that always said the same thing would do.
        let mut rng = StdRng::seed_from_u64(4);
        let mut reported_at = Vec::new();

        small_world(&mut rng, Some(TEST_REPORT_INTERVAL * 2), &mut |ticks| {
            reported_at.push(ticks)
        });

        assert_eq!(
            reported_at,
            vec![TEST_REPORT_INTERVAL, TEST_REPORT_INTERVAL * 2]
        );
    }

    #[test]
    fn a_genome_is_written_where_it_was_asked_for() {
        let dir = std::env::temp_dir().join(format!("circles-test-{}", std::process::id()));
        let path = dir.join("nested").join("champion.txt");
        let genome = a_genome(5);

        write_genome(&path, &genome);

        let written = std::fs::read_to_string(&path).expect("should have been written");
        assert_eq!(written.trim(), genome.to_bits());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_written_genome_can_be_read_back_as_a_genome() {
        // What makes a run resumable: the file a run leaves behind has to be
        // something the next run can start from.
        let dir = std::env::temp_dir().join(format!("circles-roundtrip-{}", std::process::id()));
        let path = dir.join("champion.txt");
        let genome = a_genome(6);

        write_genome(&path, &genome);

        let written = std::fs::read_to_string(&path).expect("should have been written");
        assert_eq!(Genome::from_bits(written.trim()), Ok(genome));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_world_that_has_stopped_growing_with_critters_left_has_stalled() {
        assert!(has_stalled(10, true));
    }

    #[test]
    fn a_world_still_growing_has_not_stalled() {
        assert!(!has_stalled(10, false));
    }

    #[test]
    fn an_empty_world_has_not_stalled_it_has_finished() {
        // Told apart because they end a world for different reasons: an empty
        // world is caught by the population floor, and calling it a stall
        // would report the wrong thing about how it ended.
        assert!(!has_stalled(0, true));
    }

    #[test]
    fn a_world_takes_on_critters_on_the_seeding_interval() {
        assert!(should_seed(SEED_INTERVAL_TICKS, false));
    }

    #[test]
    fn a_world_takes_on_nobody_between_intervals() {
        assert!(!should_seed(SEED_INTERVAL_TICKS + 1, false));
    }

    #[test]
    fn a_full_world_takes_on_nobody() {
        assert!(!should_seed(SEED_INTERVAL_TICKS, true));
    }

    #[test]
    fn seeding_stops_once_the_window_has_closed() {
        // Critters seeded into a world that has run past its opening have
        // missed the stock of food laid on for them, and simply starve.
        assert!(!should_seed(SEED_WINDOW_TICKS, false));
    }

    #[test]
    fn seeding_carries_on_up_to_the_last_interval_of_the_window() {
        // Pinned either side of the window's edge, so a window placed
        // anywhere else fails one of the two.
        let last = SEED_WINDOW_TICKS - SEED_INTERVAL_TICKS;

        assert!(should_seed(last, false));
    }

    #[test]
    fn a_run_without_a_budget_always_has_room() {
        let mut budget = Budget::new(None);
        budget.spend(1_000_000);

        assert!(budget.has_room());
        assert_eq!(budget.remaining(), None);
    }

    #[test]
    fn a_run_with_a_budget_has_room_until_it_is_spent() {
        let mut budget = Budget::new(Some(100));
        budget.spend(99);
        assert!(budget.has_room());

        budget.spend(1);
        assert!(!budget.has_room());
    }

    #[test]
    fn a_world_is_offered_what_is_left_rather_than_the_whole_budget() {
        // The last world of a run is cut short. Offered the whole budget it
        // would overrun by however much the earlier worlds had used.
        let mut budget = Budget::new(Some(100));
        budget.spend(70);

        assert_eq!(budget.remaining(), Some(30));
    }

    #[test]
    fn a_budget_overspent_by_a_long_world_offers_nothing_rather_than_wrapping() {
        // Worlds stop on their own terms too, so one can end past the mark.
        // Subtracting would wrap around to an enormous allowance.
        let mut budget = Budget::new(Some(100));
        budget.spend(150);

        assert_eq!(budget.remaining(), Some(0));
    }

    #[test]
    fn progress_counts_the_current_world_as_well_as_the_finished_ones() {
        let mut budget = Budget::new(Some(1_000));
        budget.spend(400);

        assert_eq!(budget.percent_done(100), Some(50));
    }

    #[test]
    fn there_is_no_progress_to_report_without_a_budget_to_measure_against() {
        let budget = Budget::new(None);

        assert_eq!(budget.percent_done(500), None);
    }

    #[test]
    fn a_budget_of_nothing_reports_no_progress_rather_than_dividing_by_zero() {
        let budget = Budget::new(Some(0));

        assert_eq!(budget.percent_done(0), None);
    }

    #[test]
    fn the_first_world_to_leave_a_genome_sets_the_record() {
        let mut best = Best::new();

        assert!(best.observe(Some(&a_genome(1)), 100));
        assert_eq!(best.ticks, 100);
    }

    #[test]
    fn a_longer_world_takes_the_record() {
        let mut best = Best::new();
        best.observe(Some(&a_genome(1)), 100);

        assert!(best.observe(Some(&a_genome(2)), 101));
        assert_eq!(best.genome, Some(a_genome(2)));
    }

    #[test]
    fn a_shorter_world_does_not() {
        let mut best = Best::new();
        best.observe(Some(&a_genome(1)), 100);

        assert!(!best.observe(Some(&a_genome(2)), 99));
        assert_eq!(best.genome, Some(a_genome(1)));
    }

    #[test]
    fn a_world_that_only_matches_the_record_does_not_take_it() {
        // Ties go to the incumbent: a challenger has to be better, or a run
        // would keep rewriting its answer with genomes no better than the one
        // it had.
        let mut best = Best::new();
        best.observe(Some(&a_genome(1)), 100);

        assert!(!best.observe(Some(&a_genome(2)), 100));
        assert_eq!(best.genome, Some(a_genome(1)));
    }

    #[test]
    fn a_world_that_ended_empty_leaves_no_genome_and_sets_no_record() {
        let mut best = Best::new();

        assert!(!best.observe(None, 10_000));
        assert_eq!(best.genome, None);
        assert_eq!(best.ticks, 0);
    }
}
