use std::time::Duration;

/// Tracks the longest a world has lasted, counting the current world's run as
/// it happens. A world holds the record while still running — so the first
/// world holds it from the outset, and its figure climbs until some later
/// world outlasts it.
pub struct WorldRecord {
    best_world: u32,
    best_elapsed: Duration,
}

impl WorldRecord {
    pub fn new() -> Self {
        Self {
            best_world: 0,
            best_elapsed: Duration::ZERO,
        }
    }

    /// Reports how long the current world has been running. The record moves
    /// to this world once it passes the standing best, and keeps moving with
    /// it for as long as it holds.
    pub fn observe(&mut self, world_number: u32, elapsed: Duration) {
        if world_number == self.best_world || elapsed > self.best_elapsed {
            self.best_world = world_number;
            self.best_elapsed = elapsed;
        }
    }

    pub fn best_world(&self) -> u32 {
        self.best_world
    }

    pub fn best_elapsed(&self) -> Duration {
        self.best_elapsed
    }
}

impl Default for WorldRecord {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn secs(seconds: u64) -> Duration {
        Duration::from_secs(seconds)
    }

    #[test]
    fn a_fresh_record_holds_nothing() {
        let record = WorldRecord::new();

        assert_eq!(record.best_world(), 0);
        assert_eq!(record.best_elapsed(), Duration::ZERO);
    }

    #[test]
    fn the_only_world_so_far_holds_the_record() {
        let mut record = WorldRecord::new();

        record.observe(1, secs(5));

        assert_eq!(record.best_world(), 1);
        assert_eq!(record.best_elapsed(), secs(5));
    }

    #[test]
    fn the_holders_figure_climbs_as_it_keeps_running() {
        // The record counts live rather than freezing at whatever the world
        // had reached when it took the lead.
        let mut record = WorldRecord::new();
        record.observe(1, secs(5));

        record.observe(1, secs(9));

        assert_eq!(record.best_world(), 1);
        assert_eq!(record.best_elapsed(), secs(9));
    }

    #[test]
    fn a_shorter_world_does_not_take_the_record() {
        let mut record = WorldRecord::new();
        record.observe(1, secs(30));

        record.observe(2, secs(4));

        assert_eq!(record.best_world(), 1);
        assert_eq!(record.best_elapsed(), secs(30));
    }

    #[test]
    fn merely_matching_the_record_does_not_take_it() {
        // A tie leaves the record where it is: the incumbent got there first.
        let mut record = WorldRecord::new();
        record.observe(1, secs(30));

        record.observe(2, secs(30));

        assert_eq!(record.best_world(), 1);
    }

    #[test]
    fn a_world_that_outlasts_the_holder_takes_the_record() {
        let mut record = WorldRecord::new();
        record.observe(1, secs(30));
        record.observe(2, secs(10));

        record.observe(2, secs(31));

        assert_eq!(record.best_world(), 2);
        assert_eq!(record.best_elapsed(), secs(31));
    }

    #[test]
    fn a_new_holder_then_counts_live_itself() {
        let mut record = WorldRecord::new();
        record.observe(1, secs(30));
        record.observe(2, secs(31));

        record.observe(2, secs(45));

        assert_eq!(record.best_world(), 2);
        assert_eq!(record.best_elapsed(), secs(45));
    }

    #[test]
    fn the_record_survives_worlds_that_never_threaten_it() {
        let mut record = WorldRecord::new();
        record.observe(1, secs(60));

        for world in 2..=6 {
            record.observe(world, secs(3));
        }

        assert_eq!(record.best_world(), 1);
        assert_eq!(record.best_elapsed(), secs(60));
    }
}
