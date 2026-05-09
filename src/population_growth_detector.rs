/// Watches the critter count and detects extended periods without growth.
/// Each call to `observe` records a population snapshot. If a snapshot is
/// strictly greater than the previous one, the timer resets; otherwise the
/// frame counter advances. Once the counter reaches the configured threshold
/// the detector reports "no growth," and the caller can take corrective
/// action (typically resetting the world).
pub struct PopulationGrowthDetector {
    threshold: u32,
    last_population: Option<usize>,
    frames_without_growth: u32,
}

impl PopulationGrowthDetector {
    pub fn new(threshold: u32) -> Self {
        Self {
            threshold,
            last_population: None,
            frames_without_growth: 0,
        }
    }

    pub fn observe(&mut self, population: usize) {
        let grew = self.last_population.is_some_and(|prev| population > prev);
        if grew {
            self.frames_without_growth = 0;
        } else {
            self.frames_without_growth += 1;
        }
        self.last_population = Some(population);
    }

    pub fn has_not_grown_in_too_long(&self) -> bool {
        self.frames_without_growth >= self.threshold
    }

    pub fn reset(&mut self) {
        self.last_population = None;
        self.frames_without_growth = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const THRESHOLD: u32 = 5;

    #[test]
    fn a_new_detector_has_not_yet_observed_a_lack_of_growth() {
        let detector = PopulationGrowthDetector::new(THRESHOLD);

        assert!(!detector.has_not_grown_in_too_long());
    }

    #[test]
    fn observing_a_constant_population_short_of_the_threshold_is_not_too_long() {
        let mut detector = PopulationGrowthDetector::new(THRESHOLD);

        for _ in 0..(THRESHOLD - 1) {
            detector.observe(50);
        }

        assert!(!detector.has_not_grown_in_too_long());
    }

    #[test]
    fn observing_a_constant_population_threshold_times_is_too_long() {
        let mut detector = PopulationGrowthDetector::new(THRESHOLD);

        for _ in 0..THRESHOLD {
            detector.observe(50);
        }

        assert!(detector.has_not_grown_in_too_long());
    }

    #[test]
    fn observing_a_decreasing_population_threshold_times_is_too_long() {
        let mut detector = PopulationGrowthDetector::new(THRESHOLD);

        for descending in (0..THRESHOLD).rev() {
            detector.observe(descending as usize);
        }

        assert!(detector.has_not_grown_in_too_long());
    }

    #[test]
    fn an_increase_in_population_resets_the_count() {
        let mut detector = PopulationGrowthDetector::new(THRESHOLD);
        for _ in 0..(THRESHOLD - 1) {
            detector.observe(50);
        }

        detector.observe(51);

        assert!(!detector.has_not_grown_in_too_long());
    }

    #[test]
    fn after_a_growth_event_the_full_threshold_is_required_again() {
        let mut detector = PopulationGrowthDetector::new(THRESHOLD);
        for _ in 0..THRESHOLD {
            detector.observe(50);
        }
        detector.observe(51);

        for _ in 0..(THRESHOLD - 1) {
            detector.observe(51);
        }

        assert!(!detector.has_not_grown_in_too_long());
    }

    #[test]
    fn reset_clears_the_no_growth_state() {
        let mut detector = PopulationGrowthDetector::new(THRESHOLD);
        for _ in 0..THRESHOLD {
            detector.observe(50);
        }

        detector.reset();

        assert!(!detector.has_not_grown_in_too_long());
    }
}
