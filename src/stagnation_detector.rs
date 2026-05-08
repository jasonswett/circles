pub struct StagnationDetector {
    threshold: u32,
    last_value: Option<u32>,
    count: u32,
}

impl StagnationDetector {
    pub fn new(threshold: u32) -> Self {
        Self {
            threshold,
            last_value: None,
            count: 0,
        }
    }

    pub fn observe(&mut self, value: u32) {
        if self.last_value == Some(value) {
            self.count += 1;
        } else {
            self.last_value = Some(value);
            self.count = 1;
        }
    }

    pub fn is_stagnant(&self) -> bool {
        self.count >= self.threshold
    }

    pub fn reset(&mut self) {
        self.last_value = None;
        self.count = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const THRESHOLD: u32 = 5;

    #[test]
    fn a_new_detector_is_not_stagnant() {
        let detector = StagnationDetector::new(THRESHOLD);

        assert!(!detector.is_stagnant());
    }

    #[test]
    fn observing_the_same_value_fewer_than_the_threshold_times_is_not_stagnant() {
        let mut detector = StagnationDetector::new(THRESHOLD);

        for _ in 0..(THRESHOLD - 1) {
            detector.observe(42);
        }

        assert!(!detector.is_stagnant());
    }

    #[test]
    fn observing_the_same_value_threshold_times_is_stagnant() {
        let mut detector = StagnationDetector::new(THRESHOLD);

        for _ in 0..THRESHOLD {
            detector.observe(42);
        }

        assert!(detector.is_stagnant());
    }

    #[test]
    fn observing_a_different_value_resets_the_count() {
        let mut detector = StagnationDetector::new(THRESHOLD);
        for _ in 0..(THRESHOLD - 1) {
            detector.observe(42);
        }

        detector.observe(43);

        assert!(!detector.is_stagnant());
    }

    #[test]
    fn after_a_value_change_stagnation_requires_threshold_more_observations() {
        let mut detector = StagnationDetector::new(THRESHOLD);
        for _ in 0..THRESHOLD {
            detector.observe(42);
        }
        detector.observe(43);

        for _ in 0..(THRESHOLD - 1) {
            detector.observe(43);
        }

        assert!(detector.is_stagnant());
    }

    #[test]
    fn reset_clears_the_stagnation_state() {
        let mut detector = StagnationDetector::new(THRESHOLD);
        for _ in 0..THRESHOLD {
            detector.observe(42);
        }

        detector.reset();

        assert!(!detector.is_stagnant());
    }
}
