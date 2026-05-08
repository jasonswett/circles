use std::time::{Duration, Instant};

pub struct FpsCounter {
    refresh_interval: Duration,
    window_start: Option<Instant>,
    frames_in_window: u32,
    reported_fps: u32,
}

impl FpsCounter {
    pub fn new(refresh_interval: Duration) -> Self {
        Self {
            refresh_interval,
            window_start: None,
            frames_in_window: 0,
            reported_fps: 0,
        }
    }

    pub fn observe_frame(&mut self, now: Instant) {
        match self.window_start {
            None => {
                self.window_start = Some(now);
                self.frames_in_window = 0;
            }
            Some(start) => {
                self.frames_in_window += 1;
                let elapsed = now - start;
                if elapsed >= self.refresh_interval {
                    let secs = elapsed.as_secs_f64();
                    self.reported_fps = (self.frames_in_window as f64 / secs).round() as u32;
                    // Reset to a fresh epoch: the next observation will become
                    // the new window's start without being counted.
                    self.window_start = None;
                    self.frames_in_window = 0;
                }
            }
        }
    }

    pub fn current_fps(&self) -> u32 {
        self.reported_fps
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    const REFRESH: Duration = Duration::from_millis(500);

    #[test]
    fn a_new_counter_reports_zero_fps() {
        let counter = FpsCounter::new(REFRESH);

        assert_eq!(counter.current_fps(), 0);
    }

    #[test]
    fn the_reported_fps_stays_at_zero_until_a_full_refresh_interval_elapses() {
        let t0 = Instant::now();
        let mut counter = FpsCounter::new(REFRESH);

        counter.observe_frame(t0);
        counter.observe_frame(t0 + Duration::from_millis(200));
        counter.observe_frame(t0 + Duration::from_millis(400));

        assert_eq!(counter.current_fps(), 0);
    }

    #[test]
    fn after_a_full_refresh_interval_the_reported_fps_is_frames_per_second() {
        // Observe 30 frames over 500ms = 60 fps.
        let t0 = Instant::now();
        let mut counter = FpsCounter::new(REFRESH);

        for i in 0..30 {
            counter.observe_frame(t0 + Duration::from_millis(i * 500 / 30));
        }
        counter.observe_frame(t0 + Duration::from_millis(500));

        assert_eq!(counter.current_fps(), 60);
    }

    #[test]
    fn the_reported_fps_holds_steady_between_refreshes() {
        let t0 = Instant::now();
        let mut counter = FpsCounter::new(REFRESH);
        for i in 0..30 {
            counter.observe_frame(t0 + Duration::from_millis(i * 500 / 30));
        }
        counter.observe_frame(t0 + Duration::from_millis(500));
        let after_first_refresh = counter.current_fps();

        // Observe a few more frames inside the next refresh window.
        counter.observe_frame(t0 + Duration::from_millis(550));
        counter.observe_frame(t0 + Duration::from_millis(600));

        assert_eq!(counter.current_fps(), after_first_refresh);
    }

    #[test]
    fn the_reported_fps_updates_after_a_second_refresh_interval() {
        // First window: 30 frames in 500ms → 60 fps.
        // Second window: 15 frames in 500ms → 30 fps.
        let t0 = Instant::now();
        let mut counter = FpsCounter::new(REFRESH);
        for i in 0..30 {
            counter.observe_frame(t0 + Duration::from_millis(i * 500 / 30));
        }
        counter.observe_frame(t0 + Duration::from_millis(500));

        for i in 0..15 {
            counter.observe_frame(t0 + Duration::from_millis(500 + i * 500 / 15));
        }
        counter.observe_frame(t0 + Duration::from_millis(1000));

        assert_eq!(counter.current_fps(), 30);
    }
}
