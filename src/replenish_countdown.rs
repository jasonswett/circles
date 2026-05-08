use std::time::Duration;

const SECONDS_PER_MINUTE: u64 = 60;

pub fn frames_until_next_replenish(frame_counter: u32, replenish_interval_frames: u32) -> u32 {
    replenish_interval_frames - (frame_counter % replenish_interval_frames)
}

pub fn format_minutes_seconds(remaining: Duration) -> String {
    let total_seconds = remaining.as_secs();
    let minutes = total_seconds / SECONDS_PER_MINUTE;
    let seconds = total_seconds % SECONDS_PER_MINUTE;
    format!("{minutes}:{seconds:02}")
}

#[cfg(test)]
mod tests {
    use super::*;

    mod frames_until_next_replenish {
        use super::*;

        #[test]
        fn at_frame_zero_a_full_interval_remains() {
            assert_eq!(frames_until_next_replenish(0, 1800), 1800);
        }

        #[test]
        fn after_one_frame_one_fewer_frame_remains() {
            assert_eq!(frames_until_next_replenish(1, 1800), 1799);
        }

        #[test]
        fn at_one_frame_before_the_boundary_one_frame_remains() {
            assert_eq!(frames_until_next_replenish(1799, 1800), 1);
        }

        #[test]
        fn at_the_boundary_a_full_interval_remains_for_the_next_cycle() {
            assert_eq!(frames_until_next_replenish(1800, 1800), 1800);
        }

        #[test]
        fn part_way_through_the_second_cycle_the_remainder_is_correct() {
            assert_eq!(frames_until_next_replenish(2000, 1800), 1600);
        }
    }

    mod format_minutes_seconds {
        use super::*;

        #[test]
        fn zero_seconds_renders_as_zero_zero_zero() {
            assert_eq!(format_minutes_seconds(Duration::from_secs(0)), "0:00");
        }

        #[test]
        fn under_a_minute_only_seconds_are_shown() {
            assert_eq!(format_minutes_seconds(Duration::from_secs(27)), "0:27");
        }

        #[test]
        fn seconds_are_zero_padded_to_two_digits() {
            assert_eq!(format_minutes_seconds(Duration::from_secs(5)), "0:05");
        }

        #[test]
        fn at_one_minute_the_minutes_field_increments() {
            assert_eq!(format_minutes_seconds(Duration::from_secs(60)), "1:00");
        }

        #[test]
        fn over_a_minute_both_fields_are_shown() {
            assert_eq!(format_minutes_seconds(Duration::from_secs(125)), "2:05");
        }

        #[test]
        fn fractional_seconds_are_truncated() {
            assert_eq!(format_minutes_seconds(Duration::from_millis(1500)), "0:01");
        }
    }
}
