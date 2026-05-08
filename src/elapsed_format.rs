use std::time::Duration;

const SECONDS_PER_MINUTE: u64 = 60;
const MINUTES_PER_HOUR: u64 = 60;
const SECONDS_PER_HOUR: u64 = SECONDS_PER_MINUTE * MINUTES_PER_HOUR;

pub fn format_elapsed(elapsed: Duration) -> String {
    let total_seconds = elapsed.as_secs();
    let hours = total_seconds / SECONDS_PER_HOUR;
    let minutes = (total_seconds % SECONDS_PER_HOUR) / SECONDS_PER_MINUTE;
    let seconds = total_seconds % SECONDS_PER_MINUTE;
    format!("{hours}:{minutes:02}:{seconds:02}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_elapsed_renders_as_zero_zero_zero() {
        assert_eq!(format_elapsed(Duration::from_secs(0)), "0:00:00");
    }

    #[test]
    fn under_a_minute_only_seconds_are_shown() {
        assert_eq!(format_elapsed(Duration::from_secs(23)), "0:00:23");
    }

    #[test]
    fn seconds_are_zero_padded_to_two_digits() {
        assert_eq!(format_elapsed(Duration::from_secs(5)), "0:00:05");
    }

    #[test]
    fn at_one_minute_the_minutes_field_increments() {
        assert_eq!(format_elapsed(Duration::from_secs(60)), "0:01:00");
    }

    #[test]
    fn minutes_are_zero_padded_to_two_digits() {
        assert_eq!(format_elapsed(Duration::from_secs(305)), "0:05:05");
    }

    #[test]
    fn at_one_hour_the_hours_field_increments() {
        assert_eq!(format_elapsed(Duration::from_secs(3600)), "1:00:00");
    }

    #[test]
    fn hours_are_not_padded() {
        assert_eq!(format_elapsed(Duration::from_secs(36000)), "10:00:00");
    }

    #[test]
    fn fractional_seconds_are_truncated() {
        assert_eq!(format_elapsed(Duration::from_millis(1500)), "0:00:01");
    }
}
