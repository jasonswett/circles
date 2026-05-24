use crate::{format_elapsed, Genome};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Formats a snapshot block: a UTC timestamp, the world's generation
/// counter, how long this world has been running, the genome's bit string,
/// and a trailing blank line so consecutive blocks are visually separated.
/// Each fact gets its own line — same shape the on-screen overlay uses.
pub fn format_block(
    timestamp: SystemTime,
    world_number: u32,
    world_elapsed: Duration,
    genome: &Genome,
) -> String {
    format!(
        "{}\nWorld: {world_number}\nTime: {elapsed}\n{bits}\n\n",
        format_timestamp(timestamp),
        elapsed = format_elapsed(world_elapsed),
        bits = genome.to_bits(),
    )
}

/// Renders a `SystemTime` as a fixed-width "YYYY-MM-DD HH:MM:SS UTC" string.
/// We do this manually (no `chrono`) because the simulation has no other
/// need for a date library.
fn format_timestamp(time: SystemTime) -> String {
    let seconds_since_epoch = time
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let (year, month, day, hour, minute, second) = civil_from_unix(seconds_since_epoch);
    format!("{year:04}-{month:02}-{day:02} {hour:02}:{minute:02}:{second:02} UTC")
}

/// Converts unix seconds (UTC) to (year, month, day, hour, minute, second).
/// Uses Howard Hinnant's `days_from_civil` algorithm in reverse — public
/// domain, well-tested. Avoids pulling in a date crate just for this.
#[mutants::skip]
fn civil_from_unix(unix_seconds: u64) -> (i32, u32, u32, u32, u32, u32) {
    let days = (unix_seconds / 86400) as i64;
    let seconds_in_day = (unix_seconds % 86400) as u32;
    let hour = seconds_in_day / 3600;
    let minute = (seconds_in_day % 3600) / 60;
    let second = seconds_in_day % 60;

    // Days since 0000-03-01 (Hinnant's epoch shift): unix epoch is 1970-01-01,
    // which is 719468 days after 0000-03-01.
    let z = days + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = (z - era * 146097) as u64; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365; // [0, 399]
    let y = (yoe as i64 + era * 400) as i32;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let day = (doy - (153 * mp + 2) / 5 + 1) as u32; // [1, 31]
    let month = if mp < 10 {
        (mp + 3) as u32
    } else {
        (mp - 9) as u32
    };
    let year = if month <= 2 { y + 1 } else { y };

    (year, month, day, hour, minute, second)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Instruction;
    use std::time::Duration;

    fn instant_at(unix_seconds: u64) -> SystemTime {
        UNIX_EPOCH + Duration::from_secs(unix_seconds)
    }

    #[test]
    fn the_block_has_timestamp_world_time_and_bits_each_on_their_own_line() {
        let genome = Genome::all(Instruction::Split);
        // 2026-05-24 00:00:00 UTC. Chosen to match the project's "today" so
        // the assertion reads obviously.
        let timestamp = instant_at(1779580800);

        let block = format_block(timestamp, 3, Duration::from_secs(83), &genome);

        let expected = format!(
            "2026-05-24 00:00:00 UTC\nWorld: 3\nTime: 0:01:23\n{}\n\n",
            genome.to_bits(),
        );
        assert_eq!(block, expected);
    }

    #[test]
    fn the_block_ends_with_two_newlines_so_consecutive_blocks_are_separated() {
        let genome = Genome::all(Instruction::Split);

        let block = format_block(instant_at(0), 1, Duration::from_secs(0), &genome);

        assert!(block.ends_with("\n\n"));
    }

    #[test]
    fn the_timestamp_renders_unix_epoch_zero_as_1970_01_01() {
        let genome = Genome::all(Instruction::Split);

        let block = format_block(instant_at(0), 1, Duration::from_secs(0), &genome);

        assert!(
            block.starts_with("1970-01-01 00:00:00 UTC\n"),
            "unexpected block start: {block}",
        );
    }

    #[test]
    fn the_timestamp_renders_a_known_mid_year_moment_correctly() {
        let genome = Genome::all(Instruction::Split);
        // 2024-02-29 12:34:56 UTC — a leap-day moment, catches off-by-one
        // bugs in the civil-from-unix conversion.
        let timestamp = instant_at(1709210096);

        let block = format_block(timestamp, 1, Duration::from_secs(0), &genome);

        assert!(
            block.starts_with("2024-02-29 12:34:56 UTC\n"),
            "unexpected block start: {block}",
        );
    }

    #[test]
    fn the_world_number_appears_on_its_own_line() {
        let genome = Genome::all(Instruction::Split);

        let block = format_block(instant_at(0), 42, Duration::from_secs(0), &genome);

        assert!(block.contains("\nWorld: 42\n"), "unexpected block: {block}",);
    }

    #[test]
    fn the_elapsed_time_appears_on_its_own_line_using_the_shared_formatter() {
        let genome = Genome::all(Instruction::Split);

        let block = format_block(
            instant_at(0),
            1,
            Duration::from_secs(3 * 3600 + 605),
            &genome,
        );

        // 3*3600 + 605 = 10805s = 3:10:05 via format_elapsed.
        assert!(
            block.contains("\nTime: 3:10:05\n"),
            "unexpected block: {block}",
        );
    }
}
