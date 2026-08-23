use crate::{Genome, GenomeParseError};
use std::path::PathBuf;

/// What the command line is asking the app to do at startup. A struct rather
/// than a choice between modes: the flags are independent of one another, and
/// a resumed headless run is all of them at once -- carry on from this
/// genome, for this long, and leave what you find here.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Startup {
    /// A genome to seed every critter with, if one was supplied.
    pub seed: Option<Genome>,
    /// How many ticks to run before stopping. None means until stopped by
    /// something else -- a closed window, or a signal.
    pub ticks: Option<u64>,
    /// Where to write the best genome the run finds.
    pub out: Option<PathBuf>,
}

#[derive(Debug, PartialEq, Eq)]
pub enum CliError {
    MissingGenomeValue,
    MissingTicksValue,
    MissingOutValue,
    InvalidTicks(String),
    UnknownArgument(String),
    InvalidGenome(GenomeParseError),
}

impl std::fmt::Display for CliError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CliError::MissingGenomeValue => {
                write!(f, "--genome requires a bit-string value")
            }
            CliError::MissingTicksValue => {
                write!(f, "--ticks requires a number of ticks")
            }
            CliError::MissingOutValue => {
                write!(f, "--out requires a path to write to")
            }
            CliError::InvalidTicks(value) => {
                write!(f, "--ticks value '{value}' is not a number")
            }
            CliError::UnknownArgument(arg) => {
                write!(f, "unknown argument '{arg}'")
            }
            CliError::InvalidGenome(error) => {
                write!(f, "invalid --genome value: {error}")
            }
        }
    }
}

impl std::error::Error for CliError {}

/// Parses the command-line arguments (minus the program name in argv\[0\]).
/// Recognized: `--genome <bit-string>` to seed the world, `--ticks <n>` to
/// stop after a while, and `--out <path>` to say where the best genome goes.
pub fn parse<I, S>(args: I) -> Result<Startup, CliError>
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    let mut iter = args.into_iter().map(Into::into);
    let mut startup = Startup::default();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--genome" => {
                let value = iter.next().ok_or(CliError::MissingGenomeValue)?;
                startup.seed = Some(Genome::from_bits(&value).map_err(CliError::InvalidGenome)?);
            }
            "--ticks" => {
                let value = iter.next().ok_or(CliError::MissingTicksValue)?;
                startup.ticks = Some(value.parse().map_err(|_| CliError::InvalidTicks(value))?);
            }
            "--out" => {
                let value = iter.next().ok_or(CliError::MissingOutValue)?;
                startup.out = Some(PathBuf::from(value));
            }
            other => return Err(CliError::UnknownArgument(other.to_string())),
        }
    }
    Ok(startup)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Instruction;

    #[test]
    fn no_arguments_yields_default_startup() {
        let result = parse(Vec::<String>::new());

        assert_eq!(result, Ok(Startup::default()));
    }

    #[test]
    fn a_valid_genome_argument_yields_a_seed_genome_startup() {
        let seed = Genome::all(Instruction::Split);

        let result = parse(vec!["--genome".to_string(), seed.to_bits()]);

        assert_eq!(result.map(|s| s.seed), Ok(Some(seed)));
    }

    #[test]
    fn a_missing_genome_value_is_an_error() {
        let result = parse(vec!["--genome".to_string()]);

        assert_eq!(result, Err(CliError::MissingGenomeValue));
    }

    #[test]
    fn an_invalid_genome_value_is_an_error() {
        let result = parse(vec!["--genome".to_string(), "not bits".to_string()]);

        assert!(matches!(result, Err(CliError::InvalidGenome(_))));
    }

    #[test]
    fn a_tick_budget_is_read_from_the_command_line() {
        // What a headless run needs that a windowed one does not: some way to
        // say when to stop, there being no window to close.
        let result = parse(vec!["--ticks".to_string(), "1000".to_string()]);

        assert_eq!(result.map(|s| s.ticks), Ok(Some(1_000)));
    }

    #[test]
    fn no_tick_budget_means_running_until_stopped() {
        let result = parse(Vec::<String>::new());

        assert_eq!(result.map(|s| s.ticks), Ok(None));
    }

    #[test]
    fn a_missing_tick_value_is_an_error() {
        let result = parse(vec!["--ticks".to_string()]);

        assert_eq!(result, Err(CliError::MissingTicksValue));
    }

    #[test]
    fn a_tick_value_that_is_not_a_number_is_an_error() {
        let result = parse(vec!["--ticks".to_string(), "soon".to_string()]);

        assert_eq!(result, Err(CliError::InvalidTicks("soon".to_string())));
    }

    #[test]
    fn an_output_path_is_read_from_the_command_line() {
        let result = parse(vec!["--out".to_string(), "genomes/best".to_string()]);

        assert_eq!(
            result.map(|s| s.out),
            Ok(Some(PathBuf::from("genomes/best")))
        );
    }

    #[test]
    fn a_missing_output_path_is_an_error() {
        let result = parse(vec!["--out".to_string()]);

        assert_eq!(result, Err(CliError::MissingOutValue));
    }

    #[test]
    fn the_flags_combine() {
        // A resumed run is all three at once: carry on from this genome, for
        // this long, and leave what you find here.
        let seed = Genome::all(Instruction::Split);

        let result = parse(vec![
            "--genome".to_string(),
            seed.to_bits(),
            "--ticks".to_string(),
            "50".to_string(),
            "--out".to_string(),
            "best.txt".to_string(),
        ])
        .expect("should parse");

        assert_eq!(result.seed, Some(seed));
        assert_eq!(result.ticks, Some(50));
        assert_eq!(result.out, Some(PathBuf::from("best.txt")));
    }

    #[test]
    fn an_unknown_argument_is_an_error() {
        let result = parse(vec!["--mystery".to_string()]);

        assert_eq!(
            result,
            Err(CliError::UnknownArgument("--mystery".to_string()))
        );
    }

    #[test]
    fn missing_genome_value_error_displays_a_helpful_message() {
        let rendered = format!("{}", CliError::MissingGenomeValue);

        assert!(
            rendered.contains("--genome"),
            "unexpected rendering: {rendered}",
        );
    }

    #[test]
    fn unknown_argument_error_displays_the_offending_argument() {
        let rendered = format!("{}", CliError::UnknownArgument("--bogus".to_string()));

        assert!(
            rendered.contains("--bogus"),
            "unexpected rendering: {rendered}",
        );
    }

    #[test]
    fn invalid_genome_error_displays_the_underlying_parse_error() {
        let rendered = format!(
            "{}",
            CliError::InvalidGenome(GenomeParseError::WrongLength {
                expected: 319,
                actual: 5,
            })
        );

        assert!(
            rendered.contains("319") && rendered.contains('5'),
            "unexpected rendering: {rendered}",
        );
    }
}
