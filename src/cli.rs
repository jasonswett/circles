use crate::{Genome, GenomeParseError};

/// What the command line is asking the app to do at startup. Today there's
/// just one option besides "default behavior" — seed every critter with a
/// supplied genome — but this enum is the natural place to grow if we ever
/// add more startup-time flags.
#[derive(Debug, PartialEq, Eq)]
pub enum Startup {
    Default,
    SeedGenome(Genome),
}

#[derive(Debug, PartialEq, Eq)]
pub enum CliError {
    MissingGenomeValue,
    UnknownArgument(String),
    InvalidGenome(GenomeParseError),
}

impl std::fmt::Display for CliError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CliError::MissingGenomeValue => {
                write!(f, "--genome requires a bit-string value")
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
/// Recognized: `--genome <bit-string>` to seed the world. No flag at all
/// means default behavior.
pub fn parse<I, S>(args: I) -> Result<Startup, CliError>
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    let mut iter = args.into_iter().map(Into::into);
    let mut startup = Startup::Default;
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--genome" => {
                let value = iter.next().ok_or(CliError::MissingGenomeValue)?;
                let genome = Genome::from_bits(&value).map_err(CliError::InvalidGenome)?;
                startup = Startup::SeedGenome(genome);
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

        assert_eq!(result, Ok(Startup::Default));
    }

    #[test]
    fn a_valid_genome_argument_yields_a_seed_genome_startup() {
        let seed = Genome::all(Instruction::Split);

        let result = parse(vec!["--genome".to_string(), seed.to_bits()]);

        assert_eq!(result, Ok(Startup::SeedGenome(seed)));
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
