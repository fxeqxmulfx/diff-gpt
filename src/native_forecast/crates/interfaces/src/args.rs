//! The options of a command: `--name value` pairs.

use std::collections::BTreeMap;
use std::str::FromStr;

use diff_gpt_forecast_application::Result;

/// The options given to a command, each taken at most once.
#[derive(Debug)]
pub struct Options {
    values: BTreeMap<String, String>,
}

impl Options {
    /// Parses `--name value` pairs.
    ///
    /// # Errors
    ///
    /// If an argument is not an option, lacks its value or comes twice.
    pub fn parse(arguments: impl IntoIterator<Item = String>) -> Result<Self> {
        let mut values = BTreeMap::new();
        let mut arguments = arguments.into_iter();
        while let Some(argument) = arguments.next() {
            let Some(name) = argument.strip_prefix("--") else {
                return Err(format!("expected an option, found {argument:?}").into());
            };
            let Some(value) = arguments.next() else {
                return Err(format!("--{name} needs a value").into());
            };
            if values.insert(name.to_owned(), value).is_some() {
                return Err(format!("--{name} is given twice").into());
            }
        }
        Ok(Self { values })
    }

    /// Takes the value of `--name`, or `default` if it was not given.
    ///
    /// # Errors
    ///
    /// If the value does not parse.
    pub fn take<T>(&mut self, name: &str, default: T) -> Result<T>
    where
        T: FromStr,
        T::Err: core::fmt::Display,
    {
        match self.values.remove(name) {
            None => Ok(default),
            Some(value) => value
                .parse()
                .map_err(|error| format!("--{name} {value:?}: {error}").into()),
        }
    }

    /// Takes the comma-separated values of `--name`, if it was given.
    pub fn take_list(&mut self, name: &str) -> Option<Vec<String>> {
        let value = self.values.remove(name)?;
        Some(value.split(',').map(str::to_owned).collect())
    }

    /// Checks that every option given was taken.
    ///
    /// # Errors
    ///
    /// If an option is left: the command does not know it.
    pub fn finish(self) -> Result<()> {
        match self.values.keys().next() {
            None => Ok(()),
            Some(name) => Err(format!("unknown option --{name}").into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn options(arguments: &[&str]) -> Result<Options> {
        Options::parse(arguments.iter().map(|&argument| argument.to_owned()))
    }

    #[test]
    fn options_are_taken_once_and_typed() {
        let mut given = options(&["--series", "12", "--sources", "a,b"]).expect("parsed");
        assert_eq!(given.take("series", 5_usize).expect("a number"), 12);
        assert_eq!(given.take("level", 0.5_f64).expect("the default"), 0.5);
        assert_eq!(
            given.take_list("sources"),
            Some(vec!["a".to_owned(), "b".to_owned()])
        );
        assert_eq!(given.take_list("sources"), None);
        given.finish().expect("nothing left");
    }

    #[test]
    fn mistakes_are_reported() {
        assert!(options(&["series"]).is_err());
        assert!(options(&["--series"]).is_err());
        assert!(options(&["--series", "1", "--series", "2"]).is_err());
        let mut given = options(&["--series", "many", "--extra", "1"]).expect("parsed");
        assert!(given.take("series", 5_usize).is_err());
        assert!(
            given
                .finish()
                .expect_err("an option is left")
                .to_string()
                .contains("--extra")
        );
    }
}
