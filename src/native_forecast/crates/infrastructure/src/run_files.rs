//! A run on disk: a directory of texts and `.npy` tables, and the clock of
//! the process.

use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use diff_gpt_forecast_application::Result;
use diff_gpt_forecast_application::ports::{Clock, RunStore};
use diff_gpt_forecast_domain::table::Table;

use crate::npy;

/// The directory of a run. A text or a table is written beside its name
/// and then moved onto it, so that a crash leaves the old file or the new.
#[derive(Clone, Debug)]
pub struct RunFiles {
    directory: PathBuf,
}

impl RunFiles {
    /// The run kept in `directory`, which is made if it is not there.
    ///
    /// # Errors
    ///
    /// If the directory cannot be made.
    pub fn new(directory: impl Into<PathBuf>) -> io::Result<Self> {
        let directory = directory.into();
        fs::create_dir_all(&directory)?;
        Ok(Self { directory })
    }

    /// The directory.
    #[must_use]
    pub fn directory(&self) -> &Path {
        &self.directory
    }

    /// Writes a file with `write` and moves it onto `name`.
    fn replace(&self, name: &str, write: impl FnOnce(&Path) -> io::Result<()>) -> Result<()> {
        let (path, partial) = (
            self.directory.join(name),
            self.directory.join(format!("{name}.partial")),
        );
        write(&partial).map_err(|error| format!("{}: {error}", partial.display()))?;
        fs::rename(&partial, &path).map_err(|error| format!("{}: {error}", path.display()))?;
        Ok(())
    }

    /// What `read` finds under `name`, or nothing if no file is there.
    fn find<T>(&self, name: &str, read: impl FnOnce(&Path) -> io::Result<T>) -> Result<Option<T>> {
        let path = self.directory.join(name);
        match read(&path) {
            Ok(found) => Ok(Some(found)),
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(None),
            Err(error) => Err(format!("{}: {error}", path.display()).into()),
        }
    }
}

impl RunStore for RunFiles {
    fn text(&self, name: &str) -> Result<Option<String>> {
        self.find(name, |path| fs::read_to_string(path))
    }

    fn write_text(&self, name: &str, text: &str) -> Result<()> {
        self.replace(name, |path| fs::write(path, text))
    }

    fn append_text(&self, name: &str, text: &str) -> Result<()> {
        let path = self.directory.join(name);
        let appended = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .and_then(|mut file| file.write_all(text.as_bytes()));
        Ok(appended.map_err(|error| format!("{}: {error}", path.display()))?)
    }

    fn table(&self, name: &str) -> Result<Option<Table>> {
        self.find(name, npy::read)
    }

    fn write_table(&self, name: &str, table: &Table) -> Result<()> {
        self.replace(name, |path| npy::write(path, table))
    }
}

/// The clock of a process: the seconds since it was started.
#[derive(Clone, Copy, Debug)]
pub struct Stopwatch(Instant);

impl Stopwatch {
    /// A clock that starts now.
    #[must_use]
    pub fn start() -> Self {
        Self(Instant::now())
    }
}

impl Clock for Stopwatch {
    fn seconds(&self) -> f64 {
        self.0.elapsed().as_secs_f64()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_run_keeps_its_texts_and_tables() {
        let directory = std::env::temp_dir().join(format!("forecast-run-{}", std::process::id()));
        let run = RunFiles::new(directory.join("nested")).expect("made");
        assert_eq!(run.text("notes.txt").expect("read"), None);
        assert_eq!(run.table("weights.npy").expect("read"), None);
        run.write_text("notes.txt", "one\n").expect("written");
        run.append_text("notes.txt", "two\n").expect("appended");
        run.append_text("log.csv", "a,b\n")
            .expect("appended to nothing");
        assert_eq!(
            run.text("notes.txt").expect("read").as_deref(),
            Some("one\ntwo\n")
        );
        assert_eq!(run.text("log.csv").expect("read").as_deref(), Some("a,b\n"));
        run.write_text("notes.txt", "three\n").expect("replaced");
        assert_eq!(
            run.text("notes.txt").expect("read").as_deref(),
            Some("three\n")
        );
        let table = Table::from_values(3, vec![1.0, 2.0, 3.0, -4.0, 5.5, 6.0]);
        run.write_table("weights.npy", &table).expect("written");
        assert_eq!(run.table("weights.npy").expect("read"), Some(table));
        // Nothing is left beside the files.
        let mut names: Vec<_> = fs::read_dir(run.directory())
            .expect("listed")
            .map(|entry| {
                entry
                    .expect("an entry")
                    .file_name()
                    .into_string()
                    .expect("a name")
            })
            .collect();
        names.sort();
        assert_eq!(names, ["log.csv", "notes.txt", "weights.npy"]);
        fs::remove_dir_all(directory).expect("removed");
        let clock = Stopwatch::start();
        assert!(clock.seconds() >= 0.0 && clock.seconds() < 60.0);
    }
}
