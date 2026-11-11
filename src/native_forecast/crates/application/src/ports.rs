//! The ports of the use cases.

use diff_gpt_forecast_domain::synth::Source;
use diff_gpt_forecast_domain::table::Table;

use crate::Result;

/// A split of a corpus.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Split {
    /// The series a model is trained on.
    Train,
    /// The series settings are chosen on; never trained on.
    Holdout,
}

impl Split {
    /// Both splits.
    pub const ALL: [Self; 2] = [Self::Train, Self::Holdout];

    /// The name of the split in a corpus.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Train => "train",
            Self::Holdout => "holdout",
        }
    }

    /// The split of that name.
    #[must_use]
    pub fn named(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|split| split.name() == name)
    }

    /// The word of the split in the key of a random stream.
    #[must_use]
    pub const fn id(self) -> u64 {
        match self {
            Self::Train => 0,
            Self::Holdout => 1,
        }
    }
}

/// Runs independent jobs, at once if it can. A use case hands its parallel
/// work to this port and starts no thread itself.
pub trait Executor: Sync {
    /// The results of `job(0), …, job(count − 1)`, in that order.
    fn map<T, F>(&self, count: usize, job: F) -> Vec<T>
    where
        T: Send,
        F: Fn(usize) -> T + Sync;
}

/// The executor that runs the jobs one after another.
#[derive(Clone, Copy, Debug, Default)]
pub struct Serial;

impl Executor for Serial {
    fn map<T, F>(&self, count: usize, job: F) -> Vec<T>
    where
        T: Send,
        F: Fn(usize) -> T + Sync,
    {
        (0..count).map(job).collect()
    }
}

/// Workers that run a job side by side, each on a processor of its own. A
/// use case that needs the same worker to come back to its own memory, as
/// the trainer does, numbers its work by the worker.
pub trait Crew: Sync {
    /// The number of workers.
    fn size(&self) -> usize;

    /// Runs `job(worker)` for every worker, `0` to `size − 1`, at once if
    /// it can, and returns when all have finished.
    fn all(&self, job: &(dyn Fn(usize) + Sync));
}

/// The crew whose workers take their turns one after another.
#[derive(Clone, Copy, Debug)]
pub struct InTurn(pub usize);

impl Crew for InTurn {
    fn size(&self) -> usize {
        self.0
    }

    fn all(&self, job: &(dyn Fn(usize) + Sync)) {
        (0..self.0).for_each(job);
    }
}

/// Where a run keeps what it reports and what it goes on from: named texts
/// and tables. A write replaces what the name held, all of it or nothing.
pub trait RunStore {
    /// The text under `name`, if there is one.
    ///
    /// # Errors
    ///
    /// If the store cannot be read.
    fn text(&self, name: &str) -> Result<Option<String>>;

    /// Puts `text` under `name`.
    ///
    /// # Errors
    ///
    /// If the store cannot be written.
    fn write_text(&self, name: &str, text: &str) -> Result<()>;

    /// Adds `text` to the end of the text under `name`.
    ///
    /// # Errors
    ///
    /// If the store cannot be written.
    fn append_text(&self, name: &str, text: &str) -> Result<()>;

    /// The table under `name`, if there is one.
    ///
    /// # Errors
    ///
    /// If the store cannot be read.
    fn table(&self, name: &str) -> Result<Option<Table>>;

    /// Puts `table` under `name`.
    ///
    /// # Errors
    ///
    /// If the store cannot be written.
    fn write_table(&self, name: &str, table: &Table) -> Result<()>;
}

/// The time a run has taken.
pub trait Clock {
    /// The seconds since the clock was started.
    fn seconds(&self) -> f64;
}

/// A corpus of series to read: some sources of some splits.
pub trait SeriesReader: Sync {
    /// The number of series of `source` in `split`.
    ///
    /// # Errors
    ///
    /// If the corpus does not hold the source or cannot be read.
    fn series(&self, split: Split, source: Source) -> Result<usize>;

    /// The series `first, …, first + count − 1` of `source` in `split`.
    ///
    /// # Errors
    ///
    /// If the corpus does not hold them or cannot be read.
    fn read(&self, split: Split, source: Source, first: usize, count: usize) -> Result<Table>;
}

/// A corpus of series to write, shard by shard.
pub trait SeriesWriter: Sync {
    /// Whether shard `shard` of `source` in `split` is already written.
    fn has(&self, split: Split, source: Source, shard: usize) -> bool;

    /// Writes shard `shard` of `source` in `split`, whole or not at all.
    ///
    /// # Errors
    ///
    /// If the shard cannot be written.
    fn write(&self, split: Split, source: Source, shard: usize, series: &Table) -> Result<()>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_have_names_and_the_serial_executor_keeps_order() {
        for split in Split::ALL {
            assert_eq!(Split::named(split.name()), Some(split));
        }
        assert_ne!(Split::Train.id(), Split::Holdout.id());
        assert_eq!(Serial.map(4, |index| index * index), [0, 1, 4, 9]);
    }

    #[test]
    fn a_crew_in_turn_runs_every_worker_once() {
        let seen = std::sync::Mutex::new(Vec::new());
        let crew = InTurn(3);
        crew.all(&|worker| seen.lock().expect("no panic").push(worker));
        assert_eq!(
            (crew.size(), seen.into_inner().expect("no panic")),
            (3, vec![0, 1, 2])
        );
    }
}
