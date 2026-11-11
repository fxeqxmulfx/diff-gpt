//! Corpora on disk.
//!
//! [`Reference`] reads the corpus that TempoPFN's Python generators made, one
//! file per source: `<root>/<split>/<source>.npy`. [`Shards`] reads and
//! writes the corpus of this workspace, a folder of shards per source:
//! `<root>/<split>/<source>/shard_<k>.npy`.

use std::fs::File;
use std::path::{Path, PathBuf};

use diff_gpt_forecast_application::Result;
use diff_gpt_forecast_application::ports::{SeriesReader, SeriesWriter, Split};
use diff_gpt_forecast_domain::synth::Source;
use diff_gpt_forecast_domain::table::Table;

use crate::npy;

/// The corpus made by TempoPFN's Python generators, exported to `.npy`.
#[derive(Clone, Debug)]
pub struct Reference {
    root: PathBuf,
}

impl Reference {
    /// The reference corpus under `root`.
    #[must_use]
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    fn path(&self, split: Split, source: Source) -> PathBuf {
        self.root
            .join(split.name())
            .join(format!("{}.npy", source.name()))
    }

    fn open(&self, split: Split, source: Source) -> Result<(File, npy::Layout)> {
        let path = self.path(split, source);
        let file = File::open(&path).map_err(|error| format!("{}: {error}", path.display()))?;
        let layout = npy::layout(&file).map_err(|error| format!("{}: {error}", path.display()))?;
        Ok((file, layout))
    }
}

impl SeriesReader for Reference {
    fn series(&self, split: Split, source: Source) -> Result<usize> {
        Ok(self.open(split, source)?.1.rows)
    }

    fn read(&self, split: Split, source: Source, first: usize, count: usize) -> Result<Table> {
        let (file, layout) = self.open(split, source)?;
        Ok(npy::read_rows(&file, layout, first, count)?)
    }
}

/// The corpus of this workspace: shards of equal size per source.
#[derive(Clone, Debug)]
pub struct Shards {
    root: PathBuf,
}

impl Shards {
    /// The corpus under `root`.
    #[must_use]
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// The folder of the corpus.
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    fn directory(&self, split: Split, source: Source) -> PathBuf {
        self.root.join(split.name()).join(source.name())
    }

    fn path(&self, split: Split, source: Source, shard: usize) -> PathBuf {
        self.directory(split, source)
            .join(format!("shard_{shard:05}.npy"))
    }

    /// The shards of a source that are written, counted from shard 0 up to
    /// the first missing one, with the layout of each.
    fn layouts(&self, split: Split, source: Source) -> Result<Vec<(File, npy::Layout)>> {
        let mut shards = Vec::new();
        loop {
            let path = self.path(split, source, shards.len());
            let Ok(file) = File::open(&path) else {
                return Ok(shards);
            };
            let layout =
                npy::layout(&file).map_err(|error| format!("{}: {error}", path.display()))?;
            shards.push((file, layout));
        }
    }
}

impl SeriesWriter for Shards {
    fn has(&self, split: Split, source: Source, shard: usize) -> bool {
        self.path(split, source, shard).is_file()
    }

    fn write(&self, split: Split, source: Source, shard: usize, series: &Table) -> Result<()> {
        std::fs::create_dir_all(self.directory(split, source))?;
        Ok(npy::write(&self.path(split, source, shard), series)?)
    }
}

impl SeriesReader for Shards {
    fn series(&self, split: Split, source: Source) -> Result<usize> {
        Ok(self
            .layouts(split, source)?
            .iter()
            .map(|(_, layout)| layout.rows)
            .sum())
    }

    fn read(&self, split: Split, source: Source, first: usize, count: usize) -> Result<Table> {
        let shards = self.layouts(split, source)?;
        let mut table: Option<Table> = None;
        let (mut skip, mut left) = (first, count);
        for (file, layout) in &shards {
            if skip >= layout.rows {
                skip -= layout.rows;
                continue;
            }
            let take = left.min(layout.rows - skip);
            let rows = npy::read_rows(file, *layout, skip, take)?;
            table
                .get_or_insert_with(|| Table::new(rows.length()))
                .append(&rows);
            skip = 0;
            left -= take;
            if left == 0 {
                break;
            }
        }
        match table {
            Some(table) if left == 0 => Ok(table),
            _ => Err(format!(
                "{}/{} holds fewer than {} series",
                split.name(),
                source.name(),
                first + count
            )
            .into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!("forecast-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        root
    }

    fn table(first: f32, rows: usize) -> Table {
        let values = (0..rows * 4).map(|index| first + index as f32).collect();
        Table::from_values(4, values)
    }

    #[test]
    fn shards_are_written_once_and_read_across_files() {
        let shards = Shards::new(scratch("shards"));
        let (split, source) = (Split::Train, Source::Step);
        assert!(!shards.has(split, source, 0));
        assert_eq!(shards.series(split, source).expect("counted"), 0);
        shards
            .write(split, source, 0, &table(0.0, 3))
            .expect("written");
        shards
            .write(split, source, 1, &table(100.0, 3))
            .expect("written");
        assert!(shards.has(split, source, 1) && !shards.has(Split::Holdout, source, 1));
        assert_eq!(shards.series(split, source).expect("counted"), 6);
        let across = shards.read(split, source, 2, 3).expect("read");
        assert_eq!(across.rows(), 3);
        assert_eq!(across.row(0), [8.0, 9.0, 10.0, 11.0]);
        assert_eq!(across.row(2), [104.0, 105.0, 106.0, 107.0]);
        assert!(shards.read(split, source, 4, 3).is_err());
        std::fs::remove_dir_all(shards.root()).expect("removed");
    }

    #[test]
    fn the_reference_is_one_file_per_source() {
        let root = scratch("reference");
        std::fs::create_dir_all(root.join("holdout")).expect("created");
        npy::write(&root.join("holdout/spike.npy"), &table(1.0, 5)).expect("written");
        let reference = Reference::new(&root);
        assert_eq!(
            reference
                .series(Split::Holdout, Source::Spike)
                .expect("counted"),
            5
        );
        let rows = reference
            .read(Split::Holdout, Source::Spike, 3, 2)
            .expect("read");
        assert_eq!(rows.row(0), [13.0, 14.0, 15.0, 16.0]);
        let missing = reference
            .series(Split::Train, Source::Spike)
            .expect_err("no file");
        assert!(missing.to_string().contains("spike.npy"), "{missing}");
        std::fs::remove_dir_all(root).expect("removed");
    }
}
