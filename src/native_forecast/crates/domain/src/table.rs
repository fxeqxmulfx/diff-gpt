//! A table of series of one length, stored row after row.

/// Series of one length in single precision, as the corpus stores them.
#[derive(Clone, Debug, PartialEq)]
pub struct Table {
    length: usize,
    values: Vec<f32>,
}

impl Table {
    /// An empty table of series of `length` steps.
    ///
    /// # Panics
    ///
    /// If `length` is 0.
    #[must_use]
    pub fn new(length: usize) -> Self {
        assert!(length > 0, "series have at least one step");
        Self {
            length,
            values: Vec::new(),
        }
    }

    /// The table of the series stored in `values` row after row.
    ///
    /// # Panics
    ///
    /// If `values` is not a whole number of series of `length` steps.
    #[must_use]
    pub fn from_values(length: usize, values: Vec<f32>) -> Self {
        assert!(
            length > 0 && values.len().is_multiple_of(length),
            "{} values are not series of {length} steps",
            values.len()
        );
        Self { length, values }
    }

    /// The length of each series.
    #[must_use]
    pub const fn length(&self) -> usize {
        self.length
    }

    /// The number of series.
    #[must_use]
    pub const fn rows(&self) -> usize {
        self.values.len() / self.length
    }

    /// Series `index`.
    #[must_use]
    pub fn row(&self, index: usize) -> &[f32] {
        &self.values[index * self.length..(index + 1) * self.length]
    }

    /// The series, in order.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = &[f32]> {
        self.values.chunks_exact(self.length)
    }

    /// All the values, row after row.
    #[must_use]
    pub fn values(&self) -> &[f32] {
        &self.values
    }

    /// Appends a series given in double precision, rounding it.
    ///
    /// # Panics
    ///
    /// If its length is not that of the table.
    pub fn push(&mut self, series: &[f64]) {
        assert_eq!(series.len(), self.length, "a series of another length");
        self.values.extend(series.iter().map(|&value| value as f32));
    }

    /// Appends the series of `other`.
    ///
    /// # Panics
    ///
    /// If its series have another length.
    pub fn append(&mut self, other: &Self) {
        assert_eq!(other.length, self.length, "series of another length");
        self.values.extend_from_slice(&other.values);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rows_come_back_in_order() {
        let mut table = Table::new(3);
        table.push(&[1.0, 2.0, 3.0]);
        table.push(&[4.0, 5.0, 6.5]);
        assert_eq!((table.rows(), table.length()), (2, 3));
        assert_eq!(table.row(1), [4.0, 5.0, 6.5]);
        assert_eq!(table.iter().count(), 2);
        let mut both = Table::from_values(3, table.values().to_vec());
        both.append(&table);
        assert_eq!(both.rows(), 4);
        assert_eq!(both.row(3), table.row(1));
    }
}
