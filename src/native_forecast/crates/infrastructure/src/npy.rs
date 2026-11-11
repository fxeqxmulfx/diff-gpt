//! Tables of series as NumPy `.npy` files: little-endian `float32`, one row
//! per series, C order — the layout of `datasets/tempopfn-synthetic/npy`.

use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::os::unix::fs::FileExt;
use std::path::Path;

use diff_gpt_forecast_domain::table::Table;

const MAGIC: &[u8; 6] = b"\x93NUMPY";

/// Where the rows of a file are and how many it holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Layout {
    /// The number of series.
    pub rows: usize,
    /// The length of a series.
    pub length: usize,
    /// The offset of the first value in the file.
    pub offset: u64,
}

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

/// The text between `key` and the next `end` of a header, trimmed.
fn field<'a>(header: &'a str, key: &str, end: char) -> io::Result<&'a str> {
    let start = header
        .find(key)
        .ok_or_else(|| invalid(format!("no {key} in the header {header:?}")))?
        + key.len();
    let rest = &header[start..];
    let stop = rest
        .find(end)
        .ok_or_else(|| invalid(format!("unfinished {key} in the header {header:?}")))?;
    Ok(rest[..stop].trim())
}

/// Reads the layout of an open `.npy` file of two-dimensional `float32`.
///
/// # Errors
///
/// If the file is not such a file.
pub fn layout(file: &File) -> io::Result<Layout> {
    let mut preamble = [0_u8; 12];
    file.read_exact_at(&mut preamble[..10], 0)?;
    if &preamble[..6] != MAGIC {
        return Err(invalid("not a .npy file"));
    }
    let (header_length, header_start) = match preamble[6] {
        1 => (
            usize::from(u16::from_le_bytes([preamble[8], preamble[9]])),
            10,
        ),
        2 | 3 => {
            file.read_exact_at(&mut preamble[10..], 10)?;
            let bytes = [preamble[8], preamble[9], preamble[10], preamble[11]];
            (u32::from_le_bytes(bytes) as usize, 12)
        }
        version => return Err(invalid(format!(".npy version {version}"))),
    };
    let mut header = vec![0_u8; header_length];
    file.read_exact_at(&mut header, header_start)?;
    let header = String::from_utf8(header).map_err(|_| invalid("a header that is not text"))?;

    let descr = field(&header, "'descr':", ',')?;
    if descr.trim_matches(['\'', '"']) != "<f4" {
        return Err(invalid(format!("values of type {descr}, not '<f4'")));
    }
    if field(&header, "'fortran_order':", ',')? != "False" {
        return Err(invalid("a table in Fortran order"));
    }
    let shape: Vec<usize> = field(&header, "'shape': (", ')')?
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(|part| {
            part.parse()
                .map_err(|_| invalid(format!("the shape {part:?}")))
        })
        .collect::<io::Result<_>>()?;
    let [rows, length] = shape[..] else {
        return Err(invalid(format!("a table of {} dimensions", shape.len())));
    };
    Ok(Layout {
        rows,
        length,
        offset: (header_start as usize + header_length) as u64,
    })
}

/// Reads the rows `first, …, first + count − 1` of an open `.npy` file.
///
/// # Errors
///
/// If the file does not hold them.
pub fn read_rows(file: &File, layout: Layout, first: usize, count: usize) -> io::Result<Table> {
    if first + count > layout.rows {
        return Err(invalid(format!(
            "rows {first}..{} of a table of {}",
            first + count,
            layout.rows
        )));
    }
    let mut bytes = vec![0_u8; count * layout.length * 4];
    file.read_exact_at(
        &mut bytes,
        layout.offset + (first * layout.length * 4) as u64,
    )?;
    let values = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| f32::from_le_bytes(*word))
        .collect();
    Ok(Table::from_values(layout.length.max(1), values))
}

/// Reads a whole `.npy` file.
///
/// # Errors
///
/// If the file is missing or is not a two-dimensional `float32` table.
pub fn read(path: &Path) -> io::Result<Table> {
    let file = File::open(path)?;
    let layout = layout(&file)?;
    read_rows(&file, layout, 0, layout.rows)
}

/// Writes `table` to `path` as a `.npy` file, through a temporary file beside
/// it, so that the file exists only whole.
///
/// # Errors
///
/// If the file cannot be written.
pub fn write(path: &Path, table: &Table) -> io::Result<()> {
    let mut header = format!(
        "{{'descr': '<f4', 'fortran_order': False, 'shape': ({}, {}), }}",
        table.rows(),
        table.length()
    );
    // The values start at a multiple of 64 bytes; the header ends with a newline.
    let unpadded = MAGIC.len() + 4 + header.len() + 1;
    header.push_str(&" ".repeat(unpadded.next_multiple_of(64) - unpadded));
    header.push('\n');
    let header_length =
        u16::try_from(header.len()).map_err(|_| invalid("a header too long for .npy 1.0"))?;

    let mut temporary = path.as_os_str().to_owned();
    temporary.push(".part");
    let temporary = Path::new(&temporary);
    let mut out = BufWriter::new(File::create(temporary)?);
    out.write_all(MAGIC)?;
    out.write_all(&[1, 0])?;
    out.write_all(&header_length.to_le_bytes())?;
    out.write_all(header.as_bytes())?;
    for value in table.values() {
        out.write_all(&value.to_le_bytes())?;
    }
    out.into_inner()
        .map_err(io::IntoInnerError::into_error)?
        .sync_all()?;
    std::fs::rename(temporary, path)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> std::path::PathBuf {
        let directory = std::env::temp_dir().join(format!("forecast-npy-{}", std::process::id()));
        std::fs::create_dir_all(&directory).expect("a temporary directory");
        directory.join(name)
    }

    #[test]
    fn a_table_survives_the_file() {
        let path = scratch("round.npy");
        let values: Vec<f32> = (0..15).map(|v| v as f32 * 0.5 - 2.0).collect();
        let table = Table::from_values(5, values);
        write(&path, &table).expect("written");
        assert_eq!(read(&path).expect("read"), table);
        let file = File::open(&path).expect("opened");
        let layout = layout(&file).expect("a layout");
        assert_eq!((layout.rows, layout.length, layout.offset % 64), (3, 5, 0));
        let middle = read_rows(&file, layout, 1, 2).expect("rows");
        assert_eq!(middle.row(0), table.row(1));
        assert!(read_rows(&file, layout, 2, 2).is_err());
        assert!(!path.with_extension("npy.part").exists());
    }

    #[test]
    fn other_files_are_refused() {
        let path = scratch("text.npy");
        std::fs::write(&path, b"not a table at all").expect("written");
        assert!(read(&path).is_err());
        // A header as NumPy writes it for float64: refused for its type.
        let mut bytes = MAGIC.to_vec();
        let header = "{'descr': '<f8', 'fortran_order': False, 'shape': (1, 2), }\n";
        bytes.extend([1, 0]);
        bytes.extend((header.len() as u16).to_le_bytes());
        bytes.extend(header.as_bytes());
        bytes.extend([0_u8; 16]);
        std::fs::write(&path, &bytes).expect("written");
        let error = read(&path).expect_err("float64 is refused");
        assert!(error.to_string().contains("<f8"), "{error}");
    }
}
