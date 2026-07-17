//! Parquet input: one document per row, text taken from a string or binary
//! column. Parquet pages are decoded through the arrow reader, so documents
//! are materialized as owned buffers rather than borrowed from an mmap like
//! the byte-stream formats. Null rows become empty documents so results
//! stay row-aligned with the source table. Row groups are the parallel work
//! units; documents always come back in row order.

use std::collections::HashMap;
use std::fs::File;
use std::io;
use std::path::Path;

use arrow_array::cast::AsArray;
use arrow_array::types::{ByteArrayType, ByteViewType};
use arrow_array::{Array, GenericByteArray, GenericByteViewArray, RecordBatch};
use arrow_schema::DataType;
use parquet::arrow::ProjectionMask;
use parquet::arrow::arrow_reader::{ParquetRecordBatchReader, ParquetRecordBatchReaderBuilder};
use rayon::prelude::*;
use rustc_hash::FxBuildHasher;

use crate::pretokenize::pretokenize_as_iter;

fn parquet_err(path: &Path, err: impl std::fmt::Display) -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidData,
        format!("{}: {err}", path.display()),
    )
}

fn open_builder(path: &Path) -> io::Result<ParquetRecordBatchReaderBuilder<File>> {
    let file = File::open(path)
        .map_err(|e| io::Error::new(e.kind(), format!("{}: {e}", path.display())))?;
    ParquetRecordBatchReaderBuilder::try_new(file).map_err(|e| parquet_err(path, e))
}

pub fn n_row_groups(path: &Path) -> io::Result<usize> {
    Ok(open_builder(path)?.metadata().num_row_groups())
}

/// Reader over `row_groups` (all when None) that decodes only `column`.
/// `column` must be a top-level field of the file's schema.
fn open_reader(
    path: &Path,
    column: &str,
    row_groups: Option<Vec<usize>>,
) -> io::Result<ParquetRecordBatchReader> {
    let builder = open_builder(path)?;
    if builder.schema().index_of(column).is_err() {
        let available: Vec<&str> = builder
            .schema()
            .fields()
            .iter()
            .map(|f| f.name().as_str())
            .collect();
        return Err(parquet_err(
            path,
            format!(
                "no column {column:?} (available: {})",
                available.join(", ")
            ),
        ));
    }
    let mask = ProjectionMask::columns(builder.parquet_schema(), [column]);
    let builder = builder.with_projection(mask);
    let builder = match row_groups {
        Some(rgs) => builder.with_row_groups(rgs),
        None => builder,
    };
    builder.build().map_err(|e| parquet_err(path, e))
}

/// Visit each row of the batch's single projected column as document bytes.
/// Null rows visit as empty slices, keeping documents row-aligned.
fn for_each_row(
    batch: &RecordBatch,
    path: &Path,
    column: &str,
    f: &mut impl FnMut(&[u8]),
) -> io::Result<()> {
    fn visit_bytes<T: ByteArrayType>(arr: &GenericByteArray<T>, f: &mut impl FnMut(&[u8]))
    where
        T::Native: AsRef<[u8]>,
    {
        for i in 0..arr.len() {
            if arr.is_null(i) {
                f(b"");
            } else {
                f(arr.value(i).as_ref());
            }
        }
    }
    fn visit_views<T: ByteViewType>(arr: &GenericByteViewArray<T>, f: &mut impl FnMut(&[u8]))
    where
        T::Native: AsRef<[u8]>,
    {
        for i in 0..arr.len() {
            if arr.is_null(i) {
                f(b"");
            } else {
                f(arr.value(i).as_ref());
            }
        }
    }

    let col = batch.column(0);
    match col.data_type() {
        DataType::Utf8 => visit_bytes(col.as_string::<i32>(), f),
        DataType::LargeUtf8 => visit_bytes(col.as_string::<i64>(), f),
        DataType::Utf8View => visit_views(col.as_string_view(), f),
        DataType::Binary => visit_bytes(col.as_binary::<i32>(), f),
        DataType::LargeBinary => visit_bytes(col.as_binary::<i64>(), f),
        DataType::BinaryView => visit_views(col.as_binary_view(), f),
        other => {
            return Err(parquet_err(
                path,
                format!(
                    "column {column:?} has unsupported type {other} \
                     (expected a string or binary column)"
                ),
            ));
        }
    }
    Ok(())
}

/// Visit every document (row) of `column` in the given row groups (all when
/// None), in row order, on the calling thread.
pub fn for_each_doc(
    path: &Path,
    column: &str,
    row_groups: Option<Vec<usize>>,
    mut f: impl FnMut(&[u8]),
) -> io::Result<()> {
    for batch in open_reader(path, column, row_groups)? {
        let batch = batch.map_err(|e| parquet_err(path, e))?;
        for_each_row(&batch, path, column, &mut f)?;
    }
    Ok(())
}

/// All documents of `column` in `path`, one owned buffer per row, in row
/// order. Parallel over row groups when `parallel`; strictly on the calling
/// thread otherwise (the serial encode paths must never touch rayon).
pub fn read_docs(path: &Path, column: &str, parallel: bool) -> io::Result<Vec<Vec<u8>>> {
    if !parallel {
        let mut docs = Vec::new();
        for_each_doc(path, column, None, |d| docs.push(d.to_vec()))?;
        return Ok(docs);
    }
    let per_group: Vec<Vec<Vec<u8>>> = (0..n_row_groups(path)?)
        .into_par_iter()
        .map(|rg| {
            let mut docs = Vec::new();
            for_each_doc(path, column, Some(vec![rg]), |d| docs.push(d.to_vec()))?;
            Ok(docs)
        })
        .collect::<io::Result<_>>()?;
    Ok(per_group.into_iter().flatten().collect())
}

/// Parallel pretokenization of `column`, one rayon task per row group.
/// Documents are visited batch-by-batch without materializing the file.
pub fn pretokenize_par(
    path: &Path,
    column: &str,
) -> io::Result<HashMap<Vec<u8>, usize, FxBuildHasher>> {
    (0..n_row_groups(path)?)
        .into_par_iter()
        .map(|rg| {
            let mut counts: HashMap<Vec<u8>, usize, FxBuildHasher> = HashMap::default();
            for_each_doc(path, column, Some(vec![rg]), |doc| {
                for pretoken in pretokenize_as_iter(doc) {
                    *counts.entry(pretoken.as_ref().to_vec()).or_default() += 1;
                }
            })?;
            Ok(counts)
        })
        .try_reduce(HashMap::default, |mut acc, counts| {
            if acc.is_empty() {
                return Ok(counts);
            }
            for (k, v) in counts {
                *acc.entry(k).or_default() += v;
            }
            Ok(acc)
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{BinaryArray, Int64Array, StringArray};
    use parquet::arrow::ArrowWriter;
    use parquet::file::properties::WriterProperties;
    use std::sync::Arc;

    /// Write a single-row-group-capped parquet file with a nullable string
    /// "text" column, a binary "raw" column, and an int "id" column.
    fn write_fixture(path: &Path, texts: &[Option<&str>], max_row_group_size: usize) {
        let schema = Arc::new(arrow_schema::Schema::new(vec![
            arrow_schema::Field::new("id", DataType::Int64, false),
            arrow_schema::Field::new("text", DataType::Utf8, true),
            arrow_schema::Field::new("raw", DataType::Binary, true),
        ]));
        let ids = Int64Array::from((0..texts.len() as i64).collect::<Vec<_>>());
        let text_col = StringArray::from(texts.to_vec());
        let raw_col = BinaryArray::from(
            texts
                .iter()
                .map(|t| t.map(|s| s.as_bytes()))
                .collect::<Vec<_>>(),
        );
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(ids), Arc::new(text_col), Arc::new(raw_col)],
        )
        .unwrap();
        let props = WriterProperties::builder()
            .set_max_row_group_row_count(Some(max_row_group_size))
            .build();
        let file = File::create(path).unwrap();
        let mut writer = ArrowWriter::try_new(file, schema, Some(props)).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
    }

    const TEXTS: [Option<&str>; 7] = [
        Some("The quick brown fox"),
        Some("jumps over the lazy dog."),
        None,
        Some("She sells seashells"),
        Some(""),
        Some("by the seashore."),
        Some("Peter Piper picked a peck"),
    ];

    fn expected_docs() -> Vec<Vec<u8>> {
        TEXTS
            .iter()
            .map(|t| t.unwrap_or("").as_bytes().to_vec())
            .collect()
    }

    #[test]
    fn test_read_docs_row_order_and_nulls() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("docs.parquet");
        write_fixture(&path, &TEXTS, 1024);
        assert_eq!(read_docs(&path, "text", false).unwrap(), expected_docs());
    }

    #[test]
    fn test_read_docs_parallel_multiple_row_groups() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("docs.parquet");
        write_fixture(&path, &TEXTS, 2); // 4 row groups for 7 rows
        assert!(n_row_groups(&path).unwrap() > 1);
        assert_eq!(read_docs(&path, "text", true).unwrap(), expected_docs());
        assert_eq!(read_docs(&path, "text", false).unwrap(), expected_docs());
    }

    #[test]
    fn test_read_binary_column() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("docs.parquet");
        write_fixture(&path, &TEXTS, 3);
        assert_eq!(read_docs(&path, "raw", true).unwrap(), expected_docs());
    }

    #[test]
    fn test_missing_column_lists_available() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("docs.parquet");
        write_fixture(&path, &TEXTS, 1024);
        let err = read_docs(&path, "content", false).unwrap_err().to_string();
        assert!(err.contains("no column \"content\""), "{err}");
        assert!(err.contains("text"), "{err}");
    }

    #[test]
    fn test_non_string_column_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("docs.parquet");
        write_fixture(&path, &TEXTS, 1024);
        let err = read_docs(&path, "id", false).unwrap_err().to_string();
        assert!(err.contains("unsupported type"), "{err}");
    }

    #[test]
    fn test_pretokenize_matches_per_doc_counts() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("docs.parquet");
        write_fixture(&path, &TEXTS, 2);
        let mut expected: HashMap<Vec<u8>, usize, FxBuildHasher> = HashMap::default();
        for doc in expected_docs() {
            for pretoken in pretokenize_as_iter(&doc) {
                *expected.entry(pretoken.as_ref().to_vec()).or_default() += 1;
            }
        }
        assert_eq!(pretokenize_par(&path, "text").unwrap(), expected);
    }
}
