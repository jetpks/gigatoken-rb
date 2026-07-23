//! `Gigatoken::Native::{Text,Jsonl,Parquet}FileSource`: paths plus how to
//! split their bytes into documents, for `BPETokenizer#encode_files`.
//! Mirrors the pyo3 `FileSource` family (`src/bindings/sources.rs`) —
//! same constructor options, same `DocFormat` payload — but each format
//! gets its own Ruby class instead of a `FileSource` base class, since
//! magnus ties one wrapped Rust type to exactly one Ruby class.
//!
//! `encode_files_ragged` below mirrors the pyo3 path's loading scaffold
//! (`src/bindings/sources.rs::encode_files_ragged`), now that the crate root
//! re-exports `batch::encode_files_docs`/`_serial` (`src/lib.rs`): parquet
//! rows can't be split out of raw file bytes, so they're read directly as
//! owned documents via `input::parquet::read_docs` and encoded through the
//! whole-document path; text/JSONL files are loaded whole (mmapped, or
//! decompressed into memory for .gz/.zst — `load_file` handles both) and
//! handed straight to the fused chunk+extract+encode core as byte regions,
//! with no separate single-threaded document-splitting pass.

use std::path::PathBuf;

use gigatoken_rs::input::file_source::{DocFormat, LoadedFile, load_file};
use gigatoken_rs::input::parquet;
use magnus::{
    prelude::*,
    scan_args::{get_kwargs, scan_args},
    Error, RClass, RHash, RModule, RString, Ruby, TryConvert, Value,
};

use crate::error::raise;

/// Paths plus how to split their bytes into documents — the payload shared
/// by the three FileSource classes below.
#[derive(Clone)]
pub(crate) struct FileSource {
    pub(crate) paths: Vec<PathBuf>,
    pub(crate) format: DocFormat,
}

fn rstring_bytes(s: RString) -> Vec<u8> {
    // SAFETY: read-only, for the duration of this synchronous call.
    unsafe { s.as_slice() }.to_vec()
}

/// Plain-text files. With `separator`, documents are the pieces between
/// separator occurrences; without one, each file is a single document.
#[magnus::wrap(class = "Gigatoken::Native::TextFileSource", free_immediately, size)]
pub struct TextFileSource(pub(crate) FileSource);

impl TextFileSource {
    fn new(args: &[Value]) -> Result<Self, Error> {
        let args = scan_args::<(Vec<PathBuf>,), (), (), (), RHash, ()>(args)?;
        let (paths,) = args.required;
        let kw = get_kwargs::<_, (), (Option<Option<RString>>,), ()>(args.keywords, &[], &["separator"])?;
        let (separator,) = kw.optional;
        let separator = separator.flatten();
        Ok(Self(FileSource {
            paths,
            format: DocFormat::Text {
                separator: separator.map(rstring_bytes),
            },
        }))
    }
}

/// JSON Lines files: one document per line, text taken from `field`
/// (default `"text"`).
#[magnus::wrap(class = "Gigatoken::Native::JsonlFileSource", free_immediately, size)]
pub struct JsonlFileSource(pub(crate) FileSource);

impl JsonlFileSource {
    fn new(args: &[Value]) -> Result<Self, Error> {
        let args = scan_args::<(Vec<PathBuf>,), (), (), (), RHash, ()>(args)?;
        let (paths,) = args.required;
        let kw = get_kwargs::<_, (), (Option<Option<String>>,), ()>(args.keywords, &[], &["field"])?;
        let (field,) = kw.optional;
        let field = field.flatten();
        Ok(Self(FileSource {
            paths,
            format: DocFormat::Jsonl {
                field: field.unwrap_or_else(|| "text".to_string()),
            },
        }))
    }
}

/// Parquet files: one document per row, text taken from `column` (default
/// `"text"`); null rows become empty documents.
#[magnus::wrap(class = "Gigatoken::Native::ParquetFileSource", free_immediately, size)]
pub struct ParquetFileSource(pub(crate) FileSource);

impl ParquetFileSource {
    fn new(args: &[Value]) -> Result<Self, Error> {
        let args = scan_args::<(Vec<PathBuf>,), (), (), (), RHash, ()>(args)?;
        let (paths,) = args.required;
        let kw = get_kwargs::<_, (), (Option<Option<String>>,), ()>(args.keywords, &[], &["column"])?;
        let (column,) = kw.optional;
        let column = column.flatten();
        Ok(Self(FileSource {
            paths,
            format: DocFormat::Parquet {
                column: column.unwrap_or_else(|| "text".to_string()),
            },
        }))
    }
}

/// Resolve an `encode_files` argument to its `FileSource`: a
/// `TextFileSource`, `JsonlFileSource`, or `ParquetFileSource`. Bare paths
/// are wrapped into a `TextFileSource` on the Ruby side
/// (`Gigatoken::Tokenizer#encode_files`), so this only needs to handle the
/// three native classes.
pub(crate) fn resolve(ruby: &Ruby, source: Value) -> Result<FileSource, Error> {
    if let Ok(s) = <&TextFileSource>::try_convert(source) {
        return Ok(s.0.clone());
    }
    if let Ok(s) = <&JsonlFileSource>::try_convert(source) {
        return Ok(s.0.clone());
    }
    if let Ok(s) = <&ParquetFileSource>::try_convert(source) {
        return Ok(s.0.clone());
    }
    Err(raise(
        ruby,
        format!(
            "expected a TextFileSource, JsonlFileSource, or ParquetFileSource, got {}",
            source.class()
        ),
    ))
}

/// Load every document named by `source`, in file/row order, and hand its
/// contents and document format to `encode` — the fused chunk+extract+encode
/// core (`batch::encode_files_docs`/`_serial`, called by `tokenizer.rs`).
/// Parquet rows can't be split out of raw file bytes, so they are
/// materialized as owned documents here and encoded through the
/// whole-document path (each buffer one document); other formats are loaded
/// whole and left for `encode` to split.
pub(crate) fn encode_files_ragged(
    source: &FileSource,
    parallel: bool,
    encode: impl FnOnce(&[&[u8]], &DocFormat) -> (Vec<u32>, Vec<i64>),
) -> std::io::Result<(Vec<u32>, Vec<i64>)> {
    if let DocFormat::Parquet { column } = &source.format {
        let docs = load_parquet_docs(&source.paths, column, parallel)?;
        let bytes: Vec<&[u8]> = docs.iter().map(Vec::as_slice).collect();
        return Ok(encode(&bytes, &DocFormat::Text { separator: None }));
    }
    let files = load_files(&source.paths, parallel)?;
    let bytes: Vec<&[u8]> = files.iter().map(LoadedFile::as_bytes).collect();
    Ok(encode(&bytes, &source.format))
}

/// Load `column` of every parquet file as one owned document per row, files
/// in argument order, rows in row order. Parallel across files and row
/// groups with rayon, or fully on the calling thread when `parallel` is
/// false (the sequential encode paths must never touch the rayon pool).
fn load_parquet_docs(
    paths: &[PathBuf],
    column: &str,
    parallel: bool,
) -> std::io::Result<Vec<Vec<u8>>> {
    use rayon::prelude::*;
    let per_file: Vec<Vec<Vec<u8>>> = if parallel {
        paths
            .par_iter()
            .map(|p| parquet::read_docs(p, column, true))
            .collect::<std::io::Result<_>>()?
    } else {
        paths
            .iter()
            .map(|p| parquet::read_docs(p, column, false))
            .collect::<std::io::Result<_>>()?
    };
    Ok(per_file.into_iter().flatten().collect())
}

/// Load all files: mmap when stored uncompressed, decompress .gz/.zst into
/// memory otherwise (parallel chunking needs random access). In parallel
/// with rayon, or serially on the calling thread when `parallel` is false
/// (the sequential encode paths must never touch the rayon pool).
fn load_files(paths: &[PathBuf], parallel: bool) -> std::io::Result<Vec<LoadedFile>> {
    use rayon::prelude::*;
    let load = |p: &PathBuf| {
        load_file(p).map_err(|e| std::io::Error::new(e.kind(), format!("{}: {e}", p.display())))
    };
    if parallel {
        paths.par_iter().map(load).collect()
    } else {
        paths.iter().map(load).collect()
    }
}

pub fn init(ruby: &Ruby, native: RModule) -> Result<(), Error> {
    let text: RClass = native.define_class("TextFileSource", ruby.class_object())?;
    text.define_singleton_method("new", magnus::function!(TextFileSource::new, -1))?;

    let jsonl: RClass = native.define_class("JsonlFileSource", ruby.class_object())?;
    jsonl.define_singleton_method("new", magnus::function!(JsonlFileSource::new, -1))?;

    let parquet: RClass = native.define_class("ParquetFileSource", ruby.class_object())?;
    parquet.define_singleton_method("new", magnus::function!(ParquetFileSource::new, -1))?;

    Ok(())
}
