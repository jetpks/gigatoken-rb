//! `Gigatoken::Native::{Text,Jsonl,Parquet}FileSource`: paths plus how to
//! split their bytes into documents, for `BPETokenizer#encode_files`.
//! Mirrors the pyo3 `FileSource` family (`src/bindings/sources.rs`) —
//! same constructor options, same `DocFormat` payload — but each format
//! gets its own Ruby class instead of a `FileSource` base class, since
//! magnus ties one wrapped Rust type to exactly one Ruby class.
//!
//! `src/batch.rs::encode_files_docs` — the pyo3 path's ragged core, which
//! fuses byte-region chunking with document-boundary extraction during the
//! parallel encode itself — lives in a `pub(crate) mod batch` and so is not
//! reachable from this crate (see the builder report's DISAGREEMENTS).
//! `load_docs` below builds the same documents from the public `input`
//! surface instead: it splits each loaded file into documents up front
//! (single-threaded), then hands them to the public `encode_docs_ragged`,
//! the same parallel ragged core `BPETokenizer#encode_batch` already uses.

use std::path::PathBuf;

use gigatoken_rs::input::file_source::{DocFormat, load_file};
use gigatoken_rs::input::jsonl::JsonLinesSlice;
use gigatoken_rs::input::parquet;
use gigatoken_rs::input::DocumentIter;
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
        let kw = get_kwargs::<_, (), (Option<RString>,), ()>(args.keywords, &[], &["separator"])?;
        let (separator,) = kw.optional;
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
        let kw = get_kwargs::<_, (), (Option<String>,), ()>(args.keywords, &[], &["field"])?;
        let (field,) = kw.optional;
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
        let kw = get_kwargs::<_, (), (Option<String>,), ()>(args.keywords, &[], &["column"])?;
        let (column,) = kw.optional;
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

/// Load every document named by `source`, in file/row order. Parquet rows
/// are read directly as owned documents (parallel across row groups); text
/// and JSONL files are loaded whole (mmapped when uncompressed, decompressed
/// into memory otherwise — `input::file_source::load_file` handles .gz/.zst
/// transparently) and then split into documents on the calling thread.
pub(crate) fn load_docs(source: &FileSource) -> std::io::Result<Vec<Vec<u8>>> {
    if let DocFormat::Parquet { column } = &source.format {
        let mut docs = Vec::new();
        for path in &source.paths {
            docs.extend(parquet::read_docs(path, column, true)?);
        }
        return Ok(docs);
    }
    let mut docs: Vec<Vec<u8>> = Vec::new();
    for path in &source.paths {
        let loaded = load_file(path)
            .map_err(|e| std::io::Error::new(e.kind(), format!("{}: {e}", path.display())))?;
        let bytes = loaded.as_bytes();
        match &source.format {
            DocFormat::Jsonl { field } => {
                docs.extend(JsonLinesSlice::new(bytes, field).map(|d| d.as_ref().to_vec()));
            }
            DocFormat::Text {
                separator: Some(sep),
            } if !sep.is_empty() => {
                docs.extend(DocumentIter::new(bytes, sep).map(<[u8]>::to_vec));
            }
            DocFormat::Text { .. } => docs.push(bytes.to_vec()),
            DocFormat::Parquet { .. } => unreachable!("handled above"),
        }
    }
    Ok(docs)
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
