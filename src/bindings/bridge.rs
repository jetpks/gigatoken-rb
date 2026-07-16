//! Python<->Rust bridging shared by the encode/decode bindings: document
//! and token-id extraction from Python objects, vocab/merges conversion the
//! other way, and the shared front-ends that resolve an encode_batch input
//! and hand the ragged result back to Python.

use crate::input::file_source::DocFormat;
use crate::token::TokenId;
use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::pybacked::{PyBackedBytes, PyBackedStr};
use pyo3::types::{IntoPyDict, PyBytes, PyDict, PyList};
use std::path::PathBuf;

/// A document to encode: str (UTF-8 text) or bytes. Both variants borrow
/// the Python object's buffer without copying and are usable with the GIL
/// released. Paths are deliberately not accepted here — encoding from files
/// goes through `encode_files`, which mmaps and chunks them.
pub(crate) enum EncodeInput {
    Text(PyBackedStr),
    Bytes(PyBackedBytes),
}

impl EncodeInput {
    pub(crate) fn as_bytes(&self) -> &[u8] {
        match self {
            EncodeInput::Text(s) => s.as_bytes(),
            EncodeInput::Bytes(b) => b,
        }
    }
}

/// Extract one document, pointing path-holders at encode_files.
pub(crate) fn extract_doc(obj: &Bound<'_, PyAny>) -> PyResult<EncodeInput> {
    if let Ok(s) = obj.extract::<PyBackedStr>() {
        return Ok(EncodeInput::Text(s));
    }
    if let Ok(b) = obj.extract::<PyBackedBytes>() {
        return Ok(EncodeInput::Bytes(b));
    }
    let hint = if obj.extract::<PathBuf>().is_ok() {
        "; to encode files, use encode_files"
    } else {
        ""
    };
    Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
        "expected str or bytes, got {}{hint}",
        obj.get_type()
    )))
}

/// decode() input resolved to a `TokenId` slice. A numpy uint32 array is
/// borrowed in place (`TokenId` is repr(transparent) over u32); other
/// integer dtypes are converted in one checked pass; anything else falls
/// back to generic per-element sequence extraction.
pub(crate) enum TokenIds<'py> {
    Borrowed(PyReadonlyArray1<'py, u32>),
    Owned(Vec<TokenId>),
}

impl TokenIds<'_> {
    pub(crate) fn as_slice(&self) -> PyResult<&[TokenId]> {
        match self {
            TokenIds::Borrowed(arr) => {
                let ids: &[u32] = arr.as_slice()?;
                // SAFETY: TokenId is #[repr(transparent)] over u32.
                Ok(unsafe { std::slice::from_raw_parts(ids.as_ptr().cast(), ids.len()) })
            }
            TokenIds::Owned(ids) => Ok(ids),
        }
    }
}

/// Convert a non-u32 integer numpy array in one pass, rejecting values that
/// do not fit a token ID.
fn cast_token_ids<T>(arr: PyReadonlyArray1<'_, T>) -> PyResult<Vec<TokenId>>
where
    T: numpy::Element + Copy + TryInto<u32> + std::fmt::Display,
{
    arr.as_array()
        .iter()
        .map(|&t| {
            t.try_into().map(TokenId).map_err(|_| {
                PyErr::new::<pyo3::exceptions::PyOverflowError, _>(format!(
                    "token id {t} does not fit in uint32"
                ))
            })
        })
        .collect()
}

/// Extract decode() input: a numpy integer array or any sequence of ints.
pub(crate) fn extract_token_ids<'py>(tokens: &Bound<'py, PyAny>) -> PyResult<TokenIds<'py>> {
    if let Ok(arr) = tokens.cast::<PyArray1<u32>>() {
        let arr = arr.readonly();
        return Ok(if arr.as_slice().is_ok() {
            TokenIds::Borrowed(arr)
        } else {
            // Non-contiguous (e.g. a strided view): gather instead.
            TokenIds::Owned(arr.as_array().iter().map(|&t| t.into()).collect())
        });
    }
    // Other integer dtypes (e.g. int64 model outputs) get a single
    // vectorized pass instead of per-element Python iteration.
    if let Ok(arr) = tokens.cast::<PyArray1<i64>>() {
        return Ok(TokenIds::Owned(cast_token_ids(arr.readonly())?));
    }
    if let Ok(arr) = tokens.cast::<PyArray1<i32>>() {
        return Ok(TokenIds::Owned(cast_token_ids(arr.readonly())?));
    }
    if let Ok(arr) = tokens.cast::<PyArray1<u64>>() {
        return Ok(TokenIds::Owned(cast_token_ids(arr.readonly())?));
    }
    Ok(TokenIds::Owned(
        tokens
            .extract::<Vec<u32>>()?
            .into_iter()
            .map(Into::into)
            .collect(),
    ))
}

/// Build a `vocab` getter's dict from `(id, bytes)` entries.
pub(crate) fn vocab_to_pydict<'py, 'a>(
    py: Python<'py>,
    entries: impl Iterator<Item = (u32, &'a [u8])>,
) -> PyResult<Bound<'py, PyDict>> {
    entries
        .map(|(id, bytes)| (id, PyBytes::new(py, bytes)))
        .into_py_dict(py)
}

/// Build a `merges` getter's list from `(left, right)` byte pairs.
pub(crate) fn merges_to_pylist<'py>(
    py: Python<'py>,
    entries: Vec<(&[u8], &[u8])>,
) -> Vec<(Bound<'py, PyBytes>, Bound<'py, PyBytes>)> {
    entries
        .into_iter()
        .map(|(a, b)| (PyBytes::new(py, a), PyBytes::new(py, b)))
        .collect()
}

/// Flat uint8 content array and per-document byte counts.
type FlatDocs<'py> = (
    Bound<'py, numpy::PyArray1<u8>>,
    Bound<'py, numpy::PyArray1<i64>>,
);

/// If `inputs` is an awkward Array of strings or bytestrings, pull out its
/// flat uint8 content and per-document counts directly — no per-document
/// Python objects are materialized. Returns None when `inputs` is not an
/// awkward Array (or awkward is not importable).
fn extract_awkward_docs<'py>(inputs: &Bound<'py, PyAny>) -> PyResult<Option<FlatDocs<'py>>> {
    let py = inputs.py();
    let Ok(ak) = py.import("awkward") else {
        return Ok(None);
    };
    if !inputs.is_instance(&ak.getattr("Array")?)? {
        return Ok(None);
    }
    let type_err = |_| {
        PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "awkward input must be an array of strings or bytestrings",
        )
    };
    // Stripping the string/bytestring parameters turns the array into plain
    // lists of uint8, whose flattened content and row lengths are views of
    // the existing buffers.
    let raw = ak
        .call_method1("without_parameters", (inputs,))
        .map_err(type_err)?;
    let flat = ak.call_method1("flatten", (&raw,)).map_err(type_err)?;
    let content = ak
        .call_method1("to_numpy", (flat,))?
        .cast_into::<PyArray1<u8>>()
        .map_err(|e| type_err(e.into()))?;
    let counts = ak
        .call_method1("to_numpy", (ak.call_method1("num", (&raw,))?,))?
        .cast_into::<PyArray1<i64>>()
        .map_err(|e| type_err(e.into()))?;
    Ok(Some((content, counts)))
}

/// Hand a ragged token batch to Python as an `awkward.Array`: one flat
/// contents array plus per-document counts — two allocations total instead
/// of one numpy array per document. Falls back to a list of zero-copy numpy
/// views when awkward is not importable.
pub(crate) fn ragged_to_python<'py>(
    py: Python<'py>,
    flat: Vec<u32>,
    counts: Vec<i64>,
) -> PyResult<Bound<'py, PyAny>> {
    let n_rows = counts.len();
    let content = flat.into_pyarray(py);
    let counts = counts.into_pyarray(py);
    match py.import("awkward") {
        Ok(ak) => ak.call_method1("unflatten", (content, counts)),
        Err(_) => {
            if n_rows == 0 {
                return Ok(pyo3::types::PyList::empty(py).into_any());
            }
            let np = py.import("numpy")?;
            let bounds = np.call_method1("cumsum", (&counts,))?;
            let split_at = bounds.get_item(pyo3::types::PySlice::new(py, 0, -1, 1))?;
            np.call_method1("split", (content, split_at))
        }
    }
}

/// Format for every non-BytesSource encode_batch input: each byte region
/// handed to `encode` is exactly one document.
const WHOLE_DOCS: DocFormat = DocFormat::Text { separator: None };

/// Shared front-end of encode_batch: extract the documents (a BytesSource,
/// a list of str, a list of bytes, or an awkward Array of strings — whose
/// flat buffer is used directly, with no per-document Python objects) and
/// run `encode` on the resolved byte slices with the GIL released,
/// returning the ragged result as one flat id buffer plus per-document row
/// lengths. `encode`'s `DocFormat` argument says how its byte regions split
/// into documents: one document per region for everything except a
/// BytesSource, whose buffers carry its separator format and are split
/// during the encode itself. Bytes-shaped inputs are trusted to be valid
/// UTF-8 by the consumers that care (the SentencePiece backend) — nothing
/// is validated here or downstream.
pub(crate) fn encode_batch_flat<'py>(
    py: Python<'py>,
    inputs: &Bound<'py, PyAny>,
    encode: impl Fn(&[&[u8]], &DocFormat) -> PyResult<(Vec<u32>, Vec<i64>)> + Send + Sync,
) -> PyResult<(Vec<u32>, Vec<i64>)> {
    // BytesSource: hand the borrowed buffers over as byte regions together
    // with the source's separator format.
    if let Ok(source) = inputs.cast::<super::sources::BytesSource>() {
        let source = source.get();
        return py.detach(|| {
            let regions: Vec<&[u8]> = source.buffers.iter().map(|b| &**b).collect();
            encode(&regions, &source.format)
        });
    }

    // Awkward input: encode straight from the flat content buffer.
    if let Some((content, in_counts)) = extract_awkward_docs(inputs)? {
        let content = content.readonly();
        let bytes: &[u8] = content.as_slice()?;
        let in_counts = in_counts.readonly();
        let in_counts: &[i64] = in_counts.as_slice()?;
        return py.detach(|| -> PyResult<_> {
            let mut docs = Vec::with_capacity(in_counts.len());
            let mut pos = 0usize;
            for &n in in_counts {
                docs.push(&bytes[pos..pos + n as usize]);
                pos += n as usize;
            }
            encode(&docs, &WHOLE_DOCS)
        });
    }

    let inputs: Vec<Bound<'py, PyAny>> = inputs.extract().map_err(|_| {
        PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "expected a list of str, a list of bytes, a BytesSource, or an awkward Array of strings",
        )
    })?;
    if inputs.is_empty() {
        return Ok((vec![], vec![]));
    }
    let mut docs = Vec::with_capacity(inputs.len());
    docs.push(extract_doc(&inputs[0])?);
    for obj in &inputs[1..] {
        let doc = extract_doc(obj)?;
        if std::mem::discriminant(&doc) != std::mem::discriminant(&docs[0]) {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "all documents in a batch must be of the same type \
                 (a list of str or a list of bytes)",
            ));
        }
        docs.push(doc);
    }
    py.detach(|| {
        let slices: Vec<&[u8]> = docs.iter().map(|d| d.as_bytes()).collect();
        encode(&slices, &WHOLE_DOCS)
    })
}

/// encode_batch: `encode_batch_flat`, handed to Python as an awkward.Array.
pub(crate) fn encode_batch_ragged<'py>(
    py: Python<'py>,
    inputs: &Bound<'py, PyAny>,
    encode: impl Fn(&[&[u8]], &DocFormat) -> PyResult<(Vec<u32>, Vec<i64>)> + Send + Sync,
) -> PyResult<Bound<'py, PyAny>> {
    let (flat, counts) = encode_batch_flat(py, inputs, encode)?;
    ragged_to_python(py, flat, counts)
}

/// How `_encode_batch_list_compat` assembles each row, plus its fused
/// forbidden-specials scan — the compat wrappers' per-call options, bundled
/// like `PadTruncate` so the entrypoint stays one argument wide. A frozen
/// pyclass: fields are validated once at construction, and each call costs
/// a typed downcast rather than a keyword list. Underscore-named on the
/// Python side: an implementation detail of the compat layer.
#[pyclass(frozen, name = "_WrapTruncate")]
pub(crate) struct WrapTruncate {
    /// Token ids written before / after every row's tokens.
    prefix: Vec<u32>,
    suffix: Vec<u32>,
    /// Keep at most this many encoded ids per row, not counting
    /// prefix/suffix (the caller has already budgeted for those).
    max_tokens: Option<usize>,
    /// Drop ids from the start of a row instead of the end.
    truncate_left: bool,
    /// Scan every document for these patterns before encoding and raise
    /// SpecialTokenFound on any hit (the tiktoken-compat specials check).
    forbid: Option<Py<crate::bindings::matcher::SubstringMatcher>>,
}

#[pymethods]
impl WrapTruncate {
    #[new]
    #[pyo3(signature = (*, prefix = Vec::new(), suffix = Vec::new(), max_tokens = None, truncate_left = false, forbid = None))]
    fn new(
        prefix: Vec<u32>,
        suffix: Vec<u32>,
        max_tokens: Option<usize>,
        truncate_left: bool,
        forbid: Option<Py<crate::bindings::matcher::SubstringMatcher>>,
    ) -> Self {
        Self {
            prefix,
            suffix,
            max_tokens,
            truncate_left,
            forbid,
        }
    }
}

/// encode_batch assembled into plain Python lists in Rust — one list of ints
/// per document, built directly from the flat ragged buffers — for callers
/// that need lists, which would otherwise convert the awkward result one
/// Python object at a time. `options` carries the compat wrappers' row
/// assembly: optional truncation to `max_tokens` ids (from the left when
/// `truncate_left`), wrapping in `prefix`/`suffix`, and the fused `forbid`
/// scan — every document is first checked for the matcher's patterns (in
/// parallel over documents when `parallel` is set, still with the GIL
/// released) and SpecialTokenFound is raised on any hit. `None` means plain
/// rows.
pub(crate) fn encode_batch_pylist<'py>(
    py: Python<'py>,
    inputs: &Bound<'py, PyAny>,
    options: Option<&WrapTruncate>,
    parallel: bool,
    encode: impl Fn(&[&[u8]], &DocFormat) -> PyResult<(Vec<u32>, Vec<i64>)> + Send + Sync,
) -> PyResult<Bound<'py, PyList>> {
    static PLAIN: WrapTruncate = WrapTruncate {
        prefix: Vec::new(),
        suffix: Vec::new(),
        max_tokens: None,
        truncate_left: false,
        forbid: None,
    };
    let opts = options.unwrap_or(&PLAIN);
    let (prefix, suffix) = (opts.prefix.as_slice(), opts.suffix.as_slice());
    let truncate_left = opts.truncate_left;
    let (flat, counts) = encode_batch_flat(py, inputs, |docs, format| {
        if let Some(matcher) = &opts.forbid {
            matcher.get().scan_docs(docs, parallel)?;
        }
        encode(docs, format)
    })?;
    let cap = opts.max_tokens.unwrap_or(usize::MAX);
    let wrap = !prefix.is_empty() || !suffix.is_empty();
    let mut row_buf: Vec<u32> = Vec::new();
    let mut rows = Vec::with_capacity(counts.len());
    let mut pos = 0usize;
    for &n in &counts {
        let n = n as usize;
        let tokens = &flat[pos..pos + n];
        pos += n;
        let kept = n.min(cap);
        let tokens = if truncate_left {
            &tokens[n - kept..]
        } else {
            &tokens[..kept]
        };
        if wrap {
            row_buf.clear();
            row_buf.extend_from_slice(prefix);
            row_buf.extend_from_slice(tokens);
            row_buf.extend_from_slice(suffix);
            rows.push(PyList::new(py, &row_buf)?);
        } else {
            // Nothing to wrap (the tiktoken path): build the row straight
            // from the flat slice, skipping the row_buf copy.
            rows.push(PyList::new(py, tokens)?);
        }
    }
    PyList::new(py, rows)
}
