//! `Gigatoken::Native::BPETokenizer`: a `gigatoken::Tokenizer` plus the
//! `WorkerPool` that backs batch encoding. Mirrors the pyo3 `BPETokenizer` in
//! the core crate's `src/lib.rs` (the `python` feature), minus the
//! numpy/awkward-array machinery that has no Ruby analog.

use std::cell::RefCell;
use std::os::raw::c_long;

use gigatoken_rs::load_tokenizer::hf::HfTokenizer;
use gigatoken_rs::load_tokenizer::{hf, tiktoken};
use gigatoken_rs::{
    GatherBuf, GatherOutcome, Tokenizer, WorkerPool, encode_docs_into, encode_docs_ragged,
    encode_files_docs, encode_files_docs_serial,
};
use magnus::{
    Error, RArray, RClass, RHash, RModule, RString, Ruby, Value, function, method, prelude::*,
    scan_args::{get_kwargs, scan_args},
};
use rb_sys::{RSTRING_PTR, rb_str_set_len};

use crate::error::raise;
use crate::gvl::without_gvl;
use crate::sources;

pub(crate) fn binary_string(ruby: &Ruby, bytes: &[u8]) -> RString {
    ruby.enc_str_new(bytes, ruby.ascii8bit_encoding())
}

/// Reinterpret a magnus `RString` as the raw `rb_sys::VALUE` it wraps, for
/// direct rb-sys calls magnus doesn't expose (`RSTRING_PTR`/`rb_str_set_len`
/// below). Sound because every magnus value type — `RString` included — is
/// `#[repr(transparent)]` down to the same `VALUE` rb-sys itself uses
/// (verified against the installed magnus-0.8.2 source: `RString(NonZeroValue)`,
/// `NonZeroValue(NonZeroUsize, ..)`, both `#[repr(transparent)]`); the
/// compiler enforces the size match this transmute depends on. Magnus does
/// expose this conversion directly (`magnus::rb_sys::AsRawValue`), but only
/// behind its own `rb-sys` Cargo feature, which isn't enabled — turning it
/// on would mean editing `ext/gigatoken/Cargo.toml`'s magnus dependency,
/// which belongs to the other lane.
fn raw_value(s: RString) -> rb_sys::VALUE {
    unsafe { std::mem::transmute(s) }
}

/// Reinterpret a `Vec<u32>` as raw bytes in the host's native byte order.
/// Safe: `u32` has no padding or niches, so any of its byte patterns is a
/// valid `u8`, and `u8`'s alignment (1) never exceeds `u32`'s.
fn u32_vec_as_bytes(values: &[u32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values)) }
}

/// Marshal a ragged `(flat, lens)` encode result into a ragged Ruby Array of
/// Arrays, one per document — the shape `encode_batch`/`encode_files` return.
pub(crate) fn ragged_result(ruby: &Ruby, flat: Vec<u32>, lens: Vec<i64>) -> Result<RArray, Error> {
    let result = ruby.ary_new_capa(lens.len());
    let mut offset = 0usize;
    for len in lens {
        let len = len as usize;
        result.push(flat[offset..offset + len].to_vec())?;
        offset += len;
    }
    Ok(result)
}

/// Freeze `string` (already ASCII-8BIT-encoded and holding exactly the
/// packed token bytes, native byte order — every realistic Ruby platform is
/// little-endian, and `IO::Buffer#get_value(s)`'s own `:u32` format is
/// always little-endian, so the two agree without any byte-swapping) and
/// wrap it zero-copy as `[IO::Buffer, lens]` (per Ruby's own docs: "If the
/// string is frozen, it will create a read-only buffer which cannot be
/// modified"). magnus 0.8 has no `IO::Buffer` wrapper (verified against the
/// installed magnus-0.8.2 source), so the buffer is built via `funcall`.
/// `lens` is a plain per-document Ruby Array — small, ergonomics over
/// purity. Shared by `packed_result` (the copy path) and the BPE packed
/// paths' zero-copy gather (`BPETokenizer::encode_batch_packed`).
fn finish_packed(ruby: &Ruby, string: RString, lens: Vec<i64>) -> Result<RArray, Error> {
    string.freeze();
    let io_buffer: RClass = ruby
        .class_object()
        .const_get::<_, RClass>("IO")?
        .const_get("Buffer")?;
    let buffer: Value = io_buffer.funcall("for", (string,))?;

    let lens_ary = ruby.ary_new_capa(lens.len());
    for len in lens {
        lens_ary.push(len)?;
    }

    let result = ruby.ary_new_capa(2);
    result.push(buffer)?;
    result.push(lens_ary)?;
    Ok(result)
}

/// Marshal a ragged `(flat, lens)` encode result into `[IO::Buffer, lens]`
/// by copying `flat`'s raw bytes into a fresh `RString` (see
/// `finish_packed`). Used directly by the SentencePiece packed paths (whose
/// gather has no overlapped-commit machinery to hand a destination to — see
/// `encode_chunks_into` in the core crate) and as the BPE packed paths'
/// fallback when the zero-copy gather's reservation overruns (see
/// `BPETokenizer::encode_batch_packed`).
pub(crate) fn packed_result(ruby: &Ruby, flat: Vec<u32>, lens: Vec<i64>) -> Result<RArray, Error> {
    let string = binary_string(ruby, u32_vec_as_bytes(&flat));
    finish_packed(ruby, string, lens)
}

#[magnus::wrap(class = "Gigatoken::Native::BPETokenizer", free_immediately, size)]
pub struct BPETokenizer {
    tokenizer: RefCell<Tokenizer>,
    workers: WorkerPool,
}

impl BPETokenizer {
    pub(crate) fn from_tokenizer(tokenizer: Tokenizer) -> Self {
        Self {
            tokenizer: RefCell::new(tokenizer),
            workers: WorkerPool::new(),
        }
    }

    fn from_hf_json(ruby: &Ruby, data: RString) -> Result<Self, Error> {
        // SAFETY: the slice is only read, and only for the duration of this
        // synchronous call (no GVL release, no Ruby allocation in between).
        let bytes = unsafe { data.as_slice() };
        match hf::load_hf_slice(bytes) {
            Ok(HfTokenizer::Bpe(tokenizer)) => Ok(Self::from_tokenizer(tokenizer)),
            Ok(HfTokenizer::SentencePiece(_)) => Err(raise(
                ruby,
                "SentencePiece tokenizer.json data loads as a SentencePieceTokenizer, not a \
                 BPETokenizer — use Gigatoken::Native.load_hf_json instead",
            )),
            Err(e) => Err(raise(ruby, e.to_string())),
        }
    }

    fn from_tiktoken(ruby: &Ruby, path: String) -> Result<Self, Error> {
        match tiktoken::load_tiktoken(&path) {
            Ok(tokenizer) => Ok(Self::from_tokenizer(tokenizer)),
            Err(e) => Err(raise(ruby, e.to_string())),
        }
    }

    fn encode(&self, input: RString) -> Vec<u32> {
        // SAFETY: read-only, for the duration of this synchronous call.
        let bytes = unsafe { input.as_slice() };
        let mut out = Vec::new();
        self.tokenizer
            .borrow_mut()
            .encode_with_added_tokens_flat(bytes, &mut out);
        out
    }

    /// Encode a batch on the core worker pool, with the GVL released for the
    /// parallel encode itself (see `gvl::without_gvl`). Every input string is
    /// copied into an owned buffer before release: nothing Ruby-managed may
    /// be touched once the GVL is gone.
    fn encode_batch_ragged(rb_self: &Self, inputs: RArray) -> Result<(Vec<u32>, Vec<i64>), Error> {
        // SAFETY: values are read (checked-converted to `RString`, then
        // copied into owned buffers below) before anything else runs.
        let docs: Vec<Vec<u8>> = unsafe { inputs.as_slice() }
            .iter()
            .map(|&v| RString::try_convert(v).map(|s| unsafe { s.as_slice() }.to_vec()))
            .collect::<Result<_, _>>()?;
        let doc_slices: Vec<&[u8]> = docs.iter().map(Vec::as_slice).collect();
        let tokenizer = rb_self.tokenizer.borrow();
        let tokenizer: &Tokenizer = &tokenizer;
        let workers = &rb_self.workers;
        Ok(without_gvl(|| encode_docs_ragged(workers, tokenizer, &doc_slices)))
    }

    fn encode_batch(ruby: &Ruby, rb_self: &Self, inputs: RArray) -> Result<RArray, Error> {
        let (flat, lens) = Self::encode_batch_ragged(rb_self, inputs)?;
        ragged_result(ruby, flat, lens)
    }

    /// The packed analog of `encode_batch`: gathers straight into a
    /// pre-allocated Ruby string's backing store instead of copying `flat`
    /// into a fresh one afterward (see `packed_result`), eliminating the
    /// packed path's whole-result copy. The destination is allocated with
    /// the GVL still held (Ruby object allocation requires it — this is why
    /// `encode_files_packed` cannot take the same route: its total input
    /// size is only known after loading files, which happens inside a
    /// single `without_gvl` call in `sources::encode_files_ragged`) and
    /// stays unexposed to Ruby (not frozen, not wrapped) until the gather
    /// below has fully completed, so nothing else can observe it mid-write.
    /// `string` itself is never captured by the `without_gvl` closure (only
    /// the raw `dest` pointer is — see `GatherBuf`'s `Send` impl); it stays
    /// alive as an ordinary local across the call, conservatively reachable
    /// from this frame the same way any pointer obtained via `RSTRING_PTR`
    /// and used across a released-GVL blocking call already is (the pattern
    /// Ruby's own C extensions use for e.g. blocking `read(2)` into a
    /// string's buffer). Falls back to `packed_result`'s copy path on the
    /// same NFC-expansion overflow escape `encode_docs_into` documents.
    fn encode_batch_packed(ruby: &Ruby, rb_self: &Self, inputs: RArray) -> Result<RArray, Error> {
        // SAFETY: values are read (checked-converted to `RString`, then
        // copied into owned buffers below) before anything else runs.
        let docs: Vec<Vec<u8>> = unsafe { inputs.as_slice() }
            .iter()
            .map(|&v| RString::try_convert(v).map(|s| unsafe { s.as_slice() }.to_vec()))
            .collect::<Result<_, _>>()?;
        let doc_slices: Vec<&[u8]> = docs.iter().map(Vec::as_slice).collect();
        let total_bytes: usize = doc_slices.iter().map(|d| d.len()).sum();

        // A token consumes >= 1 input byte, so total_bytes tokens (4 bytes
        // each) is the same reservation bound the core's owned gather uses
        // — see `encode_docs_into`. `str_buf_new` is already "binary"
        // (ASCII-8BIT) encoded per Ruby's own C API docs for
        // rb_str_buf_new (ruby/internal/intern/string.h).
        let string = ruby.str_buf_new(total_bytes * std::mem::size_of::<u32>());
        // SAFETY: `string` was just allocated with exactly this many tokens
        // of capacity and is not yet exposed to Ruby (not returned, not
        // frozen, not wrapped), so nothing else can read or write through it
        // concurrently.
        let ptr = unsafe { RSTRING_PTR(raw_value(string)) as *mut u32 };
        // SAFETY: `ptr` is valid for `total_bytes` disjoint u32 writes for
        // the duration of the gather below (see the allocation above).
        let dest = unsafe { GatherBuf::new(ptr, total_bytes) };

        let tokenizer = rb_self.tokenizer.borrow();
        let tokenizer: &Tokenizer = &tokenizer;
        let workers = &rb_self.workers;
        match without_gvl(|| encode_docs_into(workers, tokenizer, &doc_slices, dest)) {
            GatherOutcome::Committed(total_tokens, lens) => {
                // SAFETY: `encode_docs_into` only returns `Committed` once
                // every one of `total_tokens` u32s at `ptr` has been
                // written.
                unsafe {
                    rb_str_set_len(
                        raw_value(string),
                        (total_tokens * std::mem::size_of::<u32>()) as c_long,
                    );
                }
                finish_packed(ruby, string, lens)
            }
            // The reservation overran (NFC-expansion pathologies) — `string`
            // is discarded unused; fall back to the classic copy path.
            GatherOutcome::Fallback(flat, lens) => packed_result(ruby, flat, lens),
        }
    }

    /// Encode every document named by `source` (a TextFileSource,
    /// JsonlFileSource, or ParquetFileSource) with the GVL released for the
    /// whole run — loading the files and the encode itself. Files are cut
    /// into chunks at document boundaries and encoded by the core worker
    /// pool in one fused pass (`batch::encode_files_docs`, the same core the
    /// pyo3 bindings' `encode_files` uses — see `sources::encode_files_ragged`).
    /// `parallel: false` loads and encodes everything on the calling thread
    /// instead, with identical output, never touching the worker pool.
    fn encode_files_ragged(ruby: &Ruby, rb_self: &Self, args: &[Value]) -> Result<(Vec<u32>, Vec<i64>), Error> {
        let args = scan_args::<(Value,), (), (), (), RHash, ()>(args)?;
        let (source,) = args.required;
        let kw = get_kwargs::<_, (), (Option<Value>,), ()>(args.keywords, &[], &["parallel"])?;
        let (parallel,) = kw.optional;
        let parallel = match parallel {
            Some(v) if !v.is_nil() => bool::try_convert(v)?,
            _ => true,
        };

        let source = sources::resolve(ruby, source)?;
        let tokenizer = rb_self.tokenizer.borrow();
        let tokenizer: &Tokenizer = &tokenizer;
        let workers = &rb_self.workers;
        let encoded: std::io::Result<(Vec<u32>, Vec<i64>)> = without_gvl(|| {
            sources::encode_files_ragged(&source, parallel, |files, format| {
                Ok(if parallel {
                    encode_files_docs(workers, tokenizer, files, format)
                } else {
                    encode_files_docs_serial(workers, tokenizer, files, format)
                })
            })
        });
        encoded.map_err(|e| raise(ruby, e.to_string()))
    }

    fn encode_files(ruby: &Ruby, rb_self: &Self, args: &[Value]) -> Result<RArray, Error> {
        let (flat, lens) = Self::encode_files_ragged(ruby, rb_self, args)?;
        ragged_result(ruby, flat, lens)
    }

    /// The packed analog of `encode_files`: same encode core (including the
    /// `parallel:` nil-equals-omitted contract), but the flat token ids are
    /// handed to Ruby as one `IO::Buffer` instead of being re-chunked into
    /// per-document Arrays (see `packed_result`).
    fn encode_files_packed(ruby: &Ruby, rb_self: &Self, args: &[Value]) -> Result<RArray, Error> {
        let (flat, lens) = Self::encode_files_ragged(ruby, rb_self, args)?;
        packed_result(ruby, flat, lens)
    }

    fn decode(ruby: &Ruby, rb_self: &Self, tokens: RArray) -> Result<RString, Error> {
        let ids: Vec<u32> = tokens.to_vec()?;
        let ids: Vec<_> = ids.into_iter().map(Into::into).collect();
        let bytes: Vec<u8> = rb_self.tokenizer.borrow().decode(&ids).collect();
        Ok(binary_string(ruby, &bytes))
    }

    fn vocab_size(&self) -> usize {
        self.tokenizer.borrow().vocab_size()
    }

    fn vocab(ruby: &Ruby, rb_self: &Self) -> Result<RHash, Error> {
        let tokenizer = rb_self.tokenizer.borrow();
        let hash = ruby.hash_new();
        for (id, bytes) in tokenizer.vocab_entries() {
            hash.aset(id, binary_string(ruby, bytes))?;
        }
        Ok(hash)
    }

    fn merges(ruby: &Ruby, rb_self: &Self) -> Result<RArray, Error> {
        let tokenizer = rb_self.tokenizer.borrow();
        let entries = tokenizer.merge_entries();
        let result = ruby.ary_new_capa(entries.len());
        for (a, b) in entries {
            result.push((binary_string(ruby, a), binary_string(ruby, b)))?;
        }
        Ok(result)
    }
}

pub fn init(ruby: &Ruby, native: RModule) -> Result<(), Error> {
    let class: RClass = native.define_class("BPETokenizer", ruby.class_object())?;
    class.define_singleton_method("from_hf_json", function!(BPETokenizer::from_hf_json, 1))?;
    class.define_singleton_method("from_tiktoken", function!(BPETokenizer::from_tiktoken, 1))?;
    class.define_method("encode", method!(BPETokenizer::encode, 1))?;
    class.define_method("encode_batch", method!(BPETokenizer::encode_batch, 1))?;
    class.define_method("encode_batch_packed", method!(BPETokenizer::encode_batch_packed, 1))?;
    class.define_method("encode_files", method!(BPETokenizer::encode_files, -1))?;
    class.define_method("encode_files_packed", method!(BPETokenizer::encode_files_packed, -1))?;
    class.define_method("decode", method!(BPETokenizer::decode, 1))?;
    class.define_method("vocab_size", method!(BPETokenizer::vocab_size, 0))?;
    class.define_method("vocab", method!(BPETokenizer::vocab, 0))?;
    class.define_method("merges", method!(BPETokenizer::merges, 0))?;
    Ok(())
}
