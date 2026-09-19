//! `Gigatoken::Native::SentencePieceTokenizer`: a `gigatoken::SentencePieceBPE`
//! plus its persistent pretoken-cache `EncodeState`, and no `WorkerPool` —
//! the `sp_*` cores parallelize internally instead of through a pool.
//! Mirrors the pyo3 `SentencePieceTokenizer` in the core crate's `src/lib.rs`
//! (the `python` feature), minus the numpy/awkward-array machinery that has
//! no Ruby analog; marshaling helpers are shared with
//! `Gigatoken::Native::BPETokenizer` (see `crate::tokenizer`).
//!
//! Unlike the pyo3 binding, which trusts Python's `str` guarantee, every
//! document/file byte region handed in from Ruby is validated as UTF-8
//! before it reaches the `&str`-typed SentencePiece cores — a Ruby String
//! carries no such guarantee even when tagged UTF-8 — raising
//! `Gigatoken::Error` instead of ever calling `str::from_utf8_unchecked` on
//! Ruby-supplied bytes.

use std::sync::Mutex;

use gigatoken_rs::input::file_source::DocFormat;
use gigatoken_rs::{EncodeState, SentencePieceBPE, sp_encode_docs_ragged, sp_encode_files_docs, sp_encode_files_docs_serial};
use magnus::{
    Error, RArray, RClass, RHash, RModule, RString, Ruby, Value, method, prelude::*,
    scan_args::{get_kwargs, scan_args},
};

use crate::error::raise;
use crate::gvl::without_gvl;
use crate::sources;
use crate::tokenizer::{
    binary_string, packed_result, ragged_result, snapshot_entry, snapshot_inputs,
};

/// Validate that `bytes` is UTF-8, raising `Gigatoken::Error` otherwise.
fn require_utf8<'a>(ruby: &Ruby, bytes: &'a [u8]) -> Result<&'a str, Error> {
    std::str::from_utf8(bytes).map_err(|e| raise(ruby, format!("invalid UTF-8: {e}")))
}

#[magnus::wrap(class = "Gigatoken::Native::SentencePieceTokenizer", free_immediately, size)]
pub struct SentencePieceTokenizer {
    // `SentencePieceBPE`'s encode methods take `&self` (only `EncodeState`
    // is mutated), so the model needs no interior mutability at all — every
    // path here reads it, including the ones that release the GVL.
    tokenizer: SentencePieceBPE,
    // The one mutable piece. A `Mutex` rather than a `RefCell` so the wrapped
    // object is `Sync`: Ruby hands the same instance to every thread, and a
    // `RefCell` shared that way is unsound. It is only ever taken with the
    // GVL held — the batch paths build their own per-call encoders — so it
    // cannot actually block, which is what `BPETokenizer`'s invariant asks
    // of any lock on this side of the boundary.
    state: Mutex<EncodeState>,
}

impl SentencePieceTokenizer {
    /// Wrap a loaded model, applying the global cache budget to it and to
    /// the single-document encode state — mirrors pyo3's `with_model`.
    pub(crate) fn from_tokenizer(tokenizer: SentencePieceBPE) -> Self {
        let tokenizer = crate::cache::apply_max_cache_bytes_sp(tokenizer);
        let state = EncodeState::with_budget(tokenizer.max_cache_bytes());
        Self {
            tokenizer,
            state: Mutex::new(state),
        }
    }

    fn encode(ruby: &Ruby, rb_self: &Self, input: RString) -> Result<Vec<u32>, Error> {
        // SAFETY: read-only, for the duration of this synchronous call.
        let bytes = unsafe { input.as_slice() };
        let text = require_utf8(ruby, bytes)?;
        let mut ids: Vec<u32> = Vec::new();
        let mut state = rb_self.state.lock().unwrap_or_else(|e| e.into_inner());
        rb_self.tokenizer.encode_raw_cb(&mut state, text, &mut |tokens| {
            ids.extend(tokens.iter().map(|&t| u32::from(t)))
        });
        Ok(ids)
    }

    /// Encode a batch on the core's internal (rayon) parallelism, with the
    /// GVL released — like `BPETokenizer#encode_batch`, always parallel
    /// (only `encode_files` exposes a `parallel:` toggle). Every input
    /// string is validated as UTF-8 and copied into an owned `String`
    /// before release: nothing Ruby-managed may be touched once the GVL is
    /// gone.
    ///
    /// The walk reads a [`snapshot_inputs`] copy of the caller's array, one
    /// slot at a time — never a Rust borrow of the array's buffer held
    /// across `try_convert`, which runs `to_str` and with it arbitrary Ruby
    /// code, including code that re-homes the array out from under the
    /// walk. Same contract as the BPE path: the result reflects the array
    /// as it was at entry.
    fn encode_batch_ragged(ruby: &Ruby, rb_self: &Self, inputs: RArray) -> Result<(Vec<u32>, Vec<i64>), Error> {
        let snapshot = snapshot_inputs(inputs);
        let docs: Vec<String> = (0..snapshot.len())
            .map(|i| {
                let s = RString::try_convert(snapshot_entry(snapshot, i))?;
                // SAFETY: validated and copied into an owned buffer right
                // here, with no Ruby call in between to invalidate it.
                let bytes = unsafe { s.as_slice() };
                require_utf8(ruby, bytes).map(str::to_owned)
            })
            .collect::<Result<_, _>>()?;
        let doc_refs: Vec<&str> = docs.iter().map(String::as_str).collect();
        let tokenizer: &SentencePieceBPE = &rb_self.tokenizer;
        without_gvl(|| sp_encode_docs_ragged(tokenizer, &doc_refs))
    }

    fn encode_batch(ruby: &Ruby, rb_self: &Self, inputs: RArray) -> Result<RArray, Error> {
        let (flat, lens) = Self::encode_batch_ragged(ruby, rb_self, inputs)?;
        ragged_result(ruby, flat, lens)
    }

    /// The packed analog of `encode_batch`: same encode core, but the flat
    /// token ids are handed to Ruby as one `IO::Buffer` instead of being
    /// re-chunked into per-document Arrays (see `tokenizer::packed_result`).
    fn encode_batch_packed(ruby: &Ruby, rb_self: &Self, inputs: RArray) -> Result<RArray, Error> {
        let (flat, lens) = Self::encode_batch_ragged(ruby, rb_self, inputs)?;
        packed_result(ruby, flat, lens)
    }

    /// Encode every document named by `source` in parallel, with the GVL
    /// released for the whole run — loading the files and the encode
    /// itself. A `separator:` on a `TextFileSource` must be valid UTF-8
    /// (checked up front, like the pyo3 binding's own separator check):
    /// an arbitrary byte separator could cut a document mid-character and
    /// break the SentencePiece cores' `&str` contract. `parallel: false`
    /// loads and encodes everything on the calling thread instead, with
    /// identical output, never touching rayon.
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
        if let DocFormat::Text { separator: Some(sep) } = &source.format {
            if std::str::from_utf8(sep).is_err() {
                return Err(raise(
                    ruby,
                    "the SentencePiece backend requires a separator that is valid UTF-8",
                ));
            }
        }

        let tokenizer: &SentencePieceBPE = &rb_self.tokenizer;
        let encoded = without_gvl(|| {
            sources::encode_files_ragged(&source, parallel, |files, format| {
                for &region in files {
                    std::str::from_utf8(region).map_err(|e| {
                        std::io::Error::new(std::io::ErrorKind::InvalidData, format!("invalid UTF-8 in file contents: {e}"))
                    })?;
                }
                Ok(if parallel {
                    sp_encode_files_docs(tokenizer, files, format)
                } else {
                    sp_encode_files_docs_serial(tokenizer, files, format)
                })
            })
        })?;
        encoded.map_err(|e| raise(ruby, e.to_string()))
    }

    fn encode_files(ruby: &Ruby, rb_self: &Self, args: &[Value]) -> Result<RArray, Error> {
        let (flat, lens) = Self::encode_files_ragged(ruby, rb_self, args)?;
        ragged_result(ruby, flat, lens)
    }

    /// The packed analog of `encode_files`: same encode core (including the
    /// `parallel:` nil-equals-omitted contract), but the flat token ids are
    /// handed to Ruby as one `IO::Buffer` instead of being re-chunked into
    /// per-document Arrays (see `tokenizer::packed_result`).
    fn encode_files_packed(ruby: &Ruby, rb_self: &Self, args: &[Value]) -> Result<RArray, Error> {
        let (flat, lens) = Self::encode_files_ragged(ruby, rb_self, args)?;
        packed_result(ruby, flat, lens)
    }

    fn decode(ruby: &Ruby, rb_self: &Self, tokens: RArray) -> Result<RString, Error> {
        let ids: Vec<u32> = tokens.to_vec()?;
        let ids: Vec<_> = ids.into_iter().map(Into::into).collect();
        let bytes = rb_self.tokenizer.decode(&ids);
        Ok(binary_string(ruby, &bytes))
    }

    fn vocab_size(&self) -> usize {
        self.tokenizer.vocab_size()
    }

    fn vocab(ruby: &Ruby, rb_self: &Self) -> Result<RHash, Error> {
        let tokenizer = &rb_self.tokenizer;
        let hash = ruby.hash_new();
        for (id, bytes) in tokenizer.vocab_entries() {
            hash.aset(id, binary_string(ruby, bytes))?;
        }
        Ok(hash)
    }

    fn merges(ruby: &Ruby, rb_self: &Self) -> Result<RArray, Error> {
        let tokenizer = &rb_self.tokenizer;
        let entries = tokenizer.merge_entries();
        let result = ruby.ary_new_capa(entries.len());
        for (a, b) in entries {
            result.push((binary_string(ruby, a), binary_string(ruby, b)))?;
        }
        Ok(result)
    }

    /// Cached unit entries on the single-document `encode` path's state
    /// (batch encoders are per-call); see `BPETokenizer::cache_entries`.
    fn cache_entries(&self) -> usize {
        self.state.lock().unwrap_or_else(|e| e.into_inner()).cache_size()
    }
}

pub fn init(ruby: &Ruby, native: RModule) -> Result<(), Error> {
    let class: RClass = native.define_class("SentencePieceTokenizer", ruby.class_object())?;
    class.define_method("encode", method!(SentencePieceTokenizer::encode, 1))?;
    class.define_method("encode_batch", method!(SentencePieceTokenizer::encode_batch, 1))?;
    class.define_method("encode_batch_packed", method!(SentencePieceTokenizer::encode_batch_packed, 1))?;
    class.define_method("encode_files", method!(SentencePieceTokenizer::encode_files, -1))?;
    class.define_method("encode_files_packed", method!(SentencePieceTokenizer::encode_files_packed, -1))?;
    class.define_method("decode", method!(SentencePieceTokenizer::decode, 1))?;
    class.define_method("vocab_size", method!(SentencePieceTokenizer::vocab_size, 0))?;
    class.define_method("vocab", method!(SentencePieceTokenizer::vocab, 0))?;
    class.define_method("merges", method!(SentencePieceTokenizer::merges, 0))?;
    class.define_method("cache_entries", method!(SentencePieceTokenizer::cache_entries, 0))?;
    Ok(())
}
