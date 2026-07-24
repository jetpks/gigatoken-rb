//! `Gigatoken::Native::BPETokenizer`: a `gigatoken::Tokenizer` plus the
//! `WorkerPool` that backs batch encoding. Mirrors the pyo3 `BPETokenizer` in
//! the core crate's `src/lib.rs` (the `python` feature), minus the
//! numpy/awkward-array machinery that has no Ruby analog.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::os::raw::c_long;

use gigatoken_rs::load_tokenizer::hf::HfTokenizer;
use gigatoken_rs::load_tokenizer::{hf, tiktoken};
use gigatoken_rs::{
    GatherBuf, GatherOutcome, Tokenizer, WorkerPool, encode_docs_into, encode_docs_ragged,
    encode_files_docs, encode_files_docs_serial,
};
use magnus::{
    Error, RArray, RClass, RHash, RModule, RString, Ruby, Value, function, method, prelude::*,
    rb_sys::{AsRawValue, FromRawValue},
    scan_args::{get_kwargs, scan_args},
};
use rb_sys::{RSTRING_PTR, rb_ary_dup, rb_str_locktmp, rb_str_set_len, rb_str_unlocktmp};

use crate::error::raise;
use crate::gvl::without_gvl;
use crate::sources;

pub(crate) fn binary_string(ruby: &Ruby, bytes: &[u8]) -> RString {
    ruby.enc_str_new(bytes, ruby.ascii8bit_encoding())
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

/// One document's bytes as marshaled by `marshal_inputs`: either an owned
/// copy, or a zero-copy borrow of a heap `RString`'s own buffer.
enum DocSource {
    Owned(Vec<u8>),
    Borrowed { ptr: *const u8, len: usize },
}

/// Whether `value` (already confirmed to be a `T_STRING`) stores its bytes
/// in a separate heap allocation (`RSTRING_NOEMBED` set) rather than inside
/// the RVALUE itself. Heap buffers don't move under GC compaction (only the
/// RVALUE header can); embedded ones live inside the header and do — so only
/// a heap string's buffer is safe to borrow across a GVL release. Neither
/// magnus nor rb-sys expose a public query for this; this reads `RBasic`'s
/// `flags` field directly, the same technique rb-sys's own (private)
/// stable-API string accessors use internally (`rstring_ptr` in
/// rb-sys-0.9.128's `src/stable_api/ruby_4_0.rs`). Verified live against
/// Ruby 4.0.6: Variable Width Allocation raises the embed threshold well
/// past the classic ~23 bytes (bisected empirically: 512 B still embeds,
/// 1024 B doesn't), so this flag — not a size heuristic — is the only
/// reliable test.
fn is_heap_rstring(value: rb_sys::VALUE) -> bool {
    // SAFETY: every Ruby object's memory begins with an `RBasic` (flags,
    // klass) header regardless of its concrete type's trailing fields, and
    // `value` is a live `T_STRING` VALUE for the duration of this
    // synchronous, GVL-held call.
    let flags = unsafe { (*(value as *const rb_sys::RBasic)).flags };
    flags & (rb_sys::ruby_rstring_flags::RSTRING_NOEMBED as rb_sys::VALUE) != 0
}

/// Lift `rstr`'s current buffer out as a `DocSource::Borrowed`. Caller must
/// have already confirmed `rstr` is a heap `RString` that is either frozen
/// (permanently immutable) or successfully locked via `rb_str_locktmp` (any
/// mutation attempt for as long as the lock holds raises rather than
/// touching the buffer) — either guard keeps the buffer stable and
/// unmutated for as long as the borrow lives, decoupled from any Rust
/// borrow of `rstr` itself, across the `without_gvl` release that follows.
fn borrowed_doc(rstr: RString) -> DocSource {
    // SAFETY: see above — the buffer neither moves nor mutates while the
    // caller's guard (frozen, or this call's lock) holds.
    let bytes = unsafe { rstr.as_slice() };
    DocSource::Borrowed {
        ptr: bytes.as_ptr(),
        len: bytes.len(),
    }
}

/// Marshaled batch inputs, ready for `as_slices` to hand to the core encode.
/// `snapshot` is the `rb_ary_dup` copy `marshal_inputs` takes of the
/// caller's input Array at entry (see its doc comment) — classification,
/// `to_str` conversion, locking, and this Drop's unlock all read
/// exclusively from `snapshot`'s slots, never the caller's original array.
/// Dropping this unlocks every string `marshal_inputs` locked, by
/// re-reading the *current* value out of `snapshot` at each locked index
/// rather than reusing any `VALUE` captured before a possible GVL release:
/// GC compaction can move a locked heap string's RVALUE header while the
/// GVL is released (only its separate, malloc'd byte buffer is guaranteed
/// to stay put), and `snapshot` — a precisely-marked Ruby Array, kept alive
/// and pinned in place by conservative scanning of this struct's stack
/// frame for the whole call — is the only reference that stays correct
/// across that.
struct InputDocs {
    snapshot: RArray,
    should_unlock: Vec<bool>,
    docs: Vec<DocSource>,
}

impl InputDocs {
    fn as_slices(&self) -> Vec<&[u8]> {
        self.docs
            .iter()
            .map(|doc| match *doc {
                DocSource::Owned(ref bytes) => bytes.as_slice(),
                // SAFETY: see `borrowed_doc` and this struct's own doc
                // comment — the buffer is stable and unmutated for as long
                // as `self` (and thus its lock, if any) is alive, which
                // outlives every use of the slices this returns.
                DocSource::Borrowed { ptr, len } => unsafe { std::slice::from_raw_parts(ptr, len) },
            })
            .collect()
    }
}

impl Drop for InputDocs {
    fn drop(&mut self) {
        if !self.should_unlock.iter().any(|&locked| locked) {
            return;
        }
        let mut unlocked = HashSet::new();
        for (i, &locked) in self.should_unlock.iter().enumerate() {
            if !locked {
                continue;
            }
            // Re-read this slot fresh (see this struct's doc comment for
            // why a value captured earlier isn't safe to reuse here).
            let value = snapshot_entry(self.snapshot, i);
            let Some(rstr) = RString::from_value(value) else {
                continue;
            };
            let raw = rstr.as_raw();
            if unlocked.insert(raw) {
                // A failure here would mean the string was already
                // unlocked out from under us — not something we can
                // recover from in `drop`, and not expected to happen since
                // we're the ones who locked it and dedup by `raw` above.
                let _ = magnus::rb_sys::protect(|| unsafe { rb_str_unlocktmp(raw) });
            }
        }
    }
}

/// Read the `Value` currently at `index` in `snapshot`, fresh — via the
/// bounds-checked `rb_ary_entry` C accessor, never by reusing a `Value` read
/// before a Ruby-code-running call (`to_str` conversion, or any other call
/// that can allocate and trigger GC compaction), since compaction rewrites a
/// moved element's slot in place. `Value`'s own `TryConvert` is an
/// infallible identity conversion, so this can't fail.
fn snapshot_entry(snapshot: RArray, index: usize) -> Value {
    snapshot.entry(index as isize).expect("Value's TryConvert is infallible")
}

/// Marshal `inputs` (an Array of Strings, or objects converting to one via
/// `to_str`) into `InputDocs`, borrowing zero-copy wherever it's sound
/// instead of copying.
///
/// The first thing this does is `rb_ary_dup` `inputs` into a snapshot: a
/// C-level shallow copy of the array's slots that runs no user code (not
/// even `initialize_copy`). From that point on, every pass below —
/// classification, `to_str` conversion, locking, and `InputDocs`'s Drop —
/// reads exclusively from the snapshot's slots; `inputs` itself is never
/// read again. This closes three hazards a direct-`inputs` version has (I19
/// Verdict): (1) a `to_str` conversion for one element can run arbitrary
/// Ruby code, including code that mutates or replaces later elements of the
/// caller's array out from under an in-progress classify/lock pass; (2)
/// while the GVL is released for the encode itself, another Ruby thread can
/// replace or clear the caller's slots, making a borrowed string
/// collectible mid-encode; (3) holding one Rust borrow of the caller's
/// array buffer across a `to_str` call is unsound if that call resizes the
/// array. None of these can reach the snapshot: no Ruby code holds a
/// reference to it (so nothing can mutate, resize, or replace its slots
/// from Ruby), and the snapshot `RArray` local stays alive as a
/// conservatively-scanned stack root the whole call through — via
/// `InputDocs::snapshot`, kept in the calling frame including across the
/// `without_gvl` window — which is also, incidentally, why compaction never
/// relocates the snapshot array itself. The snapshot's *slots* remain
/// ordinary, precisely-marked Ruby state, though, and compaction does
/// rewrite a slot in place when that element's RVALUE moves — so every pass
/// below re-reads a slot fresh (`snapshot_entry`) rather than reusing a
/// `Value` obtained before a Ruby-code-running call, and never holds a
/// `snapshot.as_slice()` borrow across one.
///
/// One consequence is now a pinned public contract: `encode_batch`'s result
/// reflects the input array as it was *at this call's entry*. A
/// pathological `to_str` that mutates the caller's array mid-marshal can no
/// longer change which documents get encoded — the caller's own mutations
/// remain visible to the caller afterwards, just no longer to this encode.
///
/// - a heap (non-embedded) `RString` that's frozen is borrowed unlocked —
///   its immutability is itself the guard, and `rb_str_locktmp` isn't legal
///   on a frozen string in the first place (verified live: it raises
///   `FrozenError`);
/// - a heap `RString` that isn't frozen (including a "chilled" literal —
///   verified live on Ruby 4.0.6: it reports `frozen? == false` and
///   `rb_str_locktmp` succeeds on it exactly like any other mutable string)
///   is borrowed under a dedup'd, `protect`-wrapped `rb_str_locktmp`; if the
///   lock is refused (already locked by other code), it's copied instead —
///   never borrow a string whose lock isn't held;
/// - an embedded `RString`, or a `to_str` conversion result (a new object
///   reachable only from this call, never borrowable), is always copied.
///
/// A first pass classifies every slot, doing every `to_str` conversion (the
/// only step that can run arbitrary Ruby code and allocate) along the way;
/// a slot that's already a heap, non-frozen `RString` is left `NeedsLock`
/// rather than resolved immediately, since a *later* slot's `to_str` call
/// can still run before this pass finishes. That first pass's `NeedsLock`
/// verdict is a hint, not truth, by the time the second (lock) pass gets to
/// it: a later `to_str` conversion can freeze or mutate the string in the
/// meantime (e.g. `clear` can re-embed a heap string), so the lock pass
/// re-reads and re-classifies each `NeedsLock` index's snapshot slot from
/// scratch instead of trusting the first pass. Only indices are carried
/// between the two passes — never a bare `RString`/`Value` — since a
/// `Value` read before a `to_str` call can go stale under compaction before
/// the lock pass gets to it.
fn marshal_inputs(inputs: RArray) -> Result<InputDocs, Error> {
    // SAFETY: `inputs.as_raw()` is a live, array-typed VALUE for the
    // duration of this synchronous, GVL-held call; `rb_ary_dup` only reads
    // it and allocates a fresh Array via a C-level shallow slot copy that
    // runs no user code, so nothing here can run arbitrary Ruby code or
    // raise. Its result is always an Array, per the Ruby C API.
    let snapshot = RArray::from_value(unsafe { Value::from_raw(rb_ary_dup(inputs.as_raw())) })
        .expect("rb_ary_dup's result is always an Array");
    let len = snapshot.len();

    enum Classified {
        Done(DocSource),
        NeedsLock,
    }

    let mut classified = Vec::with_capacity(len);
    for i in 0..len {
        let value = snapshot_entry(snapshot, i);
        let item = match RString::from_value(value) {
            Some(rstr) if !is_heap_rstring(rstr.as_raw()) => {
                Classified::Done(DocSource::Owned(unsafe { rstr.as_slice() }.to_vec()))
            }
            Some(rstr) if rstr.is_frozen() => Classified::Done(borrowed_doc(rstr)),
            Some(_) => Classified::NeedsLock,
            None => {
                let converted = RString::try_convert(value)?;
                Classified::Done(DocSource::Owned(unsafe { converted.as_slice() }.to_vec()))
            }
        };
        classified.push(item);
    }

    let mut should_unlock = vec![false; len];
    let mut docs = Vec::with_capacity(len);
    let mut lock_cache: HashMap<rb_sys::VALUE, bool> = HashMap::new();
    for (i, item) in classified.into_iter().enumerate() {
        let source = match item {
            Classified::Done(source) => source,
            Classified::NeedsLock => {
                // Re-read and re-classify: a `to_str` conversion that ran
                // after this index was first classified may have frozen or
                // mutated this exact string (see this fn's doc comment).
                let value = snapshot_entry(snapshot, i);
                let rstr = RString::from_value(value).expect(
                    "this slot classified as a String in the first pass, and nothing but this \
                     call's Rust frame holds a reference to the snapshot to replace it",
                );
                if rstr.is_frozen() {
                    borrowed_doc(rstr)
                } else if !is_heap_rstring(rstr.as_raw()) {
                    DocSource::Owned(unsafe { rstr.as_slice() }.to_vec())
                } else {
                    let raw = rstr.as_raw();
                    let locked = *lock_cache
                        .entry(raw)
                        .or_insert_with(|| magnus::rb_sys::protect(|| unsafe { rb_str_locktmp(raw) }).is_ok());
                    if locked {
                        should_unlock[i] = true;
                        borrowed_doc(rstr)
                    } else {
                        // SAFETY: read-only; the lock attempt failing
                        // doesn't change anything about the buffer's
                        // validity here.
                        DocSource::Owned(unsafe { rstr.as_slice() }.to_vec())
                    }
                }
            }
        };
        docs.push(source);
    }

    Ok(InputDocs {
        snapshot,
        should_unlock,
        docs,
    })
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
    /// parallel encode itself (see `gvl::without_gvl`). Each input string is
    /// borrowed zero-copy where `marshal_inputs` finds it sound to, and
    /// copied into an owned buffer otherwise; either way, only raw byte
    /// slices — never a Ruby `VALUE` — are captured once the GVL is gone.
    fn encode_batch_ragged(rb_self: &Self, inputs: RArray) -> Result<(Vec<u32>, Vec<i64>), Error> {
        let marshaled = marshal_inputs(inputs)?;
        let doc_slices = marshaled.as_slices();
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
        let marshaled = marshal_inputs(inputs)?;
        let doc_slices = marshaled.as_slices();
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
        let ptr = unsafe { RSTRING_PTR(string.as_raw()) as *mut u32 };
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
                        string.as_raw(),
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
