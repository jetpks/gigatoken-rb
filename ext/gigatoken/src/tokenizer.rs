//! `Gigatoken::Native::BPETokenizer`: a `gigatoken::Tokenizer` plus the
//! `WorkerPool` that backs batch encoding. Mirrors the pyo3 `BPETokenizer` in
//! the core crate's `src/lib.rs` (the `python` feature), minus the
//! numpy/awkward-array machinery that has no Ruby analog.

use std::cell::RefCell;

use gigatoken_rs::load_tokenizer::hf::HfTokenizer;
use gigatoken_rs::load_tokenizer::{hf, tiktoken};
use gigatoken_rs::{Tokenizer, WorkerPool, encode_docs_ragged};
use magnus::{Error, RArray, RClass, RHash, RModule, RString, Ruby, function, method, prelude::*};

use crate::error::raise;
use crate::gvl::without_gvl;

fn binary_string(ruby: &Ruby, bytes: &[u8]) -> RString {
    ruby.enc_str_new(bytes, ruby.ascii8bit_encoding())
}

#[magnus::wrap(class = "Gigatoken::Native::BPETokenizer", free_immediately, size)]
pub struct BPETokenizer {
    tokenizer: RefCell<Tokenizer>,
    workers: WorkerPool,
}

impl BPETokenizer {
    fn from_tokenizer(tokenizer: Tokenizer) -> Self {
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
                "SentencePiece not supported in v1, see BRIEF §3.4",
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
    fn encode_batch(ruby: &Ruby, rb_self: &Self, inputs: RArray) -> Result<RArray, Error> {
        // SAFETY: values are read (checked-converted to `RString`, then
        // copied into owned buffers below) before anything else runs.
        let docs: Vec<Vec<u8>> = unsafe { inputs.as_slice() }
            .iter()
            .map(|&v| RString::try_convert(v).map(|s| unsafe { s.as_slice() }.to_vec()))
            .collect::<Result<_, _>>()?;
        let doc_slices: Vec<&[u8]> = docs.iter().map(Vec::as_slice).collect();
        let tokenizer = rb_self.tokenizer.borrow();
        let (flat, lens) =
            without_gvl(|| encode_docs_ragged(&rb_self.workers, &tokenizer, &doc_slices));

        let result = ruby.ary_new_capa(lens.len());
        let mut offset = 0usize;
        for len in lens {
            let len = len as usize;
            result.push(flat[offset..offset + len].to_vec())?;
            offset += len;
        }
        Ok(result)
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
    class.define_method("decode", method!(BPETokenizer::decode, 1))?;
    class.define_method("vocab_size", method!(BPETokenizer::vocab_size, 0))?;
    class.define_method("vocab", method!(BPETokenizer::vocab, 0))?;
    class.define_method("merges", method!(BPETokenizer::merges, 0))?;
    Ok(())
}
