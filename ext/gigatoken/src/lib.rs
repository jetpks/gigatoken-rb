use gigatoken_rs::load_tokenizer::hf::{self, HfTokenizer};
use magnus::{Error, Module, RString, Ruby, Value, function};

// XZM-WORKAROUND: macOS 26's xzm malloc zone SIGTRAPs on multi-GB Rust chunk
// frees (`_xzm_reclaim_mark_used_locked` assertion); routing Rust allocations
// through mimalloc avoids the xzm zone entirely.
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

mod error;
mod gvl;
mod sentencepiece;
mod sources;
mod tokenizer;

use error::raise;
use sentencepiece::SentencePieceTokenizer;
use tokenizer::BPETokenizer;

// The gigatoken core crate exposes no version constant of its own, so this
// is the ext crate's (gigatoken-rb's) version — see the builder report.
fn crate_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Load a tokenizer from in-memory HuggingFace `tokenizer.json` contents.
/// Returns a `SentencePieceTokenizer` when the model uses `byte_fallback`, a
/// `BPETokenizer` otherwise — the same split as pyo3's `load_hf_json` and
/// the two classes' own `from_hf_json` constructors.
fn load_hf_json(ruby: &Ruby, data: RString) -> Result<Value, Error> {
    // SAFETY: read-only, for the duration of this synchronous call.
    let bytes = unsafe { data.as_slice() };
    match hf::load_hf_slice(bytes) {
        Ok(HfTokenizer::Bpe(tokenizer)) => Ok(ruby.into_value(BPETokenizer::from_tokenizer(tokenizer))),
        Ok(HfTokenizer::SentencePiece(tokenizer)) => Ok(ruby.into_value(SentencePieceTokenizer::from_tokenizer(tokenizer))),
        Err(e) => Err(raise(ruby, e.to_string())),
    }
}

#[magnus::init]
fn init(ruby: &Ruby) -> Result<(), Error> {
    let gigatoken = ruby.define_module("Gigatoken")?;
    let native = gigatoken.define_module("Native")?;
    native.define_module_function("crate_version", function!(crate_version, 0))?;
    native.define_module_function("load_hf_json", function!(load_hf_json, 1))?;
    sources::init(ruby, native)?;
    tokenizer::init(ruby, native)?;
    sentencepiece::init(ruby, native)?;
    Ok(())
}
