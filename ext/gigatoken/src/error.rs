//! `Gigatoken::Error` and its subclasses are defined in Ruby
//! (`lib/gigatoken.rb`), loaded before this extension is required — looked up
//! fresh at each raise site rather than cached, so no magnus `Value` needs a
//! GC-registered static home.

use magnus::{exception::ExceptionClass, prelude::*, Error, RModule, Ruby};

fn error_class(ruby: &Ruby, name: &str) -> Result<ExceptionClass, Error> {
    ruby.class_object()
        .const_get::<_, RModule>("Gigatoken")?
        .const_get(name)
}

fn raise_as(ruby: &Ruby, name: &str, message: impl Into<String>) -> Error {
    match error_class(ruby, name) {
        Ok(class) => Error::new(class, message.into()),
        Err(e) => e,
    }
}

/// Raise `Gigatoken::Error` with `message` — what a failure that is neither
/// the caller's document nor the model itself surfaces through (never a Rust
/// panic across the Ruby boundary). Prefer [`input_error`] or [`model_error`]
/// where one of them fits.
pub fn raise(ruby: &Ruby, message: impl Into<String>) -> Error {
    raise_as(ruby, "Error", message)
}

/// Raise `Gigatoken::InputError`: a document the tokenizer cannot take —
/// invalid UTF-8 on the SentencePiece path, an id outside the vocabulary in
/// `decode`.
pub fn input_error(ruby: &Ruby, message: impl Into<String>) -> Error {
    raise_as(ruby, "InputError", message)
}

/// Raise `Gigatoken::ModelError`: a tokenizer that cannot be loaded — bad or
/// hostile JSON, an unknown pretokenizer scheme, a malformed `.tiktoken`.
pub fn model_error(ruby: &Ruby, message: impl Into<String>) -> Error {
    raise_as(ruby, "ModelError", message)
}
