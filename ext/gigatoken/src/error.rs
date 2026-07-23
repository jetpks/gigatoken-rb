//! `Gigatoken::Error` is defined in Ruby (`lib/gigatoken.rb`), loaded before
//! this extension is required — looked up fresh at each raise site rather
//! than cached, so no magnus `Value` needs a GC-registered static home.

use magnus::{exception::ExceptionClass, prelude::*, Error, RModule, Ruby};

fn error_class(ruby: &Ruby) -> Result<ExceptionClass, Error> {
    ruby.class_object()
        .const_get::<_, RModule>("Gigatoken")?
        .const_get("Error")
}

/// Raise `Gigatoken::Error` with `message` (core load/encode failures surface
/// through this — never a Rust panic across the Ruby boundary).
pub fn raise(ruby: &Ruby, message: impl Into<String>) -> Error {
    match error_class(ruby) {
        Ok(class) => Error::new(class, message.into()),
        Err(e) => e,
    }
}
