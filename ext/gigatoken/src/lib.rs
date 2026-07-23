use magnus::{Error, Module, Ruby, function};

mod error;
mod gvl;
mod tokenizer;

// The gigatoken core crate exposes no version constant of its own, so this
// is the ext crate's (gigatoken-rb's) version — see the builder report.
fn crate_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

#[magnus::init]
fn init(ruby: &Ruby) -> Result<(), Error> {
    let gigatoken = ruby.define_module("Gigatoken")?;
    let native = gigatoken.define_module("Native")?;
    native.define_module_function("crate_version", function!(crate_version, 0))?;
    tokenizer::init(ruby, native)?;
    Ok(())
}
