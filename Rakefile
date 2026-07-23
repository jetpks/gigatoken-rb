# frozen_string_literal: true

require "bundler/gem_tasks"
require "rb_sys/extensiontask"

GEMSPEC = Gem::Specification.load("gigatoken.gemspec")

# rb_sys/extensiontask wraps rake-compiler with cross-compilation tasks. The
# crate name here must match ext/gigatoken/Cargo.toml's [package] name
# ("gigatoken-rb"), not the gem name: rb_sys locates the extension's Cargo
# manifest by matching this name against `cargo metadata`'s workspace
# packages, and the core crate at the workspace root is itself already
# named "gigatoken".
RbSys::ExtensionTask.new("gigatoken-rb", GEMSPEC) do |ext|
  ext.lib_dir = "lib/gigatoken"
end

require "rspec/core/rake_task"
RSpec::Core::RakeTask.new(:spec)

require "standard/rake"

task default: [:compile, :spec]
