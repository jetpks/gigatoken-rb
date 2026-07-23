require "mkmf"
require "rb_sys/mkmf"

# Hide install_name_tool from rb_sys's fixup_libnames (rb_sys-0.9.128's
# mkmf.rb): on some hosts `install_name_tool -id "" $(DLLIB)` corrupts the
# compiled bundle's LC_ID_DYLIB load command (dyld then refuses to load it:
# "load command #N string extends beyond end of load command"), reproduced
# independent of this crate on a trivial unrelated cdylib. Skipping the
# rewrite leaves the bundle's original (absolute) install name in place,
# which `Kernel#require`'s dlopen does not care about.
def find_executable(bin, path = nil)
  return nil if bin == "install_name_tool"
  super
end

# Target basename must match the compiled artifact: the crate's [package]
# name is "gigatoken-rb" (the workspace root package already owns
# "gigatoken"), so Cargo's default lib name is "gigatoken_rb".
create_rust_makefile("gigatoken/gigatoken_rb")
