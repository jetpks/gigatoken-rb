# frozen_string_literal: true

# The packed path uses IO::Buffer, which warns as experimental.
Warning[:experimental] = false

require "gigatoken"

# CI runs the suite a second time with GC_STRESS set: a GC at every
# allocation, to catch a native object read after it was freed or moved. A
# minor GC (flag 0x01, no major) is what catches an unmarked young object —
# the C-extension bug class — at ~14x the suite's runtime; a full GC per
# allocation is ~700x and takes hours.
GC.stress = 1 if %w[1 true yes].include?(ENV["GC_STRESS"].to_s.downcase)

Dir[File.join(__dir__, "support", "**", "*.rb")].sort.each { |f| require f }

RSpec.configure do |config|
  config.mock_with :rspec do |mocks|
    mocks.verify_partial_doubles = true
  end

  # enable flags like --only-failures and --next-failure
  config.example_status_persistence_file_path = "tmp/rspec_status.txt"

  config.filter_run focus: true
  config.run_all_when_everything_filtered = true

  # disable RSpec exposing methods globally on `Module` and `main`
  config.disable_monkey_patching!
end
