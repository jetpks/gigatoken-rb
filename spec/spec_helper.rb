# frozen_string_literal: true

# The packed path uses IO::Buffer, which warns as experimental.
Warning[:experimental] = false

require "gigatoken"

# GC_STRESS=1 runs the whole suite with a minor GC at every allocation (flag
# 0x01), to catch a native object read after it was freed or moved. It is a
# manual hard mode — ~14x the suite's runtime on 4.0, an hour-plus on 3.4 —
# so CI relies on spec/gigatoken/gc_stress_spec.rb, which puts every native
# path under the same stress with small inputs in seconds.
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
