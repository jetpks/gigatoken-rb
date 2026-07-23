# frozen_string_literal: true

# The packed path uses IO::Buffer, which warns as experimental.
Warning[:experimental] = false

require "gigatoken"

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
