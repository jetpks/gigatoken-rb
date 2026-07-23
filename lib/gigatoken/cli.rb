# frozen_string_literal: true

require "gigatoken"
require "dry/cli"

require_relative "cli/bench"
require_relative "cli/validate"

module Gigatoken
  module CLI
    module Commands
      extend Dry::CLI::Registry

      register "bench", Gigatoken::CLI::Bench
      register "validate", Gigatoken::CLI::Validate
    end
  end
end
