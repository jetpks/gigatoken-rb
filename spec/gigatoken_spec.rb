# frozen_string_literal: true

require_relative "spec_helper"

RSpec.describe Gigatoken do
  it "has a version number" do
    expect(Gigatoken::VERSION).to match(/\A\d+\.\d+\.\d+\z/)
  end

  it "loads the native extension and can call into it" do
    version = Gigatoken::Native.crate_version
    expect(version).to be_a(String)
    expect(version).not_to be_empty
  end
end
