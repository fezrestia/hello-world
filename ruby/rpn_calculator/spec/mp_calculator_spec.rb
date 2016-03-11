require "spec_helper"

RSpec.describe RpnCalculator do
    describe ".run" do
        subject { -> { RpnCalculator.run(["2.7", "1.1", "-", "-5.2", "0.9", "+", "*"]) } }
        it { expect(subject).to be_output "-6.88\n" }
    end
end
