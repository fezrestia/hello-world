require "spec_helper"

RSpec.describe RpnCalculator::StackCalculator do
    describe "#calc" do
        # Simple patterns

        context "-2 3 +" do
            subject { RpnCalculator::StackCalculator.new(["-2", "3", "+"]).calc }
            it { expect(subject).to eq 1.0 }
        end

        context "-2 3 -" do
            subject { RpnCalculator::StackCalculator.new(["-2", "3", "-"]).calc }
            it { expect(subject).to eq -5.0 }
        end

        context "-2 3 *" do
            subject { RpnCalculator::StackCalculator.new(["-2", "3", "*"]).calc }
            it { expect(subject).to eq -6.0 }
        end

        context "-2 3 /" do
            subject { RpnCalculator::StackCalculator.new(["-2", "3", "/"]).calc }
            it { expect(subject).to eq (-2.to_f / 3.to_f) }
        end

        # Complex patterns

        context "2.7 1.1 - -5.2 0.9 + *" do
            subject { RpnCalculator::StackCalculator.new(["2.7", "1.1", "-", "-5.2", "0.9", "+", "*"]).calc }
            it { expect(subject).to eq -6.88 }
        end
    end

    describe "#numeric?" do
        context "the argument can be converted to float" do
            subject { RpnCalculator::StackCalculator.new([]).send(:numeric?, "3.14") }
            it { expect(subject).to be true }
        end

        context "the argument can not be converted to float" do
            subject { RpnCalculator::StackCalculator.new([]).send(:numeric?, "+") }
            it { expect(subject).to be false }
        end
    end
end

