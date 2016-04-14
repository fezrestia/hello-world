require "rpn_calculator/version"
require "rpn_calculator/stack_calculator"

module RpnCalculator
    class << self
        def run(args)
            calculator = StackCalculator.new(args)
            puts "RESULT = #{calculator.calc}"
        end
    end
end

