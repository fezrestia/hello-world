# ENCODING: UTF-8

module RpnCalculator
    class StackCalculator
        public

        def initialize(formula)
            @formula = formula
            @stack = []
        end

        def calc
            @formula.each do |e|
                if numeric?(e)
                    @stack.push(e)
                else
                    op1 = @stack.pop.to_f # right
                    op2 = @stack.pop.to_f # left
                    result = op2.send(e, op1) # left calc right

                    @stack.push(result)
                end
            end

            @stack.first
        end

        private

        def numeric?(s)
            begin
                Float(s)
                true
            rescue ArgumentError
                false
            end
        end
    end
end

