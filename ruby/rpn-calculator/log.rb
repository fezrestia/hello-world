#!/usr/bin/env ruby
#! ruby
# ENCODING: UTF-8

# Log Functionalities

class Log
    class << self

#private
        def output(level:, timestamp: 0, thread: "unknown", tag:, event:)
            puts "#{level} | #{timestamp} | #{thread} | #{tag}:#{event}"
        end

    end # class << self
end # class Log
