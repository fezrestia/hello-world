#!/usr/bin/env ruby
#! ruby
# ENCODING: UTF-8

module Walkable
    def Walk
        puts "Tote Tote"
    end
end

class Duck
    include Walkable
end

class Hamster
    include Walkable
end

class Rabbit
    include Walkable
    include Comparable

    attr_accessor :name
    attr_reader :color
    attr_reader :length_of_ears

    @@count = 0

    DEFAULT_NAME = "usachan"
    DEFAULT_COLOR = :white
    DEFAULT_LENGTH_OF_EAR = 10

    DESCRIPTION = "Rabitt is pretty animal with long ears."

    public

    # CONSTRUCTOR
    def initialize(
            name: DEFAULT_NAME,
            color: DEFAULT_COLOR,
            length_of_ears: DEFAULT_LENGTH_OF_EAR)
        @name = name
        @color = color
        @length_of_ears = length_of_ears

        @@count += 1
    end

    def jump
        puts "Pyon! Pyon!"
    end

    def pound_streamed_rice_into_rice_cake
        puts "Pettan! Pettan!"
    end

    def say_name
        puts "Hello, I am #{@name} !"
    end

    def print_ears
        puts "^-^"
    end

    def print_description
        puts "#{DESCRIPTION}"
    end

    #Override
    def to_s
        "Name=#{@name}, Color=#{@color}, LengthOfEars=#{@length_of_ears}, AllCount=#{@@count}"
    end

    #Override
    def Walk
        puts "Pyoko Pyoko"
    end

    #Override
    def <=>(other)
        @length_of_ears <=> other.length_of_ears
    end

    protected

    def protectedFunction
        puts "protectedFunction is called"
    end

    private

    def privateFunction
        puts "privateFunction is called"
    end

end

class LopEar < Rabbit
    #Override
    DESCRIPTION = "Lop ear is rabit with lopped ear."

    #Override
    def print_ears
        puts "U-U"
    end

    def print_description
        puts "#{DESCRIPTION}"
    end

    def print_class_constants_and_variables
        puts "DEFAULT_NAME: #{DEFAULT_NAME}"
        puts "DEFAULT_COLOR: #{DEFAULT_COLOR}"
        puts "DEFAULT_LENGTH_OF_EAR: #{DEFAULT_LENGTH_OF_EAR}"
        puts "DESCRIPTION: #{DESCRIPTION}"
        puts "@@count: #{@@count}"
    end
end
