#!/usr/bin/env ruby
#! ruby
# ENCODING: UTF-8

# HELLO WORLD

=begin
Injected Document
=end


puts 'HELLO WORLD !'
#print("hello world !\n")

puts "\n"

puts "ARGV[0] : #{ARGV[0]}"
puts "ARGV[1] : #{ARGV[1]}"

#print("KCODE=", $KCODE)

if ARGV[0] == '0'
    puts "ARGV[0] == 0"
else
    puts "ARGV[0] != 0"
end

puts ""

argNum = ARGV[0].to_i
if argNum < -10 or 10 < argNum
    puts "not +=10"
else
    if argNum == 0
        puts "Just 0"
    else
        puts "in +=10"
    end
end

puts ""

unless argNum == 0
    puts "argNum is not 0"
else
    puts "argNum is 0"
end

puts ""

case argNum
    when 0
        puts "Jus 0"
    when 1
        puts "== 1"
    when 2, 3
        puts "== 2 or 3"
    else
        puts "else"
end

puts ""

i = 0
while i < 10
    i += 1

    next if i % 3 == 0
    puts i

    break if i == 5
end

puts ""

until i == 0
    i -= 1

    puts i
end

puts ""

strArray = ["alpha", "beta", "chary"]
for str in strArray
    puts str
end

for ii in 1..10
    puts ii

#    redo
end

puts ""

array = []
array.push(1)
array << 2
array << "3"

puts array

printf("SIZE=#{array.size}\n")
printf("LEN =#{array.length}\n")

puts ""

puts ".."
for i in 1..5
    p i
end

puts "..."
for ii in 1...5
    p ii
end

Array.new
Hash.new
Range.new(1, 3)

case argNum
    when -Float::INFINITY..-10
        puts "INF to -10"
    when -9..9
        puts "in +=10"
    when 10..Float::INFINITY
        puts "10 to INF"
    else
        puts "UNEXPECTED"
end

puts ""

range = Range.new(2, 8)
range.step(2) {|i| puts i}

array = ["alpha", "beta", "char", "delta", "eta"]
array.each do |item|
    puts item
end

array.each_with_index do |item, index|
    puts "#{index} : #{item}"
end

array.reverse_each do |item|
    puts item
end

puts ""

"Hello World !".each_char do |char|
    puts char
end

"Hello\nWorld".each_line do |line|
    puts line
end

puts ""

if ARGV[1] != nil
    if ARGV[0] =~ Regexp.new(ARGV[1])
        puts "Regular Expression OK"
    else
        puts "Regular Expression NG"
    end
end

if ARGV[1] != nil
    if ARGV[0] !~ Regexp.new(ARGV[1])
        puts "RegExp NG"
    else
        puts "RegExp OK"
    end
end

puts ""

require_relative "./rabbit"

rabbits = []

rabbits.push(Rabbit.new)
rabbits.push(Rabbit.new(name: "pyonkichi"))
rabbits.push(Rabbit.new(name: "inaba", color: :brown, length_of_ears: 7))

rabbits.each do |rabbit|
    puts rabbit.name
    puts rabbit.color
    puts rabbit.length_of_ears

    rabbit.say_name
    puts ""
end

rabbits[0].name = "wooser"
puts rabbits[0].name

puts ""

rabbit = Rabbit.new
lop = LopEar.new(name: "lopchan")

[rabbit, lop].each do |r|
    r.say_name
    r.print_ears
end

puts ""

rabbit = Rabbit.new
puts rabbit.to_s

lop = LopEar.new
puts lop.to_s

lop.print_class_constants_and_variables

rabbit.print_description
lop.print_description

puts ""

Rabbit.new.Walk
Duck.new.Walk
Hamster.new.Walk

puts ""

rabbits = Array.new.tap do |array|
    [7.0, 1.0, 5.0, 1.0].each do |length|
        array << Rabbit.new(length_of_ears: length)
    end
end

rabbits.each_with_index do |rabbit, i|
    puts "rabbit#{i}: #{rabbit.length_of_ears}"
end

puts rabbits.sort

puts ""

require_relative("./log.rb")
Log.new.output(level: "ERR", tag: "TAG", event: "EVENT")

def repeat(n)
    for i in 1..n
        puts "#{i} ->"
        yield
    end
end
repeat(3) {puts "pyonpyon"}

puts ""

def arithmetic_sequence(init: 1, diff: 1, count: 10)
    current = init
    if block_given?
        count.times do
            yield(current)
            current += diff
        end
    else
        Array.new.tap do |a|
            count.times do
                a << current
                current += diff
            end
        end
    end
end

arithmetic_sequence(init: 2, diff: 3, count: 5) do |n|
    puts n
end

p arithmetic_sequence(init: 2, diff: 3, count: 5)

puts ""

procedure1 = Proc.new { |word| puts word }
procedure2 = proc { |word| puts word }
procedure3 = lambda { |word| puts word }
procedure4 = ->(word) { puts word }
procedure5 = -> { puts "cluck" }

procedure1.call("meow")
procedure2.call("quack")
procedure3.call("oink")
procedure4.call("bowwow")
procedure5.call

puts ""

def repeat1(n)
    for i in 1..n
        puts "#{i} ->"
        yield
    end
end

def repeat2(n, &procedure)
    for i in 1..n
        puts "#{i} ->"
        procedure.call
    end
end

procedure = -> { puts "quack" }
repeat1(3, &procedure)
repeat2(3) { puts "meow" }

puts ""

def hoge
    procedure = Proc.new { return "return from Proc.new" }
    procedure.call
    return "return from hoge"
end

def fuga
    procedure = -> { return "return from lambda" }
    procedure.call
    return "return from fuga"
end

puts hoge
puts fuga

puts ""

def increase
    n = 0;
    -> { n += 5 }
end

3.times {
    func = increase
    puts func
    p func.call
}

puts "============"

increase_proc = increase
3.times {
    puts increase_proc
    p increase_proc.call
}

puts ""

class UnacceptableRequidError < StandardError
    attr_reader :obj

    # CONSTRUCTOR
    def initialize(obj)
        @obj = obj
    end
end

class Vessel
    def pour_out
        requid = @requid
        @requid = nil
        requid
    end

    def pour_in(requid)
        if requid.to_s == acceptance
            @requid = requid
        else
            raise UnacceptableRequidError.new(requid), "unacceptable"
        end
    end

    def acceptance
        raise NotImplementedError.new, "#acceptance is not implemented !"
    end
end

class Teapot < Vessel
    # CONSTRUCTOR
    def initialize(requid)
        pour_in(requid)
    end

    def acceptance
        "tea"
    end
end

class Decanter < Vessel
    # CONSTRUCTOR
    def initialize(requid)
        pour_in(requid)
    end

    def acceptance
        "wine"
    end
end

class Kettle < Vessel
    # CONSTRUCTOR
    def initialize(requid)
        pour_in(requid)
    end
end

puts ""

begin
#    raise "something wrong"
#    bottle = Bottle.new("water")
#    teapot = Teapot.new("coffee")
#    puts teapot.pour_out

    retry_count ||= 0
    puts "ReTry..."

    teapot = Teapot.new(%w(coffee tea beer wine).sample)
    puts teapot.pour_out
rescue UnacceptableRequidError => e
    puts "Rescued in UnacceptableRequidError"
    p e
    p e.obj

    retry_count += 1
    retry if retry_count < 3

rescue NoMethodError, ZeroDivisionError => e
    puts "Rescued in NoMethodError / ZeroDivisionError"
    p e
    p $!
rescue StandardError => e
    puts "Rescued in StandardError"
    p e
else
    puts "There are no errors"
ensure
    puts "Ensure"
end

puts ""

#print "INPUT : "
#p gets.chomp

puts ""

#while line = gets
#    puts "//#{line.chomp}//"
#end

puts ""

#file = File.open("LOG.TXT", "r")
#file.each do |line|
#    puts "//#{line.chomp}//"
#end
#file.close

#file = File.open("LOG.TXT", "w")
#file.print("print test\n")
#file.puts("puts test\n")
#file.printf("%s\n", "printf test")
#file.close

puts ""

#STDIN.each.with_index(1) do | line, i |
#    puts "#{i} : #{line.chomp}"
#end

puts ""

names = %w(alpha beta char delta eta)
threads = []

names.each do |name|
    threads << Thread.new do
        3.times do |i|
            print "#{name}:#{i}\n"
        end
    end
end

threads.each do |t|
    t.join
end

puts "\nAll threads are terminated."

puts ""

require "io/console"

view_thread = Thread.new do
    loop do
        print "\33[2K\r#{Time.now.strftime("%F %T")}"
        sleep 1
    end
end

input_thread = Thread.new do
    while STDIN.getch != "q"
    end

    puts

    view_thread.kill
end

view_thread.join
input_thread.join

puts ""

g = Fiber.new do |x|
    loop {
        Fiber.yield(x)
        x += 1
    }
end

5.times {
    puts g.resume(0)
}

puts

#pid = Process.spawn("call sleep 5\necho terminated")
#puts "waiting..."
#puts "waiting..."
#puts "waiting..."
#Process.waitpid(pid)

puts

IO.popen("ping -n 3 localhost", "r") do |pipe|
    while line = pipe.gets
        line.match(/time=(\d\.\d+) ms/)
        if $1.to_f < 0.1
            puts line
        end
    end
end

puts

f = 0.0
startedTime = Time.now.to_i

3.times do
#    fork do
        1000000.times {
            f += 0.001
        }
        printf("%.3f\n", f)
#    end
end

Process.waitall
puts "it takes #{Time.now.to_i - startedTime} sec"
