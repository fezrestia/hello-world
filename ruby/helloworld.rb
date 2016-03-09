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

#print("KCODE=", $KCODE)

#if ARGV[0] == '0'
#    puts "ARGV[0] == 0"
#else
#    puts "ARGV[0] != 0"
#end

argNum = ARGV[0].to_i
#if argNum < -10 or 10 < argNum
#    puts "not +=10"
#else
#    if argNum == 0
#        puts "Just 0"
#    else
#        puts "in +=10"
#    end
#end

#unless argNum == 0
#    puts "argNum is not 0"
#else
#    puts "argNum is 0"
#end


#case argNum
#    when 0
#        puts "Jus 0"
#    when 1
#        puts "== 1"
#    when 2, 3
#        puts "== 2 or 3"
#    else
#        puts "else"
#end


#i = 0
#while i < 10
#    i += 1
#
#    next if i % 3 == 0
#    puts i
#
#    break if i == 5
#end


#until i == 0
#    i -= 1
#
#    puts i
#end



#strArray = ["alpha", "beta", "chary"]
#for str in strArray
#    puts str
#end

#for ii in 1..10
#    puts ii
#
##    redo
#end


array = []
array.push(1)
array << 2
array << "3"

#puts array

#printf("SIZE=#{array.size}\n")
#printf("LEN =#{array.length}\n")



=begin
puts ".."
for i in 1..5
    p i
end

puts "..."
for ii in 1...5
    p ii
end
=end


Array.new
Hash.new
Range.new(1, 3)


=begin
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
=end


=begin

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

=end


=begin
"Hello World !".each_char do |char|
    puts char
end

"Hello\nWorld".each_line do |line|
    puts line
end
=end



=begin
if ARGV[0] =~ Regexp.new(ARGV[1])
    puts "Regular Expression OK"
else
    puts "Regular Expression NG"
end

if ARGV[0] !~ Regexp.new(ARGV[1])
    puts "RegExp NG"
else
    puts "RegExp OK"
end
=end



require_relative "./rabbit"

=begin
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
=end

=begin
rabbit = Rabbit.new
puts rabbit.to_s

lop = LopEar.new
puts lop.to_s

lop.print_class_constants_and_variables

rabbit.print_description
lop.print_description
=end


#Rabbit.new.Walk
#Duck.new.Walk
#Hamster.new.Walk


rabbits = Array.new.tap do |array|
    [7.0, 1.0, 5.0, 1.0].each do |length|
        array << Rabbit.new(length_of_ears: length)
    end
end

rabbits.each_with_index do |rabbit, i|
    puts "rabbit#{i}: #{rabbit.length_of_ears}"
end

puts rabbits.sort
