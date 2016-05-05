
# Without line end '\n'
print "HELLO WORLD !"

# With line end '\n'
puts "HELLO WORLD ! (puts)"

# With data format
p "HELLO WORLD !"



msg = "Message"
p msg

ADMIN_EMAIL = "admin@mail.gom"
#ADMIN_EMAIL = "hogehoge"



p "Hello World".length



x = 10
y = 20.5
z = 1/3r
p x % 3
p x ** 3
p z * 2

p x += 5
p y.round



name = "meme"
x = "He\tllo\n World #{name}"
y = 'hello world #{name}'
puts x
puts y
puts "Hello world" + " meme"
puts "Hello" * 5



# ! Destructable method
# ? true or false
s = "meme"
puts s.upcase
puts s
puts s.upcase!
puts s
p s.empty?
s = ''
p s.empty?



sales = [5, 8, 4]
p sales[1]
sales[1] = 10
p sales[0..2]
p sales[0...2]
p sales[-1]
p sales[1, 2]



sales[0...2] = [1, 2]
p sales
sales[1, 0] = [10, 11, 12]
p sales
sales[0, 2] = []
p sales
p sales.size
p sales.sort
p sales.sort.reverse
p sales.push(100)
p sales << 100 << 102



sales = {"taguchi" => 200, "fujio" => 100}
p sales["taguchi"]

sales = {:taguchi => 300, :fujio => 400}
p sales[:taguchi]

sales = {taguchi: 500, fujio: 600}
p sales[:taguchi]

p sales.size
p sales.keys
p sales.values
p sales.has_key?(:taguchi)



a = 10
b = "5"
p a + b.to_i
p a + b.to_f
p a.to_s + b

h = {tauchi: 100, fkoji: 200}
p h.to_a
p h.to_a.to_h



s = "Hel\"lo"
s = %Q(he"llo)
p s
s = %q(he"llo)
p s
s = %(hel"lo)
p s
a = ["a", "b", "c"]
p a
a = %W(a b c)
p a
a = ['a', 'b', 'c']
p a
a = %w(a b c)
p a



score = 50
if 60 < score
    puts "OK!"
elsif 40 < score
    puts "SoSo"
else
    puts "NG!"
end

puts "Put if" if 40 < score



b = 10
c = 20
b, c = 10, 20
a = b < c ? c : b
p a



signal = "blue"
case signal
    when "red"
        puts "STOP!"
    when "yellow"
        puts "CAUTION"
    when "green", "blue"
        puts "GO!"
    else
        puts "Wrong Signal"
end



# Times
3.times do |i|
    puts "#{i} Hello"
end

# While
i = 0
while i < 3 do
    puts "#{i} World"
    i += 1
end

# Break Next
3.times do |i|
    if i == 1
        break
    end
    puts "#{i} Hello World"
end

3.times do |i|
    if i == 1
        next
    end
    puts "#{i} Hello World"
end



for i in 0..2 do
    puts i
end
for color in ["red", "green", "blue"] do
    puts color
end
["red", "green", "blue"].each do |color|
    puts color
end
{"red" => 200, "green" => 300, "blue" => 400}.each do |color, price|
    puts "#{color} = #{price}"
end



def sayHi(name = "Steve")
    s = "Hello!" + name
    puts s
    return s
end
sayHi("Tom")
sayHi("Bob")
tweet = sayHi()
puts tweet



class User

    @@count = 0

    attr_accessor :name
    attr_reader :name
    attr_writer :name

    # CONSTRUCTOR
    def initialize(name)
        @name = name
        @@count += 1
    end

    def sayHi
        puts "Hello my name is #{@name}"
    end

    def User.sayHello
        puts "Hello from User Class : #{@@count}"
    end

    def name
        return @name
    end

    def setName(newName)
        @name = newName
    end

end

User.sayHello()

tom = User.new("Tom")
tom.sayHi()

User.sayHello()

bob = User.new("Bob")
bob.sayHi()

User.sayHello()



class SuperUser < User
    def shout
        puts "HELLO! from #{@name}"
    end
end

lucy = User.new("Lucy")
lucy.sayHi()
superLucy = SuperUser.new("Super Lucy")
superLucy.sayHi()
superLucy.shout()



p lucy.name
lucy.setName("LUCY")
p lucy.name



t = Time.now
p t
p t.year
p t.month

t = Time.new(2014, 12, 25, 12, 23, 45)
p t

t += 10
p t

p t.strftime("Updated: %Y-%m-%d")

