import sys

args = sys.argv

print(args)
print(args[0])
print(args[1])
print(args[2])

num = 1
string = 'abc'
array = [1, 2, 3]
hash = {'a': 1, 'b': 2, 'c': 3}

print(num)
print(string)
print(array)
print(hash)

print('----')

def method(x, y, **args):
    print(x)
    print(y)
    print(args)

args = {'a': 1, 'b': 2}
method(10, 20, **args)

