import collections

d = collections.deque(maxlen=3)

d.append('a')
d.append('b')
d.append('c')
d.append('d')

print(d)