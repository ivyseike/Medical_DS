import random

batch = [(1, 10), (2, 5), (3, 4)]
batch_ = random.sample(batch, 2)
print(batch_)
a, b = zip(*batch_)
print(a)
print(b)
