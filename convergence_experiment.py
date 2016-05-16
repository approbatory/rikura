from pylab import *
from example import example

sample_curve_standard = lambda: example(hidden=100, examples=500, epochs=50, eta=0.04, binary=False, embedded=True)[1]
sample_curve_binary = lambda: example(hidden=100, examples=500, epochs=50, eta=1, binary=True, embedded=True)[1]
N = 10

s_collection = []
b_collection = []

for i in range(N):
    s_collection.append(sample_curve_standard())
    b_collection.append(sample_curve_binary())



