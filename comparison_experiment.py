from pylab import *
import example


binary = example.example(100, 500, 50, 1, None, True, True, True)[1]

standard = {}
eta_vals = 10.0**(-arange(25.0)/5.0)

for eta in eta_vals:
    standard[eta] = example.example(100, 500, 50, eta, None, False, True, True)[1]

