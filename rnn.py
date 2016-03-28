from pylab import *
from common import *

#some convenience functions for forward propagation of rnn
def linear_step(w, inp, b):
    return w.dot(inp) + b
def forward_prop(sigma, sigma_, w, inp, b):
    z = linear_step(w, inp, b)
    return (sigma(z), sigma_(z))
def double_forward_prop(sigma, sigma_, w1, inp1, b1, w2, inp2, b2):
    z = linear_step(w1, inp1, b1) + linear_step(w2, inp2, b2)
    return (sigma(z), sigma_(z))

class SRNN(RNN):
    def sigma(self, x):
        return tanh(x)
    def prime(self, x):
        return 1-x**2
    def sigma_(self, x):
        return 1-tanh(x)**2
    def cost_deriv(self, x, a):
        return ce_logit_deriv(x,a)
    def cost(self, x, a):
        #return ce_logit_cost(x,a)
        return any(around(exp(-x)/sum(exp(-x)) * sum(a)) != a) #alternative error
    def __init__(self, input_size, recurrent_size, output_size):
        self.input_size = input_size
        self.recurrent_size = recurrent_size
        self.output_size = output_size
        self.params = {}
        self.params['w'] = randn(recurrent_size, recurrent_size) * 0.1 / sqrt(recurrent_size)
        self.params['wi'] = randn(recurrent_size, input_size) * 0.1 / sqrt(input_size)
        self.params['bi'] = zeros((recurrent_size,))
        self.params['wo'] = randn(output_size, recurrent_size) * 0.1 / sqrt(recurrent_size)
        self.params['bo'] = zeros((output_size,))
    #forward propagate the network using an initial state of init, from time ti to tf in the input list ins, erases previous context
    def fprop(n, ins, init=None, ti=0, tf=None):
        init, tf, c = init, tf, n.context = zeros((n.recurrent_size,)) if init is None else init, len(ins)-1 if tf is None else tf, Context()
        a, a_ = init, 0
        for t in xrange(ti, tf+1):
            c.a[t], c.a_[t] = a, a_ = double_forward_prop(n.sigma, n.sigma_, n.params['wi'], ins[t], n.params['bi'], n.params['w'], a, 0)
            c.out[t], c.out_[t] = linear_step(n.params['wo'], a, n.params['bo']), 1
        return n
    #back propagate ONCE, assumes n is already forward propagated (uses preexisting context)
    def backprop(n, ins, outs, eta, ti=0, tf=None):
        c, tf = n.context, len(ins)-1 if tf is None else tf
        delta_from_right = zeros(n.params['bi'].shape)
        delta = {}
        sum_delta = dict(wo=0,bo=0,w=0,wi=0,bi=0)
        for t in xrange(tf, ti-1, -1):
            delta_from_top = top_backprop_signal(n.cost_deriv(c.out[t], outs[t]), c.out_[t])
            delta['wo'], delta['bo'] = layer_updates(delta_from_top, c.a[t], eta)
            delta_to_left_or_down = backprop_step(delta_from_top, n.params['wo'], c.a_[t]) + backprop_step(delta_from_right, n.params['w'] , c.a_[t])
            delta['w'], _ = (0,0) if t == ti else layer_updates(delta_to_left_or_down, c.a[t-1], eta)
            delta['wi'], delta['bi'] = layer_updates(delta_to_left_or_down, ins[t], eta)
            sum_delta = { k : (sum_delta[k] + delta[k]) for k in delta.keys()}
        n.params = { k : (n.params[k] + sum_delta[k]) for k in sum_delta.keys()}
        return n
