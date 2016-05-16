from pylab import *
from common import *

#some convenience functions for forward propagation of binary rnn, refer to formula sheet
def binary_linear_step(w, w_, inp, b, is_first=False):
    K = 1.0*(len(inp) + 1)
    zmu = 1/sqrt(K) * (w.dot(inp) + b)
    zsigma = sqrt(1/K * (1 + (not is_first) * sum(1 - inp**2) + w_.dot(inp**2)))
    return (zmu, zsigma)
def binary_nonlinear_step(zmu, zsigma):
    out = 2 * phi(zmu / zsigma) - 1
    out_ = N0(zmu, zsigma)
    return (out, out_)
def binary_forward_prop(w, w_, inp, b, is_first=False):
    zmu, zsigma = binary_linear_step(w, w_, inp, b, is_first)
    return binary_nonlinear_step(zmu, zsigma)
def binary_add_layers(lin1, lin2):
    m1,s1 = lin1
    m2,s2 = lin2
    return (m1+m2, sqrt(s1**2+s2**2))
def binary_double_forward_prop(w1, w1_, inp1, b1, w2, w2_, inp2, b2, is_first1=False, is_first2=False):
    lin1 = binary_linear_step(w1, w1_, inp1, b1, is_first1)
    lin2 = binary_linear_step(w2, w2_, inp2, b2, is_first2)
    zmu, zsigma = binary_add_layers(lin1, lin2)
    return binary_nonlinear_step(zmu, zsigma)
def sample_array(arr):
    return sign(arr - (2*sample(shape(arr))-1))


class BRNN(RNN):
    def sigma(self, x):
        return tanh(x)
    def prime(self, x):
        return 1-x**2
    def sigma_(self, x):
        return 1 - tanh(x)**2
    def cost_deriv(self, z_out, y):
        return y/phi(y*z_out)
    def cost(self, z_out, y):
        return any(sign(z_out) != y)
    def __init__(self, input_size, recurrent_size, output_size):
        self.input_size = input_size
        self.recurrent_size = recurrent_size
        self.output_size = output_size
        self.params = {}
        self.params['h'] = randn(recurrent_size, recurrent_size) * 0.1 / sqrt(recurrent_size)
        self.params['hi'] = randn(recurrent_size, input_size) * 0.1 / sqrt(input_size)
        self.params['bi'] = zeros((recurrent_size,))
        self.params['ho'] = randn(output_size, recurrent_size) * 0.1 / sqrt(recurrent_size)
        self.params['bo'] = zeros((output_size,))
        self.aux, self.aux_ = {}, {}
        self.aux['h'] , self.aux_['h']  = self.sigma(self.params['h'] ), self.sigma_(self.params['h'] )
        self.aux['ho'], self.aux_['ho'] = self.sigma(self.params['ho']), self.sigma_(self.params['ho'])
        self.aux['hi'], self.aux_['hi'] = self.sigma(self.params['hi']), self.sigma_(self.params['hi'])
    def fprop(n, ins, init=None, ti=0, tf=None):
        init, tf, c = init, tf, n.context = zeros((n.recurrent_size,)) if init is None else init, len(ins)-1 if tf is None else tf, Context()
        a, a_ = init, 0
        for t in xrange(ti, tf+1):
            c.a[t], c.a_[t] = a, a_ = binary_double_forward_prop(n.aux['hi'], n.aux_['hi'], ins[t], n.params['bi'],
                                                                 n.aux['h'] , n.aux_['h'] , a     , 0,    is_first1=True)
            zmu_out, zsigma_out = binary_linear_step(n.aux['ho'], n.aux_['ho'], a, n.params['bo'])
            c.out[t] = zmu_out / zsigma_out
            _, c.out_[t] = binary_nonlinear_step(zmu_out, zsigma_out)
        return n
    def backprop(n, ins, outs, eta, ti=0, tf=None):
        c, tf = n.context, len(ins)-1 if tf is None else tf
        delta_from_right = zeros(n.params['bi'].shape)
        delta = {}
        sum_delta = dict(ho=0,bo=0,h=0,hi=0,bi=0)
        for t in xrange(tf, ti-1, -1):
            delta_from_top = top_backprop_signal(n.cost_deriv(c.out[t], outs[t]), c.out_[t])
            nanz = ~isfinite(delta_from_top)
            delta_from_top[nanz] = 0 #TODO???
            delta['ho'], delta['bo'] = layer_updates(delta_from_top, c.a[t], eta, binary=True)
            delta_to_left_or_down = backprop_step(delta_from_top, n.aux['ho'], c.a_[t], binary=True) + backprop_step(delta_from_right, n.aux['h'], c.a_[t], binary=True)
            delta['h'], _ = (0,0) if t == ti else layer_updates(delta_to_left_or_down, c.a[t-1], eta, binary=True)
            delta['hi'], delta['bi'] = layer_updates(delta_to_left_or_down, ins[t], eta, binary=True)
            sum_delta = { k : (sum_delta[k] + delta[k]) for k in delta.keys()}
        n.params = { k : (n.params[k] + sum_delta[k]) for k in sum_delta.keys()}
        n.aux = { k : n.sigma(n.params[k]) for k in n.aux.keys()}
        n.aux_= { k : n.prime(n.aux[k])    for k in n.aux_.keys()}
        return n

    def fprop_det(n, ins, init=None, ti=0, tf=None):
        init, tf, c = init, tf, n.context = zeros((n.recurrent_size,)) if init is None else init, len(ins)-1 if tf is None else tf, Context()
        a = init
        for t in xrange(ti, tf+1):
            c.a[t] = a = sign( sign(n.params['hi']).dot(ins[t]) + sign(n.params['h']).dot(a) + n.params['bi'])
            c.out[t] = sign( sign(n.params['ho']).dot(a) + n.params['bo'])
        return n

    def fprop_resample(n, ins, init=None, ti=0, tf=None):
        init, tf, c = init, tf, n.context = zeros((n.recurrent_size,)) if init is None else init, len(ins)-1 if tf is None else tf, Context()
        a = init
        for t in xrange(ti, tf+1):
            c.a[t] = a = sign( sample_array(n.aux['hi']).dot(ins[t]) + sample_array(n.aux['h']).dot(a) + n.params['bi'])
            c.out[t] = sign( sample_array(n.aux['ho']).dot(a) + n.params['bo'])
        return n

    def fprop_per_layer_avg(n, ins, init=None, ti=0, tf=None, reps=1000):
        init, tf, c = init, tf, n.context = zeros((n.recurrent_size,)) if init is None else init, len(ins)-1 if tf is None else tf, Context()
        a = init
        for t in xrange(ti, tf+1):
            c.a[t] = 0
            for i in xrange(reps):
                c.a[t] += sign( sample_array(n.aux['hi']).dot(ins[t]) + sample_array(n.aux['h']).dot(a) + n.params['bi'])
            c.a[t] = 1.0 * c.a[t] / reps
            a = c.a[t]

            c.out[t] = 0
            for i in xrange(reps):
                c.out[t] += sign( sample_array(n.aux['ho']).dot(a) + n.params['bo'])
            c.out[t] = 1.0 * c.out[t] / reps
        return n




    def fprop_single_sample(n, ins, init=None, ti=0, tf=None):
        init, tf, c = init, tf, n.context = zeros((n.recurrent_size,)) if init is None else init, len(ins)-1 if tf is None else tf, Context()
        a = init
        wi = sample_array(n.aux['hi']); w = sample_array(n.aux['h']); wo = sample_array(n.aux['ho'])
        for t in xrange(ti, tf+1):
            c.a[t] = a = sign( wi.dot(ins[t]) + w.dot(a) + n.params['bi'])
            c.out[t] = sign( wo.dot(a) + n.params['bo'])
        return n


    def fprop_multi_resample(n, ins, init=None, ti=0, tf=None, reps=1000):
        action = lambda: n.fprop_resample(ins, init=None, ti=0, tf=None)
        out = action().context.out
        for i in xrange(1,reps):
            temp_out = action().context.out
            out = { k : out[k] + temp_out[k] for k in out.keys() }
        out = { k : 1.0*out[k]/reps for k in out.keys() }
        n.context.out = out
        return n

    def fprop_multi_single_sample(n, ins, init=None, ti=0, tf=None, reps=1000):
        action = lambda: n.fprop_single_sample(ins, init=None, ti=0, tf=None)
        out = action().context.out
        for i in xrange(1,reps):
            temp_out = action().context.out
            out = { k : out[k] + temp_out[k] for k in out.keys() }
        out = { k : 1.0*out[k]/reps for k in out.keys() }
        n.context.out = out
        return n
