from pylab import *
from scipy.stats import norm as norm_dist


#backprop core functions, these are combined in the back propagation stage
#These are essentially common between the RNN and BRNN
def layer_updates(delta, mat_input, eta, binary=False):
    delta_b = (1/sqrt(len(mat_input)+1) if binary else -1) * eta * delta
    return outer(delta_b, mat_input), delta_b
def top_backprop_signal(cost_grad, q_top):
    return cost_grad * q_top
def backprop_step(delta_upper, w_upper, q_lower, binary=False):
    return (2/sqrt(len(q_lower)+1) if binary else 1) * w_upper.T.dot(delta_upper) * q_lower


#commonly used standard functions:
N0 = lambda mu,sigma: norm_dist.pdf(0, mu, sigma)
phi = lambda x: norm_dist.cdf(x,0,1)

mse_deriv = lambda x,a: x-a
mse_cost  = lambda x,a: norm(x-a)**2

ce_logit_deriv = lambda x,a: a/sum(a) - exp(-x)/sum(exp(-x))
def ce_logit_cost(x,a):
    a = a/sum(a)
    ce = a.dot(x) + log(sum(exp(-x)))
    a[a == 0] = 1
    return ce + a.dot(log(a))

#definition of the computational context of RNN or BRNN
class Context:
    def __init__(self):
        self.a, self.a_ = {}, {}
        self.out, self.out_ = {}, {}



class RNN:
    #common method of calculating error for RNN and BRNN
    def calculate_cost(rnn, outs, ti=0, tf=None):
        c, tf = rnn.context, len(outs)-1 if tf is None else tf
        cost = 0
        for t in xrange(ti, tf+1):
            cost += rnn.cost(c.out[t], outs[t])
        return cost*1.0 / (tf-ti+1)
    def train(self, ins, outs, eta):
        self.fprop(ins)
        self.backprop(ins, outs, eta)
        return self.calculate_cost(outs)
    def train_session(self, data, eta, epoch_iterable, progress=None):
        res = []
        for _ in epoch_iterable:
            count, accum = 0, 0
            for ins, outs in data:
                accum += self.train(ins,outs,eta)
                if not (progress is None):
                    next(progress)
                count += 1
            res.append(accum*1.0/count)
        return res

        #res = [sum(self.train(ins,outs,eta) for ins, outs in data) for _ in epoch_iterable]


