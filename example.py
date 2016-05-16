from pylab import *
from rnn import *
from brnn import *
from common import *

def example(hidden=10, examples=1000, epochs=100, eta=0.001, rnn=None, binary=False, progress=True, embedded=False):
    import reber
    data_source = reber.get_n_embedded_examples if embedded else reber.get_n_examples
    DATA = map((lambda x: 2*x-1) if binary else (lambda x: x), map(np.array, data_source(examples)))
    if rnn is None:
        rnn = BRNN(7, hidden, 7) if binary else SRNN(7, hidden, 7)
    pbar = gen_pbar() if progress else (lambda x: x)
    costs = rnn.train_session(DATA, eta, pbar(xrange(epochs)))

    #validate:
    eta=0
    DATA = map((lambda x: 2*x-1) if binary else (lambda x: x), map(np.array, data_source(examples)))

    pbar = gen_pbar() if progress else (lambda x: x)
    validation_costs = rnn.train_session(DATA, eta, pbar(xrange(epochs)))

    return rnn, costs, validation_costs

def compare_embedded(hidden=100, embedded=True, examples=1000, epochs=100):
    eta_srnn = 0.001

    _, costs_srnn, val_costs_srnn = example(hidden, examples, epochs, eta_srnn, binary=False, embedded=embedded)
    _, costs_brnn, val_costs_brnn = example(hidden, examples, epochs, 1, binary=True, embedded=embedded)
    return (costs_srnn, costs_brnn), (val_costs_srnn, val_costs_brnn)

def triple_comparison():
    import reber
    #data_source = lambda ex: map(lambda x: 2*x-1,map(np.array,reber.get_n_embedded_examples(ex)))
    data_source = lambda ex: map(lambda x: 2*x-1,map(np.array,reber.get_n_examples(ex)))
    word_len = 7
    examples = 200
    epochs = 100
    hidden = 100
    data = data_source(examples)
    rnn = BRNN(word_len, hidden, word_len)
    pbar = gen_pbar()
    #train
    costs = rnn.train_session(data, 1, pbar(xrange(epochs)))

    #validate / measure performance
    #get new data
    data = data_source(examples)
    #pbar = gen_pbar()
    #lazy_method_costs = rnn.train_session(DATA, 0, pbar(xrange(epochs)))

    funcs = dict(prob=rnn.fprop, det=rnn.fprop_multi_single_sample, resample_per_layer_avg=rnn.fprop_per_layer_avg)
    error_bins = dict(prob=[], det=[], resample_per_layer_avg=[])
    for k in funcs:
        for ins, outs in data:
            funcs[k](ins)
            error_bins[k].append(rnn.calculate_cost(outs))

    return rnn, costs, error_bins




def experiment():
    import progressbar as pb
    hiddens = arange(1,101)
    pbar = gen_pbar()
    residuals = array([ example(h, 500, 50, 1, None, True, False)[1] for h in pbar(hiddens) ])
    return hiddens,residuals

def text(fname='aiw.txt', hidden=10, seq_length=10, epochs=10, eta=1, rnn=None, binary=False):
    # Data I/O
    data = open(fname, 'r').read()  # Use this source file as input for RNN
    chars = sorted(list(set(data)))
    data_size, vocab_size = len(data), len(chars)
    print('Data has %d characters, %d unique.' % (data_size, vocab_size))
    char_to_ix = {ch: i for i, ch in enumerate(chars)}
    ix_to_char = {i: ch for i, ch in enumerate(chars)}


    def one_hot(v):
        return np.eye(vocab_size)[v]
    def text_to_repr(text):
        if binary:
            return -1 + 2*one_hot([char_to_ix[ch] for ch in text])
        else:
            return        one_hot([char_to_ix[ch] for ch in text])

    if rnn is None:
        if binary:
            rnn = BRNN(vocab_size, hidden, vocab_size)
        else:
            rnn = SRNN(vocab_size, hidden, vocab_size)

    dataset = [(text_to_repr(data[j  :j+seq_length]),
                text_to_repr(data[j+1:j+seq_length] + data[(j+seq_length+1)%data_size])) for j in xrange(0,data_size,seq_length)]
    costs = rnn.train_session(dataset, eta, xrange(epochs), gen_pbar()(xrange(epochs*len(dataset))))
    return rnn, costs, dataset



def gen_pbar():
    import progressbar as pb
    return pb.ProgressBar(widgets=[pb.Percentage(), pb.Bar(marker=pb.RotatingMarker()),' ',pb.ETA(),' time to learn'])
