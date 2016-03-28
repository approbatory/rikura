from pylab import *
from rnn import *
from brnn import *
from common import *

def example(hidden=10, examples=1000, epochs=100, eta=0.001, rnn=None, binary=False, progress=True):
    import reber
    DATA = map((lambda x: 2*x-1) if binary else (lambda x: x), map(np.array, reber.get_n_examples(examples)))
    if rnn is None:
        rnn = BRNN(7, hidden, 7) if binary else SRNN(7, hidden, 7)
    pbar = gen_pbar() if progress else (lambda x: x)
    costs = rnn.train_session(DATA, eta, pbar(xrange(epochs)))
    return rnn, costs

def experiment():
    import progressbar as pb
    hiddens = arange(1,101)
    pbar = gen_pbar()
    residuals = array([ example(h, 500, 50, 1e-3, None, False, False)[1] for h in pbar(hiddens) ])
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
