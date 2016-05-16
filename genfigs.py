from example import *

if False:
    figure()
    h = 70
    rnn, costs = example(hidden=h, examples=1000, epochs=100, eta=1, rnn=None, binary=True, progress=True)
    plot(costs)
    title('Binary RNN with %d Hidden Units' % h)
    xlabel('Epoch #')
    ylabel('Discrete Error')
    savefig('error_curve.svg')

    figure()
    hist(ravel(rnn.aux['h']))
    title('Histogram of Recurrent Weight Average Values')
    savefig('histogram.svg')

if True:
    figure()
    hids, resids = experiment()
    plots(hids, resids[:,-1])
    title('Residual Error for BRNN vs. Hidden Layer Size')
    xlabel('Hidden layer size')
    ylabel('Residual error')
    savefig('residual_error.svg')
