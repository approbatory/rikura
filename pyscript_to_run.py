import example
import h5py
import time
from datetime import datetime


BINARY = True
epochs = 500
if BINARY:
	errs = example.text(fname='rj_intro.txt', hidden=500, seq_length=10, epochs=epochs, eta=1, binary=True, progress=False)[1]
	label = 'Train stats: binary'
else:
	errs = example.text(fname='rj_intro.txt', hidden=500, seq_length=10, epochs=epochs, eta=1e-3, binary=False, progress=False)[1]
	label = 'Train stats: standard'

start_date = datetime.fromtimestamp(time.time())
f = h5py.File('runs/' + start_date.isoformat() + '.hdf5', 'w')
run_stats = f.create_group(label)
run_stats.create_dataset('errors', data=errs, compression='gzip')

f.close()
