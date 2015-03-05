"""
Usage
python ~/Projects/Itheano/Python/test/test_plot_mmnet_cost.py --load-file=/opt/visal/tmp/for_sijin/tmp/tmp_theano
"""
from init_test import *
import iread.myio as mio
import pylab as pl
import iutils as iu
import options
from time import time
def get_cost(d, key, cost_name):
    if key in d['model_state']:
        l = d['model_state'][key]
    else:
        l = d['solver_params'][key]
    return [e[cost_name] for e in l]
def mark_peak_point(x,y,c):
    x,y = np.array(x).flatten(), np.array(y).flatten()
    assert(x.size == y.size)
    pre = np.inf
    for i in range(x.size):
        if y[i] > pre and i + 1 < x.size and y[i] > y[i+1]:
            pl.plot(x[i],y[i],marker='*', c=c)
        pre = y[i]
def get_sampling_index(op, n):
    if n > 6000:
        return range(0, 6000)
    max_num = 5000
    if n > max_num:
        step = n // max_num 
        return range(0, n, step)
    else:
        return range(n)
def make_equal(x1,x2, freq):
    t1,t2 = len(x1), len(x2)
    if t1 == t2 * freq:
        return x1,x2
    elif t1 > t2 * freq:
        return x1[:t2*freq], x2
    else:
        t2 = t1 // freq
        return x1[:t2*freq], x2[:t2]
    return [],[]

def plot_cost(op, d, cost_name):
    train_error = get_cost(d, 'train_error', cost_name)
    test_error = get_cost(d, 'test_error', cost_name)
    testing_freq = d['solver_params']['testing_freq']
    print len(train_error), len(test_error), testing_freq
    if (len(train_error) != len(test_error) * testing_freq):
        print 'The length is not equal'
        train_error, test_error = make_equal(train_error, test_error, testing_freq)
    test_error = iu.concatenate_list([ [elem] * testing_freq for elem in test_error ])
    print 'Testing freq is {}'.format(testing_freq)
    
    ndata = len(train_error)
    indexes = get_sampling_index(op, ndata)
    x_values = indexes
    y_train_error = [train_error[k] for k in indexes]
    y_test_error = [test_error[k] for k in indexes]
    if type(y_train_error[0]) is list or type(y_train_error[0]) is tuple:
        n_elem = len(y_train_error[0])
        for k in range(n_elem):
            if k != 0:
                break
            pl.plot(x_values, [y[k] for y in y_train_error] ,c='r', label='train[%d]' % k)
            pl.plot(x_values, [y[k] for y in y_test_error], c='g', label='test[%d]' % k)
    else:
        pl.plot(x_values, y_train_error,c='r', label='train')
        # mark_peak_point(range(ndata), train_error, 'r')
        pl.plot(x_values, y_test_error, c='g', label='test')
    pl.title(cost_name)
    pl.legend()
def process(op):
    data_folder = op.get_value('load_file')
    # data_folder = '/public/sijinli2/ibuffer/2015-01-16/net2_test_for_stat_2000'
    all_files = iu.getfilelist(data_folder, '\d+@\d+$')
    print all_files
    d = mio.unpickle(iu.fullfile(data_folder, all_files[0]))
    ms = d['model_state']
    if op.get_value('cost_name') is not None:
        cost_names = op.get_value('cost_name').split(',')
        n_cost = len(cost_name)
    else:
        n_cost = len(d['solver_params']['train_error'][0])
        cost_names = d['solver_params']['train_error'][0].keys()
    print 'Start to plot'
    start_time = time()
    for i in range(n_cost):
        pl.subplot(n_cost, 1, i + 1)
        plot_cost(op, d, cost_names[i])
    print 'Cost {} seconds '.format(time()- start_time)
    pl.show()
    
def main():
    op = options.OptionsParser()
    op.add_option('load-file', 'load_file', options.StringOptionParser, 'load file folder', default=None,excuses=options.OptionsParser.EXCLUDE_ALL)
    op.add_option('cost-name', 'cost_name', options.StringOptionParser, 'the cost name', default=None)
    op.parse()
    op.eval_expr_defaults()
    process(op)

if __name__ == '__main__':
    main()