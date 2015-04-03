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
import imgproc

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
    save_path = op.get_value('save_path')
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
    if save_path:
        imgproc.imsave_tight(save_path)
    pl.show()

def show_cnn_filters(lay, op):
    # The filter shape is numfilter  x input_dimension  x sy  x sx
    weights = lay['weights']
    idx = op.get_value('weight_idx')
    W = weights[idx]
    nfilter, ndim, SY, SX = W.shape
    n_col = 8
    n_row = (nfilter - 1) // n_col + 1
    nc = 0
    fig = pl.figure()
    for r in range(n_row):
        for c in range(n_col):
            ax = fig.add_subplot(n_row, n_col, nc + 1)
            curF = W[nc, ...]
            img = curF.transpose([1,2,0]) if curF.shape[0] == 3 else curF.mean(axis=0)
            ax.imshow(imgproc.maptorange(-img, [0,1]))
            iu.print_common_statistics(img)
            pl.title('filter idx {:02d}'.format(nc))
            nc = nc + 1
    pl.show()
    
def show_fc_filters(lay, op):
    pass
def show_filter(op):
    import isolver
    lay_name = op.get_value('layer_name')
    data_folder = op.get_value('load_file')
    model = isolver.Solver.get_saved_model(data_folder)
    layers = model['net_dic'].items()[0][1]['layers']
    cur_lay = layers[lay_name][2]
    print 'Processing layer: {}'.format(lay_name)
    print 'Layer type is {}'.format(cur_lay['type'])
    t = cur_lay['type']
    if t in 'convdnn':
        show_cnn_filters(cur_lay, op)
    else:
        show_fc_filters(cur_lay,op)
def main():
    op = options.OptionsParser()
    op.add_option('load-file', 'load_file', options.StringOptionParser, 'load file folder', default=None,excuses=options.OptionsParser.EXCLUDE_ALL)
    op.add_option('cost-name', 'cost_name', options.StringOptionParser, 'the cost name', default=None)
    op.add_option('save-path', 'save_path', options.StringOptionParser, 'The path to save plot', default=None)
    op.add_option('mode', 'mode', options.StringOptionParser, 'The mode of plot', default='show_cost')
    op.add_option('layer-name','layer_name', options.StringOptionParser, 'The layer to be analyzed', default=None)
    op.add_option('weight-idx', 'weight_idx', options.IntegerOptionParser, 'The index of filter to be displayed', default=0)
    op.parse()
    op.eval_expr_defaults()
    mode = op.get_value('mode')
    if mode == 'show_cost':
        process(op)
    elif mode == 'show_filter':
        show_filter(op)
        
if __name__ == '__main__':
    main()