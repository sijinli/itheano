from init_test import *
import theano
from theano import tensor as tensor
from ilayer import *
import iread.myio as mio
from isolver import *
from idata import *
from igraphparser import GraphParser
import pylab as pl
import iutils as iu
import dhmlpe_utils as dutils
from isolver_ext import *


def analyze_mmls_outputs(solver, output_layer_names):
    cur_data = solver.get_next_batch(train=True)
    most_violated_data = solver.find_most_violated(cur_data, train=True)
    alldata = [solver.gpu_require(e.T) for e in most_violated_data[2][1:]]
    output_layers = solver.train_net.get_layer_by_names(output_layer_names)
    outputs= sum([e.outputs for e in output_layers],[])
    f = theano.function(inputs=solver.train_net.inputs,
                        outputs=outputs, on_unused_input='ignore')
    res = f(*alldata)
    max_col = 3
    nbin=100
    n_row = (len(res) - 1)//max_col + 1
    idx = 0

    for e,name in zip(res, output_layer_names):
        idx = idx + 1
        ndata = e.shape[0]
        print 'Layer {} output {} nodes'.format(name, ndata)
        iu.print_common_statistics(e)
        pl.subplot(n_row, max_col, idx)
        pl.hist(e.flatten(), bins=nbin)
        pl.title('Layer({})'.format(name))
    pl.show()
    
    
        
def main():
    solver_loader = MMSolverLoader()
    solver= solver_loader.parse()
    
    output_layer_names = ['net1_fc0', 'net2_fc0', 'net1_fc1', 'net2_fc1', 'net1_fc2',
                          'net2_fc2', 'net2_fc2_dropout', 'net1_fc2_dropout']
    analyze_mmls_outputs(solver, output_layer_names)

if __name__ == '__main__':
    main()