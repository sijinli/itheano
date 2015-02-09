"""
Copy from HMLPE part (It I have time, make it into a class)

"""
from init_test import *
from  ipyml.regression import *
import iread.myio as mio
import pylab as pl
import numpy as np
import dhmlpe_utils as dutils
import iutils as iu
def simpleDP(X,Y):
    def f():
        yield X,Y
    return f
def read_inputs():
    d = mio.unpickle('/opt/visal/tmp/for_sijin/Data/H36M/H36MExp/folder_FCJ0_act_12/batches.meta')
    info = d['info']
    print info.keys()
    indexes = info['indexes']
    Y = d['feature_list'][0]
    X = d['feature_list'][1]
    train_range = range(0,76048)
    test_range = range(76048,105368)

    print min(indexes[train_range]), max(indexes[train_range])
    print min(indexes[test_range]), max(indexes[test_range])

    print 'X '
    iu.print_common_statistics(X)
    
    X_train = X[..., train_range]
    Y_train = Y[..., train_range]

    
    feature_dim = X_train.shape[0]
    
    X_test = X[..., test_range]
    Y_test = Y[..., test_range]

    params = {'Sigma':np.ones(feature_dim + 1) * 0.0001}
    r = LinearRegression(params)
    r.fit(simpleDP(X_train,Y_train))
    Y_pred = r.apply(X_test)
    print Y_pred.shape
    print Y_test[:5,:5]
    print Y_pred[:5,:5]
    diff = Y_test - Y_pred
    print 'abs diff = {}'.format(np.sum(diff.flatten()**2))
    mpjpe = dutils.calc_mpjpe_from_residual(diff,17)
    
    print 'average mpjpe  {}'.format(np.mean(mpjpe.flatten()))
    
if __name__ == '__main__':
    read_inputs()

