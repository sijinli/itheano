"""
This file defines some extra intialization methods
Usage:
initWfunc=initfunc.gwfp(/opt/visal/tmp/for_sijin/Data/saved/theano_models/2015_02_02_acm_act_14_exp_2_19_graph_0012/)
initBfunc=initfunc.gbfp(/opt/visal/tmp/for_sijin/Data/saved/theano_models/2015_02_02_acm_act_14_exp_2_19_graph_0012/)
"""
from init_src import *
import numpy as np
import iutils as iu
import iread.myio as mio
import os
import re
from isolver import *
def constant(name, sp, params):
    s = float(params[0])
    if type(sp[0]) is list or type(sp[0]) is tuple:
        return [np.ones(p)* s for p in sp]
    return np.ones(sp) * s
def tryint(s):
    try:
        return int(s)
    except:
        return s
def alphanum_key(s):
    return [tryint(c) for c in re.split('([0-9]+)', s)]
def get_layers(model):
    return model['net_dic'].items()[0][1]['layers']
def get_w_fconv2(name, sp_list, params):
    return get_wb_from_convnet2_checkpoint(name, sp_list, params)
def get_b_fconv2(name, sp, params):
    res = get_wb_from_convnet2_checkpoint(name, sp, params)
    return res.reshape(sp, order='F')
def get_wb_from_convnet2_checkpoint(name, sp, params):
    """
    params = [check_point_path, layer_name, ]
    """
    sys.path.append('/home/grads/sijinli2/Projects/cuda-convnet2/')
    checkpath= params[0]
    layer_name= name if len(params) == 1 else params[1]
    filepath = os.path.join(checkpath, sorted(os.listdir(checkpath), key=alphanum_key)[-1])
    saved = mio.unpickle(filepath)
    model_state = saved['model_state']
    layer = model_state['layers'][layer_name]
    if type(sp[0]) is list or type(sp[0]) is tuple:
        # weights
        n_w = len(sp)
        print '    init from convnet {}--------'.format(checkpath)
        idx_list = range(n_w) if len(params) < 3 else iu.get_int_list_from_str(params[2])
        print ','.join(['{}'.format(layer['weights'][k].shape) for k in idx_list])
        print '--------------\n---------\n'
        return [layer['weights'][k] for k in idx_list]  
    else:
        return layer['biases']
def gwfp(name, sp_list, params):
    """
    get weights from saved model
    params = [model_path, layer_name, [idx,...]]
    """
    n_w = len(sp_list)
    model_path = params[0]
    layer_name = name if len(params) == 1 else params[1]
    model = Solver.get_saved_model(model_path)
    layers = get_layers(model)
    lay = layers[layer_name][2]
    if len(params) < 3:
        idx_list = range(n_w)
    else:
        idx_list = iu.get_int_list_from_str(params[2])
    return [lay['weights'][k] for k in idx_list]
def gbfp(name, sp, params):
    """
    get biases from saved model
    params = [model_path, layer_name, [idx,...]]
    """
    model_path = params[0]
    layer_name = name if len(params) == 1 else params[1]
    model =Solver.get_saved_model(model_path)
    layers = get_layers(model)
    lay = layers[layer_name][2]
    return lay['biases'][0]
def gwns(name, sp_list, params):
    """
    get weights for combining norm and scale layer
    params[0] = model_folder
    params[1] = norm layer name  [source]
    params[2] = scale layer name [source]
    """
    model_folder, norm_name, scale_name = params[0], params[1], params[2]
    stat_folder = iu.fullfile(model_folder, 'stat')
    stat_path = Solver.get_saved_model_path(stat_folder)
    stat = mio.unpickle(stat_path)
    model  = Solver.get_saved_model(model_folder)
    layers = get_layers(model)
    W= layers[scale_name][2]['weights']
    if 'epsilon' in layers[norm_name][2]: 
        epsilon = layers[norm_name][2]['epsilon']
    else:
        epsilon = 1e-6
    # u = stat['layers'][norm_name]['u'].flatten()
    var = stat['layers'][norm_name]['var'].flatten()
    return [W[0] / np.sqrt(var + epsilon)]
def gbns(name,sp, params):
    """
    get bias for combining norm and scale layer
    params[0] = model_folder
    params[1] = norm layer name  [source]
    params[2] = scale layer name [source]
    """
    model_folder, norm_name, scale_name = params[0], params[1], params[2]
    stat_folder = iu.fullfile(model_folder, 'stat')
    stat_path = Solver.get_saved_model_path(stat_folder)
    stat = mio.unpickle(stat_path)
    model  = Solver.get_saved_model(model_folder)
    layers = get_layers(model)
    W= layers[scale_name][2]['weights'][0]
    b= layers[scale_name][2]['biases'][0]
    print 'W-------------'
    iu.print_common_statistics(W)
    print 'b'
    iu.print_common_statistics(b)
    if 'epsilon' in layers[norm_name][2]: 
        epsilon = layers[norm_name][2]['epsilon']
    else:
        epsilon = 1e-6
    u = stat['layers'][norm_name]['u'].flatten()
    var = stat['layers'][norm_name]['var'].flatten()
    return b - W * u / (np.sqrt(var + epsilon))