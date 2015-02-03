"""
This file is for testing the 
"""
from init_test import *
from ilayer import *
from idata import *
from isolver import *

from igraphparser import *

import iutils as iu
def create_network(layers):
    config_dic = {}
def print_dims(alldata):     # debug use
    for i,e in enumerate(alldata):
        print 'Dim {}: \t shape {} \t type {}'.format(i, e.shape, type(e))
def pre_process_data(d):
    d['feature_list'][0] = d['feature_list'][0] / 1200
def merge_dic(s1,s2):
    s = s1.copy()
    if s2:
        for e in s2:
            s[e] = s2[e]
    return s
def create_dp(train_ext_params=None, test_ext_params=None):
    meta_path = '/opt/visal/tmp/for_sijin/Data/H36M/H36MExp/folder_SP_t004_act_12/batches.meta'
    d = mio.unpickle(meta_path)
    print 'data format'
    print_dims(d['feature_list'])
    train_range = range(0, 76048)
    test_range = range(76048, 105368)
    params = {'batch_size':1024}
    train_params = merge_dic(params, train_ext_params)
    test_params = merge_dic(params, test_ext_params)
    pre_process_data(d)
    
    train_dp = MemoryDataProvider(data_dic=d, train=True, data_range=train_range, params=train_params)
    test_dp = MemoryDataProvider(d, train=False, data_range=test_range, params=test_params)
    print 'Create Data Provider Successfully'
    return train_dp, test_dp
def create_dp2(train_ext_params=None, test_ext_params=None):
    meta_path = '/opt/visal/tmp/for_sijin/Data/H36M/H36MExp/folder_SP_t004_act_14/batches.meta'
    d = mio.unpickle(meta_path)
    print 'data format'
    print_dims(d['feature_list'])
    train_range = range(0, 132744)
    test_range = range(132744, 162008)
    params = {'batch_size':1024}
    train_params = merge_dic(params, train_ext_params)
    test_params = merge_dic(params, test_ext_params)
    pre_process_data(d)
    
    train_dp = MemoryDataProvider(data_dic=d, train=True, data_range=train_range, params=train_params)
    test_dp = MemoryDataProvider(d, train=False, data_range=test_range, params=test_params)
    print 'Create Data Provider Successfully'
    return train_dp, test_dp
def create_layers():
    s_img_feature = tensor.fmatrix('s_img_feature')
    s_target_feature = tensor.fmatrix('s_target_feature')
    s_candidate_feature = tensor.fmatrix('s_candidate_feature')
    s_gt_margin = tensor.fmatrix('s_gt_margin')
    s_candidate_margin = tensor.fmatrix('s_candidate_margin')

    l_img_feature = DataLayer(inputs=[s_img_feature], param_dic={'input_dims':[1024],
                                                                 'name':'l_img_feature'
                                                             })
    l_target_feature = DataLayer(inputs=[s_target_feature], param_dic={'input_dims':[1024],
                                                                       'name':'l_target_feature'
                                                                   })
    l_candidate_feature = DataLayer(inputs=[s_candidate_feature],param_dic={'input_dims':[1024],
                                                                        'name':'l_candidate_feature'})
    l_gt_margin = DataLayer(inputs=[s_gt_margin],param_dic={'input_dims':[1],
                                                            'name':'l_gt_margin'
                                                        })
    l_candidate_margin = DataLayer(inputs=[s_candidate_margin],param_dic={'input_dims':[1],
                                                                          'name':'l_candidate_margin'
                                                                      })


    net1_fc1 = FCLayer(inputs=l_img_feature.outputs + l_target_feature.outputs,
                  param_dic={'input_dims':[1024,1024], 'output_dims':[1024],
                             'name':'net1_fc1', 'wd':[0.00001, 0.00001], 'initW':[0.1,0.1],
                             'initb':[0], 'actfunc':{'name':'relu','act_params':None},
                         }
    )

    net1_fc2 = FCLayer(inputs=net1_fc1.outputs,
                  param_dic={'input_dims':net1_fc1.param_dic['output_dims'],
                             'output_dims':[1],
                             'name':'net1_fc2', 'wd':[0.00001],'initW':[0.1], 'initb':[0],
                             'actfunc':{'name':'relu','act_params':None},
                         }
    )

    net1_score = ElementwiseSumLayer(inputs=net1_fc2.outputs + l_gt_margin.outputs,
                                     param_dic={'name':'net1_score',
                                                'coeffs':[1,1],
                                                'output_dims':[1],
                                                'input_dims':[1,1]
                                            }
    )


    net2_fc1 = FCLayer(inputs=l_img_feature.outputs + l_candidate_feature.outputs,
                       param_dic={'input_dims':[1024,1024], 'output_dims':[1024],
                                  'name':'net2_fc1', 'wd':[0.00001, 0.00001], 'initW':[0.1,0.1],
                                  'initb':[0],'actfunc':{'name':'relu','act_params':None},
                                  'weights':net1_fc1.W_list,
                                  'biases':net1_fc1.b
                         }
    )

    net2_fc2 = FCLayer(inputs=net2_fc1.outputs,
                  param_dic={'input_dims':net2_fc1.param_dic['output_dims'],
                             'output_dims':[1],
                             'name':'net2_fc2', 'wd':[0.00001],'initW':[0.01],
                             'initb':[0], 'actfunc':{'name':'relu','act_params':None},
                             'weights':net1_fc2.W_list,
                             'biases':net1_fc2.b
                         }
    )

    net2_score = ElementwiseSumLayer(inputs=net2_fc2.outputs +  l_candidate_margin.outputs,
                                     param_dic={'name':'net2_score',
                                                'coeffs':[1,1],
                                                'output_dims':[1],
                                                'input_dims':[1,1]
                                            }
    )
    
    net2_mmcost = MaxMarginCostLayer(inputs=net1_score.outputs + net2_score.outputs,param_dic={'name':'net2_mmcost', 'input_dims':[1,1]})



    #---------------- Net 3
    net3_eltsum1 = ElementwiseSumLayer(inputs=l_img_feature.outputs +
                                       l_target_feature.outputs,
                                       param_dic={'name':'net3_eltsum1',
                                                  'coeffs':[1,-1],
                                                  'actfunc':{'name':'abs','act_params':None},
                                                  'output_dims':[1024],
                                                  'input_dims':[1024,1024]
                                              }
    )

    net3_fc2 = FCLayer(inputs=net3_eltsum1.outputs,
                  param_dic={'input_dims':net3_eltsum1.param_dic['output_dims'],
                             'output_dims':[1],
                             'input_dims':[1024],
                             'name':'net3_fc2', 'wd':[0.00001],'initW':[1], 'initb':[0],
                             'actfunc':{'name':'relu','act_params':None},
                         }
    )

    net3_score = ElementwiseSumLayer(inputs=net3_fc2.outputs +  l_gt_margin.outputs,
                                     param_dic={'coeffs':[1,1],
                                                'output_dims':[1],
                                            'name':'net3_score','input_dims':[1,1]}
    )


    net4_eltsum1 = ElementwiseSumLayer(inputs=l_img_feature.outputs +
                                   l_candidate_feature.outputs,
                                   param_dic={'name':'net4_eltsum1',
                                              'coeffs':[1,-1],
                                              'actfunc':{'name':'abs','act_params':None},
                                              'output_dims':[1024],
                                              'input_dims':[1024,1024]
                                          }
    )

    net4_fc2 = FCLayer(inputs=net4_eltsum1.outputs,
                  param_dic={'input_dims':net4_eltsum1.param_dic['output_dims'],
                             'output_dims':[1],
                             'name':'net4_fc2', 'wd':[0.01],'initW':[1],
                             'initb':[0], 'actfunc':{'name':'relu','act_params':None},
                             'weights':net3_fc2.W_list,
                             'biases':net3_fc2.b
                         }
    )    
    net4_score = ElementwiseSumLayer(inputs=net4_fc2.outputs + l_candidate_margin.outputs,
                                     param_dic={'name':'net4_score',
                                                'coeffs':[1,1],
                                                'output_dims':[1],
                                                'input_dims':net4_fc2.param_dic['output_dims']
                                            })
    net4_mmcost = MaxMarginCostLayer(inputs=net3_score.outputs + net4_score.outputs,param_dic={'name':'net4_mmcost','input_dims':[1,1]})
    #----------------
    

    
    layers = {'l_img_feature':[[],[], l_img_feature],
              'l_target_feature':[[],[], l_target_feature],
              'l_candidate_feature':[[],[], l_candidate_feature],
              'l_gt_margin':[[],[], l_gt_margin],
              'l_candidate_margin':[[],[], l_candidate_margin],
              'net1_score':[[],[], net1_score],
              'net2_score':[[],[], net2_score],
              'net1_fc1':[[], [], net1_fc1],
              'net1_fc2':[[], [], net1_fc2],
              'net2_fc1':[[], [], net2_fc1],
              'net2_fc2':[[], [], net2_fc2],
              'net2_mmcost':[[], [], net2_mmcost],
              'net3_eltsum1':[[],[], net3_eltsum1],
              'net3_fc2':[[],[], net3_fc2],
              'net4_eltsum1':[[],[], net4_eltsum1],
              'net4_fc2':[[],[], net4_fc2],
              'net3_score':[[],[], net3_score],
              'net4_score':[[],[], net4_score],
              'net4_mmcost':[[],[], net4_mmcost]
    }
    def run1():
        net1_config_dic = {'cost_layer_names':[],
                           'data_layer_names':['l_img_feature', 'l_target_feature', 'l_gt_margin'],
                           'output_layer_names':['net1_score'],
                           'layer_with_weights':['net1_fc1', 'net1_fc2']
        }
        net2_config_dic = {'cost_layer_names':['net2_mmcost'],
                       'data_layer_names':['l_img_feature', 'l_target_feature',
                                           'l_candidate_feature', 'l_gt_margin',
                                           'l_candidate_margin'],
                       'output_layer_names':['net1_fc2', 'net2_fc1'],
                       'layer_with_weights':['net1_fc1', 'net1_fc2']
        }
        eval_net = Network(layers, net1_config_dic)
        train_net = Network(layers, net2_config_dic)
        print eval_net
        print train_net
        print '-----------------'
        print 'Create layer succussfully'
        # for a in ['inputs', 'layers', 'outputs', 'costs']:
        #     print 'The type of {} :\t{}\t'.format(a, type(getattr(eval_net,a))),
        #     if type(getattr(eval_net,a)) is list:
        #         print 'element type is {}'.format(type(getattr(eval_net,a)[0]))
        #     print '.'
        return layers, eval_net, train_net
    def run2():
        net3_config_dic = {'cost_layer_names':[],
                           'data_layer_names':['l_img_feature', 'l_target_feature', 'l_gt_margin'],
                           'output_layer_names':['net3_score'],
                           'layer_with_weights':['net3_fc2']
        }
        net4_config_dic = {'cost_layer_names':['net4_mmcost'],
                           'data_layer_names':['l_img_feature', 'l_target_feature',
                                               'l_candidate_feature', 'l_gt_margin',
                                               'l_candidate_margin'],
                           'output_layer_names':['net3_fc2', 'net4_fc2'],
                           'layer_with_weights':['net3_fc2']
        }
        eval_net = Network(layers, net3_config_dic)
        train_net = Network(layers, net4_config_dic)
        return layers, eval_net, train_net
    return run1()

def create_layers2():
    file_path = '/home/grads/sijinli2/Projects/Itheano/doc/netdef/graph_def_0001.cfg'
    from igraphparser import GraphParser
    a = GraphParser(file_path)
    layers = a.layers
    net1_config_dic = a.network_config['network1']
    net2_config_dic = a.network_config['network2']
    eval_net = Network(layers, net1_config_dic)
    train_net = Network(layers, net2_config_dic)
    return layers, eval_net, train_net
def process():
    train_dp, test_dp = create_dp2()
    layers, eval_net, train_net = create_layers2()
    solver_params = {'num_epoch':200, 'save_path':'/opt/visal/tmp/for_sijin/tmp/itheano_test_act14',
                     'testing_freq':1, 'K_candidate':2000, 'max_num':10
    }
    iu.ensure_dir(solver_params['save_path'])
    solvers = MMLSSolver([eval_net, train_net], train_dp, test_dp, solver_params)
    GraphParser.print_graph_connections(layers)
    solvers.train()
def main():
    process()

if __name__ == '__main__':
    main()
