"""
Examples:
python --data-path=/opt/visal/tmp/for_sijin/Data/H36M/H36MExp/folder_SP_t004_act_14 --num-epoch=200 --data-provider=mem --layer-def=/home/grads/sijinli2/Projects/Itheano/doc/netdef/graph_def_0002.cfg --testing-freq=1 --batch-size=1024 --train-range=0-132743 --test-range=132744-162007 --save-path=/public/sijinli2/ibuffer/2015-01-16/net2_test_for_stat --K-candidate=200

python ~/Projects/Itheano/Python/task/train_mmls.py --data-path=/opt/visal/tmp/for_sijin/Data/H36M/H36MExp/folder_SP_t004_act_14 --num-epoch=200 --data-provider=mem --layer-def=/home/grads/sijinli2/Projects/Itheano/doc/netdef/graph_def_0003.cfg --testing-freq=1 --batch-size=1024 --train-range=0-132743 --test-range=132744-162007 --save-path=/public/sijinli2/ibuffer/2015-01-16/net3_test_for_stat --K-candidate=200

python ~/Projects/Itheano/Python/task/train_mmls.py --data-path=/opt/visal/tmp/for_sijin/Data/H36M/H36MExp/folder_SP_t004_act_14 --num-epoch=200 --data-provider=mem --layer-def=/home/grads/sijinli2/Projects/Itheano/doc/netdef/graph_def_0009.cfg --testing-freq=1 --batch-size=1024 --train-range=0-132743 --test-range=132744-162007 --save-path=/opt/visal/tmp/for_sijin/tmp/tmp_theano --K-candidate=200 --K-most-violated=100 --margin-func=rbf,0.0416666667 --max-num=20

python ~/Projects/Itheano/Python/task/train_mmls.py --data-path=/opt/visal/tmp/for_sijin/Data/H36M/H36MExp/folder_SP_t004_act_14 --num-epoch=200 --data-provider=mem --layer-def=/home/grads/sijinli2/Projects/Itheano/doc/netdef/graph_def_0009.cfg --testing-freq=1 --batch-size=1024 --train-range=0-132743 --test-range=132744-162007 --save-path=/opt/visal/tmp/for_sijin/tmp/tmp_theano --K-candidate=200 --K-most-violated=100 --max-num=20

python ~/Projects/Itheano/Python/task/train_mmls.py --data-path=/opt/visal/tmp/for_sijin/Data/H36M/H36MExp/folder_SP_t004_act_14 --num-epoch=200 --data-provider=mem --layer-def=/home/grads/sijinli2/Projects/Itheano/doc/netdef/graph_def_0009.cfg --testing-freq=1 --batch-size=1024 --train-range=0-132743 --test-range=132744-162007 --save-path=/opt/visal/tmp/for_sijin/tmp/tmp_theano_t --K-candidate=200 --K-most-violated=100 --max-num=20

"""
from init_task import *
import options
import iread.myio as mio
import options
from ilayer import *
from idata import *
from isolver import *
from igraphparser import *
from isolver_ext import *
import sys

def pre_process_data(d):
    print 'I have preprocess the data by dividing feature_0 '
    d['feature_list'][0] = d['feature_list'][0] / 1200
    

            
def copy_dic_by_key(target, source, key_list):
    for k in key_list:
        target[k] = source[k]
def create_dp(op, saved_model = None):
    try:
        if saved_model and 'dp_params' in saved_model['solver_params']:
            dp_params = saved_model['solver_params']['dp_params']
        else:
            dp_params = None
        required_fields = ['data_path', 'data_provider', 'train_range', 'test_range',
                           'batch_size']
        d = dict()
        for e in required_fields:
            if op.get_value(e):
                d[e] = op.get_value(e)
            else:
                d[e] = dp_params[e]
    except Exception as e:
        print e
        sys.exit(1)
    meta= mio.unpickle(iu.fullfile(d['data_path'], 'batches.meta'))
    param1, param2 = dict(), dict()
    if saved_model:
        copy_dic_by_key(param1, saved_model['model_state']['train'], ['epoch', 'batchnum'])
        copy_dic_by_key(param2, saved_model['model_state']['test'], ['epoch', 'batchnum'])
    copy_dic_by_key(param1, d, ['batch_size'])
    copy_dic_by_key(param2, d, ['batch_size'])
    dp_type = d['data_provider']
    pre_process_data(meta) # < ---- This should be used with care
    train_dp = dp_dic[dp_type](data_dic=meta, train=True, data_range=d['train_range'], params=param1)
    test_dp = dp_dic[dp_type](data_dic=meta, train=False, data_range=d['test_range'], params=param2)
    return d, train_dp, test_dp

def create_layers(op, saved_model = None):
    try:
        cfg_file_path = op.get_value('layer_def')
    except Exception as e:
        print e
        sys.exit(1)
    g= GraphParser(cfg_file_path)
    layers = g.layers
    if saved_model:
        Solver.clone_weights(layers, saved_model.train_net['layers'])
    ## In the future, this part will depends on the inputs
    ## now it assumes the structure of maxmimum-margin network
    net1_config_dic = g.network_config['network1']
    net2_config_dic = g.network_config['network2']
    eval_net = Network(layers, net1_config_dic)
    train_net = Network(layers, net2_config_dic)
    return layers, eval_net, train_net
def create_solver_params(op, saved_model = None):
    # solver_params = {'max_num':10}
    solver_params = dict()
    saved_params = saved_model['solver_params'] if saved_model else None 
    required_list = ['num_epoch', 'testing_freq', 'K_candidate',
                     'max_num', 'save_path', 'K_most_violated']
    # It is able to change save_path ???? < OK, it is fine.
    for e in required_list:
        if (not e in solver_params):
            solver_params[e] = op.get_value(e)
        elif saved_params:
            solver_params[e] = saved_params[e]
    if op.get_value('margin_func') is not None:
        s = op.get_value('margin_func').split(',')
        name = s[0]
        margin_params = None if len(s) == 1 else [float(x) for x in s[1:]]
        solver_params['margin_func'] = {'name':name, 'params':margin_params}
    else:
        solver_params['margin_func'] = {'name':'mpjpe', 'params':None}
    print solver_params
    return solver_params
    # try:
    #     # allow to change ['num_epoch', testing_freq, K_candidate, max_num]
    #     if saved_model:

                
def main():
    op = options.OptionsParser()
    op.add_option("load-file", "load_file", \
                  options.StringOptionParser, 'The experiment name', \
                  default='', excuses=options.OptionsParser.EXCLUDE_ALL)
    op.add_option("data-path", "data_path", \
                  options.StringOptionParser, 'the path for data',default='')
    op.add_option("num-epoch", "num_epoch", \
                  options.IntegerOptionParser, 'The number of epoch',default='')
    op.add_option("data-provider", "data_provider", \
                  options.StringOptionParser, 'The type of Data Provider',default='')
    op.add_option("layer-def", "layer_def", \
                  options.StringOptionParser, 'layer definition file (.cfg)',default='')
    op.add_option('testing-freq', 'testing_freq', options.IntegerOptionParser, 'testing frequency',
                  default=1)
    op.add_option('batch-size', 'batch_size', options.IntegerOptionParser, 'batch size',
                  default=None)
    op.add_option('train-range', 'train_range', options.RangeOptionParser, 'train range',
                  default=None)
    op.add_option('test-range', 'test_range', options.RangeOptionParser, 'test range',
                  default=None)
    op.add_option('save-path', 'save_path', options.StringOptionParser, 'save path', default=None)
    op.add_option('K-candidate', 'K_candidate', options.IntegerOptionParser, 'the number of candidate', default=None)
    op.add_option('K-most-violated', 'K_most_violated', options.IntegerOptionParser, 'the size of the hold-on set', default=None)
    op.add_option('max-num', 'max_num', options.IntegerOptionParser, 'The maximum number of sample to be processed')
    op.add_option('margin-func', 'margin_func', options.StringOptionParser, 'the parameters for marginl', default='mpjpe')
    op.parse()
    op.eval_expr_defaults()
    if op.get_value('load_file'):
        saved_model = Solver.get_saved_model(op.get_value('load_file'))
    else:
        saved_model = None
    dp_params, train_dp, test_dp = create_dp(op, saved_model)
    layers, eval_net, train_net = create_layers(op, saved_model)
    GraphParser.print_graph_connections(layers)
    solver_params = create_solver_params(op, saved_model)
    solver_params['dp_params'] = dp_params # Just for easy resuming the network
    iu.ensure_dir(solver_params['save_path'])
    solvers = MMLSSolver([eval_net, train_net], train_dp, test_dp, solver_params)
    solvers.train()
def main_test():
    mmlsloader = MMSolverLoader()
    solver = mmlsloader.parse()
    solver.train()
if __name__ == '__main__':
    # main()
    main_test()