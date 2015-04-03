from init_src import *
import theano
import theano.tensor as tensor
import numpy as np
import dhmlpe_utils as dutils
from time import time
import options
import sys

import iutils as iu
from time import gmtime, strftime
import os
import iread.myio as mio

import idata
import igraphparser
from ilayer import Network # Maybe I will move to other files latter
class SolverLoader(object):
    def __init__(self, op = None):
        if op is None:
            self.op = options.OptionsParser()
        else:
            self.op = op
        self.add_default_options(self.op)
        self.dp_params = None
    def add_default_options(self, op):
        op.add_option("load-file", "load_file", \
                      options.StringOptionParser, 'The experiment name', \
                      default='', excuses=options.OptionsParser.EXCLUDE_ALL)
        op.add_option("data-path", "data_path", \
                      options.StringOptionParser, 'the path for data',default=None)
        op.add_option("num-epoch", "num_epoch", \
                      options.IntegerOptionParser, 'The number of epoch',default=None)
        op.add_option("data-provider", "data_provider", \
                      options.StringOptionParser, 'The type of Data Provider',default=None)
        op.add_option("layer-def", "layer_def", \
                      options.StringOptionParser, 'layer definition file (.cfg)',default=None)
        op.add_option('testing-freq', 'testing_freq', options.IntegerOptionParser,
                      'testing frequency', default=1)
        op.add_option('force-shuffle', 'force_shuffle', options.BooleanOptionParser, 'Whether to force the dataprovider to shuffle data', default=False)
        op.add_option('mini', 'mini', options.IntegerOptionParser, 'mini batch size',
                      default=-1)
        op.add_option('batch-size', 'batch_size', options.IntegerOptionParser, 'batch size',
                      default=None)
        op.add_option('train-range', 'train_range', options.RangeOptionParser, 'train range',
                      default=None)
        op.add_option('test-range', 'test_range', options.RangeOptionParser, 'test range',
                      default=None)
        op.add_option('save-path', 'save_path', options.StringOptionParser, 'save path',
                      default=None)
        op.add_option('solver-type', 'solver_type', options.StringOptionParser, 'solver type',
                      default=None)
        op.add_option('selective-load', 'selective_load', options.BooleanOptionParser, 'whether to allow for new layers to be added in when using load file', default=False)
        op.add_option('external-meta-path', 'external_meta_path', options.StringOptionParser, 'external meta path', default='')
    def parse(self):
        self.op.parse()
        self.op.eval_expr_defaults()
        if self.op.get_value('load_file'):
            saved_model = Solver.get_saved_model(self.op.get_value('load_file'))
        else:
            saved_model = None
        train_dp, test_dp, dp_params = self.create_dp(self.op, saved_model)
        net_dic, layers = self.create_networks(self.op, saved_model)
        self.dp_params = dp_params
        solver = self.create_solver(self.op, net_dic, train_dp, test_dp, saved_model)
        return solver
    @classmethod
    def copy_dic_by_key(cls, target, source, key_list):
        for k in key_list:
            target[k] = source[k]
    def load_data_dic(self, dp_params):
        return None
    def create_dp(self, op, saved_model = None):
        try:
            if saved_model and 'dp_params' in saved_model['solver_params']:
                dp_params = saved_model['solver_params']['dp_params']
            else:
                dp_params = None
            required_fields = ['data_path', 'data_provider', 'train_range',
                               'test_range', 'batch_size', 'force_shuffle']
            optional_field = ['external_meta_path']
            default_dic = {'force_shuffle': False}
            d = dict()
            # print saved_model['solver_params'].keys()
            for e in required_fields:
                if op.get_value(e) is not None:
                    d[e] = op.get_value(e)
                elif dp_params and e in dp_params:
                    d[e] = dp_params[e]
                elif e in default_dic:
                    d[e] = default_dic[e]
                else:
                    raise Exception('Field {} is missing'.format(e))
            for e in optional_field:
                if op.get_value(e) is not None:
                    d[e] = op.get_value(e)
        except Exception as err:
            print 'SolverLoader: create_dp {}'.format(err)
            sys.exit(1)
        train_param, test_param = dict(), dict()
        if saved_model:
            self.copy_dic_by_key(train_param, saved_model['model_state']['train'],
                                 ['epoch', 'batchnum'])
            self.copy_dic_by_key(test_param, saved_model['model_state']['test'],
                                              ['epoch', 'batchnum'])
        self.copy_dic_by_key(train_param, d, ['batch_size', 'data_path', 'force_shuffle'])
        self.copy_dic_by_key(test_param, d, ['batch_size', 'data_path', 'force_shuffle'])
        dp_type = d['data_provider']
        data_dic = self.load_data_dic(d)
        train_dp = idata.dp_dic[dp_type](data_dic=data_dic, train=True,
                                         data_range=d['train_range'], params=train_param)
        test_dp = idata.dp_dic[dp_type](data_dic=data_dic, train=False, data_range=d['test_range'], params=test_param)
        return train_dp, test_dp, d
    def create_networks(self, op, saved_model):
        try:
            cfg_file_path = op.get_value('layer_def')
            if not cfg_file_path:
                dummy, saved_net = saved_model['net_dic'].items()[0]
                cfg_file_path = saved_net['config_dic']['layer_def_path']
        except Exception as err:
            print err
            sys.exit(1)
        g = igraphparser.GraphParser(cfg_file_path)
        layers = g.layers
        if saved_model:
            netname,net = saved_model['net_dic'].items()[0]
            selective_load = op.get_value('selective_load')
            print 'Selective load is {}'.format('on' if selective_load else 'off')
            Solver.clone_weights(layers, net['layers'], selective_load)
        if len(g.network_config) == 0:
            raise Exception('Can not find network configuration')
        net_dic = dict()
        for netname in g.network_config:
            net_dic[netname] = Network(layers, g.network_config[netname])
        return net_dic, layers
    def parse_solver_params(self, solver_params, op):
        if self.dp_params:
            solver_params['dp_params'] = self.dp_params
    @classmethod
    def is_valid_value(cls, op, e):
        v = op.get_value(e)
        if v is None or (type(v) is str and len(v) == 0):
            return False
        return True
    def create_solver(self, op, net_dic, train_dp, test_dp, saved_model):
        saved_params = saved_model['solver_params'] if saved_model else dict()
        solver_params = dict()
        self.parse_solver_params(solver_params, op)
        solver_type = op.get_value('solver_type')
        if solver_type is None and 'solver_type' in saved_params:
            solver_type = saved_params['solver_type']
        _cls = solver_dic[solver_type] 
        required_list = _cls._required_field
        default_dic = dict(_cls._default_list)
        # params in saved_params has least priority
        
        for e in required_list:
            if (not e in solver_params) and self.is_valid_value(op, e):
                solver_params[e] = op.get_value(e)
            elif e in default_dic:
                solver_params[e] = default_dic[e]
            elif e in saved_params:
                solver_params[e] = saved_params[e]
        if 'net_order' in solver_params:
            net_list = [ net_dic[nname] for nname in solver_params['net_order']]
        else:
            net_list = [net for nname, net in net_dic.items()]
        for e in _cls._resuming_field:
            if e in saved_params:
                solver_params[e] = saved_params[e]
        iu.ensure_dir(solver_params['save_path'])
        print 'save path = {}'.format(solver_params['save_path'])
        solver = _cls(net_list, train_dp, test_dp, solver_params)
        return solver

class MMSolverLoader(SolverLoader):
    def add_default_options(self, op):
        SolverLoader.add_default_options(self, op)
        op.add_option('K-candidate', 'K_candidate', options.IntegerOptionParser, 'the number of candidate', default=None)
        op.add_option('K-most-violated', 'K_most_violated', options.IntegerOptionParser, 'the size of the hold-on set', default=None)
        op.add_option('K-top-update', 'K_top_update', options.IntegerOptionParser, 'Using the top K most violoated candidate for updating the cost', default=1)
        op.add_option('max-num', 'max_num', options.IntegerOptionParser, 'The maximum number of sample to be processed')
        op.add_option('margin-func', 'margin_func', options.StringOptionParser, 'the parameters for marginl', default='mpjpe')
        op.add_option('candidate-mode', 'candidate_mode', options.StringOptionParser, 'the parameters for marginl', default='random')
        op.add_option('cumulate-update-num', 'cumulate_update_num', options.IntegerOptionParser, 'the number of trial to cumulate data', default=-1)
        op.add_option('candidate-feat-pca-path', 'candidate_feat_pca_path', options.StringOptionParser, 'The path for storing the pca results of candidate features',default='')
        op.add_option('candidate-feat-pca-noise', 'candidate_feat_pca_noise', options.FloatOptionParser, 'The sigma level of pca', default=0)
        op.add_option('opt-num', 'opt_num', options.IntegerOptionParser, 'the number of times to update after finding the most violated one', default=1)
    def parse_solver_params(self, solver_params, op):
        SolverLoader.parse_solver_params(self, solver_params, op)
        if (not 'margin_func' in solver_params) and (op.get_value('margin_func') is not None):
            s = op.get_value('margin_func').split(',')
            name = s[0]
            margin_params = None if len(s) == 1 else [float(x) for x in s[1:]]
            solver_params['margin_func'] = {'name':name, 'params':margin_params}
        else:
            solver_params['margin_func'] = {'name':'mpjpe', 'params':None}

class Solver(object):
    _default_list = [('mini', -1)]
    _required_field =  ['num_epoch', 'save_path', 'testing_freq', 'solver_type']
    _required_attributes =  ['num_epoch', 'save_path', 'testing_freq']
    _resuming_field  = ['train_error', 'test_error']
    _solver_type = 'solver'
    def __init__(self, net_obj_list, train_dp, test_dp, solver_params = None):
        self.train_dp = train_dp
        self.test_dp = test_dp
        self.net_obj_list = net_obj_list
        self.parse_params(solver_params)
        self.data_idx = None
    def safe_set_attr(self, solver_params, name, default=None):
        if name in solver_params:
            setattr(self, name, solver_params[name])
        else:
            setattr(self, name, default)
    def mergefrom(cls, source_dic, keylist, target_dic):
        for e in keylist:
            target_dic[e] = source_dic[e]
    def get_num_batches_done(self, train):
        dp = self.train_dp if train else self.test_dp
        r =  dp.get_num_batches_done(dp.epoch, dp.batchnum)
        return r
    def pre_advance_batch(self, train=True):
        dp = self.train_dp if train else self.test_dp
        return dp.pre_advance_batch()
    @classmethod
    def set_params(cls, params, params_on_host):
        """
        params is the list of shared variable
        params_on_host is the list of numpy ndarray 
        """
        if len(params)!= len(params_on_host):
            raise Exception('The length of params {} != length of params_on_host {}'.format(
                len(params), len(params_on_host)
            ))
        for p, p_h in zip(params, params_on_host):
            p.set_value(np.cast[theano.config.floatX](p_h))
    @classmethod
    def gpu_require(cls, X):
        return np.require(X, dtype=theano.config.floatX)
    def get_next_batch(self, train=True):
        dp = self.train_dp if train else self.test_dp
        return dp.get_next_batch()
    def parse_params(self, solver_params):
        self.solver_params = dict()
        default_dic = dict(self._default_list)
        for e in self._required_field:
            if e in solver_params:
                self.solver_params[e]= solver_params[e]
            else:
                self.solver_params[e] = default_dic[e]
        for e in self._required_attributes:
            setattr(self, e, solver_params[e])
        for e in ['train_error', 'test_error']:
            self.safe_set_attr(solver_params, e, [])
        if 'dp_params' in solver_params:
            self.solver_params['dp_params']= solver_params['dp_params']
        self.solver_params['solver_type'] = self._solver_type
    def print_cost(self, train=True):
        st = 'Train:\t' if train else 'Test:\t'
        costs = self.train_error[-1] if train else self.test_error[-1]
        st += ', '.join(['{}:{}'.format(name, value) for name, value in costs.items()])
        print st
    def print_iteration(self):
        print '------------------------'
        print 'In iteration {}.{} of {}'.format(self.epoch, self.batchnum,self.num_epoch)
    def _save(self, to_save):
        model_name = '{:d}@{:d}'.format(self.epoch, self.batchnum)
        allfiles = iu.getfilelist(self.save_path, '\d+@\d+$')
        save_path = iu.fullfile(self.save_path, model_name)
        mio.pickle(save_path, to_save)
        print '    Saved model to {}'.format(save_path)
        for fn in allfiles:
            os.remove(iu.fullfile(self.save_path, fn))
    @classmethod
    def write_features(cls, dp, net, save_folder, output_layer_names = None,
                       test_one=False, inputs=None, dataidx=None):
        if inputs is None:
            inputs = net.inputs
        if dataidx is None:
            dataidx = range(len(inputs))
        num_batch = dp.num_batch
        res = dict()
        tmp_res = []
        if output_layer_names is None:
            outputs = net.outputs
        else:
            output_layers = net.get_layer_by_names(output_layer_names)
            outputs = sum([lay.outputs for lay in output_layers], [])
        func = theano.function(inputs=inputs,outputs=outputs,
                               on_unused_input='ignore')
        iu.ensure_dir(save_folder)
        saved = dict()
        saved['info'] = dict()
        saved['info']['feature_names'] = [e.name for e in outputs]
        net.set_train_mode(False)
        dp.reset()
        for b in range(num_batch):
            epoch, batchnum, alldata = dp.get_next_batch()
            input_data = [Solver.gpu_require(alldata[k].T) for k in dataidx]
            res = func(*input_data)
            tmp_res += [res]
            save_path = iu.fullfile(save_folder, 'batch_feature_%d' % batchnum)
            saved['feature_list'] = [e.T for e in res]
            saved['feature_dim'] = [e.shape[:-1] for e in saved['feature_list']]
            saved['info']['indexes'] = dp.get_batch_indexes()
            cur_indexes = saved['info']['indexes']
            print 'write range [{}, {}]'.format(min(cur_indexes), max(cur_indexes))
            mio.pickle(save_path, saved)
            ndata = input_data[0].shape[0]
            print 'Finish batch {} (ndata = {} shape is {}'.format(batchnum,
                                                                  ndata,saved['feature_dim'])
            if test_one:
                break
    @classmethod
    def get_saved_model_path(cls, saved_folder):
        allfiles = iu.getfilelist(saved_folder, '\d+@\d+$')
        def f(s):
            return [int(x) for x in s.split('@')]
        allfiles = sorted(allfiles,key=lambda x: f(x))
        return iu.fullfile(saved_folder, allfiles[-1])
    @classmethod
    def get_saved_model(cls, saved_folder):
        model_path = Solver.get_saved_model_path(saved_folder)
        return mio.unpickle(model_path)
    @classmethod
    def clone_weights(cls, layers, saved_layers, selective_load=False):
        l_keys = sorted(layers.keys())
        l_keys1 = sorted(saved_layers.keys())
        if (not selective_load) and (l_keys != l_keys1):
            # print 'Inconsistent number of layers {} vs {}'.format(len(l_keys),len(l_keys1))
            raise Exception('Inconsistent number of layers or different components')
        for l in l_keys:
            if l in saved_layers:
                layers[l][2].copy_from_saved_layer(saved_layers[l][2])
    def train(self):
        pass
    def prepare_data(self, cur_data):
        """
        This function will select the 
        """
        if self.data_idx:
            return [self.gpu_require(cur_data[k].T) for k in self.data_idx]
        else:
            return [self.gpu_require(x.T) for x in cur_data]
class MMSolver(Solver):
    """
    Abstract class for maximum margin solver
    """
    _default_list = [('candidate_mode', 'random'),
                     ('cumulate_update_num', -1), ('K_top_update', 1),
                     ('candidate_feat_pca_noise',0.0), ('opt_num', 1),
                     ('candidate_feat_pca_path',None)] + Solver._default_list
    _required_field = ['K_candidate', 'max_num', 'K_most_violated', 'K_top_update', 'candidate_mode', 'opt_num',
                       'margin_func', 'cumulate_update_num', 'candidate_feat_pca_path', 'candidate_feat_pca_noise'] + Solver._required_field
    def get_repindexes(cls, n, k):
        return np.tile(np.array(range(n)).reshape((1,n)), [k, 1]).flatten(order='F')
    def create_candidate_indexes(self,  ndata, dp, train=True):
        """
        return K_candidate, canddiate_indexes
        The returned index is the index within (training set | candidate set) before shuffle
        
        """
        data_range = dp.data_range
        n_train = len(data_range)
        candidate_mode = self.solver_params['candidate_mode']
        if candidate_mode == 'random':
            K_candidate = self.solver_params['K_candidate']
            indexes = np.random.randint(low=0, high=n_train, size = K_candidate * ndata)
        elif candidate_mode == 'all':
            print 'Use All training as candidates'
            K_candidate = n_train
            indexes = np.array(range(0, n_train) * ndata)
        elif candidate_mode == 'random2':
            K_candidate = self.solver_params['K_candidate']
            indexes = np.concatenate([np.random.choice(range(0, n_train), size=K_candidate, replace=False) for k in range(ndata)])
        elif candidate_mode.find('randomneighbor_') != -1:
            # print 'Candidate mode is {} '.format(candidate_mode)
            t = time()
            K_candidate = self.solver_params['K_candidate']
            if train == False:
                indexes = np.random.randint(low=0, high=n_train, size = K_candidate * ndata)
            else:
                indexes = np.random.randint(low=0, high=n_train, size = K_candidate * ndata).reshape((K_candidate, ndata),order='F')
                t1 = time() - t
                t = time()
                K_neighbor = int(candidate_mode[15:])
                assert(n_train >= K_neighbor)
                offset = data_range[0]
                cur_data_indexes = np.asarray(self.cur_batch_indexes).flatten()
                hf = K_neighbor//2
                s_indexes = [ max(0, min(k - offset - hf, n_train - K_neighbor)) for k in cur_data_indexes]
                index_neighbor = np.concatenate([np.array(range(k, k + K_neighbor)).reshape((K_neighbor, 1),order='F') for k in s_indexes], axis=1)
                K_candidate = K_candidate + K_neighbor
                indexes = np.concatenate([indexes, index_neighbor], axis=0).flatten(order='F')
                # t2 = time() - t
                # print '''
                #      Cost {} ({}, {})seconds to generate candidates \n \n
                # '''.format(t1+t2,t1,t2 )
        elif candidate_mode == 'testrandom':
            print '''
            Only used to check the performance on test samples\n Should not be used for normal
            training or testing \n \n !!!!!!!!!!!!!!\n\n Even for testing, Ensure train test continuous
            '''
            K_candidate = self.solver_params['K_candidate']
            n_all = len(self.train_dp.data_range) + len(self.test_dp.data_range)
            indexes = np.random.randint(low=0, high=n_all, size = K_candidate * ndata)
        else:
            raise Exception('Can not recognize the candidate_mode type {}'.format(candidate_mode))
        return K_candidate, indexes
    def analyze_num_sv_ext(self, alldata):
        """
        Here will analyze the number of support vector
        """
        res = self.train_forward_func(*alldata)[0]
        act_ind = res.sum(axis=1,keepdims=True) > 0
        return res, act_ind
    def analyze_num_sv(self, alldata):
        """
        Here will analyze the number of support vector
        """
        res, act_ind = self.analyze_num_sv_ext(alldata)
        ntot= act_ind.size
        nsv = ntot - np.sum(act_ind.flatten())
        iu.print_common_statistics(res)
        print '{}[{}]:\t mm-cost {} #correct {} [{:.2f}%]'.format(res.dtype,
                                                                  res.shape,
                                                                  res.sum(axis=1,keepdims=True).mean(), nsv, nsv * 100.0/ntot)
    @classmethod
    def add_holdon_candidates(cls, candidate_indexes, holdon_indexes, num):
        if holdon_indexes.size == 0:
            return candidate_indexes
        holdon_indexes_ext = np.tile(holdon_indexes.reshape((-1,1),order='F'),[1,num])
        t = np.concatenate([candidate_indexes.reshape((-1, num),order='F'),
                            holdon_indexes_ext],axis=0)
        return t.flatten(order='F')
    @classmethod
    def zero_margin(cls, residuals, margin_dim=1):
        ndata = residuals.shape[-1]
        return np.zeros((margin_dim, ndata))
    def make_margin_func(self, func_name, func_params):
        """
        Any valid margin function should satisfy \Delta(y,y) = 0   
        """
        if func_name == 'rbf':
            print '    use rbf as margin'
            sigma = func_params[0]
            return lambda X: 1 - dutils.calc_RBF_score(X, sigma, group_size=3)
        elif func_name == 'mpjpe':
            return lambda X: dutils.calc_mpjpe_from_residual(X, num_joints=17)
        elif func_name == 'jpe':
            return lambda X: dutils.calc_jpe_from_residual(X, num_joints=17)
        elif func_name == 'exp_mpjpe':
            return lambda X: np.exp(dutils.calc_mpjpe_from_residual(X, num_joints=17)) - 1
        else:
            raise Exception('Unsupported margin type {}'.format(func_name))
    def parse_params(self, solver_params):
        Solver.parse_params(self, solver_params)
        ## For defaults
        if 'margin_func' in self.solver_params:
            self.margin_func = self.make_margin_func(self.solver_params['margin_func']['name'],
                                                self.solver_params['margin_func']['params'])
        else:
            raise Exception('I can not find margin_func')
            self.margin_func = lambda X:self.calc_margin(X)

        if solver_params['candidate_feat_pca_path'] and solver_params['candidate_feat_pca_noise'] != 0:
            feat_pca = mio.unpickle(solver_params['candidate_feat_pca_path'])
            E = feat_pca['eigvector'] * np.sqrt(feat_pca['eigvalue']).reshape((1,-1),order='F')
            self.candidate_feat_E = E *  solver_params['candidate_feat_pca_noise']
        else:
            self.candidate_feat_E = None
        self.opt_num = solver_params['opt_num']
    @classmethod
    def concatenate_data(cls, data_list, theano_order=True):
        axis = 0 if theano_order else -1
        n_element = len(data_list[0])
        return [np.concatenate([d[k] for d in data_list ],
                               axis=axis) for k in range(n_element)]
    @classmethod
    def collect_sv(cls, ind, data, theano_order=True):
        f_ind = ind.flatten()
        if theano_order:
            return [e[f_ind,...] for e in data]
        else:
            return [e[..., f_ind] for e in data]
class MMLSSolver(MMSolver):
    """
    The solver for Maxmimum Margin && linesearch
    train_forward_func: 
    """
    _solver_type='mmls'
    def __init__(self, net_obj_list, train_dp, test_dp, solver_params = None):
        MMSolver.__init__(self, net_obj_list, train_dp, test_dp, solver_params)
        self.eval_net, self.train_net = net_obj_list[0], net_obj_list[1]
        self.grad = tensor.grad(self.train_net.costs[0], self.train_net.params)
        # print theano.pp(self.grad[0])
        self.grad_func = theano.function(inputs=self.train_net.inputs,
                                         outputs=self.grad
        )
        self.params = self.train_net.params
        self.test_func = theano.function(inputs=self.train_net.inputs,
                                          outputs=self.train_net.costs
        )
        #  For evaluating the outputs
        self.eval_func = theano.function(inputs=self.eval_net.inputs,
                                          outputs=self.eval_net.outputs,
        )
        flayer_name = 'net2_mmcost'
        self.train_forward_func = theano.function(inputs=self.train_net.inputs,
                                                  outputs=self.train_net.layers[flayer_name][2].outputs)
        self.margin_dim = self.train_net.layers[flayer_name][2].param_dic['margin_dim']
        # For debug usage
        self.stat = dict()
        n_train = len(self.train_dp.data_range)
        self.stat['sample_candidate_counts'] = np.zeros((n_train))  #
        self.stat['most_violated_counts'] = np.zeros((n_train))
        self.data_idx = self.train_net.data_idx
        self.cur_batch_indexes = None
    def get_best_steps(cost_list):
        return np.argmin(np.array([x[0] for x in cost_list]))
    def get_search_steps(self):
        num_step = int(7)
        return np.power(10.0,range(-num_step,3))
    # def get_updates(self, step_size):
    #     """
    #     get the list of updates
    #     """
    #     gradients = tensor.grad(self.net_obj.costs[0], self.net_obj.all_params)
    #     params = self.net_obj.all_params
    #     return [[p, p - p * step_size * g]  for p,g in zip(params, gradients)]
    def get_next_batch(self, train=True):
        dp = self.train_dp if train else self.test_dp
        epoch, batchnum, alldata =  dp.get_next_batch()
        if self.data_idx:
            alldata = [alldata[i] for i in self.data_idx]
        return epoch, batchnum, alldata
    @classmethod
    def calc_margin(cls, residuals):
        return dutils.calc_mpjpe_from_residual(residuals, 17)
    def print_layer_outputs(self,alldata):   # FOR DEBUG
        import iutils as iu
        t = time()
        show_layer_name = ['net1_score', 'net1_fc2', 'net2_score', 'net2_fc2', 'net2_mmcost']
        net= self.train_net
        outputs = sum([self.train_net.layers[name][2].outputs
                             for name in show_layer_name],[])
        func = theano.function(inputs=self.train_net.inputs,
                               outputs=outputs,
                               on_unused_input='warn'
        )
        res = self.call_func(func,alldata)
        print '<<<<<<<<<<<<<< evaluation cost %.6f sec.' % (time() - t)
        for name, r in zip(show_layer_name, res):
            print '---output layer %s' % name
            iu.print_common_statistics(r)
    def print_dims(self,alldata):     # debug use
        for i,e in enumerate(alldata):
            print 'Dim {}: \t shape {} \t type {} {}'.format(i, e.shape, type(e), e.dtype if type(e) is np.ndarray else '' )
    def find_most_violated(self, data, train=True):
        alldata, dummy_list = self.find_most_violated_ext(data,
                                                          use_zero_margin=False, train=train)
        return alldata
    def get_all_candidates(self, dp):
        fl = dp.data_dic['feature_list']
        if self.data_idx:
            fl = [fl[k] for k in self.data_idx]
        if self.candidate_mode in ['testrandom']:
            raise Exception('Not implemented')
        return fl
        
    def find_most_violated_ext(self, data, use_zero_margin=False, train=True):
        """
        data = [gt, imgfeature, gt_jtfeature, margin]
        return [gt, imgfeature, gt_jtfeature, mv_jtfeature, margin0, margin1]
        """
        # K_candidate = self.solver_params['K_candidate']
        
        self.eval_net.set_train_mode(train=False) # Always use the test mode for searching
        K_mv = self.solver_params['K_most_violated']
        if train:
            K_update = self.solver_params['K_top_update']
        else:
            K_update = 1
        max_num = int(self.solver_params['max_num'])
        calc_margin = (lambda R:self.zero_margin(R, self.margin_dim)) if use_zero_margin else self.margin_func
        train_dp =  self.train_dp
        if (self.candidate_feat_E is not None) and train:
            # need to ensure it has max_depth
            feat_E = self.candidate_feat_E / train_dp.max_depth
        else:
            feat_E = None
            
        n_train = len(train_dp.data_range)
        ndata = data[0].shape[-1] 
        num_mini_batch = (ndata - 1) / max_num  + 1
        K_candidate, candidate_indexes = self.create_candidate_indexes(ndata, train_dp, train)
        if train:
            cur_counts, dummy = np.histogram(candidate_indexes, bins=range(0, n_train + 1))
            self.stat['sample_candidate_counts'] += cur_counts
        mvc = self.stat['most_violated_counts']
        sorted_indexes = sorted(range(n_train), key=lambda k:mvc[k],reverse=True)
        holdon_indexes = np.array(sorted_indexes[:K_mv])            

        fl = self.get_all_candidates(train_dp)
        selected_indexes = []
        all_candidate_indexes = []
        for mb in range(num_mini_batch):
            start, end = mb * max_num, min(ndata, (mb + 1) * max_num)
            start_indexes, end_indexes = start * K_candidate, end * K_candidate
            cur_candidate_indexes = candidate_indexes[start_indexes:end_indexes]

            cur_candidate_indexes = self.add_holdon_candidates(cur_candidate_indexes,
                                                              holdon_indexes, end-start)
            candidate_targets = fl[0][..., cur_candidate_indexes]
            candidate_features = fl[2][..., cur_candidate_indexes]
            cur_num = end - start
            K_tot = K_candidate + K_mv
            gt_target = np.tile(data[0][..., start:end], [K_tot, 1]).reshape((-1, K_tot * cur_num), order='F')
            imgfeatures = np.tile(data[1][..., start:end], [K_tot, 1]).reshape((-1, K_tot * cur_num ), order='F')
            # margin = self.calc_margin(gt_target - candidate_targets)
            margin = calc_margin(gt_target - candidate_targets)

            ##@ adding noise
            if feat_E is not None:
                dim_X = candidate_targets.shape[0]
                candidate_targets = candidate_targets + np.dot(feat_E, np.random.randn(dim_X, candidate_targets.shape[-1]))                
            # #
            alldata = [self.gpu_require(imgfeatures.T),
                       self.gpu_require(candidate_features.T),
                       self.gpu_require(margin.T)]
            outputs = self.eval_func(*alldata)[0]            
            outputs = outputs.reshape((K_tot, cur_num),order='F')
            # m_indexes = np.argmax(outputs, axis=0).flatten() + \
            #             np.array(range(0, cur_num)) * K_tot
            ##@
            m_indexes = np.argpartition(-outputs, K_update, axis=0)[:K_update,:] + np.array(range(0, cur_num)).reshape((1,-1),order='F') * K_tot
            m_indexes = m_indexes.flatten(order='F')
            
            selected_indexes += cur_candidate_indexes[m_indexes].tolist()
            all_candidate_indexes += [cur_candidate_indexes]
        if train:
            most_violated_cnt, dummy = np.histogram(selected_indexes,bins=range(0, n_train + 1))
            self.stat['most_violated_counts'] += most_violated_cnt
        most_violated_features = fl[2][..., selected_indexes]
        most_violated_targets = fl[0][..., selected_indexes]
        ##@
        if K_update !=1:
            rep_indexes = self.get_repindexes(ndata, K_update)
            imgfeats, gt, gtfeats = data[1][..., rep_indexes], data[0][..., rep_indexes], data[2][..., rep_indexes]
        else:
            # data = [gt, imgfeature, gt_jtfeature, margin]
            # return [gt, imgfeature, gt_jtfeature, mv_jtfeature, margin0, margin1]
            imgfeats, gt, gtfeats = data[1] ,data[0], data[2]
        print 'data len ={}<<<<'.format(len(data))
        # mv_margin = self.calc_margin(gt - most_violated_targets)
        mv_margin = calc_margin(gt - most_violated_targets)
        gt_margin = np.zeros((self.margin_dim, ndata * K_update), dtype=np.single)
        alldata = [imgfeats, gt, gtfeats,
                   most_violated_features, gt_margin, mv_margin]
        # extra information
        all_candidate_indexes_arr = np.concatenate(all_candidate_indexes)
        return alldata, [most_violated_targets, all_candidate_indexes_arr, selected_indexes]
    @classmethod
    def call_func(cls, func, params):
        """
        """
        return func(*params)
    def analyze_net_params(self, param_list):
        import iutils as iu
        for w in param_list:
            iu.print_common_statistics(w)
    def analyze_param_vs_gradients(self, param_list, gradients_list, sym_params_list):
        for w,g,s in zip(param_list, gradients_list, sym_params_list):
            avgw = np.mean(np.abs(w.flatten()))
            avgg = np.mean(np.abs(g.flatten()))
            print('{}:\t v:{:+.6e} \t g: {:+.6e} \t [{:+.6e}]'.format(s.name,
                                                                      avgw, avgg,
                                                                      avgg/avgw))
    def do_opt(self, alldata):
        compute_time_py = time()
        self.train_net.set_train_mode(train=True)
        self.eval_net.set_train_mode(train=True)
        params = self.train_net.params
        params_eps = [e.get_value() for e in self.train_net.params_eps]
        params_host = [v.get_value(borrow=False) for v in params] # backup
        n_data = alldata[0].shape[0]
        steps = self.get_search_steps()
        info = None
        for k in range(self.opt_num):
            cur_gradiens = self.call_func(self.grad_func, alldata)
            cost_list = []
            for s in steps:
                ss = s # no need to do the normalization any more
                inner_params_host = [p - (ss * eps) * g for p,g,eps in zip(params_host,
                                                                           cur_gradiens,
                                                                           params_eps)]
                self.set_params(params, inner_params_host)
                cur_cost = self.call_func(self.test_func, alldata)
                cost_list.append([cur_cost[0]])
            min_idx = np.argmin(cost_list)
            best_step = steps[min_idx]
            print '    cost = {} \t [max: {}]'.format(cost_list[min_idx], max(cost_list))
            print '    best step is {}'.format(best_step)
            if k == self.opt_num - 1:
                g_update = [(best_step * eps) * g for g,eps in zip(cur_gradiens, params_eps)]
                self.analyze_param_vs_gradients(params_host, g_update, params )
                params_host = [p -  g for p,g in zip(params_host, g_update)]
                self.set_params(params, params_host)
                info = {'mmcost':cost_list[min_idx]}
            else:
                inner_params_host = [p - (best_step * eps) * g for p,g,eps in zip(params_host,
                                                                                  cur_gradiens,
                                                                                  params_eps)]
                params_host = inner_params_host
                self.set_params(params, inner_params_host)
        print 'Optimization {} times:\t (%.3f sec)' % (self.opt_num, time()- compute_time_py)
        return info
    def train(self):
        cumulate_update_num = self.solver_params['cumulate_update_num']
        pre_batch_done = self.get_num_batches_done(train=True)
        cur_data = self.get_next_batch(train=True)
        K_top_update = self.solver_params['K_top_update']
        self.cur_batch_indexes = self.train_dp.get_batch_indexes()
        self.epoch, self.batchnum = cur_data[0], cur_data[1]
        num_updates = 0
        while True: 
            if cumulate_update_num > 0:
                collected_indexes = []
                data_list = []
                nsv_cum = 0
                last_mv_data = None
                print 'Cumulate update'
                for ct in range(cumulate_update_num):
                    print '[{}]:------'.format(ct)
                    self.epoch, self.batchnum = cur_data[0], cur_data[1]
                    self.print_iteration()
                    compute_time_py = time()
                    most_violated_data = self.find_most_violated(cur_data[2], train=True)
                    tmp_data = [self.gpu_require(e.T) for e in most_violated_data[1:]]
                    print 'Searching the most violated cost %.3f sec' % (time() - compute_time_py)
                    res, act_ind = self.analyze_num_sv_ext(tmp_data)
                    nsv = np.sum(act_ind.flatten())
                    print 'support vector {} of {}:\t{}%\t mmcost={}'.format(nsv, act_ind.size,
                                                                             nsv * 100.0/act_ind.size, res.mean())
                    last_mv_data = tmp_data
                    nsv_cum += nsv
                    if nsv:
                        data_list.append(self.collect_sv(act_ind, tmp_data))
                        cur_indexes = self.train_dp.get_batch_indexes()
                        collected_indexes += np.array(cur_indexes)[act_ind]
                    if nsv_cum >= tmp_data[0].shape[0] * K_top_update:
                        break
                    next_epoch, next_batchnum = self.train_dp.epoch, self.train_dp.batchnum
                    if next_epoch  == self.num_epoch: 
                        break
                    if ct != cumulate_update_num - 1:
                        cur_data = self.get_next_batch(train=True)
                        self.cur_batch_indexes = self.train_dp.get_batch_indexes()
                if len(data_list) == 0:
                    alldata = last_mv_data
                    self.cur_batch_indexes = self.train_dp.get_batch_indexes()
                else:
                    alldata = self.concatenate_data(data_list)
                    self.cur_batch_indexes = collected_indexes
            else:
                self.print_iteration()
                compute_time_py = time()
                most_violated_data = self.find_most_violated(cur_data[2], train=True)
                alldata = [self.gpu_require(e.T) for e in most_violated_data[1:]]
                print 'Searching the most violated cost %.3f sec' % (time() - compute_time_py)

            # Inside model the data are interpreted as [ndata x dim]
            self.do_opt(alldata)
            # deprecated
            # steps = self.get_search_steps()
            # compute_time_py = time()
            # self.train_net.set_train_mode(train=True)
            # self.eval_net.set_train_mode(train=True)
            # cur_gradiens = self.call_func(self.grad_func, alldata)
            # params = self.train_net.params
            # params_eps = [e.get_value() for e in self.train_net.params_eps]
            # params_host = [v.get_value(borrow=False) for v in params] # backup
            # cost_list = []
            # n_data = alldata[0].shape[0]
            # for s in steps:
            #     ss = s # no need to do the normalization any more
            #     inner_params_host = [p - (ss * eps) * g for p,g,eps in zip(params_host,
            #                                                                cur_gradiens,
            #                                                                params_eps)]
            #     self.set_params(params, inner_params_host)
            #     cur_cost = self.call_func(self.test_func, alldata)
            #     cost_list += [cur_cost[0]]
            # min_idx = np.argmin(cost_list)
            # best_step = steps[min_idx]
            # print 'cost = {} \t [max: {}]'.format(cost_list[min_idx], max(cost_list))
            
            # self.train_error.append({'mmcost':cost_list[min_idx]})
            # # self.print_layer_outputs(alldata)
            # # update params_host
            # g_update = [(best_step * eps) * g for g,eps in zip(cur_gradiens, params_eps)]
            # print 'best step is {}'.format(best_step)
            # self.analyze_param_vs_gradients(params_host, g_update, params )
            # params_host = [p -  g for p,g in zip(params_host, g_update)]
            # # self.analyze_net_params(params_host)
            # self.set_params(params, params_host) # copyback
            print 'analyze_num_sv---train'
            self.analyze_num_sv(alldata)
            
            compute_time_py = time()
            cur_batch_done = self.get_num_batches_done(True)
            save_batch_idx = (cur_batch_done // self.testing_freq ) * self.testing_freq
            num_updates += 1
            if num_updates % self.testing_freq == 0:
                mmcost = self.get_test_error()
                self.test_error.append({'mmcost':mmcost[0]})
                print 'test cost = {}'.format(mmcost[0])
            if save_batch_idx > pre_batch_done:
                self.save_model()
            pre_batch_done = cur_batch_done
            self.epoch, self.batchnum = self.train_dp.epoch, self.train_dp.batchnum
            if self.epoch  == self.num_epoch: 
                break
            cur_data = self.get_next_batch(train=True)
            self.cur_batch_indexes = self.train_dp.get_next_batch()
    def get_test_error(self):
        self.train_net.set_train_mode(train=False)
        self.eval_net.set_train_mode(train=False)
        test_data = self.get_next_batch(train=False)
        most_violated_data = self.find_most_violated(test_data[2], train=False)
        alldata = [self.gpu_require(e.T) for e in most_violated_data[1:]]
        ndata = test_data[2][0].shape[-1]
        mmcost = self.call_func(self.test_func, alldata)
        print 'analyze_num_sv---test'
        self.analyze_num_sv(alldata)
        return [mmcost[0]]
    def save_model(self):
        net_dic = {'eval_net': self.eval_net.save_to_dic(),
                    'train_net': self.train_net.save_to_dic()
        }
        net_dic = {'eval_net': self.eval_net.save_to_dic()}
        net_dic['train_net'] = self.train_net.save_to_dic(ignore=set(['layers']))
        net_dic['train_net']['layers'] = net_dic['eval_net']['layers']
        model_state = {'train':{'epoch':self.train_dp.epoch, 'batchnum':self.train_dp.batchnum},
                       'test':{'epoch':self.test_dp.epoch, 'batchnum':self.test_dp.batchnum},
                       'stat':self.stat}
        solver_params = self.solver_params
        solver_params['train_error'] = self.train_error
        solver_params['test_error'] = self.test_error
        to_save = {'net_dic':net_dic, 'model_state':model_state, 'solver_params':self.solver_params}
        self._save(to_save)
       
class BasicBPSolver(Solver):
    """
    This Solver need a network with following property
          costs: The total cost function to optimize
          update: use
                   weights_inc(t) = weights_inc(t-1) * mom - eps * g_weights(t) 
                   weights(t)     =  weights(t-1) + weights_inc(t)
                                  =  weights(t-1) + weights_inc(t-1) * mom - eps * g_weights(t) 
    DONE : Add W_inc_list, b_inc into layer with weigths Need to check
    """
    _solver_type='basicbp'
    def __init__(self, net_list, train_dp, test_dp, solver_params=None):
        Solver.__init__(self, net_list, train_dp, test_dp, solver_params)
        # I need the output of all cost
        self.net = net_list[0]
        # self.params_all = self.net.params + self.net.params_inc
        self.lr = theano.shared(np.cast[theano.config.floatX](1.0))
        self.grads = tensor.grad(self.net.costs[0], self.net.params)
        p_inc_new = [p_inc * mom - (self.lr * eps) * g for p_inc, mom, eps, g in
                     zip(self.net.params_inc, self.net.params_mom, self.net.params_eps,
                         self.grads)]
        param_updates = [ (p, p + inc) for p, inc in zip(self.net.params, p_inc_new)]
        param_inc_updates = [(p_inc, inc) for p_inc, inc in zip(self.net.params_inc, p_inc_new)]
        self.monitor_list = self.net.cost_list + [abs(p).mean() for p in self.net.params] + [abs(g).mean() for g in p_inc_new]
        self.monitor_idx = np.cumsum([0, len(self.net.cost_list),
                            len(self.net.params), len(p_inc_new)])
        self.monitor_name = ['cost_list', 'mean_weights', 'mean_inc']
        self.train_func = theano.function(inputs=self.net.inputs,
                                          outputs=self.monitor_list,
                                          updates= param_updates + param_inc_updates,
        )
        self.test_func = theano.function(inputs=self.net.inputs,
                                         outputs= self.net.cost_list)
        self.grad_check_func = theano.function(inputs=self.net.inputs,
                                               outputs=p_inc_new)
        self.data_idx = self.net.data_idx
    def pack_cost(self, cost_list, net):
        num_cost = len(net.cost_names)
        cli = net.cost_list_idx
        packed_list = [cost_list[cli[k]:cli[k+1]] for k in range(num_cost)]
        assert(len(net.cost_names) == len(packed_list))
        return dict(zip(net.cost_names, packed_list))
    def monitor_params_vs_gradients(self, input_data):
        start_time = time()
        grads = self.grad_check_func(*input_data)
        for w, g in zip(self.net.params, grads):
            t1 = np.mean(np.abs(w.get_value()).flatten())
            t2=  np.mean(np.abs(g).flatten())
            print '    {}:\t avg value={}\t avg gradients={}\t [{}]'.format(w.name, t1,t2,t2/t1)
        print 'Gradient monitoring cost {} seconds'.format(time()- start_time)
    def print_monitor_info(self, info, names):
        w_name = names
        avg_w = info[self.monitor_idx[1]:self.monitor_idx[2]]
        avg_g = info[self.monitor_idx[2]:self.monitor_idx[3]]
        assert(len(avg_w) == len(avg_g) and len(avg_w) == len(w_name))
        for wn, aw, ag in zip(w_name, avg_w, avg_g):
            print '    {}:\t avg value={:.2e}\t avg gradients={:.2e}\t [{:.2e}]'.format(wn, aw[()], ag[()], ag[()]/aw[()])
    def process_data(self, input_data, train):
        func = self.train_func if train else self.test_func
        n_size = input_data[0].shape[0]
        mini = self.solver_params['mini'] if self.solver_params['mini'] > 0 else n_size
        n_mini_batch = n_size
        info_l = []
        for b in range(n_mini_batch):
            indexes = range(b * mini, min((b + 1)* mini, n_size))
            cur_input = [e[indexes , ...] for e in  input_data]
            info = func(*cur_input)
            info_l += [info]
        # <<<<<<<<<<<<<< How to deal with cost and info separately?>
    def DEBUG_show_layer_statistcs(self, input_data):
        names = ['sqdiffcost_reconstruct', 'joints', 'fc_f1']
        outputs = sum([self.net.layers[name][2].outputs for name in names], [])
        func = theano.function(inputs=self.net.inputs,
                               outputs=outputs)
        res = func(*input_data)
        for e in self.net.inputs:
            print 'input = {}'.format(e)
        print 'HHHHH See input\n\n\n'
        iu.print_common_statistics(input_data[0])
        for e_res,e_out in zip(res, outputs):
            print 'layer name = {}'.format(e_out.name)
            iu.print_common_statistics(e_res)
    def extract_cost(self, info):
        return info[:self.monitor_idx[1]]
    def train(self):
        cur_data = self.get_next_batch(train=True)
        self.epoch, self.batchnum=cur_data[0], cur_data[1]
        while True:
            self.net.set_train_mode(train=True)
            self.print_iteration()
            compute_time_py = time()
            input_data = self.prepare_data(cur_data[2])
            # Althouth there is no need to change every time
            self.lr.set_value(np.cast[theano.config.floatX](1.0))

            info = self.train_func(*input_data)
            # self.DEBUG_show_layer_statistcs(input_data)
            
            costs = self.extract_cost(info)
            self.train_error.append(self.pack_cost(costs, self.net))
            print '{} seconds '.format(time()- compute_time_py)
            if self.get_num_batches_done(True) % self.testing_freq == 0:
                self.print_monitor_info(info, [w.name for w in self.net.params])
                self.print_cost(train=True)
                
                compute_time_py = time()
                test_costs = self.get_test_error()
                normalized_test_cost = test_costs
                self.test_error.append(self.pack_cost(normalized_test_cost, self.net))
                self.print_cost(train=False)
                self.save_model()
                print '{} seconds '.format(time()- compute_time_py)
            self.epoch, self.batchnum = self.train_dp.epoch, self.train_dp.batchnum
            if self.epoch == self.num_epoch:
                self.save_model()
                break
            compute_time_py = time()
            cur_data = self.get_next_batch(train=True)
            print 'Loading data:\t{:.2f} seconds'.format(time()- compute_time_py)
    def get_test_error(self):
        self.net.set_train_mode(train=False)
        cur_data = self.get_next_batch(train=False)
        input_data = self.prepare_data(cur_data[2])
        return self.test_func(*input_data)
    def save_model(self):
        net_dic = {'net':self.net.save_to_dic()}
        model_state = {'train':self.train_dp.get_state_dic(),
                       'test':self.test_dp.get_state_dic()}
        solver_params = self.solver_params
        solver_params['train_error'] = self.train_error
        solver_params['test_error'] = self.test_error
        to_save = {'net_dic':net_dic, 'model_state':model_state, 'solver_params':solver_params}
        self._save(to_save)

class ImageMMSolver(BasicBPSolver, MMSolver):
    """
    This is a image maximum margin solver

    network: train_net, cnn_net, 
           :
    dataprovider: img, target, ...              <------- input_data
           
    -->           
    train_net       : inputs=img, gt_target, candidate_target, margin_gt, margin_candidate, ...
    feature_net     : inputs=img
    score_net       : inputs=img_feature, candidate, margin

    eval_func  < ------------      score_net
    train_func < ------------      train_net
    test_func  < ------------      train_net
    calc_img_feature_func <------- feature_net
    train_forward_func  < -------  trainnet    use to get the outputs of maxmargin cost layer
    """
    _default_list = [('opt_method', 'bp')] + MMSolver._default_list
    _required_field = ['opt_method'] + MMSolver._required_field
    _solver_type='imgmm'
    def __init__(self, net_obj_list, train_dp, test_dp, solver_params = None):
        Solver.__init__(self, net_obj_list, train_dp, test_dp, solver_params)
        self.train_net, self.feature_net, self.score_net = net_obj_list
        self.net_obj_list = net_obj_list
        self.lr = theano.shared(np.cast[theano.config.floatX](1.0))
        self.grads = tensor.grad(self.train_net.costs[0], self.train_net.params)

        self.opt_method = self.solver_params['opt_method']
        if self.opt_method == 'bp':
            p_inc_new = [p_inc * mom - (self.lr * eps) * g for p_inc, mom, eps, g in
                         zip(self.train_net.params_inc, self.train_net.params_mom,
                             self.train_net.params_eps, self.grads)]
        else:
            assert(self.opt_method == 'ls')
            p_inc_new = [ - (self.lr * eps) * g for  eps, g in
                         zip(self.train_net.params_eps, self.grads)]
        param_updates = [ (p, p + inc) for p, inc in zip(self.train_net.params, p_inc_new)]
        param_inc_updates = [(p_inc, inc) for p_inc, inc in zip(self.train_net.params_inc,
                                                                p_inc_new)]
        self.monitor_list = self.train_net.cost_list + \
                            [abs(p).mean() for p in self.train_net.params] +\
                            [abs(g).mean() for g in p_inc_new]
        self.monitor_idx = np.cumsum([0, len(self.train_net.cost_list),
                            len(self.train_net.params), len(p_inc_new)])
        self.monitor_name = ['cost_list', 'mean_weights', 'mean_inc']
        self.train_func = theano.function(inputs=self.train_net.inputs,
                                          outputs=self.monitor_list,
                                          updates=param_updates + param_inc_updates,
        )
        self.train_cost_func = theano.function(inputs=self.train_net.inputs,
                                               outputs=self.train_net.costs)
        self.train_grad_func = theano.function(inputs=self.train_net.inputs,
                                               outputs=self.grads)
        self.test_func = theano.function(inputs=self.train_net.inputs,
                                         outputs= self.train_net.cost_list)
        self.grad_check_func = theano.function(inputs=self.train_net.inputs,
                                               outputs=p_inc_new)
        self.calc_img_feature_func = theano.function(inputs=self.feature_net.inputs,
                                                     outputs=self.feature_net.outputs)
        self.eval_func = theano.function(inputs=self.score_net.inputs,
                                         outputs=self.score_net.outputs)
        flayer_name = 'net2_mmcost'
        self.train_forward_func = theano.function(inputs=self.train_net.inputs,
                                                  outputs=self.train_net.layers[flayer_name][2].outputs)
        self.margin_dim = self.train_net.layers[flayer_name][2].param_dic['margin_dim']
        self.data_idx = self.train_net.data_idx
        self.stat = dict()
        n_train = len(self.train_dp.data_range)
        self.stat['sample_candidate_counts'] = np.zeros((n_train))  #
        self.stat['most_violated_counts'] = np.zeros((n_train))
        self.cur_batch_indexes = None
    def set_train_mode(self, train=True):
        for net in self.net_obj_list:
            net.set_train_mode(train)
    def calc_image_features(self, input_data):
        # use test mode for feature extraction
        self.set_train_mode(train=False)
        res = self.calc_img_feature_func(*input_data)
        return res[0]
    def get_all_candidates(self, dp):
        default_idx = 1
        if self.data_idx:
            default_idx = self.data_idx[default_idx]
        if self.solver_params['candidate_mode'] in ['testrandom']:
            fl = [np.concatenate([self.train_dp.get_all_data_at(default_idx),
                                  self.test_dp.get_all_data_at(default_idx)], axis=1)]
        else:
            fl = [dp.get_all_data_at(default_idx)]
        return fl
    def find_most_violated_ext(self, fdata, use_zero_margin=False, train=True):
        """
        fdata: [imgfeature, gt_target]  gt_target_feature can be the same as gt_target
        return [imgfeature, gt_target, mv_target, gt_margin, mv_margin]
        """
        self.set_train_mode(False)  # Always use test mode for searching       
        K_mv = self.solver_params['K_most_violated']
        if train:
            K_update = self.solver_params['K_top_update']
        else:
            K_update = 1
        max_num = int(self.solver_params['max_num'])
        calc_margin = (lambda R:self.zero_margin(R, self.margin_dim)) if use_zero_margin else self.margin_func
        
        dp =  self.train_dp
        #@@@ for adding noise
        if (self.candidate_feat_E is not None)  and train:
            feat_E = self.candidate_feat_E / dp.max_depth
        else:
            feat_E = None
        #
         
        n_train = len(dp.data_range)
        ndata = fdata[0].shape[-1]
        
        num_mini_batch = (ndata - 1) // max_num  + 1
        K_candidate, candidate_indexes = self.create_candidate_indexes(ndata, dp, train)
        
        if train:
            cur_counts, dummy = np.histogram(candidate_indexes, bins=range(0, n_train + 1))
            self.stat['sample_candidate_counts'] = self.stat['sample_candidate_counts'] + cur_counts
        mvc = self.stat['most_violated_counts']
        sorted_indexes = sorted(range(n_train), key=lambda k:mvc[k],reverse=True)
        holdon_indexes = np.array(sorted_indexes[:K_mv])            
        fl = self.get_all_candidates(dp)
        selected_indexes = []
        all_candidate_indexes = []
        for mb in range(num_mini_batch):
            start, end = mb * max_num, min(ndata, (mb + 1) * max_num)
            start_indexes, end_indexes = start * K_candidate, end * K_candidate
            cur_candidate_indexes = candidate_indexes[start_indexes:end_indexes]
            cur_num = end - start
            cur_candidate_indexes = self.add_holdon_candidates(cur_candidate_indexes,
                                                              holdon_indexes, cur_num)
            candidate_targets  =  fl[0][..., cur_candidate_indexes]
            # candidate_features =  candidate_targets

            K_tot = K_candidate + K_mv
            
            gt_target = np.tile(fdata[1][..., start:end], [K_tot, 1]).reshape((-1, K_tot * cur_num), order='F')
            imgfeatures = np.tile(fdata[0][..., start:end], [K_tot, 1]).reshape((-1, K_tot * cur_num ), order='F')
            margin = calc_margin(gt_target - candidate_targets)

            ##@ adding noise
            if feat_E is not None:
                dim_X = candidate_targets.shape[0]
                tmp = candidate_targets + np.dot(feat_E, np.random.randn(dim_X, candidate_targets.shape[-1]))
                candidate_targets = tmp
                # print '\n\n---- mpjpe test {}----\n\n'.format(np.mean(mpjpe_test.flatten()))
            # #
            
            alldata = [self.gpu_require(imgfeatures.T),
                       self.gpu_require(candidate_targets.T),
                       self.gpu_require(margin.T)]
            
            outputs = self.eval_func(*alldata)[0]
            outputs = outputs.reshape((K_tot, cur_num),order='F')
            # m_indexes = np.argmax(outputs, axis=0).flatten() + \
            #             np.array(range(0, cur_num)) * K_tot
            ##@@ 
            m_indexes = np.argpartition(-outputs, K_update, axis=0)[:K_update,:] + np.array(range(0, cur_num)).reshape((1,-1),order='F') * K_tot
            m_indexes = m_indexes.flatten(order='F')
            ##
            selected_indexes += cur_candidate_indexes[m_indexes].tolist()
            all_candidate_indexes += [cur_candidate_indexes]
        if train:
            most_violated_cnt, dummy = np.histogram(selected_indexes,bins=range(0, n_train + 1))
            self.stat['most_violated_counts'] = self.stat['most_violated_counts'] + most_violated_cnt
            
        # most_violated_features = fl[0][..., selected_indexes]
        most_violated_targets = fl[0][..., selected_indexes]

        ##@
        if K_update != 1:
            gt = np.tile(fdata[1], [K_update,1]).reshape((-1, K_update * ndata),order='F')
            imgfeats = np.tile(fdata[0], [K_update,1]).reshape((-1, K_update*ndata),order='F')
        else:
            imgfeats,gt = fdata[0],fdata[1]
        mv_margin = calc_margin(gt - most_violated_targets)
        gt_margin = np.zeros((self.margin_dim, K_update * ndata ), dtype=np.single)

        alldata = [imgfeats, gt, most_violated_targets, gt_margin, mv_margin]
        # extra information
        all_candidate_indexes_arr = np.concatenate(all_candidate_indexes)
        return alldata, [most_violated_targets, all_candidate_indexes_arr, selected_indexes]
    def find_most_violated(self, fdata, train=True):
        mvdata, dummy =  self.find_most_violated_ext(fdata, use_zero_margin=False, train=train)
        return mvdata
    def get_search_steps(self):
        num_step = int(7)
        return np.power(10.0,range(-num_step,3))
    def do_opt(self, mv_input_data):
        if self.opt_method == 'bp':
            self.lr.set_value(np.cast[theano.config.floatX](1.0))
            info_list = [self.train_func(*mv_input_data) for k in range(self.opt_num)]
            info = info_list[-1]
        else: # Do line search ha ha ha
            steps = self.get_search_steps()
            params_eps = [e.get_value(borrow=False) for e in self.train_net.params_eps]
            params = self.train_net.params
            params_host = [v.get_value(borrow=False) for v in params]

            for k in range(self.opt_num):
                cur_gradients = self.train_grad_func(*mv_input_data) 
                cost_list = []
                n_data = mv_input_data[0].shape[0]
                for s in steps:
                    ss = s 
                    inner_params_host = [p - (ss * eps) * g for p,g, eps in zip(params_host,
                                                                                cur_gradients,
                                                                                params_eps)]
                    self.set_params(params, inner_params_host)
                    cur_cost = self.train_cost_func(*mv_input_data)
                    cost_list.append(cur_cost[0])
                min_idx = np.argmin(cost_list)
                best_step = steps[min_idx]
                print '    Cost = {} \t [max :{}]'.format(cost_list[min_idx], max(cost_list))
                print '    best step = {}'.format(best_step)
                if k == self.opt_num - 1:
                    self.lr.set_value(np.cast[theano.config.floatX](best_step))
                    info = self.train_func(*mv_input_data)
                else:
                    inner_params_host = [p - (best_step * eps) * g for p,g, eps in
                                         zip(params_host,
                                             cur_gradients,
                                             params_eps)]
                    params_host = inner_params_host
                    self.set_params(params, inner_params_host)
        return info
    def train(self):
        cumulate_update_num = self.solver_params['cumulate_update_num']
        pre_batch_done = self.get_num_batches_done(train=True)
        cur_data = self.get_next_batch(train=True)
        K_top_update = self.solver_params['K_top_update']
        self.cur_batch_indexes = self.train_dp.get_batch_indexes()
        self.epoch, self.batchnum = cur_data[0], cur_data[1]
        num_updates = 0
        while True:
            self.set_train_mode(train=True)
            if cumulate_update_num > 0:
                data_list = []
                nsv_cum = 0
                print 'Cumulate update'
                last_mv_data = None
                for ct in range(cumulate_update_num):
                    print '[{}]:------'.format(ct)
                    self.epoch, self.batchnum = cur_data[0], cur_data[1]
                    self.print_iteration()
                    compute_time_py = time()
                    input_data = self.prepare_data(cur_data[2])
                    imgfeatures = self.calc_image_features([input_data[0]])
                    fdata = [imgfeatures.T, input_data[1].T]
                    mvdata = self.find_most_violated(fdata, train=True)
                    mini_mv_input_data = [self.gpu_require(e.T) for e in mvdata]
                    if K_top_update == 1:
                        mini_mv_input_data[0] = input_data[0]
                    else:
                        rep_indexes = np.tile(np.array(range(input_data[0].shape[0])).reshape((1, input_data[0].shape[0])), [K_top_update, 1]).flatten(order='F')
                        mini_mv_input_data[0] = input_data[0][rep_indexes, ...]
                    res, act_ind = self.analyze_num_sv_ext(mini_mv_input_data)
                    nsv = np.sum(act_ind.flatten())
                    print 'support vector {} of {}:\t{}%\t mmcost={}'.format(nsv, act_ind.size,
                                                                             nsv * 100.0/act_ind.size, res.mean())
                    last_mv_data = mini_mv_input_data
                    nsv_cum += nsv
                    if nsv:
                        data_list.append(self.collect_sv(act_ind, mini_mv_input_data))
                    if nsv_cum >=mini_mv_input_data[0].shape[0] * K_top_update:
                        break
                    next_epoch, next_batchnum = self.train_dp.epoch, self.train_dp.batchnum
                    if next_epoch == self.num_epoch:
                        break
                    if ct != cumulate_update_num - 1:
                        cur_data = self.get_next_batch(train=True)
                        self.cur_batch_indexes = self.train_dp.get_batch_indexes()
                if len(data_list) == 0:
                    mv_input_data = last_mv_data
                else:
                    mv_input_data = self.concatenate_data(data_list)
            else:
                self.epoch, self.batchnum = cur_data[0], cur_data[1]
                self.print_iteration()
                input_data = self.prepare_data(cur_data[2]) 
                imgfeatures = self.calc_image_features([input_data[0]])
                fdata = [imgfeatures.T, input_data[1].T]            
                mvdata = self.find_most_violated(fdata, train=True)
                mv_input_data = [self.gpu_require(e.T) for e in mvdata]
                if K_top_update == 1:
                    mv_input_data[0] = input_data[0]
                else:
                    rep_indexes = self.get_repindexes(input_data[0].shape[0], K_top_update)
                    mv_input_data[0] = input_data[0][rep_indexes, ...]
            compute_time_py = time()

            self.set_train_mode(train=True)
            info = self.do_opt(mv_input_data)
            
            cur_costs = self.extract_cost(info)
            print 'Analyze-num-sv\t [Train]'
            t0 = time()
            self.analyze_num_sv(mv_input_data)
            print '   Analyze-num-sv costs {} seconds'.format(time() - t0)
            self.train_error.append(self.pack_cost(cur_costs, self.train_net))
            print 'train: {} seconds '.format(time()- compute_time_py)
            num_updates += 1
            compute_time_py = time()
            if num_updates % self.testing_freq == 0:
                self.print_monitor_info(info, [w.name for w in self.train_net.params])
                self.print_cost(train=True)
                test_costs = self.get_test_error()
                self.test_error.append(self.pack_cost(test_costs, self.train_net))
                self.print_cost(train=False)
            cur_batch_done = self.get_num_batches_done(True)
            if (cur_batch_done // self.testing_freq) * self.testing_freq > pre_batch_done:
                self.save_model()
                print 'Test && save:\t{} seconds '.format(time()- compute_time_py)
            pre_batch_done = cur_batch_done
            self.epoch, self.batchnum = self.train_dp.epoch, self.train_dp.batchnum
            if self.epoch == self.num_epoch:
                break
            compute_time_py = time()
            cur_data = self.get_next_batch(train=True)
            self.cur_batch_indexes = self.train_dp.get_batch_indexes()
            print 'Loading data:\t{:.2f} seconds'.format(time()- compute_time_py)
    def get_test_error(self):
        self.set_train_mode(False)
        test_data = self.get_next_batch(train=False)
        input_data = self.prepare_data(test_data[2])
        imgfeatures = self.calc_image_features([input_data[0]])
        fdata = [imgfeatures.T, input_data[1].T]            
        mvdata = self.find_most_violated(fdata, train=False)
        mv_input_data = [self.gpu_require(e.T) for e in mvdata]
        K_update = self.solver_params['K_top_update'] 
        if K_update == 1:
            mv_input_data[0] = input_data[0]
        else:
            ntest = input_data[0].shape[0]
            rep_indexes = np.tile(np.array(range(ntest)).reshape((1,ntest)), [K_update, 1]).flatten(order='F')
            mv_input_data[0] = input_data[0][rep_indexes,...]
        costs = self.test_func(*mv_input_data)
        return costs
    def save_model(self):
        ignore=set(['layers'])
        net_dic = {'train_net': self.train_net.save_to_dic(),
                   'feature_net': self.feature_net.save_to_dic(ignore=ignore),
                   'score_net': self.score_net.save_to_dic(ignore=ignore)}
        net_dic['score_net']['layers'] = net_dic['train_net']['layers']
        net_dic['feature_net']['layers'] = net_dic['train_net']['layers']
        model_state = {'train': self.train_dp.get_state_dic(),
                       'test':self.test_dp.get_state_dic(),
                       'state':self.stat}
        solver_params = self.solver_params
        solver_params['train_error'] = self.train_error
        solver_params['test_error'] = self.test_error
        to_save = {'net_dic':net_dic, 'model_state':model_state, 'solver_params':solver_params}
        self._save(to_save)
        
solver_dic = {'basicbp':BasicBPSolver, 'mmls':MMLSSolver, 'imgmm':ImageMMSolver} 