"""
This file contains all kinds of extension class of basic solver type for different application
Because some might need special processing
"""
from isolver import *
from ilayer import *
class MMLSSolverLoader(MMSolverLoader):
    def pre_process_data(self, d):
        print 'I have preprocess the data by dividing feature_0 '
        d['feature_list'][0] = d['feature_list'][0] / 1200.0
    def load_data_dic(self, dp_params):
        meta = mio.unpickle(iu.fullfile(dp_params['data_path'], 'batches.meta'))
        self.pre_process_data(meta)
        return meta
    def parse_solver_params(self, solver_params, op):
        MMSolverLoader.parse_solver_params(self, solver_params, op)
        solver_params['net_order'] = ['network1', 'network2']
class ImageMMSolverLoader(MMSolverLoader):
    def parse_solver_params(self, solver_params, op):
        MMSolverLoader.parse_solver_params(self, solver_params, op)
        solver_params['net_order'] = ['train_net', 'feature_net', 'score_net']
    def add_default_options(self, op):
        MMSolverLoader.add_default_options(self, op)
        op.add_option('opt-method', 'opt_method', options.StringOptionParser, 'the optimization methods [bp | ls]', default='bp')
    