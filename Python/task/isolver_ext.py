"""
This file contains all kinds of extension class of basic solver type for different application
Because some might need special processing
"""
from isolver import *
from ilayer import *
class MMSolverLoader(SolverLoader):
    def add_default_options(self, op):
        SolverLoader.add_default_options(self, op)
        op.add_option('K-candidate', 'K_candidate', options.IntegerOptionParser, 'the number of candidate', default=None)
        op.add_option('K-most-violated', 'K_most_violated', options.IntegerOptionParser, 'the size of the hold-on set', default=None)
        op.add_option('max-num', 'max_num', options.IntegerOptionParser, 'The maximum number of sample to be processed')
        op.add_option('margin-func', 'margin_func', options.StringOptionParser, 'the parameters for marginl', default='mpjpe')
        op.add_option('candidate-mode', 'candidate_mode', options.StringOptionParser, 'the parameters for marginl', default='random')
    def pre_process_data(self, d):
        print 'I have preprocess the data by dividing feature_0 '
        d['feature_list'][0] = d['feature_list'][0] / 1200.0
    def load_data_dic(self, dp_params):
        meta = mio.unpickle(iu.fullfile(dp_params['data_path'], 'batches.meta'))
        self.pre_process_data(meta)
        return meta
    def parse_solver_params(self, solver_params, op):
        SolverLoader.parse_solver_params(self, solver_params, op)
        solver_params['net_order'] = ['network1', 'network2']
        if (not 'margin_func' in solver_params) and (op.get_value('margin_func') is not None):
            s = op.get_value('margin_func').split(',')
            name = s[0]
            margin_params = None if len(s) == 1 else [float(x) for x in s[1:]]
            solver_params['margin_func'] = {'name':name, 'params':margin_params}
        else:
            solver_params['margin_func'] = {'name':'mpjpe', 'params':None}

