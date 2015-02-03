"""

python ~/Projects/Itheano/Python/task/eval_hmlpe.py --data-path=/opt/visal/tmp/for_sijin/Data/H36M/H36MExp/folder_ASM_act_14_exp_2 --num-epoch=200 --data-provider=croppedjt --layer-def=/home/grads/sijinli2/Projects/Itheano/doc/netdef/graph_def_0008.cfg --testing-freq=15 --batch-size=256 --train-range=0-132743 --test-range=132744-162007 --save-path=/opt/visal/tmp/for_sijin/tmp/test_basic_bp --solver-type=evalhmlpe --load-file=/opt/visal/tmp/for_sijin/tmp/test_basic_bp
"""
from init_task import *
import options
import iread.myio as mio
import options
from ilayer import *
from idata import *
from isolver import *
from igraphparser import *
import sys
import dhmlpe_utils as dutils

from time import time
class EvalHMLPESolver(BasicBPSolver):
    def __init__(self, net_list, train_dp, test_dp, solver_params=None):
        BasicBPSolver.__init__(self, net_list, train_dp, test_dp, solver_params)
        test_dp.epoch = test_dp.batchnum = 0
    def eval_mpjpe(self):
        test_one = False
        train = False
        output_layer_name = 'fc_j2'
        gt_idx = 1
        outputs = self.net.get_layer_by_names([output_layer_name])[0].outputs
        cur_data = self.get_next_batch(train)
        eval_func = theano.function(inputs=self.net.inputs, outputs=outputs,
                                    on_unused_input='ignore')
        mpjpe_l = []
        compute_time_py = time()
        self.epoch, self.batchnum=cur_data[0], cur_data[1]
        self.net.set_train_mode(False)
        while True:
            self.print_iteration()
            input_data = self.prepare_data(cur_data[2])
            res = eval_func(*input_data)[0].T
            gt = input_data[gt_idx].T
            
            residuals = (np.require(res,dtype=np.float64) - np.require(gt,dtype=np.float64))
            mpjpe = dutils.calc_mpjpe_from_residual(residuals, 17)
            print '----'
            print res[:3,:3]
            print '----'
            print 'mpjpe shape is {}'.format(mpjpe.shape)
            print 'cur_mean is {}'.format(np.mean(mpjpe.flatten())* 1200)
            mpjpe_l += [mpjpe * 1200]
            if test_one:
                break
            self.epoch, self.batchnum = self.pre_advance_batch(train)
            if self.epoch == 1:
                break
            cur_data = self.get_next_batch(train)
        res_arr = np.concatenate(mpjpe_l, axis=1)
        print 'Compute {} batch {} data in total: {} seconds '.format(len(mpjpe_l),
                                                                      res_arr.size,
                                                                      time() - compute_time_py)
        print 'mean mpjpe = {}'.format(np.mean(res_arr.flatten()))
def main():
    solver_dic['evalhmlpe'] = EvalHMLPESolver
    print solver_dic
    solver_loader = SolverLoader()
    solver = solver_loader.parse()
    solver.eval_mpjpe()

if __name__ == '__main__':
    main()
    