"""
Exmaple
python ~/Projects/Itheano/Python/task/eval_imgdphmlpe.py --load-file=/opt/visal/tmp/for_sijin/Data/saved/Test/2015_04_06_0063_random2_simple/ --solver-type=evaldpdhmlpe --candidate-feat-pca-path=/opt/visal/tmp/for_sijin/Data/H36M/H36MExp/folder_ASM_act_14_exp_2/rel_pca.meta --candidate-feat-pca-noise=0.0003

"""
from init_task import *
import options
import iread.myio as mio
import options
from ilayer import *
from idata import *
from isolver import *
from isolver_ext import *
from igraphparser import *
import sys
import dhmlpe_utils as dutils


class EVALDPDHMLPESolver(ImageDotProdMMSolver):
    def __init__(self, net_list, train_dp, test_dp, solver_params=None):
        ImageDotProdMMSolver.__init__(self, net_list, train_dp, test_dp, solver_params)
        test_dp.reset()
        self.prediction_layer_name='fc_j2'
        # assert(self.candidate_feat_E is not None)
        self.epoch, self.batchnum = test_dp.epoch, test_dp.batchnum
    def get_scale(self,dp):
        if isinstance(dp, CroppedDHMLPEJointDataWarper):
            return (1200.0, 1200.0, int(1))
        else:
            raise Exception('No supported')
    def add_pca_noise(self, target, feat_E, sample_num,r):
        assert(len(target.shape) == 2)
        dimX, ndata = target.shape[0], target.shape[1]
        target_ext = np.tile(target, [sample_num, 1]).reshape([dimX, sample_num, ndata],order='F')
        pca_noise = np.dot(feat_E, np.random.randn(dimX, sample_num * ndata)).reshape((dimX, sample_num, ndata),order='F')/r
        res = target_ext + pca_noise
        return res
    def eval_mm_mpjpe(self):
        """
        This function will use the score prediction as the baseline
        And add Gaussian noise to 
        """
        train=False
        feat_E = self.candidate_feat_E / self.train_dp.max_depth
        # Note that noise has already been multiplied into feat_E
        net = self.feature_net
        output_layer_name = self.prediction_layer_name
        pred_outputs = net.get_layer_by_names([output_layer_name])[0].outputs
        pred_func = theano.function(inputs=net.inputs, outputs=pred_outputs,on_unused_input='ignore')
        itm = iu.itimer()
        itm.restart()
        cur_data = self.get_next_batch(train)
        sample_num = 500
        topK = 10
        pred_mpjpe_l = []
        topK_mpjpe_l = []
        itm.tic()
        
        while True:
            self.print_iteration()
            ndata = cur_data[2][0].shape[-1]
            input_data = self.prepare_data(cur_data[2])
            gt_target = input_data[1].T
            print 'gt_target.shape  = {}'.format(gt_target.shape) 
            imgfeatures = self.calc_image_features([input_data[0]]).T
            print 'imgfeatures.shape = {}'.format(imgfeatures.shape)
            preds = pred_func(*[input_data[0]])[0].T
            print 'Prediction.shape = {}'.format(preds.shape)
            candidate_targets = self.add_pca_noise(preds, feat_E, sample_num,r=1)
            candidate_features = self.calc_target_feature_func(self.gpu_require(candidate_targets.reshape((-1, sample_num * ndata)).T))[0].T
            print 'candidate_features.shape = {}'.format(candidate_features.shape)
            scores = (candidate_features.reshape((-1, sample_num, ndata),order='F') * imgfeatures.reshape((-1, 1, ndata),order='F')).sum(axis=0)
            print 'score.shape = {}'.format(scores.shape)
            sidx = np.argpartition(-scores, topK, axis=0)[:topK,...]
            sidx_in_arr = sidx + np.array(range(ndata)) * sample_num
            topK_target_list = [ candidate_targets[:, sidx[...,k].flatten() ,k].mean(axis=1,keepdims=True) for k in range(ndata)]
            topK_target = np.concatenate(topK_target_list, axis=1)
            topK_mpjpe = dutils.calc_mpjpe_from_residual(topK_target - gt_target, 17)* 1200
            pred_mpjpe = dutils.calc_mpjpe_from_residual(preds - gt_target, 17) * 1200
            print 'topK_mpjpe = {} pred_mpjpe = {}'.format(topK_mpjpe.mean(), pred_mpjpe.mean())
            pred_mpjpe_l.append(pred_mpjpe.flatten())
            topK_mpjpe_l.append(topK_mpjpe.flatten())
            self.epoch, self.batchnum = self.test_dp.epoch, self.test_dp.batchnum
            if self.epoch == 1:
                break
            cur_data = self.get_next_batch(train)
        itm.toc()
        pred_mpjpe_arr = np.concatenate(pred_mpjpe_l)
        topK_mpjpe_arr = np.concatenate(topK_mpjpe_l)
        print 'Compute {} batch {} data in total'.format(len(pred_mpjpe_l), pred_mpjpe_arr.shape)
        print 'in total pred_mpjpe = {} \t topK mpjpe = {}'.format(pred_mpjpe_arr.mean(),
                                                                   topK_mpjpe_arr.mean()
        )
    def eval_mpjpe(self, op):
        """
        This function will simple evaluate mpjpe
        """
        train=False
        net = self.feature_net
        output_layer_name = self.prediction_layer_name
        pred_outputs = net.get_layer_by_names([output_layer_name])[0].outputs
        pred_func = theano.function(inputs=net.inputs, outputs=pred_outputs,on_unused_input='ignore')
        itm = iu.itimer()
        itm.restart()
        cur_data = self.get_next_batch(train)
        pred_mpjpe_l = []
        pred_target_l = []
        save_folder = op.get_value('save_res_path')
        assert(save_folder is not None)
        iu.ensure_dir(save_folder)
        save_path = iu.fullfile(save_folder, 'pose_prediction')
        net.set_train_mode(False) # added in Oct 27, 2015 After ICCV submission
        while True:
            self.print_iteration()
            ndata = cur_data[2][0].shape[-1]
            input_data = self.prepare_data(cur_data[2])
            gt_target = input_data[1].T
            print 'gt_target.shape  = {}'.format(gt_target.shape) 
            preds = pred_func(*[input_data[0]])[0].T
            print 'Prediction.shape = {}'.format(preds.shape)
            pred_mpjpe = dutils.calc_mpjpe_from_residual(preds - gt_target, 17) * 1200
            print 'pred_mpjpe = {}'.format(pred_mpjpe.mean())
            self.epoch, self.batchnum = self.test_dp.epoch, self.test_dp.batchnum
            pred_mpjpe_l.append(pred_mpjpe)
            pred_target_l.append(preds)
            if self.epoch == 1:
                break
            cur_data = self.get_next_batch(train)
        mio.pickle(save_path, {'preds':pred_target_l, 'mpjpe':pred_mpjpe_l})
        preds = np.concatenate(pred_target_l, axis=1)
        mpjpe = np.concatenate(pred_mpjpe_l, axis=1)
        mio.pickle(save_path, {'preds':pred_target_l, 'mpjpe':pred_mpjpe_l})
        
def add_options(op):
    op.add_option('output-layer-name', 'output_layer_name', options.StringOptionParser, 'The output layer name',\
                  default='fc_j2')
    op.add_option('save-res-path', 'save_res_path', options.StringOptionParser, 'The folder to store the results', default=None)
def main():
    solver_dic['evaldphmlpe'] = EVALDPDHMLPESolver
    solver_loader = ImageDotProdMMSolverLoader()
    op = solver_loader.op
    add_options(op)
    solver = solver_loader.parse()
    # solver.eval_mm_mpjpe()
    solver.eval_mpjpe(op)

if __name__ == '__main__':
    main()