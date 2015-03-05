"""
Example:
python ~/Projects/Itheano/Python/test/test_loadnet.py --load-file=/opt/visal/tmp/for_sijin/Data/saved/theano_models/SP_t004_act_14_0010 --data-path=/opt/visal/tmp/for_sijin/Data/H36M/H36MExp/folder_SP_t004_act_14 --data-provider=mem --layer-def=/home/grads/sijinli2/Projects/Itheano/doc/netdef/graph_def_0010.cfg --testing-freq=1 --batch-size=1024 --train-range=0-132743 --test-range=132744-162007 --solver-type=mmls


"""
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

def print_dims(alldata):     # debug use
    for i,e in enumerate(alldata):
        print 'Dim {}: \t shape {} \t type {}'.format(i, e.shape, type(e))

def show_highest_score(train, solver):
    """
    This function will load data from train or test set
    """
   
    dp = solver.train_dp if train else solver.test_dp
    data = solver.get_next_batch(train)
    alldata, ext_data  = solver.find_most_violated_ext(data[2],use_zero_margin=True,
                                                               train=train
    )
    print len(alldata)
    gt_target = alldata[0]
    gt_margin= alldata[4]
    img_features = alldata[1]
    mv_target = ext_data[0]
    batch_candidate_indexes =ext_data[1]
    print 'batch candidate indexes shape is {}'.format(batch_candidate_indexes.shape)
    mv_features = alldata[3]
    gt_features = alldata[2]
    # mv_margin = solver.calc_margin(gt_target - mv_target)
    mv_margin = alldata[5]
    print "mv shape {}, gt shape {}".format(mv_target.shape, gt_target.shape)

    fl = solver.get_all_candidates(solver.train_dp)
    batch_candidate_targets = fl[0][...,batch_candidate_indexes]
    ndata = gt_target.shape[-1]
    data_to_eval = [solver.gpu_require(img_features.T),
                    solver.gpu_require(mv_features.T),
                    solver.gpu_require(mv_margin.T)
    ]
    print 'Eval inpus are {}'.format([l.name for l in solver.eval_net.inputs])
    mv_score = solver.eval_net.outputs[0].eval(dict(zip(solver.eval_net.inputs, data_to_eval)))
    data_to_eval = [solver.gpu_require(img_features.T),
                    solver.gpu_require(gt_features.T),
                    solver.gpu_require(gt_margin.T)
    ]
    gt_score = solver.eval_net.outputs[0].eval(dict(zip(solver.eval_net.inputs, data_to_eval)))

    res_mpjpe, bmi  =  get_batch_best_match(batch_candidate_targets, gt_target, solver)
    print 'Current best match mpjpe is {}'.format(np.mean(res_mpjpe)* 1200)
    
    bmi_raw = batch_candidate_indexes[bmi]
    bm_features = fl[2][..., bmi_raw]
    bm_targets  = fl[0][..., bmi_raw]
    residuals = bm_targets - gt_target
    mpjpe = dutils.calc_mpjpe_from_residual(residuals, 17) # mpjpe for best match
    print 'mpjpe(bm_target, gt_target) is {}'.format(np.mean(mpjpe.flatten())*1200)
    data_to_eval = [solver.gpu_require(img_features.T),
                    solver.gpu_require(bm_features.T),
                    solver.gpu_require(gt_margin.T)
    ]
    bm_score = solver.eval_net.outputs[0].eval(dict(zip(solver.eval_net.inputs, data_to_eval)))
   
    all_input_data = [solver.gpu_require(e.T) for e in alldata[1:]]
    solver.analyze_num_sv(all_input_data)
    # all_input_data = [all_input_data[0], all_input_data[2], all_input_data[1],
    #                   all_input_data[4], all_input_data[3]]
    # solver.print_layer_outputs(all_input_data)

    # Ignore the use_zero margin flag
    whole_candidate_set = solver.train_dp.data_dic['feature_list'][0][..., solver.train_dp.data_range]
    # print 'Whole candidate_set shape is {}'.format(whole_candidate_set.shape)
    # what_is_the_best_match( whole_candidate_set , mv_target, solver)
    # show_what_is_best_all(solver.train_dp, solver.test_dp, solver)
    mv_margin = solver.calc_margin(gt_target - mv_target)  # MPJPE
    print 'gt_margin<======================'
    iu.print_common_statistics(gt_margin)
    print 'mv_margin<======================'
    iu.print_common_statistics(alldata[5])
    show_bm_cmp(ndata, gt_target, mv_target, bm_targets, mv_score, gt_score, bm_score, solver)
    show_masked_plot(ndata, mv_margin, mv_score, gt_score, bm_score)
    show_raw_plot(ndata, mv_margin, mv_score, gt_score)
    # print 'Strange Here: {:.6f}% is correct'.format()
    pl.show()
def get_batch_best_match(candidate_targets, mv_targets, solver):
    """
    Find the best match under mpjpe criteria
    """
    start_time = time()
    ndata = mv_targets.shape[-1]
    c_tot = candidate_targets.shape[-1]
    per = c_tot // ndata
    assert(per * ndata == c_tot)
    match_indexes = np.zeros(ndata)
    res = np.zeros(ndata)
    for i,p in enumerate(mv_targets.T):
        b_idx, e_idx = per * i, per * (i + 1)
        t = p.reshape((-1,1),order='F')
        residuals = candidate_targets[..., b_idx:e_idx] - t
        mpjpe = solver.calc_margin(residuals).flatten()
        idx = np.argmin(mpjpe)
        match_indexes[i] = idx + i * per
        res[i] = mpjpe[idx]
    match_indexes = np.array(match_indexes, dtype=np.int)
    print 'Cost {} seconds for finding the best match for current batch'.format(time() - start_time)
    return res, match_indexes
def find_best_match(candidate_set, jt_data, solver):
    ndata = jt_data.shape[-1]
    res = np.zeros(ndata)
    match_indexes = np.zeros(ndata)
    for i,p in enumerate(jt_data.T):
        t = p.reshape((-1,1),order='F')
        residuals = t - candidate_set
        mpjpe = solver.calc_margin(residuals).flatten()
        idx= np.argmin(mpjpe)
        res[i]= mpjpe[idx]
        match_indexes[i] = idx 
    return res, match_indexes
def what_is_the_best_match(candidate_set, jt_data, solver):
    start_time = time()
    res, dummy = find_best_match(candidate_set, jt_data, solver)
    print 'Average mpjpe is {}'.format(np.mean(res.flatten())* 1200)
    print 'Cost {} seconds for finding the best'.format(time() - start_time)
    return res
def show_bm_cmp_from_saved():
    save_path = '/public/sijinli2/ibuffer/2015-01-22/saved_batch_data_test_net4_K_rbf_test'
    save_path = '/public/sijinli2/ibuffer/2015-02-03/bm_vs_mv_SP_t004_act_14_0010_model'
    d = mio.unpickle(save_path)
    print d.keys()
    bm_target, gt_score, mv_score, mv_target, gt_target, bm_score = d['bm_target'], \
                                                                    d['gt_score'], \
                                                                    d['mv_score'], \
                                                                    d['mv_target'],\
                                                                    d['gt_target'],\
                                                                    d['bm_score']
    ndata = bm_target.shape[-1]
    bm_margin = MMLSSolver.calc_margin(gt_target - bm_target).flatten() * 1200
    mv_margin = MMLSSolver.calc_margin(gt_target - mv_target).flatten() * 1200

    # bm_rbf_margin = 1 - dutils.calc_RBF_score(gt_target - bm_target, 50/1200.0,3).flatten()
    # mv_rbf_margin = 1 - dutils.calc_RBF_score(gt_target - mv_target, 50/1200.0,3).flatten()
    # pl.ylabel('bm rbf margin')
    # pl.xlabel('highest score rbf margin')
    # pl.scatter(mv_rbf_margin, bm_rbf_margin, s=15, c='b')
    # pl.show()
    
    score_diff = mv_score - bm_score
    pl.ylabel('score_diff')
    pl.xlabel('mpjpe (bestmatch_pose, max_score_pose)')
    pl.scatter(bm_margin, score_diff, s=15, c='b')
    pl.show()
    
    #    show the scatter plot for mpjpe vs RBF score
    residuals = (gt_target - bm_target) * 1200
    bm_margin = bm_margin / 1200.0
    t = 0
    ncol = 4
    sigma_list = [1,5,10, 20, 50, 100, 200, 500, 1000, 1500, 2000, 5000]
    nrow = (len(sigma_list) - 1)//ncol  + 1
    # mm = np.max(bm_margin)
    for r in range(nrow):
        for c in range(ncol):
            sigma = float(sigma_list[t])
            t = t + 1
            pl.subplot(nrow, ncol, t)
            bm_rbf_score = 1 - dutils.calc_RBF_score(residuals, sigma)
            pl.xlabel('mpjpe (bestmatch_pose, max_score_pose) ')
            pl.ylabel('RBF sigma = {:.1f}'.format(sigma))
            pl.scatter(bm_margin, bm_rbf_score)
            if t == len(sigma_list):
                break
    pl.show()
    
def show_bm_cmp(ndata, gt_target, mv_target, bm_targets, mv_score, gt_score,
                bm_score, solver=None, save_path=None):
    mv_margin = MMLSSolver.calc_margin(gt_target - mv_target).flatten() * 1200
    bm_margin = MMLSSolver.calc_margin(gt_target - bm_targets).flatten() * 1200
    save_path = '/public/sijinli2/ibuffer/2015-02-04/bm_cmp_SP_t004_act_14_graph_0010'
    save_path = '/public/sijinli2/ibuffer/2015-02-14/bm_cmp_FCJ0_act_14_graph_0020_test_'

    d = {'gt_target':gt_target, 'mv_target':mv_target, 'bm_target':bm_targets,
         'mv_score':mv_score, 'gt_score':gt_score, 'bm_score':bm_score
    }
    if save_path:
        mio.pickle(save_path, d)
    ndata = mv_margin.size
    pt_size = 15
    pl.subplot(4,1,1)
    avg = np.mean(bm_margin)
    pl.scatter(range(ndata), bm_margin, s=pt_size,c='r',label='mpjpe between max_score and best match (avg={:.3f})'.format(avg))
    pl.legend()
    pl.subplot(4,1,2)
    pl.scatter(range(ndata), mv_score - bm_score, s=pt_size,c='b',label='max score - best_match_score')
    pl.legend()
    pl.subplot(4,1,3)
    pl.scatter(range(ndata), (mv_score - bm_score)/np.abs(mv_score), c='b',label='(max_score - best_match_score)/abs(max_score)')
    pl.legend()
    pl.subplot(4,1,4)
    pl.scatter(range(ndata), mv_score, s=pt_size, c='r',label='max score')
    pl.scatter(range(ndata), bm_score, s=pt_size,c='b',label='best match score')
    pl.legend()
    pl.show()
def show_masked_plot(ndata, mv_margin, mv_score, gt_score, bm_score = None):
    mv_margin, mv_score, gt_score = mv_margin.flatten(), mv_score.flatten(), gt_score.flatten()
    diff = mv_score - gt_score
    good_one = diff <= 0 
    masked_margin = mv_margin.copy()
    masked_margin[good_one] = 0
    avg_mpjpe = np.mean(mv_margin.flatten())
    print 'mpjpe() is {}'.format(avg_mpjpe * 1200)
    pl.subplot(3,1,1)
    pl.plot(range(ndata), masked_margin * 1200, label='masked_margin')
    pl.legend()
    pl.subplot(3,1,2)
    pl.plot(range(ndata), mv_score, label='mv_score')
    pl.plot(range(ndata), gt_score, label='gt_score')
    if bm_score is not None:
        pl.plot(range(ndata), bm_score.flatten(), label='best_match_score')
    pl.legend()
    pl.subplot(3,1,3)
    masked_diff = diff.copy()
    masked_diff[good_one] = 0
    grate = good_one.sum() * 100.0/good_one.size
    pl.plot(range(ndata), masked_diff, '-', label='max(0, mv_score - gt_score)\n acc={:3f}%'.format(grate))
    pl.legend()
    pl.show()
def show_raw_plot(ndata, mv_margin, mv_score, gt_score):
    pl.subplot(3,1,1)
    pl.plot(range(ndata), mv_margin.flatten() * 1200, label='margin')
    pl.legend()
    pl.subplot(3,1,2)
    pl.plot(range(ndata), mv_score.flatten(), label='mv_score')
    pl.plot(range(ndata), gt_score.flatten(), label='gt_score')
    pl.legend()
    pl.subplot(3,1,3)
    pl.plot(range(ndata), mv_score.flatten() - gt_score.flatten(), label='mv_score - gt_score')
    pl.legend()
    pl.show()
def show_what_is_best_all(train_dp, test_dp, solver):
    """
    In this funciton, I will show what is the upper bound for the KNN search like methods
    """
    candidates = train_dp.data_dic['feature_list'][0][..., train_dp.data_range]
    eval_data = train_dp.data_dic['feature_list'][0][..., test_dp.data_range]
    print 'Begin to calc the best mpjpe candidate shape {} eval shape'.format(candidates.shape, eval_data.shape)
    res = what_is_the_best_match(candidates, eval_data, solver)
    save_path = '/public/sijinli2/ibuffer/2015-01-16/best_match_act_14'
    mio.pickle(save_path, {'best_mpjpe':res})
def main():
    solver_loader = MMLSSolverLoader()
    solver = solver_loader.parse()
    # solver.solver_params['candidate_mode'] =
    solver.solver_params['K_candidate'] = 20000
    # solver.solver_params['candidate_mode'] = 'all'
    solver.solver_params['max_num'] = 1
    show_highest_score(train=False, solver=solver)
    # show_bm_cmp_from_saved()
    # test_shared_weights()
    # retuirn
    # save_again()
    # return

if __name__ == '__main__':
    main()