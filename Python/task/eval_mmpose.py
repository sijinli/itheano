"""
example:
python ~/Projects/Itheano/Python/task/eval_mmpose.py /opt/visal/tmp/for_sijin/Data/saved/theano_models/2015_02_17_ASM_act_14_exp_2_graph_0025 --solver-type=imgmm




"""
from init_task import *
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
import Image
from isolver_ext import *
import imgproc
from time import time
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
        mpjpe = solver.margin_func(residuals).flatten()
        idx = np.argmin(mpjpe)
        match_indexes[i] = idx + i * per
        res[i] = mpjpe[idx]
    match_indexes = np.array(match_indexes, dtype=np.int)
    print '    Cost {} seconds for finding the best match for current batch'.format(time() - start_time)
    return res, match_indexes
def get_feature_list(solver, dp):
    if solver._solver_type == 'mmls':
        return solver.get_all_candidates(dp)
    else:
        fl = solver.get_all_candidates(dp)
        return [fl[0],None, fl[0]]
def get_eval_func(solver):
    if solver._solver_type in ['mmls', 'imgmm', 'imgdpmm']:
        return solver.eval_func
    else:
        raise Exception('Can not find eval_func')
def show_bm_cmp(ndata, gt_target, mv_target, bm_targets, mv_score, gt_score,
                bm_score, solver=None, save_path=None):
    mv_margin = MMLSSolver.calc_margin(gt_target - mv_target).flatten() * 1200
    bm_margin = MMLSSolver.calc_margin(gt_target - bm_targets).flatten() * 1200
    
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
    # pl.show()
def get_fdata(solver, data):
    if solver._solver_type == 'mmls':
        return data[2]
    else:
        input_data = solver.prepare_data(data[2])
        imgfeatures = solver.calc_image_features([input_data[0]])
        fdata = [imgfeatures.T, input_data[1].T]
    return fdata
def parse_mvdata(solver, alldata):
    # return img_feature, gt_target, gt_margin, gt_feature, mv_features, mv_margin
    if solver._solver_type in ['imgmm','imgdpmm']:
        return alldata[0], alldata[1], alldata[3], alldata[1], alldata[2], alldata[4]
    else:
        return alldata[1], alldata[0], alldata[4], alldata[2], alldata[3], alldata[5]
def get_batch_best_match(candidate_targets, targets, solver):
    """
    Find the best match under mpjpe criteria
    """
    start_time = time()
    ndata = targets.shape[-1]
    c_tot = candidate_targets.shape[-1]
    per = c_tot // ndata
    assert(per * ndata == c_tot)
    match_indexes = np.zeros(ndata)
    res = np.zeros(ndata)
    for i,p in enumerate(targets.T):
        b_idx, e_idx = per * i, per * (i + 1)
        t = p.reshape((-1,1),order='F')
        residuals = candidate_targets[..., b_idx:e_idx] - t
        mpjpe = MMLSSolver.calc_margin(residuals).flatten()
        idx = np.argmin(mpjpe)
        match_indexes[i] = idx + i * per
        res[i] = mpjpe[idx]
    match_indexes = np.array(match_indexes, dtype=np.int)
    print 'Cost {} seconds for finding the best match for current batch'.format(time() - start_time)
    return res, match_indexes
def show_highest_score(train, solver, op):
    """
    This function will load data from train or test set
    """
    no_display = op.get_value('no_display')
    dp = solver.train_dp if train else solver.test_dp
    dp.reset()
    func = get_eval_func(solver)
    data = solver.get_next_batch(train)
    indexes = dp.get_batch_indexes()
    print indexes[:10], '<<<<<<<<<<<<<<<INdexes'
    fdata = get_fdata(solver, data)
    itm = iu.itimer()
    itm.restart()
    itm.addtag('begin')
    alldata, ext_data  = solver.find_most_violated_ext(fdata,use_zero_margin=True,
                                                               train=False
    )
    itm.addtag('find most violated ext')
    mv_target, batch_candidate_indexes = ext_data[0], ext_data[1]
    img_features, gt_target, gt_margin, gt_features, mv_features, mv_margin = parse_mvdata(solver, alldata)

    fl = get_feature_list(solver, solver.train_dp)
    batch_candidate_targets = fl[0][...,batch_candidate_indexes]
    ndata = gt_target.shape[-1]
    res_mpjpe, bmi  =  get_batch_best_match(batch_candidate_targets, gt_target, solver)
    itm.addtag('get batch best match')
    bmi_raw = batch_candidate_indexes[bmi]
    bm_features = fl[2][..., bmi_raw]
    bm_targets  = fl[0][..., bmi_raw]
    itm.print_all()
    residuals = bm_targets - gt_target
    mpjpe = dutils.calc_mpjpe_from_residual(residuals, 17) # mpjpe for best match
    print 'Residuals for best match poses {}'.format(mpjpe.mean()* 1200)
    def get_score(X):
        eval_input_data = [solver.gpu_require(img_features.T),
                           solver.gpu_require(X.T),
                           solver.gpu_require(gt_margin.T)]
        return func(*eval_input_data)[0]
    mv_score = get_score(mv_features)
    bm_score = get_score(bm_features)
    gt_score = get_score(gt_features)
    # all_input_data = [solver.gpu_require(e.T) for e in alldata[1:]]
    # solver.analyze_num_sv(all_input_data)
    mv_margin = MMLSSolver.calc_margin(gt_target - mv_target)  # MPJPE
    print 'mpjpe as mv_margin is {}'.format(mv_margin.flatten().mean()*1200)
    print 'gt_margin<======================'
    iu.print_common_statistics(gt_margin)
    print 'mv_margin<======================'
    iu.print_common_statistics(mv_margin)
    show_bm_cmp(ndata, gt_target, mv_target, bm_targets, mv_score,
                gt_score, bm_score, solver)

    # show_raw_plot(ndata, mv_margin, mv_score, gt_score)
    # print 'Strange Here: {:.6f}% is correct'.format()
    
    # for debug
    
    # d = {'gt':gt_target, 'gt_feature':gt_features, 'img_features':img_features,
    #      'gt_score':gt_score}
    # save_path  = '/public/sijinli2/ibuffer/2015-03-08/t1'
    # mio.pickle(save_path, d)
    #
    if not no_display:
        pl.show()
def eval_mpjpe_all(train, solver):
    assert(train==False)
    save_folder = '/public/sijinli2/ibuffer/2015-03-13/2015_03_10_0037_corrected_version'
    save_folder = '/public/sijinli2/ibuffer/2015-03-13/2015_03_13_0039_1_slack/'
    save_folder = '/public/sijinli2/ibuffer/2015-03-13/2015_03_12_0039_slack_again_100_epoch_25/'
    save_folder = '/public/sijinli2/ibuffer/2015-03-13/2015_03_14_0040_slack_epoch_29'
    save_folder = '/public/sijinli2/ibuffer/2015-03-13/2015_03_14_0039_4_slack/'
    save_folder = '/public/sijinli2/ibuffer/2015-03-13/2015_03_12_0039_slack_again_100_epoch_34_train'
    save_folder = '/public/sijinli2/ibuffer/2015-03-13/2015_03_17_0041_test'
    save_folder = '/public/sijinli2/ibuffer/2015-03-13/2015_03_17_0041_test'
    save_folder= '/public/sijinli2/ibuffer/2015-03-13/2015_03_20_0046_test/'
    save_folder = '/public/sijinli2/ibuffer/2015-03-13/2015_03_17_0043_test'
    save_folder = '/public/sijinli2/ibuffer/2015-03-13/2015_03_21_0042_test_sgd_randomneighbor'
    save_folder = op.get_value('save_res_path')
    iu.ensure_dir(save_folder)
    save_path  = iu.fullfile(save_folder, 'eval_mpjpe')
    dp = solver.test_dp
    func = get_eval_func(solver)
    dp.reset()
    num_batch = dp.num_batch
    mpjpe_list = []
    target_list = []
    mvtarget_list = []
    bmtarget_list = []
    gt_score_list = []
    mv_score_list = []
    def get_score(img_features, X, gt_margin):
        eval_input_data = [solver.gpu_require(img_features.T),
                           solver.gpu_require(X.T),
                           solver.gpu_require(gt_margin.T)]
        return func(*eval_input_data)[0]
    def pack_to_save(m_list, t_list, mt_list, bm_list, gt_score_list, mv_score_list, bmi_raw, mv_indexes):
        start_time = time()
        mpjpe_arr = np.concatenate(m_list)
        target_arr = np.concatenate(t_list, axis=1)
        mvtarget_arr = np.concatenate(mt_list, axis=1)
        bmtarget_arr = np.concatenate(bm_list, axis=1)
        gt_score_arr = np.concatenate(gt_score_list)
        mv_score_arr = np.concatenate(mv_score_list)
        td = {'mpjpe':mpjpe_arr, 'target':target_arr, 'mv_target':mvtarget_arr,
              'bm_target':bmtarget_arr,
              'gt_score':gt_score_arr, 'mv_score':mv_score_arr, 'bm_index':bmi_raw, 'mv_index':mv_indexes}
        mio.pickle(save_path, td)
        print '    save cost {}'.format(time() - start_time)
    fl = get_feature_list(solver, solver.train_dp)
    for b in range(num_batch):
        print 'Process batch {}'.format(b)
        data = solver.get_next_batch(train)
        fdata = get_fdata(solver, data)
        alldata, ext_data = solver.find_most_violated_ext(fdata,use_zero_margin=True,
                                                               train=False)
        mv_target, batch_candidate_indexes = ext_data[0], ext_data[1]
        img_features, gt_target, gt_margin, gt_features, mv_features, mv_margin = parse_mvdata(solver, alldata)
        mv_margin = MMLSSolver.calc_margin(gt_target - mv_target).flatten() * 1200
        gt_score = get_score(img_features, gt_features, gt_margin)
        mv_score = get_score(img_features, mv_features, gt_margin)

        batch_candidate_targets = fl[0][...,batch_candidate_indexes]
        res_mpjpe, bmi  =  get_batch_best_match(batch_candidate_targets, gt_target, solver)
        bmi_raw = batch_candidate_indexes[bmi]
        bm_target  = fl[0][..., bmi_raw]

        mv_indexes = ext_data[2]
        
        mpjpe_list.append(mv_margin)
        target_list.append(gt_target)
        mvtarget_list.append(mv_target)
        bmtarget_list.append(bm_target)
        gt_score_list.append(gt_score.flatten())
        mv_score_list.append(mv_score.flatten())
        print '    Cur mpjpe is {}'.format(np.mean(mv_margin))
        pack_to_save(mpjpe_list, target_list, mvtarget_list, bmtarget_list, gt_score_list, mv_score_list, bmi_raw, mv_indexes)
def analyze_eval_saved():
    from mpl_toolkits.mplot3d import Axes3D
    import imgproc
    import iread.h36m_hmlpe as h36m
    model_folder_name = '2015_03_20_0046_test'
    saved_path = '/public/sijinli2/ibuffer/2015-03-13/{}/eval_mpjpe'.format(model_folder_name)
    d = mio.unpickle(saved_path)
    def get_images_path():
        exp_meta_path = '/opt/visal/tmp/for_sijin/Data/H36M/H36MExp/folder_ASM_act_14_exp_2/batches.meta'
        d = mio.unpickle(exp_meta_path)
        return [d['images_path'][k] for k in range(132744,162008)]
    images_path = get_images_path()
    print 'Overall mpjpe is {}'.format(np.mean(d['mpjpe'].flatten()))
    mpjpe, target, mv_target = d['mpjpe'].flatten(), d['target'], d['mv_target']
    bm_target = d['bm_target']
    ndata = mpjpe.size
    show_best = True
    print 'There are {} data in total'.format(ndata)
    sorted_idx = np.array(sorted(np.array(range(ndata)), key=lambda k:mpjpe[k],reverse=not show_best))
    def show_plot_bm_mv():
        bm_mpjpe = MMLSSolver.calc_margin(target - bm_target).flatten()* 1200
        mv_mpjpe = MMLSSolver.calc_margin(target - mv_target).flatten() * 1200
        # diff = np.abs(mv_mpjpe[1:] - mv_mpjpe[0:-1])
        # diff_i = np.abs(diff) > 1
        # diff[diff_i] = 1
        indexes = sorted(range(bm_mpjpe.size), key=lambda k:bm_mpjpe[k])
        imgproc.set_figure_size(50,10)
        fig = pl.figure()
        ax = fig.add_subplot(111)
        # ax.plot(range(ndata-1), diff)
        # pl.show()
        
        ax.plot(range(ndata), bm_mpjpe[indexes], c='r', label='bm_mpjpe')
        ax.plot(range(ndata), mv_mpjpe[indexes], c='b', label='maximum score mpjpe')
        pl.legend()
        imgproc.imsave_tight('/public/sijinli2/ibuffer/2015-03-13/{}/{}_bm_mv_plot.png'.format(model_folder_name, model_folder_name))
        pl.show()
    def show_scatter_bm_mv():
        bm_mpjpe = MMLSSolver.calc_margin(target- bm_target).flatten()* 1200
        mv_mpjpe = mpjpe.flatten()
        fig = pl.figure()
        ax = fig.add_subplot(111)
        ax.scatter(bm_mpjpe, mv_mpjpe,s=5)
        ax.set_aspect(1)
        ax.set_xlim([0,400])
        ax.set_ylim([0,400])
        pl.xlabel('best match mpjpe')
        pl.ylabel('maximum score mpjpe')
        imgproc.imsave_tight('/public/sijinli2/ibuffer/2015-03-13/{}/{}_scatter_bm_mv_debug.png'.format(model_folder_name, model_folder_name))
        # pl.savefig('/public/sijinli2/ibuffer/2015-03-13/test.png', bbox_inches='tight')
        pl.show()
    def show_gt_mv_score():
        gt_score, mv_score = d['gt_score'], d['mv_score']
        fig = pl.figure()
        ax = fig.add_subplot(111)
        ax.scatter(gt_score, mv_score)
        lims = np.abs(gt_score.flatten()).max()*1.5
        ax.plot([-lims,lims],[-lims, lims],c='r')
        ax.set_aspect(1)
        ax.set_xlim([-lims,lims])
        ax.set_ylim([-lims,lims])
        pl.xlabel('score of ground-truth pose')
        pl.ylabel('maximum score in the candidate set')
        imgproc.imsave_tight('/public/sijinli2/ibuffer/2015-03-13/{}/{}_gt_vs_mv_score_debug.png'.format(model_folder_name, model_folder_name))
        pl.show()
    # show_plot_bm_mv()
    # show_gt_mv_score()
    # show_scatter_bm_mv()

    def show_mpjpe_hist():
        s_mpjpe = mpjpe[sorted_idx]
        pl.subplot(2,1,1)
        pl.plot(range(ndata), s_mpjpe)
        pl.title('mpjpe all avg = {}'.format(np.mean(s_mpjpe)))
        pl.subplot(2,1,2)
        pl.hist(s_mpjpe, bins=range(0, 1000, 5))
        pl.title('mpjpe hist')
        pl.show()
        return
    def show_pose_cmp():
        num_to_show = 20
        start = 2000
        step = 1000
        # show_idx = sorted_idx[start:16*step+start:step]
        show_idx = sorted_idx[start:start+num_to_show * step:step]
        n_row, n_col = 3, num_to_show
        fig = pl.figure()
        nc = 1
        limbs = h36m.part_idx
        params = {'elev':-89, 'azim':-87, 'linewidth':3}
        bm_mpjpe = MMLSSolver.calc_margin(target[..., show_idx] - bm_target[...,show_idx]).flatten() * 1200
        cur_images_path = [images_path[k] for k in show_idx]
        show_img = True
        imgproc.set_figure_size(600,600)
        for i in range(num_to_show):
            idx = show_idx[i]
            p0 = target[..., idx].reshape((3,17),order='F').T
            p1 = mv_target[..., idx].reshape((3,17),order='F').T
            p2 = bm_target[..., idx].reshape((3,17),order='F').T
            err = mpjpe[idx]
            if show_img:
                ax = fig.add_subplot(n_row, n_col, nc)
                imgproc.turn_off_axis()
                img = np.asarray(Image.open(cur_images_path[i]))
                pl.imshow(img)
                pl.title('input')
            else:
                ax = fig.add_subplot(n_row, n_col, nc, projection='3d')
                imgproc.turn_off_axis()
                dutils.show_3d_skeleton(p0, limbs, params)
                pl.title('gt-pose')
            #
            ax = fig.add_subplot(n_row, n_col, nc + num_to_show, projection='3d')
            imgproc.turn_off_axis()
            dutils.show_3d_skeleton(p1, limbs, params)
            pl.title('mv({:.2f})'.format(err))
            print 'mpjpe = {}'.format(err)
            #
            ax = fig.add_subplot(n_row, n_col, nc + num_to_show * 2, projection='3d')
            imgproc.turn_off_axis()
            dutils.show_3d_skeleton(p2, limbs, params)
            pl.title('bm({:.2f})'.format(bm_mpjpe[i]))
            nc = nc + 1
        imgproc.imsave_tight('/public/sijinli2/ibuffer/2015-03-13/{}/{}_show_pose_cmp.png'.format(model_folder_name, model_folder_name))      
    show_pose_cmp()
    pl.show()
        

def show_mpjpe_vs_score(train, solver):
    dp = solver.train_dp if train else solver.test_dp
    assert(train == False)
    func = get_eval_func(solver)
    data = solver.get_next_batch(train)
    indexes = dp.get_batch_indexes()
    fdata = get_fdata(solver, data)
    
    alldata, ext_data  = solver.find_most_violated_ext(fdata,use_zero_margin=True,
                                                               train=False
    )     
    fl = get_feature_list(solver, solver.train_dp)
    mv_target, batch_candidate_indexes = ext_data[0], ext_data[1]
    img_features, gt_target, gt_margin, gt_features, mv_features, mv_margin = parse_mvdata(solver, alldata)
    batch_candidate_targets = fl[0][...,batch_candidate_indexes]
    res_mpjpe, bmi  =  get_batch_best_match(batch_candidate_targets, gt_target, solver)
    bmi_raw = batch_candidate_indexes[bmi]
    per = batch_candidate_indexes.size // indexes.size
    bmi_each = np.asarray(bmi) - np.array(range(0, gt_features.shape[-1])) * per
    print bmi_each, max(bmi_each),min(bmi_each), 'per = {}'.format(per)
    bm_targets  = fl[0][..., bmi_raw]
    bm_mpjpe = dutils.calc_mpjpe_from_residual(bm_targets - gt_target, 17).flatten()
    mv_mpjpe = dutils.calc_mpjpe_from_residual(mv_target - gt_target, 17).flatten()
    diff = mv_mpjpe - bm_mpjpe

    ssidx = sorted(range(diff.size), key=lambda k:diff[k])
    show_idx_list = [ssidx[0], ssidx[-1]]
    print diff[show_idx_list]
    print 'bm_mpjpe is :'
    print bm_mpjpe[show_idx_list]
    print 'mv_mpjpe is :'
    print mv_mpjpe[show_idx_list]
    save_base_folder = op.get_value('save_res_path')
    subfolder_name = iu.getpath(save_base_folder)
    save_base_name = iu.fullfile(save_base_folder, '{}_mpjpg_vs_score_{}.png'.format(subfolder_name))
    # save_base_name = '/public/sijinli2/ibuffer/2015-03-13/2015_03_10_0037_corrected_version/2015_03_10_0037_corrected_version_mpjpg_vs_score_{}.png'
    # save_base_name = '/public/sijinli2/ibuffer/2015-03-13/tmp_mpjpe_vs_score'
    # show_idx_list = [0]

    fig = pl.figure()
    ax = fig.add_subplot(111)
    ax.plot(range(bm_mpjpe.size), bm_mpjpe[ssidx], c='r', label = 'bm_mpjpe')
    ax.plot(range(bm_mpjpe.size), mv_mpjpe[ssidx], c='g', label = 'mv_mpjpe')
    pl.legend()
    imgproc.imsave_tight(save_base_name.format('bm_mpjpe_vs_mv_mpjpe').replace('mpjpe_vs_score_',''))
    pl.show()
    
    for show_idx in show_idx_list:
        cur_img_feature = img_features[..., show_idx].reshape((-1, 1), order='F')
        n_per = batch_candidate_indexes.size // mv_target.shape[-1]
        cur_candidate_indexes = batch_candidate_indexes[show_idx*n_per:(show_idx+1)*n_per]

        cur_candidate_target = fl[0][..., cur_candidate_indexes]
        cur_gt_target = gt_target[..., show_idx].reshape((-1,1),order='F')
        print 'Cur_candidate_indexes.size = {}'.format(cur_candidate_indexes.size)
        def get_score(X):
            """
            note this is different from the get-score in show_hightest_score
            """
            cur_n = X.shape[-1]
            imgfeatures = np.tile(cur_img_feature, [1, cur_n])
            eval_input_data = [solver.gpu_require(imgfeatures.T),
                               solver.gpu_require(X.T),
                               solver.gpu_require(np.zeros((cur_n,1)))]
            return func(*eval_input_data)[0]
        candidate_score= get_score(cur_candidate_target).flatten()
        candidate_margin = solver.margin_func(cur_candidate_target - cur_gt_target).flatten()
        candidate_mpjpe = MMLSSolver.calc_margin(cur_candidate_target - cur_gt_target).flatten() * 1200
        sidx = np.array(sorted(range(candidate_score.size), key=lambda k:candidate_score[k],reverse=True))
        gt_score = get_score(cur_gt_target).flatten()[0]
        print 'gt_score = {}, candidate_score shape {} candidate_max_score = {}'.format(gt_score, candidate_score.shape, np.max(candidate_score))
        imgproc.set_figure_size(15,10)
        fig = pl.figure()
        ax = fig.add_subplot(2,1,1)
        cur_n = candidate_score.size
        ax.plot(range(cur_n), candidate_score[sidx], c= 'b', label='candidate_score')
        ax.plot(range(cur_n), [gt_score] * cur_n, c= 'r', label='gt score')
        ax.plot(range(cur_n), candidate_score[sidx] + candidate_margin[sidx], c = 'g', label='candidate_aug_score')
        pl.legend()
        print bmi_each[show_idx], 'bm index current'
        pl.title('mpjpe: mv({:.3f})(mv score {:.5f}), best match({:.3f})(gt score {:.5f} bm_score {:.5f})'.format(mv_mpjpe[show_idx]*1200, candidate_score.max(),  bm_mpjpe[show_idx]*1200, gt_score, candidate_score[bmi_each[show_idx]]))
        ax = fig.add_subplot(2,1,2)
        ax.plot(range(cur_n), candidate_mpjpe[sidx])
        imgproc.imsave_tight(save_base_name.format(show_idx))
        pl.title('candidate mpjpe')
        pl.show()

def get_all_candidate(solver, dp, ndata):
    K_candidate, candidate_indexes = solver.create_candidate_indexes(ndata, dp, train=True)
    mvc = solver.stat['most_violated_counts']
    K_mv = solver.solver_params['K_most_violated']
    n_train = len(solver.train_dp.data_range)
    if solver._solver_type == 'imgdpmm':
        if solver.solver_params['candidate_mode'] != 'all':
            sorted_indexes = sorted(range(n_train), key=lambda k:mvc[k],reverse=True)
            holdon_indexes = np.array(sorted_indexes[:K_mv]).flatten()
            candidate_indexes = np.tile(np.concatenate([candidate_indexes.flatten(), holdon_indexes]).reshape((K_mv + K_candidate, 1),order='F'),[1, ndata]).flatten(order='F')
            K_tot = K_candidate + K_mv
        else:
            candidate_indexes = np.tile(candidate_indexes.reshape((K_candidate,1),order='F'),
                                        [1, ndata]).flatten(order='F')
            K_tot = K_candidate
        return K_tot, candidate_indexes
    else:

        sorted_indexes = sorted(range(n_train), key=lambda k:mvc[k],reverse=True)
        holdon_indexes = np.array(sorted_indexes[:K_mv]).reshape((K_mv,1))           
        return K_candidate + K_mv, np.concatenate([candidate_indexes.reshape((K_candidate,ndata),order='F'), np.tile(holdon_indexes, [1, ndata])], axis=0).flatten(order='F')
def parse_fdata(fdata, solver):
    if solver._solver_type == 'mmls':
        return fdata[1], fdata[0]
    else:
        return fdata[0], fdata[1]
def calc_candidate_target_feature_list(solver, train, op, targets, N_candidate):
    solver.set_train_mode(train)
    assert(train== False)
    N = targets.shape[-1]
    ndata = N / N_candidate
    max_num = solver.solver_params['max_num']
    def inner_calc(cur_targets):
        s = cur_targets.shape[-1]
        mini_batches = (s - 1) // max_num + 1
        res_list = [solver.calc_target_feature_func(solver.gpu_require(cur_targets[..., k * max_num:min((k+1)*max_num, s)].T))[0].T for k in range(mini_batches)]
        return np.concatenate(res_list, axis=1)
    if solver.solver_params['candidate_mode'] == 'all':
        feats = inner_calc(targets[..., :N_candidate])
        return [feats for k in range(ndata)]
    else:
        return [inner_calc(targets[..., k * N_candidate:(k+1)*N_candidate]) for k in range(ndata)]
    
def show_topK_pose_eval(solver, train, op):
    dp = solver.train_dp if train else solver.test_dp
    assert(train == False)
    func = get_eval_func(solver)
    num_batch = dp.num_batch
    all_avg_mpjpe, all_top_mpjpe = [], []
    # save_path = '/public/sijinli2/ibuffer/2015-03-13/2015_03_17_0042_test/topK20'
    save_folder = op.get_value('save_res_path')
    topK = 20
    if len(save_folder) == 0:
        raise Exception('The savepath are not supplied')
    print 'Save path = {}'.format(save_folder)
    iu.ensure_dir(save_folder)
    save_path = iu.fullfile(save_folder, 'topK_{}'.format(topK))
    dp.reset()
    itm = iu.itimer()
    if iu.exists(save_path, 'file'):
        d = mio.unpickle(save_path)
        if d['avg_mpjpe'] is not None:
            all_avg_mpjpe = d['avg_mpjpe'].flatten().tolist()
            all_top_mpjpe = d['top_mpjpe'].flatten().tolist()
    else:
        d = {'avg_mpjpe':None, 'top_mpjpe':None, 'finished':0}
        mio.pickle(save_path, d)
    for bn in range(num_batch):
        if d['finished'] > bn:
            continue
        print 'Begin to load data batch {}'.format(bn)
        t = time()
        data = solver.get_next_batch(train)
        indexes = dp.get_batch_indexes()
        print 'Load data completed ({})'.format(time() - t)
        fdata = get_fdata(solver, data)
        img_features, gt_target = parse_fdata(fdata, solver)
        ndata = np.array(indexes).size
        N_candidate, batch_candidate_indexes_all = get_all_candidate(solver, dp, ndata)
        
        fl = get_feature_list(solver, solver.train_dp)
        batch_candidate_targets = fl[0][...,batch_candidate_indexes_all]
        batch_candidate_features = fl[2][..., batch_candidate_indexes_all]

        top_list, avg_list = [], []
        print 'batch target {} \t features {} \t img_features {}'.format(batch_candidate_targets.shape, batch_candidate_features.shape, img_features.shape)
        
        if solver._solver_type == 'imgdpmm':
            itm.tic('calculate candidate features')
            ctf_list = calc_candidate_target_feature_list(solver, train, op, batch_candidate_targets, N_candidate)
            itm.toc()
            # ^ dim x N_candidate x ndata ?
            assert(len(ctf_list) == ndata)
            # img_features  dim x ndata
            compute_time_py = time()
            score_list = [solver.dot_func(ctf_list[k].T, img_features[..., [k]])[0] for k in range(ndata)]
            score_mat = np.concatenate(score_list, axis=1)
            print 'score_mat.shape{}'.format(score_mat.shape)
            print 'Calculate score cost {} seconds'.format(time() - compute_time_py)
            sidx = np.argpartition(-score_mat, topK, axis=0)[:topK,:]
            sidx_arr = sidx + np.array(range(0, ndata)).reshape((1,ndata)) * N_candidate
            print 'sidx_arr.shape = {}'.format(sidx_arr.shape)
            top_target = batch_candidate_targets[..., sidx_arr[0,...]]
            avg_target_list = [  batch_candidate_targets[..., sidx_arr[...,k]].mean(axis=1,keepdims=True) for k in range(ndata)]
            avg_target = np.concatenate(avg_target_list, axis=1)
            print 'top_target, avg_target.shape = {}'.format(top_target.shape, avg_target.shape)
            top_mpjpe = dutils.calc_mpjpe_from_residual(top_target - gt_target, 17) * 1200
            avg_mpjpe = dutils.calc_mpjpe_from_residual(avg_target - gt_target, 17) * 1200
            print 'avg mpjpe = {}, top mpjpe = {}'.format(avg_mpjpe.mean(), top_mpjpe.mean())
            top_list =  top_mpjpe.flatten().tolist()
            avg_list =  avg_mpjpe.flatten().tolist()
        else:
            for b in range(ndata):
                # print 'Process idx = {}\t (raw index = {})'.format(b, indexes[b])
                cur_image_feature = img_features[..., b].reshape((-1,1),order='F')
                imgfeatures = np.tile(cur_image_feature, [1, N_candidate])
                s,e = N_candidate * b, N_candidate * (b + 1)
                candidate_features= batch_candidate_features[..., s:e]
                candidate_targets = batch_candidate_targets[...,s:e]
                eval_input_data = [solver.gpu_require(imgfeatures.T),
                                   solver.gpu_require(candidate_features.T),
                                   solver.gpu_require(np.zeros((N_candidate,1)))]

                candidate_score= func(*eval_input_data)[0].flatten()
                sidx = sorted(range(N_candidate), key= lambda k:candidate_score[k], reverse=True)
                top_indexes = sidx[:topK]
                avg_target = candidate_targets[..., top_indexes].mean(axis=1,keepdims=True)
                top_target = candidate_targets[..., sidx[0]].reshape((-1,1),order='F')
                cur_gt = gt_target[..., b].reshape((-1,1),order='F')
                top_mpjpe = dutils.calc_mpjpe_from_residual(cur_gt - top_target, 17) * 1200
                avg_mpjpe = dutils.calc_mpjpe_from_residual(cur_gt - avg_target, 17) * 1200
                print 'avg mpjpe = {}, top mpjpe = {}'.format(avg_mpjpe, top_mpjpe)
                top_list.append(top_mpjpe[0])
                avg_list.append(avg_mpjpe[0])
        print '    batch {} The avg mpjpe is {}'.format(bn, sum(avg_list)/ndata)
        print '    batch {} The top mpjpe is {}'.format(bn, sum(top_list)/ndata)
        all_avg_mpjpe += avg_list
        all_top_mpjpe += top_list
        d = {'avg_mpjpe':np.array(all_avg_mpjpe), 'top_mpjpe':np.array(all_top_mpjpe),
        'finished': bn + 1}
        mio.pickle(save_path, d)
    
def test_tmp():
    p1 = '/public/sijinli2/ibuffer/2015-03-08/t1'
    p2 = '/public/sijinli2/ibuffer/2015-03-08/t2'
    d1,d2 = mio.unpickle(p1), mio.unpickle(p2)
    print d1.keys(),d2.keys()
    diff = d1['gt'] - d2['gt']
    print d1['gt'].shape, d2['gt'].shape
    print np.abs(diff).sum(), '<------------ abs sum'
    diff2 = d1['gt_feature'] - d2['gt_feature']
    print np.abs(diff2).sum(), '<------------diff feature'
    print '-=- gt-features----------------------'
    print d1['gt_feature'][[0,1,2,3,4,5],0]
    
    print d2['gt_feature'][[0,1,2,3,4,5],0]
    print 'img_features-----'
    print d1['img_features'][[0,1,2,3,4,5],0]
    print d2['img_features'][[0,1,2,3,4,5],0]
    print 'gt_score'
    print d1['gt_score'][[0,1,2,3,4,5],0]
    print d2['gt_score'][[0,1,2,3,4,5],0]
class MMEvalPoseLoader(MMSolverLoader):
    _inner_loader_dic = {'mmls': MMLSSolverLoader, 'imgmm':ImageMMSolverLoader, 'imgdpmm':ImageDotProdMMSolverLoader}
    @classmethod
    def add_extra_op(cls, op):
        op.add_option('mode', 'mode', options.StringOptionParser, 'the most for testing',default='shs')
        op.add_option('mode-params', 'mode_params', options.StringOptionParser, 'the most for testing', default='')
        op.add_option('save-res-path', 'save_res_path', options.StringOptionParser, 'The path for saving results', default='')
        op.add_option('no-display', 'no_display', options.BooleanOptionParser, 'Will not display any figure', default=False)
    def add_default_options(self, op):
        MMSolverLoader.add_default_options(self, op)
        self.add_extra_op(op)
    def parse(self):
        self.op.parse()
        self.op.eval_expr_defaults()
        solver_type = self.op.get_value('solver_type')
        if solver_type is None:
            load_file_path = self.op.get_value('load_file')
            saved_model = Solver.get_saved_model(load_file_path)
            solver_type = saved_model['solver_params']['solver_type']
        print '''
        solver_type is {}
        '''.format(solver_type)
        inner_loader = self._inner_loader_dic[solver_type]()
        self.add_extra_op(inner_loader.op)
        return inner_loader.parse()
def main():
    # test_tmp()
    # analyze_eval_saved()
    # return
    # 
    loader = MMEvalPoseLoader()
    solver = loader.parse()
    # solver.solver_params['candidate_mode'] = 'all'
    # solver.solver_params['candidate_mode'] = 'random2'
    solver.solver_params['K_candidate'] = 20000
    solver.solver_params['max_num'] = 500
    solver.solver_params['K_top_update'] = 1
    mode = loader.op.get_value('mode')
    if mode == 'shs':
        show_highest_score(train=False, solver=solver, op = loader.op)
    elif mode == 'stpe':
        show_topK_pose_eval(solver, train=False, op=loader.op)
    # show_highest_score(train=False, solver=solver, op = loader.op)
    # show_highest_score(train=True, solver=solver)
    # eval_mpjpe_all(train=False, solver=solver)
    # show_mpjpe_vs_score(train=False, solver=solver)
    # show_topK_pose_eval(solver, train=False, op = loader.op) 
if __name__ == '__main__':
    main()