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
def clone_weights(layers, saved_layers):
    l_keys = sorted(layers.keys())
    l_keys1 = sorted(saved_layers.keys())
    if len(l_keys) != len(l_keys1) or l_keys != l_keys1:
        print 'Inconsistent number of layers {} vs {}'.format(len(l_keys),len(l_keys1))
        # raise GraphParserError('Inconsistent number of layers or different components')
    for l in l_keys:
        layers[l][2].copy_from_saved_layer(saved_layers[l][2])
def pre_process_data(d):
    d['feature_list'][0] = d['feature_list'][0] / 1200
    
def print_dims(alldata):     # debug use
    for i,e in enumerate(alldata):
        print 'Dim {}: \t shape {} \t type {}'.format(i, e.shape, type(e))
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
    params = {'batch_size':1024, 'data_path':meta_path}
    train_params = merge_dic(params, train_ext_params)
    test_params = merge_dic(params, test_ext_params)
    pre_process_data(d)
    train_dp = MemoryDataProvider(data_dic=d, train=True, data_range=train_range, params=train_params)
    test_dp = MemoryDataProvider(d, train=False, data_range=test_range, params=test_params)
    print 'Create Data Provider Successfully'
    return train_dp, test_dp
def save_again():
    """
    change net_list objects to saved dict
    """
    d = Solver.get_saved_model('/public/sijinli2/ibuffer/2015-01-13/itheano_test')
    save_path = '/public/sijinli2/ibuffer/2015-01-13/saved_haha'
    for net_name in d['net_list']:
        net = d['net_list'][net_name]
        res_dic = dict()
        for e in net['layers']:
            l = net['layers'][e]
            res_dic[e] = [l[0], l[1], l[2].save_to_dic()]
        d['net_list'][net_name]['layers'] = res_dic
    print d['net_list']['eval_net'].keys()
    mio.pickle(save_path, d)
def test_shared_weights():
    d = Solver.get_saved_model('/opt/visal/tmp/for_sijin/tmp/itheano_test')
    net_dic = d['net_dic']
    layers = net_dic['eval_net']['layers']
    print layers['net1_fc1'][2].keys()
    W_list_1 = layers['net1_fc2'][2]['weights']
    W_list_2 = layers['net2_fc2'][2]['weights']
    assert(len(W_list_1) == len(W_list_2))
    for w1,w2 in zip(W_list_1, W_list_2):
        diff = w1 - w2
        e = np.abs(diff.flatten()).sum()
        print 'The total difference is {}'.format(e)
def test_shared_weights_online(layers):
    W_list1 = layers['net1_fc1'][2].W_list
    W_list2 = layers['net2_fc1'][2].W_list
    for w1,w2 in zip(W_list1, W_list2):
        wv1 = w1.get_value()
        wv2 = w2.get_value()
        diff = np.abs(wv1 - wv2).sum()
        print 'Diff is {}'.format(diff)
    
def resume_solver():
    d = Solver.get_saved_model('/opt/visal/tmp/for_sijin/Data/saved/theano_models/net4_rbf_correct')
    graph_cfg_path = '/home/grads/sijinli2/Projects/Itheano/doc/netdef/graph_def_0004.cfg'
    
    net_dic, ms = d['net_dic'], d['model_state']
    saved_layers = net_dic['eval_net']['layers']
    g = GraphParser(graph_cfg_path)
    layers = g.layers
    clone_weights(layers, saved_layers)
    # Now we begin to wake up the network
    eval_net_config = net_dic['eval_net']['config_dic']
    train_net_config = net_dic['train_net']['config_dic']
    
    eval_net = Network(layers, eval_net_config)
    train_net = Network(layers, train_net_config)
    solver_params = d['solver_params']
    ############### Some hack about the solver_params
    if 'K_most_violated' not in d['solver_params']:
        solver_params['K_most_violated'] = 100
    solver_params['K_candidate'] = 10000
    # solver_params['candidate_mode'] = 'all'
    # solver_params['K_candidate'] = 200
    solver_params['K_most_violated'] = 0
    print 'KC = {}, KMV={}'.format(solver_params['K_candidate'], solver_params['K_most_violated'])
    solver_params['max_num'] = 5

    ################
    epoch, batchnum = ms['train']['epoch'],ms['train']['batchnum']
    test_epoch, test_batchnum = ms['test']['epoch'],ms['test']['batchnum']
    
    train_dp, test_dp = create_dp2({'epoch':epoch,'batchnum':batchnum},
                                  {'epoch':test_epoch, 'batchnum':test_batchnum})

    
    solver=  MMLSSolver([eval_net, train_net], train_dp, test_dp, solver_params)
    print 'Solver Loading complete'
    # print 'Solver_params {}'.format(solver.solver_params)
    return solver
def show_highest_score(train):
    """
    This function will load data from train or test set
    """
    solver = resume_solver()

    # stat = solver.stat
    # print stat.keys()
    # mvc = stat['most_violated_counts']
    # scc = stat['sample_candidate_counts']
    # print 'mvc sum = {}, scc = {}'.format(mvc.sum(), scc.sum())
    
    # test_shared_weights_online(solver.train_net.layers)
    # print '<<<<<<<<<<<<<<<<{}'.format(solver.train_net.layers is solver.eval_net.layers)
    # print 'train net inputs {}'.format(solver.train_net.inputs)
    # print 'eval net inputs {}'.format(solver.eval_net.inputs)
    # print 'eval net outputs {}'.format(solver.eval_net.outputs)
    
    # GraphParser.print_graph_connections(solver.train_net.layers)
    # return
    dp = solver.train_dp if train else solver.test_dp
    
    data = dp.get_next_batch(train)
    ep, bn, alldata, ext_data  = solver.find_most_violated_ext(data,use_zero_margin=True,
                                                               train=train
    )
    print ep, bn, len(alldata)
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
    fl = solver.train_dp.data_dic['feature_list']
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
    print 'Calc Again mpjpe is {}'.format(np.mean(mpjpe.flatten())*1200)
    data_to_eval = [solver.gpu_require(img_features.T),
                    solver.gpu_require(bm_features.T),
                    solver.gpu_require(gt_margin.T)
    ]
    bm_score = solver.eval_net.outputs[0].eval(dict(zip(solver.eval_net.inputs, data_to_eval)))
    # for evaluation
    # inputs = [solver.train_net.inputs[0],solver.train_net.inputs[2],solver.train_net.inputs[4]]
    # print 'inputs = {}'.format(inputs)
    # ff = theano.function(inputs=inputs,
    #                      outputs=solver.eval_net.layers['net2_score'][2].outputs
    # )
    # print solver.eval_net.layers['net1_score'][2].outputs
    # res = solver.call_func(ff, data_to_eval)
    # r = res[0]
    # diff = r - gt_score
    # print '=======The abs difference is {}==========='.format(np.abs(diff).sum())
    
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
    save_path = '/public/sijinli2/ibuffer/2015-01-22/saved_batch_data_test_net4_K_rbf_correct_test'
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
    save_path = '/public/sijinli2/ibuffer/2015-01-22/saved_batch_data_test_net4_K_rbf_correct_test'

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
    print 'avg_mpjpe is {}'.format(avg_mpjpe * 1200)
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
    # show_highest_score(train=False)
    show_bm_cmp_from_saved()
    # test_shared_weights()
    # return
    # save_again()
    # return

if __name__ == '__main__':
    main()