from init_test import *
from idata import *
from ilayer import *
from igraphparser import *
from isolver import *
import pylab as pl
import dhmlpe_utils as dutils
import iread.myio as mio
import iread.h36m_hmlpe as h36m
def show_the_most_violated_poses():
    from mpl_toolkits.mplot3d import Axes3D
    import imgproc
    saved_model_path = '/public/sijinli2/ibuffer/2015-01-16/net2_test_for_stat'
    saved_model_path = '/opt/visal/tmp/for_sijin/Data/saved/Test/FCJ0_act_14_graph_0028_test_KMV_1000'
    data_path = '/opt/visal/tmp/for_sijin/Data/H36M/H36MExp/folder_SP_t004_act_14/batches.meta'
    meta = mio.unpickle(data_path)
    all_pose = meta['feature_list'][0]
    ssolver = Solver.get_saved_model(saved_model_path)
    stat = ssolver['model_state']['stat']
    cnt_sample = stat['sample_candidate_counts']
    mvc = stat['most_violated_counts']
    ntrain = cnt_sample.size

    sorted_indexes = sorted(range(ntrain), key=lambda k: mvc[k], reverse=True)
    show_num = int(100)
    selected_indexes = sorted_indexes[:show_num]
    max_show_row = int(8)
    n_row = (show_num - 1)// max_show_row + 1
    nc = 0
    selected_pose = all_pose[:,selected_indexes]
    limbs = h36m.part_idx
    fig = pl.figure()
    params = {'elev':-89, 'azim':-107, 'linewidth':3}
    selected_cnt = mvc[selected_indexes]
    print n_row, max_show_row
    for r in range(n_row):
        for c in range(max_show_row):
            if nc == show_num:
                break
            p = selected_pose[...,nc].reshape((3,17),order='F').T
            nc = nc + 1
            # pl.subplot(n_row, max_show_row, c)
            ax = fig.add_subplot(n_row, max_show_row, nc, projection='3d')
            imgproc.turn_off_axis()
            dutils.show_3d_skeleton(p, limbs, params)
            pl.title('mvc={}'.format(selected_cnt[nc-1]))
    pl.show()
def show_stat_most_violated_indexes():
    saved_model_path = '/public/sijinli2/ibuffer/2015-01-16/net2_test_for_stat'
    saved_model_path = '/opt/visal/tmp/for_sijin/Data/saved/Test/FCJ0_act_14_graph_0028_test_KMV_1000'
    ssolver = Solver.get_saved_model(saved_model_path)
    stat = ssolver['model_state']['stat']
    cnt_sample = stat['sample_candidate_counts']
    mvc = stat['most_violated_counts']
    ntrain = cnt_sample.size
    pl.subplot(3,1,1)
    # ax = pl.gca()
    pl.plot(range(ntrain), cnt_sample.flatten(), label='random sample counts')
    pl.legend()
    # ax.bar(range(ntrain),cnt_sample)
    pl.subplot(3,1,2)
    pl.plot(range(ntrain), mvc.flatten(), label='most violated sample counts')
    pl.legend()
    pl.subplot(3,1,3)
    N = 200
    pl.hist(mvc.flatten(),range(0, N + 1), label='histogram for most violated sample counts')
    pl.legend()
    l, dummy = np.histogram(mvc.flatten(), range(0,N + 1))
    total = mvc.sum()
    print 'total = {}'.format(total)
    for i in range(N):
        print '{:5f}% in range [ {},{} )'.format(l[i]*100.0/total, i, i + 1)        
    pl.show()
def verify_layer_outputs():
    solver_loader = MMSolverLoader()
    solver = solver_loader.parse()
    net = solver.train_net
    output_layers = net.get_layer_by_names(['net1_fc0'])
    outputs = [lay.outputs for lay in output_layers]
    f = theano.function(inputs=net.inputs,
                        outputs=outputs, on_unused_input='ignore')
    cur_data = solver.get_next_batch(train=False)
    mvd = solver.find_most_violated(cur_data, train=False)
    alldata = [self.gpu_require(e.T) for e in most_violated_data[2][1:]]
    res = f(alldata)
    ref_meta = mio.unpickle('/opt/visal/tmp/for_sijin/Data/H36M/H36MExp/folder_SP_t004_act_14/batches.meta')
def test_tmp():
    meta = mio.unpickle('/opt/visal/tmp/for_sijin/Data/H36M/H36MExp/folder_Raw_SP_t004_act_14/batches.meta')
    f0, f2 = meta['feature_list'][0], meta['feature_list'][2]
    print f0[..., 0]
    diff = f0 - f2
    print 'diff = {}'.format(diff.flatten().sum())
    print '''
    Okay, f0, f2 is gt 
    '''
    
    
  
    
def main():
    # test_tmp()
    show_stat_most_violated_indexes()
    # show_the_most_violated_poses()


if __name__ == '__main__':
    main()
    