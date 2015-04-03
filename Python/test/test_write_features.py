from init_test import *
from isolver import *
import dhmlpe_utils as dutils
def test_bpsolver():
    solver_loader = SolverLoader()
    solver = solver_loader.parse()
    net = solver.net
    save_folder = '/opt/visal/tmp/for_sijin/tmp/tmp_saved_tt'
    output_layer_names = ['joints','fc_j0']
    output_layer_names = ['joints','fc_j0', 'fc_j1', 'fc_j2']
    solver.write_features(solver.test_dp, net, save_folder, output_layer_names)
def test_collect():
    folder = '/opt/visal/tmp/for_sijin/tmp/tmp_saved'
    allfile = sorted(iu.getfilelist(folder, 'feature_batch'), key=lambda x:dutils.extract_batch_num(x))
    ntot = 0
    for fn in allfile:
        d = mio.unpickle(iu.fullfile(folder, fn))
        ndata = d['feature_list'][0].shape[-1]
        ntot += ndata
    print 'The total number of data is {}'.format(ntot)
    meta = dutils.collect_feature_meta(folder)
    print meta.keys()
    print meta['info']['indexes'][:10]
    print meta['feature_dim']
def testtt():
    folder = '/opt/visal/tmp/for_sijin/tmp/tmp_saved_tt'
    meta = dutils.collect_feature_meta(folder)
    meta_folder = '/opt/visal/tmp/for_sijin/Data/H36M/H36MFeatures/2015_02_02_acm_act_14_exp_2_19_graph_0012'
    iu.ensure_dir(meta_folder)
    meta_path = iu.fullfile(meta_folder, 'prediction.meta')
    mio.pickle(meta_path, meta)
    # d = mio.unpickle(meta_path)
    # f1 = meta['feature_list'][1]
    # print f1.shape
    # f11 = d['feature_list'][1]
    # print f11.shape
    # ndata = f1.shape[-1]
    # f11_c = f11[..., 132744:ndata + 132744]
    # diff = f1 - f11_c
    # print np.abs(diff).sum()
    # print f1[[0,1,2],0]
    # print f11_c[[0,1,2],0]
    # print d['feature_list'][0][:,0]
def main():
    # test_bpsolver()
    # test_collect()
    testtt()
if __name__ == '__main__':
    main()