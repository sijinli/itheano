from init_test import *
from isolver import *
import dhmlpe_utils as dutils
def test_bpsolver():
    solver_loader = SolverLoader()
    solver = solver_loader.parse()
    net = solver.net
    save_folder = '/opt/visal/tmp/for_sijin/tmp/tmp_saved'
    output_layer_names = ['joints','fc_j0']
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
def main():
    test_bpsolver()
    # test_collect()
if __name__ == '__main__':
    main()