"""
This file just contains all the packing experiments codes
"""
from init_test  import *
import numpy as np
import dhmlpe_utils as dutils
import dhmlpe_features as df
import iutils as iu
def pack_01():
    """
    input:   fc_j0 feature,  rel_pose
    outputs: rel_pose, fc_j0_feature
    """
    source_meta_path = '/opt/visal/tmp/for_sijin/tmp/tmp_saved'
    
    exp_meta_path = '/opt/visal/tmp/for_sijin/Data/H36M/H36MExp/folder_ASM_act_12_exp_4/batches.meta'
    save_path = '/opt/visal/tmp/for_sijin/Data/H36M/H36MExp/folder_FCJ0_act_12'
    feature_name = 'Relative_Y3d_mono_body'
    res = dict()
    exp_meta = mio.unpickle(exp_meta_path)
    source_meta = dutils.collect_feature_meta(source_meta_path)
    rel_pose = exp_meta[feature_name]
    fc_j0_feature = source_meta['feature_list'][1]
    rel_gt = source_meta['feature_list'][0]

    diff = rel_gt.reshape((-1, rel_gt.shape[-1]),order='F') * 1200 - rel_pose
    print 'diff is {}'.format(diff.flatten().sum())
    feature_list =  [rel_pose, fc_j0_feature]
    feature_dim = [rel_pose.shape[0], fc_j0_feature.shape[0]]
    print feature_dim, '<<<feature dim'
    res = {'feature_list': feature_list, 'feature_dim':feature_dim,
           'info':{'indexes':source_meta['info']['indexes'],
                   'max_depth': 1200.0}}
    indexes = res['info']['indexes'] 
    print indexes[:10], min(indexes), max(indexes)
    print 'The number of data is {} == {}'.format(indexes.size, feature_list[0].shape[-1])
    iu.ensure_dir(save_path)
    mio.pickle(iu.fullfile(save_path, 'batches.meta'), res)
def test():
    d = mio.unpickle('/opt/visal/tmp/for_sijin/Data/H36M/H36MExp/folder_FCJ0_act_12/batches.meta')
    a = d['feature_list'][1]
    print a[...,0].flatten()
    iu.print_common_statistics(a)
def main():
    # pack_01()
    #
    test()

if __name__ == '__main__':
    main()


