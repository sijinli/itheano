
from init_test import *
import iread.myio as mio
import iutils as iu
def cvt1(source_exp_name, target_exp_name):
    print '''
    SP_t004_act_14:
    source meta [rel_gt,  img_feature_accv_fc_j0,  relskel_feature_t004]
    Raw_SP_t004_act_14:
    target meta [rel_gt,  img_feature_accv_fc_j0,  rel_gt]
    '''
    base_path = '/opt/visal/tmp/for_sijin/Data/H36M/H36MExp/'
    source_meta = mio.unpickle(iu.fullfile(base_path, 'folder_%s' % source_exp_name,
                                           'batches.meta'))
    target_meta_folder = iu.fullfile(base_path, 'folder_%s' % target_exp_name) 
    target_meta_path =  iu.fullfile(target_meta_folder, 'batches.meta') 
    d = source_meta.copy()
    print d.keys()
    d['feature_list'] = [source_meta['feature_list'][k] for k in [0, 1, 0]]
    d['feature_dim'] = [source_meta['feature_dim'][k] for k in [0, 1, 0]]
    # print d['info']
    print 'folder :{}\n path {}'.format(target_meta_folder, target_meta_path)
    iu.ensure_dir(target_meta_folder)
    mio.pickle(target_meta_path, d)

def main():
    source_exp_name = 'SP_t004_act_14'
    target_exp_name = 'Raw_SP_t004_act_14'
    cvt1(source_exp_name, target_exp_name)


if __name__ == '__main__':
    main()

    