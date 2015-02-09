from init_test import *
import iutils as iu
from idata import *
import pylab as pl
import dhmlpe_utils as dutils
import dhmlpe_features as df
def test_CroopedDHMPLEJointDataWarper():
    data_dir = '/opt/visal/tmp/for_sijin/Data/H36M/H36MExp/folder_ASM_act_14_exp_2'
    data_path  = iu.fullfile(data_dir, 'batches.meta')
    params = {'batch_size':1024, 'data_path':data_dir}
    data_dic = None
    train=False
    data_range = range(0, 132744)
    test_range = range(132744, 162008)
    dp = CroppedDHMLPEJointDataWarper(data_dic, train, data_range, params)
    epoch, batchnum, alldata = dp.get_next_batch()
    print ' {}.{} [] of {}'.format(epoch, batchnum, len(alldata))

    show_idx = 100
    img = np.require(dp.get_plottable_data(alldata[0][..., show_idx].reshape((-1,1),order='F')), dtype=np.uint8)
    sp = img.shape
    img = img.reshape((sp[0],sp[1],sp[2]),order='F')
    print np.max(img.flatten())
    print np.min(img.flatten())
    pl.subplot(2,1,1)
    pl.imshow(img)
    pl.subplot(2,1,2)
    img = dp.cropped_mean_image
    pl.imshow(img/255.0)
    pl.show()
def test_Embedding_ASM_act_14_exp_2_ACCV_fc_j0():
    """
    just check the pose displaying
    """
    from mpl_toolkits.mplot3d import Axes3D
    import imgproc
    import iread.h36m_hmlpe as h36m
    data_path = '/opt/visal/tmp/for_sijin/Data/H36M/H36MExp/folder_Embedding_ASM_act_14_exp_2_ACCV_fc_j0/batches.meta'
    data_path1 = '/opt/visal/tmp/for_sijin/Data/H36M/H36MExp/folder_SP_t004_act_14/batches.meta'
    meta = mio.unpickle(data_path)
    meta1= mio.unpickle(data_path1)
    print 'Len of feature list is {} \t dims = {}'.format(len(meta['feature_list']), meta['feature_dim'])
    for i, e in enumerate(meta['feature_list']):
        print 'idx {}:\t shapes = {}'.format(i, e.shape)
    f0 = meta['feature_list'][0]
    f0_1 = meta1['feature_list'][0]
    ndata = f0.shape[-1]
    limbs, params = h36m.part_idx, {'elev':-89, 'azim':-107, 'linewidth':3}
    fig = pl.figure()
    idx= 0
    fig.add_subplot(2,1,1, projection='3d')
    p = f0[..., idx].reshape((-1,1),order='F') * 1200
    p1 = f0_1[..., idx].reshape((-1,1),order='F')
    print p
    pp = df.convert_relskel2rel(p).reshape((-1,1),order='F')
    diff = pp.reshape((-1,1),order='F') - p1
    
    imgproc.turn_off_axis()
    dutils.show_3d_skeleton(p1, limbs, params)
    pl.show()
    print '''
    Embedding_ASM_act_14_exp_2_ACCV_fc_j0/batches.meta
    len = 2. idx-0 = rel-skel ground-truth  /1200
             idx-1 = image embedding
    '''
def main():
    # test_CroopedDHMPLEJointDataWarper()
    test_Embedding_ASM_act_14_exp_2_ACCV_fc_j0()
if __name__ =='__main__':
    main()