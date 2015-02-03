from init_test import *
import iutils as iu
from idata import *
import pylab as pl
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
def main():
    test_CroopedDHMPLEJointDataWarper()

if __name__ =='__main__':
    main()