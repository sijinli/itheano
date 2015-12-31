import numpy as np
import scipy.io as sio
import iread.myio as mio
import iutils as iu
import scipy
import sys
import os
def is_on_cluster():
    s = os.path.getcwd()
    if s.find('grads') != -1 or s.find('opt') != -1:
        return True
    else:
        return False
class DHMLPEError(Exception):
    pass
META_GENERAL_FIELD=['image_adjust_dim', 'image_sample_dim']
DHMLPE_DEFAULT_META= {'image_sample_dim':[112,112,3], \
                      'joint_indicator_map':{'filter_size':30, \
                                             'stride':10}}

class DHMLPE():
    def __init__(self):
        self.meta = DHMLPE_DEFAULT_META
    def add_meta_item(self, name, value):
        self.meta[name] = value        
    def get_convnet_meta(self, matmeta_path):
        d = self.load_matmeta(matmeta_path)
        # add meta info
        self.copy_dic(d, self.meta, override=False)
        # add new statistics
        print d.keys()
        self.fill_statistics(d)
        return d
    @classmethod
    def fill_statistics(cls, d):
        d['rgb_eigenvector'], d['rgb_eigenvalue'] = cls.rgb_eig(d['rgb_meancov'])
        d['cropped_mean_image'] = cls.calc_cropped_mean(d['mean_image'], d['image_sample_dim'])
        d['num_joints'] = d['occ_body'].shape[0]
    @classmethod
    def calc_cropped_mean(cls, ori_meanimg, new_dim):
        """
        This version is slow, it takes 0.5 seconds for one image
        However, it seems to be okay, since it will be only called once
        """
        [nr, nc , nchannel] = ori_meanimg.shape
        if nchannel != 3:
            raise BasicDataProviderError('Only RGB IMage are supported')
        [new_nr, new_nc, dummy] = new_dim
        fr,fc = nr - new_nr + 1, nc - new_nc + 1
        ind = np.tile(np.asarray(range(0,fr)).reshape((fr,1)),[1,fc]) + \
           np.asarray(range(0,fc)).reshape((1,fc)) * nr
        ind = ind.flatten(order='F')
        vimg = ori_meanimg.reshape((nr*nc*nchannel),order='F')
        new_per_channel = new_nr * new_nc
        per_channel = nr * nc
        element = (np.tile(np.asarray(range(0,new_nr)).reshape((new_nr,1)),[1, new_nc]) + np.asarray(range(0,new_nc)).reshape((1,new_nc)) * nr).reshape((new_per_channel,1),order='F')
        element = np.concatenate((element, element + per_channel, element + per_channel*2),axis=0)
        tmp = map(lambda x:np.sum(vimg[x + ind],dtype=np.float), element)
        res_img = np.asarray(tmp).reshape((new_nr, new_nc,nchannel),order='F')
        return res_img/(fr*fc)
    @classmethod
    def load_matmeta(cls, matmeta_path):
        """
        Please note that this function require scipy.version >= 0.9.0
        """
        dmat = sio.loadmat(matmeta_path, squeeze_me=False)
        d = dict()
        
        ## general information
        d['image_adjust_dim'] = dmat['image_adjust_dim']
        ##
        num_images = dmat['num_images']
        d['num_images'] = np.int64(num_images)
        d['rgb_meancov'] = dmat['rgb_meancov']
        d['images_path'] = [dmat['images_path'][0,x][0] for x in range(num_images)]
        d['Y2d_bnd_body'] = dmat['Y2d_bnd_body']
        d['Y2d_image_body'] = dmat['Y2d_image_body']
        d['oribbox'] = dmat['oribbox']
        d['Y3d_mono_body'] = dmat['Y3d_mono_body']
        d['mean_image'] = dmat['mean_image']
        d['occ_body'] = dmat['occ_body']
        d['image_dim'] = dmat['image_dim'] # The original size of image 
        d['X'] = dmat['X']                  # empty
        d['Y3d_mocap'] = dmat['Y3d_mocap']  # empty
        return d


    @classmethod
    def copy_dic(cls, d1, d2, override=False, fields = None):
        if fields is None:
            fields = d2.keys()
        if not override:
            for x in fields:
                if x not in d1:
                    d1[x] = d2[x]
        else:
            for x in fields:
                d1[x] = d2[x]
    @classmethod
    def rgb_eig(cls,covmat):

        # U,S,V = np.linalg.svd(covmat)
        # return U,S.reshape((3,1))
        # It seems that eig is more stable than svd
        # 
        S, U = np.linalg.eig(covmat)
        return U,S.reshape((3,1))
    @classmethod
    def merge_field(cls, d1, d2, res_d, fields):
        """
        Apply different ways for merging different kinds of data
        """
        if d1 is None:
            for x in fields:
                res_d[x] = d2[x]
        for x in fields:
            if (not x in d1) or (not x in d2):
                raise DHMLPEError('Need field %s requred' % x)
            if type(d1[x]) != type(d2[x]):
                raise DHMLPEError('Type are not consistent')
            if type(d1[x]) == np.ndarray:
                if d1[x].size !=0 and d2[x].size!=0:
                    res_d[x] = np.concatenate([d1[x],d2[x]], axis=-1)
            elif type(d1[x]) == list:
                res_d[x] = d1[x] + d2[x]
            else:
                raise DHMLPEError('Unsupported type for merging')
    @classmethod 
    def merge_meta_list(cls, meta_list, opt=None):
        meta = None
        if len(meta_list) == 0:
            return meta
        if len(meta_list) < 2:
            raise DHMLPEError('meta_list should contain at least two element')
        if opt is None:
            opt = {'calc_statistics':False}
        else:
            opt['calc_statistics'] = False
        for x in meta_list:
            meta = cls.merge_meta(meta, x,opt)
        if meta is not None:
            cls.fill_statistics(meta)
        return meta
    @classmethod   
    def merge_meta(cls, meta1, meta2, opt=None):
        if meta1 is None:
            return meta2
        elif meta2 is None:
            return meta1
        required_field = []
        shared_field = set(['image_adjust_dim', 'image_sample_dim', \
                            'joint_indicator_map', 'image_dim'])
        calc_statistics = True
        if opt:
            if 'calc_statistics' in opt:
                calc_statistics = opt['calc_statistics']
            if 'required_field' in opt:
                required_field = opt['required_field']
            if 'shared_field' in opt:
                shared_field = opt['shared_field']
        res = dict()
        for x in required_field:
            if (not x in meta1) or (not x in meta2):
                raise DHMLPEError('Field %d is missing' % meta1)
        for x in meta2:
            if x in shared_field:
                res[x] = meta2[x]
        cls.merge_field(meta1, meta2, res, ['images_path', 'Y2d_bnd_body', \
                                            'oribbox', \
                                        'Y3d_mono_body', 'occ_body'])        
        # 'mean_image', 'rgb_meancov', 'num_images', 'rgb_eigenvector', 'rgb_eigenvalue'
        res['num_images'] = np.int64(meta1['num_images']) + \
                            np.int64(meta2['num_images'])
        res['mean_image'] = (meta1['mean_image'] * meta1['num_images'] + \
                             meta2['mean_image'] * meta2['num_images'])/res['num_images']
        res['rgb_meancov'] = (meta1['rgb_meancov'] * meta1['num_images'] + \
                             meta2['rgb_meancov'] * meta2['num_images'])/res['num_images']
        print '%d merge %d ==> %d' % (meta1['num_images'], meta2['num_images'], \
                                      res['num_images'])
        if calc_statistics:
            cls.fill_statistics(res)
        return res

def batch_generate_pymeta(data_root_folder, force_to_generate=False):
    """
    This function will convert batches into pymeta style which can be loaded by convnet
    """
    allfolder = iu.getfolderlist(data_root_folder)
    print 'Get %d folders' % len(allfolder)
    l = []
    import sys
    for fn in allfolder:
        a = DHMLPE()
        fp = iu.fullfile(data_root_folder, fn, 'matlab_meta.mat')
        if iu.exists(fp, 'file'):
            save_fp = iu.fullfile(data_root_folder, fn, 'batches.meta')
            print '-----------------------------'
            print 'Processing ', fp
            if iu.exists(save_fp, 'file') and not force_to_generate:
                print 'Ha ha, it exists!'
            else:
                meta = a.get_convnet_meta(fp)
                mio.pickle(save_fp, meta)
            print 'Saved %s' % save_fp
        else:
            l = l + [fp]
    print '=============\n'
    print 'Here is what I cannot find (%d in total)' % len(l)
    print l

class CostFunc(object):
    def __init__(self, params):
        self.params = params
    def calc_func_value(self, X):
        pass
    def calc_func_gradient(self, X):
        pass

