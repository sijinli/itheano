# Copyright (c) 2013, Li Sijin (lisijin7@gmail.com)
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
# - Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# 
# - Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


from python_util.data import *
import numpy.random as nr
import numpy as np
from numpy import random as rd
import sys
sys.path.append('/home/grads/sijinli2/Projects/DHMLPE/Python/src')
import iutils as iu
from time import time
from PIL import Image
class BasicDataProviderError(Exception):
    pass
def iprod(x):
    res = 1
    for t in x:
        res *=t
    return res
def load_images(imagepath, imgsize, mean_image = None):
    """
    The size of all the image should be the same
    """
    dimX = imgsize[0] * imgsize[1] * imgsize[2]
    ndata = len(imagepath)
    if mean_image is None:
        res = np.zeros((dimX,ndata), dtype=np.single)
    else:
        res = np.tile(-mean_image.reshape((-1,1),order='F'), [1, ndata])
    for i,p in enumerate(imagepath):
        res[...,i] += np.asarray(Image.open(p)).reshape((dimX, 1),order='F')
    return  res
def load_cropped_images(imagepath, imgdim, cropped_mean_image, crop_dim, \
                        eigvalue, eigvector, sigma=0.1, test=False ):
    """
    Please note that mean_image should be the cropped mean
    """

    dimX = crop_dim[0] * crop_dim[1] * crop_dim[2]
    ndata = len(imagepath)
    pb = np.sqrt(eigvalue).reshape((1,-1)) * eigvector
    a =  np.dot(pb, rd.randn(3, len(imagepath))) * sigma
    res = np.tile(-cropped_mean_image.reshape((dimX, 1),order='F'),[1, ndata])
    nr, nc, dummy = np.floor((np.asarray(imgdim) - np.asarray(crop_dim))/2) + 1

    sr_list = rd.randint(nr, size=ndata) if not test else [nr-1] * ndata
    sc_list = rd.randint(nc, size=ndata) if not test else [nc-1] * ndata
    if not test:
        f = lambda curimg,r,c,i: curimg[r:r+crop_dim[0],c:c+crop_dim[1],:].reshape((dimX),order='F') + np.tile(a[...,i].reshape((1,3)),[1,dimX/3]).reshape((dimX),order='F')
    else:
        f = lambda curimg,r,c,i: curimg[r:r+crop_dim[0],c:c+crop_dim[1],:].reshape((dimX),order='F')
    for i,p in enumerate(imagepath):
        curimg = np.array(Image.open(p))
        sr = sr_list[i]
        sc = sc_list[i]
        res[...,i] += f(curimg, sr,sc,i)
    return sr_list, sc_list, res

def load_cropped_images_in_memory(imagedata, imgdim, cropped_mean_image, crop_dim, \
                                  eigvalue, eigvector, sigma=0.1, test=False):
    dimX = crop_dim[0] * crop_dim[1] * crop_dim[2]
    ndata = imagedata.shape[-1]
    pb = np.sqrt(np.abs(eigvalue)).reshape((1,-1)) * eigvector
    a =  np.dot(pb, rd.randn(3, ndata)) * sigma
    res = np.tile(-cropped_mean_image.reshape((dimX, 1),order='F'),[1, ndata])
    nr, nc, dummy = np.floor((np.asarray(imgdim) - np.asarray(crop_dim))/2) + 1
    sr_list = rd.randint(nr, size=ndata) if not test else [nr-1] * ndata
    sc_list = rd.randint(nc, size=ndata) if not test else [nc-1] * ndata
    if not test:
        f = lambda curimg,r,c,i: curimg[r:r+crop_dim[0],c:c+crop_dim[1],:].reshape((dimX),order='F') + np.tile(a[...,i].reshape((1,3)),[1,dimX/3]).reshape((dimX),order='F')
    else:
        f = lambda curimg,r,c,i: curimg[r:r+crop_dim[0],c:c+crop_dim[1],:].reshape((dimX),order='F')
    for i in range(ndata):
        curimg = imagedata[...,i]
        sr = sr_list[i]
        sc = sc_list[i]
        res[...,i] += f(curimg, sr,sc,i)
    return sr_list, sc_list, res
    
                                  
def calc_cropped_mean_global(ori_meanimg, new_dim):
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
    
class CroppedImageDataProvider(DataProvider):
    """
    
    """
    def __init__(self, data_dir, image_range, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        DataProvider.__init__(self, data_dir, range(len(image_range)), init_epoch, init_batchnum, dp_params, test)

        #crop_boarder will crop 2 * crop_boarder in each dimension
        #padding is similar. It will pad both size, i.e., adding 2x pixels
        self.image_dim = np.asarray(self.batch_meta['image_adjust_dim']).flatten()
        if dp_params['crop_border'] > 0:
            self.input_image_dim = self.batch_meta['image_adjust_dim']
            self.input_image_dim[0] -= dp_params['crop_border'] * 2
            self.input_image_dim[1] -= dp_params['crop_border'] * 2
        elif dp_params['crop_one_border'] > 0:
            self.input_image_dim = self.batch_meta['image_adjust_dim']
            self.input_image_dim[0] -= dp_params['crop_one_border']
            self.input_image_dim[1] -= dp_params['crop_one_border']
        else:
            self.input_image_dim = self.batch_meta['image_sample_dim']
        if 'fix_num_batch' in dp_params:
            # It can be used for forcing the number of batch to be fixed
            # The last batch might be smaller than previous batch
            self.fix_num_batch = dp_params['fix_num_batch']
        else:
            self.fix_num_batch = False 
        self.shuffle_data = dp_params['shuffle_data'] # determine whether to shuffle test data
        if 'external_meta_path' in dp_params and dp_params['external_meta_path']:
            import iread.myio as mio
            ext_meta = mio.unpickle(dp_params['external_meta_path'])
            print 'Print load external_meta for %s succussfully' % dp_params['external_meta_path']
            override_dic = ['mean_image', 'cropped_mean_image', 'rgb_eigenvalue', \
                            'rgb_eigenvector', 'RelativeSkel_Y3d_mono_body', \
                            'Relative_Y3d_mono_body']
            for item in override_dic:
                if item in ext_meta:
                    self.batch_meta[item] = ext_meta[item]
                    print '----Load %s from ext_meta succussfully' % item
            del ext_meta
        self.mean_image = self.batch_meta['mean_image']
        self.cropped_mean_image = self.get_cropped_mean()

        self.rgb_eigenvalue = self.batch_meta['rgb_eigenvalue']
        self.rgb_eigenvector = self.batch_meta['rgb_eigenvector']
        self.test = test        
        self.image_range = np.asarray(image_range)
        self.num_image = len(image_range)
        self.batch_size = dp_params['batch_size']
        self.keep_data_dic = False
        if self.batch_size > self.num_image or self.batch_size <= 0:
            raise BasicDataProviderError('Invaid batch_size %d (num_image=%d)' % (self.batch_size, self.num_image))
        self.num_batch = (self.num_image - 1)/ self.batch_size + 1
        # override batch_range, this is not actually the batch_range
        # just keep consistent
        self.batch_range = range(self.num_image)
        # recheck curr_batchnum 
        # (Remembering last times' batch_num will not help training), just keep batch consistant
        if self.curr_batchnum not in self.batch_range:
            self.curr_batchnum = 0
        self.curr_batchnum = min(max(self.curr_batchnum, 0), self.num_image - 1)
        # print 'Curr_batchnum = %d Test  = %s' % (self.curr_batchnum, 'True' if self.test else 'False')
        # override batch_Idx
        self.batch_idx = self.curr_batchnum
        if test and (not self.shuffle_data):
            # There is no need to shuffle testing data
            self.shuffled_image_range = self.image_range
        else:
            self.shuffled_image_range = self.image_range[rd.permutation(self.num_image)]
        if 'images_path' in self.batch_meta:
            self.images_path = self.batch_meta['images_path']
        else:
            self.images_path = None
    def get_cropped_mean(self):
        if 'cropped_mean_image' in self.batch_meta and self.batch_meta['cropped_mean_image'].shape == tuple(self.input_image_dim):
            return self.batch_meta['cropped_mean_image']
        else:
            return self.calc_cropped_mean(self.batch_meta['mean_image'], self.input_image_dim)
    @classmethod
    def calc_cropped_mean(cls, ori_meanimg, new_dim):
        return calc_cropped_mean_global(ori_meanimg, new_dim)
    def get_next_batch(self):
        if self.data_dic is None or len(self.batch_range) > 1:
            self.data_dic = self.get_batch(self.curr_batchnum)
        epoch, batchnum = self.curr_epoch, self.curr_batchnum
        self.advance_batch()
        ndata = self.data_dic['data'].shape[-1]
        alldata = [np.require(self.data_dic['data'].reshape((-1,ndata),order='F'),dtype=np.single, requirements='C')]
        return epoch, batchnum, alldata
    def __add_subbatch(self, batch_num, sub_batchnum, batch_dic):
        raise BasicDataProviderError('Not implemented')
    # _joint_batches use the parant class's method
    def get_batch(self, batch_num):
        """
        batch_num in self.image_range
        """
        dic = dict()
        if (self.test and (not self.shuffle_data)) or self.fix_num_batch:
            # test data doesn't need to circle 
            end_num = min(batch_num + self.batch_size, self.num_image)
            cur_batch_indexes = self.shuffled_image_range[batch_num:end_num]
        else:
            cur_batch_indexes = self.shuffled_image_range[ map(lambda x: x if x < self.num_image else x - self.num_image ,range(batch_num, batch_num + self.batch_size)) ]
        ## record the current batch_indexes
        dic['cur_batch_indexes'] = cur_batch_indexes.copy()
        ## Load image data
        imagepaths = []
        imagepaths = map(lambda x:self.images_path[x], cur_batch_indexes)
        
        offset_r, offset_c, dic['data'] = load_cropped_images(imagepaths, self.image_dim, self.cropped_mean_image, self.input_image_dim, self.rgb_eigenvalue, self.rgb_eigenvector, 0.1, self.test)
        self.cur_offset_r = offset_r
        self.cur_offset_c = offset_c
        return dic
    def get_data_dims(self):
        return iprod(self.input_image_dim)
    def get_plottable_data(self, imgdata):
        ndata = imgdata.shape[-1]
        dimX = imgdata.shape[0]
        res = imgdata.copy() +self.cropped_mean_image.reshape((dimX,1),order='F')
        imgdim = list(self.input_image_dim) + [ndata]
        return res.reshape(imgdim, order='F').transpose((3,0,1,2))/255.0
    def advance_batch(self):
        self.batch_idx = self.get_next_batch_idx()
        if self.batch_idx >= self.num_image:
            self.curr_epoch += 1
            if not ((self.test and (not self.shuffle_data)) or self.fix_num_batch):
                self.batch_idx -= self.num_image
            else:
                self.batch_idx = 0
        self.curr_batchnum = self.batch_idx
    def reset(self,epoch=1, batch_idx=0):             #I don't remember why it is 1 here >_<
        self.curr_batchnum = self.batch_idx = batch_idx
        self.epoch = epoch
    def get_next_batch_idx(self):
        return self.batch_idx + self.batch_size
    def get_next_batch_num(self):
        if (self.test and (self.shuffle_data == 0) ) or self.fix_num_batch :
            if self.batch_idx + self.batch_size < self.num_image:
                return self.batch_idx + self.batch_size
            else:
                return 0
        else:
            return (self.batch_idx + self.batch_size) % self.num_image
    def get_data_file_name(self, batchnum=None):
        return 'There are no data file here'
    def get_num_classes(self):
        return self.num_classes 
    @staticmethod
    def get_batch_filenames(srcdir):
        raise NoahDataProviderError('Should not use this')
    @staticmethod
    def get_batch_nums(srcdir):
        return -1
    @staticmethod
    def get_num_batches(srcdir):
        return -1

class CroppedMemoryMetaDataProvider(CroppedImageDataProvider):
    """
    All the image will be saved in meta.data
    
    """
    def __init__(self, data_dir, image_range, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        CroppedImageDataProvider.__init__(self, data_dir, range(len(image_range)), init_epoch, init_batchnum, dp_params, test)
        self.read_all_data()
    def read_all_data(self):
        """
        Read data from meta
        """
        self.data = self.batch_meta['data']
    def get_batch(self, batch_num):
        """
        batch_num in self.image_range
        """
        dic = dict()
        if self.test and self.shuffle_data == 0:
            # test data doesn't need to circle 
            end_num = min(batch_num + self.batch_size, self.num_image)
            cur_batch_indexes = self.shuffled_image_range[batch_num:end_num]
        else:
            cur_batch_indexes = self.shuffled_image_range[ map(lambda x: x if x < self.num_image else x - self.num_image ,range(batch_num, batch_num + self.batch_size)) ]
        dic['cur_batch_indexes'] = cur_batch_indexes.copy()
        cur_images = self.data[...,cur_batch_indexes]
        offset_r, offset_c, dic['data'] = load_cropped_images_in_memory(cur_images, self.image_dim, self.cropped_mean_image, self.input_image_dim, self.rgb_eigenvalue, self.rgb_eigenvector, 0.1, self.test)
        self.cur_offset_r = offset_r
        self.cur_offset_c = offset_c
        if np.any(np.isnan(dic['data'])):
            print dic['data'].shape
            print dic['data'][:4,0]
            print 'Has nan'
        return dic

class MemoryFeatureDataProvider(DataProvider):
    """
    For this data provider
    all the features will be loaded into memroy at the beginning
    batches.meta will have two field
      'feature_list'   : a list of feature
      'feature_dim': The dimension of features for all the element in features  
    """
    def __init__(self, data_dir, feature_range, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        DataProvider.__init__(self, data_dir, feature_range, init_epoch, init_batchnum, dp_params, test)
        self.shuffle_data = dp_params['shuffle_data'] # determine whether to shuffle test data
        if 'external_meta_path' in dp_params and dp_params['external_meta_path']:
            import iread.myio as mio
            ext_meta = mio.unpickle(dp_params['external_meta_path'])
            print 'Print load external_meta for %s succussfully' % dp_params['external_meta_path']
            for item in ext_meta:
                self.batch_meta[item] = ext_meta[item]
                print '----Load %s from ext_meta succussfully' % item
            del ext_meta
        self.test = test
        self.feature_range = np.asarray(feature_range)
        self.num_feature = len(feature_range)
        self.batch_size = dp_params['batch_size']
        self.keep_data_dic = False
        if self.batch_size > self.num_feature or self.batch_size <= 0:
            raise BasicDataProviderError('Invaid batch_size %d (num_image=%d)' % (self.batch_size, self.num_feature))
        self.num_batch = (self.num_feature - 1)/ self.batch_size + 1
        self.batch_range = range(self.num_feature)
        if self.curr_batchnum not in self.batch_range:
            self.curr_batchnum = 0
        self.curr_batchnum = min(max(self.curr_batchnum, 0), self.num_feature - 1)
        self.batch_idx = self.curr_batchnum
        if test and self.shuffle_data == 0:
            # There is no need to shuffle testing data
            self.shuffled_feature_range = self.feature_range
        else:
            self.shuffled_feature_range = self.feature_range[rd.permutation(self.num_feature)]
        self.num_feature_type = len(self.batch_meta['feature_dim'])
        self.feature_dim = self.batch_meta['feature_dim']
    def get_batch(self, batch_num):
        """
        batch_num in self.feature_range
        if condition is different from CroppedImageDataProvider
        """
        dic = dict()
        if self.test:            
            # test data doesn't need to circle 
            end_num = min(batch_num + self.batch_size, self.num_feature)
            cur_batch_indexes = self.shuffled_feature_range[batch_num:end_num]
        else:
            cur_batch_indexes = self.shuffled_feature_range[ map(lambda x: x if x < self.num_feature else x - self.num_feature ,range(batch_num, batch_num + self.batch_size)) ]
        ## record the current batch_indexes
        dic['cur_batch_indexes'] = cur_batch_indexes.copy()
        return dic
    def get_next_batch(self):
        if self.data_dic is None or len(self.batch_range) > 1:
            self.data_dic = self.get_batch(self.curr_batchnum)
        epoch, batchnum = self.curr_epoch, self.curr_batchnum
        self.advance_batch()
        cur_ndata = len(self.data_dic['cur_batch_indexes'])
        alldata = [np.require(self.batch_meta['feature_list'][k][..., self.data_dic['cur_batch_indexes']].reshape((-1,cur_ndata),order='F'),dtype=np.single, requirements='C') for k in range(self.num_feature_type)]
        return epoch, batchnum, alldata
    def advance_batch(self):
        self.batch_idx = self.get_next_batch_idx()
        if self.batch_idx >= self.num_feature:
            self.curr_epoch += 1
            if not (self.test and (self.shuffle_data == 0)):
                self.batch_idx -= self.num_feature
            else:
                self.batch_idx = 0
        self.curr_batchnum = self.batch_idx
    def get_next_batch_idx(self):
        return self.batch_idx + self.batch_size
    def get_next_batch_num(self):
        if self.test and (self.shuffle_data == 0):
            if self.batch_idx + self.batch_size < self.num_feature:
                return self.batch_idx + self.batch_size
            else:
                return 0
        else:
            return (self.batch_idx + self.batch_size) % self.num_feature
    def get_num_batches(self):
        return -1
    def get_data_dims(self,idx=0):
        if idx >= self.num_feature_type or idx < 0:
            raise BasicDataProviderError('Index should be in range[%d,%d]' % (0, self.num_feature_type))
        else:
            return self.feature_dim[idx]
    @staticmethod
    def get_num_batches(srcdir):
        return -1
