import numpy as np
import sys
import iread.myio as mio
import iutils as iu
class DataProvider(object):
    def __init__(self, data_dic, train, data_range, params):
        self.data_dic = data_dic
        self.param_dic = dict()
        self.required_attributes = ['batch_size']
        self.required_field = []
        self.train = train
        self.data_range = np.array(data_range)
        self.num_batch = None
        self.pre_batchnum = None
        self.batch_data = None
        self.batch_indexes = None
    def reset(self):
        self.batchnum = self.epoch = 0
    def safe_add_params(self, key_name, source_dic, default_value=None):
        if key_name in source_dic:
            self.param_dic[key_name] = source_dic[key_name]
        else:
            self.param_dic[key_name] = default_value
    def get_state_dic(self):
        return {'epoch':self.epoch, 'batchnum':self.batchnum}
    def parse_params(self, params):
        for e in self.required_attributes:
            setattr(self, e, params[e])
            self.param_dic[e] = params[e]
        for e in self.required_field:
            self.param_dic[e] = params[e]
        if 'batchnum' not in params:
            self.batchnum = 0
        else:
            self.batchnum = params['batchnum']
        if 'epoch' not in params:
            self.epoch = 0
        else:
            self.epoch = params['epoch']
        if 'force_shuffle' in params:
            self.force_shuffle = params['force_shuffle']
        else:
            self.force_shuffle = False
        print 'force_shuffle == {}'.format(self.force_shuffle)
    def advance_batch(self):
        self.epoch, self.batchnum = self.pre_advance_batch()
    def pre_advance_batch(self):
        epoch, batchnum = self.epoch, self.batchnum + 1
        if batchnum == self.num_batch:
            batchnum = 0
            epoch = epoch + 1
        return epoch, batchnum
    def get_num_batches_done(self, epoch, batchnum):
        return epoch * self.num_batch + batchnum
class MemoryDataProvider(DataProvider):
    def __init__(self, data_dic, train, data_range, params):
        DataProvider.__init__(self, data_dic, train, data_range, params)
        self.parse_params(params)

        self.generate_batch_data(self.batch_size)
    def parse_params(self, params):
        DataProvider.parse_params(self, params)
        if self.data_dic is None:
            if 'data_path' in params:
                self.data_dic = mio.unpickle(iu.fullfile(params['data_path'], 'batches.meta'))
            else:
                raise Exception('data-path is missing')
    def generate_batch_data(self, batch_size):
        """
        """
        ndata = len(self.data_range)
        self.num_batch = int(ndata - 1) / int(batch_size) + 1
        if self.batchnum >= self.num_batch or self.batchnum < 0:
            self.batchnum = 0
        batch_data = []
        if self.train or self.force_shuffle:
            self.shuffled_data_range = np.random.permutation(self.data_range)
        else:
            self.shuffled_data_range = np.array(self.data_range)
        batch_indexes = []
        for i in range(self.num_batch):
            start, end = i * batch_size, min((i + 1) * batch_size, ndata)
            indexes = self.shuffled_data_range[start:end] 
            batch_data += [[elem[..., indexes].reshape((-1,end-start),order='F')
                           for elem in self.data_dic['feature_list']]]
            batch_indexes.append(indexes)
        self.batch_data, self.batch_indexes = batch_data, batch_indexes
    def get_batch(self, batchnum):
        return self.batch_data[batchnum]
    def get_batch_indexes(self, batchnum = None):
        bn = batchnum if (batchnum is not None ) else self.pre_batchnum
        return self.batch_indexes[bn]
    def get_next_batch(self):
        data = self.get_batch(self.batchnum)
        epoch, batchnum = self.epoch, self.batchnum
        self.pre_batchnum = batchnum
        self.advance_batch()
        return epoch, batchnum, data


class DHMLPEDataWarper(DataProvider):
    """
    This is a warper class for DHMLP
    """
    _DEFAULT_MODULE_PATH='/home/grads/sijinli2/Projects/cuda-convnet2/'
    def __init__(self, data_dic, train, data_range, params):
        DataProvider.__init__(self, data_dic, train, data_range, params)
        if 'module_path' not in params:
            self.module_path = DHMLPEDataWarper._DEFAULT_MODULE_PATH
        else:
            self.module_path = params['module_path']
        sys.path.append(self.module_path)
        try:
            self.dhmlpe_convdata = __import__('dhmlpe_convdata')
        except Exception as err:
            print 'Import dhmlpe_convdata fails: {}'.format(err)
            sys.exit(1)
    def get_all_data_at(self, idx=0):
        raise Exception('Should not be  called from base class''s  ')
    @classmethod
    def imgdim_shuffle(cls, X, imgdim,inv=False):
        """                      
        Change from [dimX, ndata] -->  [helght, width,nchannel, ndata]
                                      [width, height, nchannel, ndata]
        """
        ndata = X.shape[-1]
        dims = imgdim.tolist() + [ndata]
        return np.transpose(X.reshape(dims,order='F'), [1,0,2,3])

class CroppedImageDataWarper(DHMLPEDataWarper):
    def create_inner_dp(self, data_path, data_range, epoch, init_batchnum, dp_params, test):
        return self.dhmlpe_convdata.CroppedImageDataProvider(data_path,
                                                             data_range,
                                                             epoch,
                                                             init_batchnum,
                                                             dp_params,
                                                             test=True)
    def __init__(self, data_dic, train, data_range, params):
        DHMLPEDataWarper.__init__(self, data_dic, train, data_range, params)
        self.required_attributes += ['data_path']
        self.parse_params(params)
        dp_params = {'fix_num_batch':True, 'crop_border':-1, 'crop_one_border':-1,
                     'shuffle_data':train or self.force_shuffle, 'batch_size':self.batch_size
        }
        
        init_batchnum = self.batchnum * self.batch_size
        self.inner_dp = self.create_inner_dp(self.data_path, data_range, self.epoch,
                                             init_batchnum, dp_params, True)

        self.image_dim = self.inner_dp.image_dim
        self.mean_image = self.inner_dp.mean_image
        self.input_image_dim = np.array(self.inner_dp.input_image_dim)
        self.cropped_mean_image = self.inner_dp.cropped_mean_image
        self.rgb_eigenvalue = self.inner_dp.rgb_eigenvalue.copy()

        self.inner_dp.rgb_eigenvalue[:] = 0
        self.inner_dp.input_image_dim = np.array(self.inner_dp.image_dim,dtype=np.int)
        self.inner_dp.cropped_mean_image = self.inner_dp.mean_image.copy()
        self.inner_dp.cropped_mean_image[:] =0
        self.data_buffer = dict()
        self.num_batch = self.inner_dp.num_batch
        assert( self.num_batch == (len(data_range) - 1) // self.batch_size + 1)
        self.offset_r, self.offset_c = None, None
    def parse_params(self, params):
        DHMLPEDataWarper.parse_params(self, params)
        self.safe_add_params('with_buffer', params, True)
    def reset(self):
        DHMLPEDataWarper.reset(self)
        self.inner_dp.reset()
    def get_batch_indexes(self, batchnum=None):
        return self.inner_dp.data_dic['cur_batch_indexes']
    def get_plottable_data(self, imagedata):
        ndata = imagedata.shape[-1]
        dims = self.input_image_dim.tolist() + [ndata]
        dim_mean = dims[:-1] + [1]
        dims[0], dims[1] = dims[1], dims[0] # swap height and width
        imagedata = self.imgdim_shuffle(imagedata.reshape(dims, order='F'), np.array(dims[:-1]))
        imagedata = imagedata + self.cropped_mean_image.reshape(dim_mean, order='F')
        return imagedata
    def crop_image(self, imagedata):
        ndata = imagedata.shape[-1]
        dim = self.image_dim.flatten().tolist() + [ndata]
        imagedata = imagedata.reshape(dim, order='F')
        dp = self.inner_dp
        offset_r, offset_c, cropped_images = self.dhmlpe_convdata.load_cropped_images_in_memory(imagedata, self.image_dim, self.cropped_mean_image, self.input_image_dim, self.rgb_eigenvalue, dp.rgb_eigenvector, sigma=0.1, test=not self.train)
        return offset_r, offset_c, cropped_images
    def get_next_batch(self):
        return None
class CroppedDHMLPEJointDataWarper(CroppedImageDataWarper):
    def create_inner_dp(self, data_path, data_range, epoch, init_batchnum, dp_params, test):
        return self.dhmlpe_convdata.CroppedDHMLPEJointDataProvider(data_path,
                                                                   data_range,
                                                                   epoch,
                                                                   init_batchnum,
                                                                   dp_params,
                                                                   test=True)
    def get_next_batch(self):
        epoch, batchnum = self.epoch, self.batchnum
        if batchnum in self.data_buffer:
            alldata = self.data_buffer[batchnum]
        else:
            dummy1, dummy2, alldata = self.inner_dp.get_next_batch()
            self.data_buffer[self.batchnum] = alldata
        offset_r, offset_c, cropped_images = self.crop_image(alldata[0])
        self.cur_offset_r = offset_r
        self.cur_offset_c = offset_c
        self.advance_batch()
        input_images = self.imgdim_shuffle(cropped_images,self.input_image_dim)
        return epoch, batchnum, [input_images] + alldata[1:]
    def get_all_data_at(self, idx=0):
        if idx == 0:
            raise Exception('It is too large ! You will not want to load it into memory now')
        else: 
            return self.inner_dp.get_all_data_at(idx)

class CroppedImageClassificationDataWarper(CroppedImageDataWarper):
    """
    Provide [img, label, labelind]
    """
    def __init__(self, data_dic, train, data_range, params):
        CroppedImageDataWarper.__init__(self, data_dic, train,data_range, params)
        self.labels = self.inner_dp.batch_meta['labels']
        self.min_label = 10
        self.max_label = 59
        self.num_classes = 25
        self.indlabels = self.cvtlabel2ind(self.labels, self.min_label, self.max_label,
                                           self.num_classes)
    def get_step(self):
        return (self.max_label - self.min_label ) // self.num_classes + 1
    @classmethod
    def cvtlabel2ind(cls, labels, min_label, max_label, num_classes):
        nlabel = max_label - min_label + 1
        step = (nlabel - 1) // num_classes + 1
        indmap = np.eye(num_classes, dtype=np.int)
        idx = [min(int(max(0,x - min_label)) //step, num_classes-1) for x in  labels.flatten()]
        return indmap[..., idx]
    def get_next_batch(self):
        epoch, batchnum = self.epoch, self.batchnum
        if batchnum in self.data_buffer:
            alldata = self.data_buffer[batchnum]
        else:
            dummy1, dummy2, alldata = self.inner_dp.get_next_batch()
            assert(len(alldata) == 1)
            cur_indexes = self.get_batch_indexes(self)
            labels = self.labels[..., cur_indexes]
            indlabels = self.indlabels[..., cur_indexes]
            alldata.append(labels)
            alldata.append(indlabels)
            self.data_buffer[batchnum] = alldata
        offset_r, offset_c, cropped_images = self.crop_image(alldata[0])
        self.cur_offset_r = offset_r
        self.cur_offset_c = offset_c
        self.advance_batch()
        input_images = self.imgdim_shuffle(cropped_images,self.input_image_dim)
        return epoch, batchnum, [input_images] + alldata[1:]
dp_dic = {'mem': MemoryDataProvider, 'croppedjt':CroppedDHMLPEJointDataWarper,
          'croppedimgcls':CroppedImageClassificationDataWarper}
