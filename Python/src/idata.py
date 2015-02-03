import numpy as np
import sys
class DataProvider(object):
    def __init__(self, data_dic, train, data_range, params):
        self.data_dic = data_dic
        self.param_dic = dict()
        self.required_attributes = ['batch_size']
        self.required_field = []
        self.train = train
        self.data_range = np.array(data_range)
        self.num_batch = None
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
    def generate_batch_data(self, batch_size):
        """
        """
        ndata = len(self.data_range)
        self.num_batch = int(ndata - 1) / int(batch_size) + 1
        if self.batchnum >= self.num_batch or self.batchnum < 0:
            self.batchnum = 0
        batch_data = []
        self.shuffled_data_range = np.random.permutation(self.data_range)
        for i in range(self.num_batch):
            start, end = i * batch_size, min((i + 1) * batch_size, ndata)
            indexes = self.shuffled_data_range[start:end] 
            batch_data += [[elem[..., indexes].reshape((-1,end-start),order='F')
                           for elem in self.data_dic['feature_list']]]
        self.batch_data = batch_data
    def get_batch(self, batchnum):
        return self.batch_data[batchnum]
    def get_next_batch(self):
        data = self.get_batch(self.batchnum)
        epoch, batchnum = self.epoch, self.batchnum
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
    @classmethod
    def imgdim_shuffle(cls, X, imgdim,inv=False):
        """                      
        Change from [dimX, ndata] -->  [helght, width,nchannel, ndata]
                                      [width, height, nchannel, ndata]
        """
        ndata = X.shape[-1]
        dims = imgdim.tolist() + [ndata]
        return np.transpose(X.reshape(dims,order='F'), [1,0,2,3])
class CroppedDHMLPEJointDataWarper(DHMLPEDataWarper):
    def __init__(self, data_dic, train, data_range, params):
        DHMLPEDataWarper.__init__(self, data_dic, train, data_range, params)
        self.required_attributes += ['data_path']
        self.parse_params(params)
        dp_params = {'fix_num_batch':True, 'crop_border':-1, 'crop_one_border':-1,
                     'shuffle_data':train, 'batch_size':self.batch_size
        }
        self.inner_dp = self.dhmlpe_convdata.CroppedDHMLPEJointDataProvider(self.data_path,
                                                                            data_range,
                                                                            self.epoch,
                                                                            self.batchnum,
                                                                            dp_params,
                                                                            test=True)
        self.image_dim = self.inner_dp.image_dim
        self.mean_image = self.inner_dp.mean_image
        self.input_image_dim = np.array(self.inner_dp.input_image_dim)
        self.cropped_mean_image = self.inner_dp.cropped_mean_image
        self.rgb_eigenvalue = self.inner_dp.rgb_eigenvalue.copy()
        self.inner_dp.rgb_eigenvalue[:] = 0
        # Don't crop in the inner data provider
        self.inner_dp.input_image_dim = np.array(self.inner_dp.image_dim,dtype=np.int)
        self.inner_dp.cropped_mean_image = self.inner_dp.mean_image.copy()
        self.inner_dp.cropped_mean_image[:] = 0
        # print 'cropped_mean_image shape = {}'.format(self.inner_dp.cropped_mean_image.shape)
        # print 'input_image_dim = {}'.format(self.inner_dp.input_image_dim)
        self.data_buffer = dict()
        self.num_batch = self.inner_dp.num_batch
        assert( self.num_batch == (len(data_range) - 1) // self.batch_size + 1)
        self.offset_r, self.offset_c = None, None
    def parse_params(self, params):
        DHMLPEDataWarper.parse_params(self, params)
        self.safe_add_params('with_buffer', params, True)
    def crop_image(self, imagedata):
        ndata = imagedata.shape[-1]
        dim = self.image_dim.flatten().tolist() + [ndata]
        imagedata = imagedata.reshape(dim, order='F')
        dp = self.inner_dp
        offset_r, offset_c, cropped_images = self.dhmlpe_convdata.load_cropped_images_in_memory(imagedata, self.image_dim, self.cropped_mean_image, self.input_image_dim, self.rgb_eigenvalue, dp.rgb_eigenvector, sigma=0.1, test=not self.train)
        return offset_r, offset_c, cropped_images
    def get_plottable_data(self, imagedata):
        ndata = imagedata.shape[-1]
        dims = self.input_image_dim.tolist() + [ndata]
        dim_mean = dims[:-1] + [1]
        dims[0], dims[1] = dims[1], dims[0] # swap height and width
        print dims, imagedata.shape, '0000000000000000'
        imagedata = self.imgdim_shuffle(imagedata.reshape(dims, order='F'), np.array(dims[:-1]))
        imagedata = imagedata + self.cropped_mean_image.reshape(dim_mean, order='F')
        return imagedata
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
dp_dic = {'mem': MemoryDataProvider, 'croppedjt':CroppedDHMLPEJointDataWarper}
