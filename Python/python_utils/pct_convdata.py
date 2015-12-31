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

from ibasic_convdata import *
##
import dhmlpe
import dhmlpe_pct
import indicatormap
###

class PCTDataProviderError(Exception):
    pass
class PCTDataProvider(CroppedMemoryMetaDataProvider):
    """
    This data provider will provide
    [data, pct(real), detection map]
    """
    def __init__(self, data_dir, image_range, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        CroppedMemoryMetaDataProvider.__init__(self, data_dir, image_range, init_epoch, init_batchnum, dp_params, test)
        self.detection_ind_dim = self.batch_meta['detection_ind_dim']
        self.maximum_count = self.batch_meta['maximum_count']
    def read_all_data(self):
        """
        Read all necessary data into class object
        """
        self.data = self.batch_meta['data']
        self.detection_ind = self.batch_meta['detection_ind']
        self.pct = self.batch_meta['real_cnt'] / self.batch_meta['maximum_count']
    def get_next_batch(self):
        """
        Prepare data batch for cuda core
        alldata will be a list of data
        self.data_dic['cur_batch_indexes'] specify the index of data for this batch
        """
        epoch, batchnum, alldata = CroppedMemoryMetaDataProvider.get_next_batch(self)
        self.data_dic['pct'] = self.pct[..., self.data_dic['cur_batch_indexes']]
        self.data_dic['detection_ind'] = self.detection_ind[...,self.data_dic['cur_batch_indexes']]
        # print self.data_dic['data'].shape, self.data_dic['data'].dtype
        # if np.any(np.isnan(self.data_dic['data'])):
        #     print 'I found non in data_dic'
        # print self.data_dic['pct'].shape, self.data_dic['pct'].dtype
        # print self.data_dic['detection_ind'].shape, self.data_dic['detection_ind'].dtype
        ndata = len(self.data_dic['cur_batch_indexes'])
        alldata += [np.require(self.data_dic['pct'].reshape((-1,ndata), order='F'), dtype=np.single, requirements='C')]
        alldata += [np.require(self.data_dic['detection_ind'].reshape((-1,ndata),order='F'), dtype=np.single, requirements='C')]
        return epoch, batchnum, alldata
    def get_data_dims(self, idx=0):
        if idx==0:
            return np.prod(self.input_image_dim)
        elif idx == 1:
            return 1
        else:
            return np.prod(self.detection_ind_dim)

    
