import scipy.io as sio
import myio
import iutils as iu
import numpy as np
joint8names = ['head', 'Lshoulder', 'Neck', 'Rshoulder', 'Lelbow', 'Relbow', 'Lwrist', 'Rwrist']
joint8partnames = [ 'head', 'left_shoulder','right_shoulder', 'LUA','LLA', 'RUA', 'RLA']

AHE_names = ['torso', 'LUA', 'RUA', 'LLA', 'RLA', 'Head']
part_idx = [[0,2],[1,2],[2,3],[1,4],[4,6], [3,5],[5,7] ]    
all_dataset_name = set(['buffy', 'ETHZ','FLIC', 'lsp', 'SynchronicActivities', 'WeAreFamily'])
fieldlist = ['data', 'labels', 'filenames', 'joints8', 'oribbox', 'indmap', 'indmap_para']
default_imgdata_info = {}
default_indmap_para = {'filter_size':30, 'stride':12, 'rate':0.3}
default_savedata_info = {'sample_num':32, 'max_batch_size':2000, 'newdim':(112,112,3),'start_patch_id':1, 'savename':'data_patch', 'indmap_para':default_savedata_info}
class HMLPEError(Exception):
    pass

def flip_pose8_joint(dim, points):
    import copy
    if len(points)!= 8:
        raise ConvertError(' Invalid length <flip_pose8_joint')
    points = np.asarray([[dim[1] - 1 - p[0], p[1]] for p in points])
    respoints = copy.copy(points)
    respoints[[1,3],:] = respoints[[3,1],:]
    respoints[[4,5],:] = respoints[[5,4],:]
    respoints[[6,7],:] = respoints[[7,6],:]
    return respoints
class HMLPE:
    def __init__(self):
        self.imgdata_info = default_imgdata_info
        self.savedata_info = default_savedata_info
		self.flip_joints = flip_pose8_joint
    def set_imgdata_info(self,d):
        self.mergeto(d, self.imgdata_info)
    def set_savedata_info(self,d):
        self.mergeto(d, self.savedata_info)
    def mergeto(cls, source_d, target_d):
        for item in source_d:
            target_d[item] = source_d[item]
    def check_imgdata_info(self):
        imgdata_necessary = ['imgdatapath']
        self.mergedict(default_imgdata_info, self.imgdata_info)
        for item in imgdata_necessary:
            if item not in self.imgdata_info:
                raise HMLPEError('%s not found' % item)
    def check_savedata_info(self):
        savedata_necessary = ['savedir']
        for item in savedata_necessary:
            if item not in self.savedata_info:
                raise HMLPEError('%s not found' % item)
    def check_savedata_info(self):
        return True
    def generate_data(self, generate_type, allfile = None):
        """
        generate_type = 'rt' only
        """
        if allfile is None:
            allfile = iu.getfilelist( self.imgdata_info['imgdata_path'], '\w+\.mat')
        print 'imgdatapath=%s, %d files are found' % (self.imgdata_info['imgdata_path'], len(allfile))
        iu.ensure_dir(self.savedata_info['savedir'])
        self.batch_id = self.savedata_info['start_patch_id']
        ndata = 0
        self.meta = {'imgdata_info':self.imgdata_info,'savedata_info':self.savedata_info}
        self.meta['num_vis'] = iu.prod(self.savedata_info['newdim'])
        self.meta['data_sum'] = 0
        self.meta['ndata'] = 0
        self.meta['nparts'] = len(part_idx) 
        for fn in allfile:
            if generate_type == 'rt':
                mpath = iu.fullfile(self.imgdata_info['imgdata_path'], fn)
                self.generate_rt_data(iu.fullfile(mpath))
        if self.meta['ndata'] > 0:
            self.meta['data_mean']  = self.meta['data_sum'] / self.meta['ndata']
        del self.meta['data_sum']
        myio.pickle(iu.fullfile(self.savedata_info['savedir'], 'batches.meta'), self.meta)
    def generate_rt_data(self, matpath):
        """
        in each mat
                        mat['X'] is image data
                        mat['Y'] is npart x ndata array
        """
        mat = sio.loadmat(matpath)
        dim = mat['dim'][0]
        ndata = (mat['X'].shape)[1]
        sample_num = self.savedata_info['sample_num']
        totaldata = sample_num * ndata * 2
        newdim = self.savedata_info['newdim']
        nparts = self.meta['nparts']
        filter_size = self.savedata_info['indmap_para']['filter_size']
        stride =  self.savedata_info['indmap_para']['stride']
        rate = self.savedata_info['indmap_para']['rate']
        mdim = self.get_indmapdim(newdim, filter_size, stride)
        
        if newdim[0] > dim[0] or newdim[1] > dim[1]:
            raise HMLPEError('Invalid new size ')
        if (dim[0] - newdim[0] + 1) * (dim[1] - newdim[1] + 1) < sample_num:
            raise HMLPEError(' Invalid sample_num')
        res = {}
        per_size = min(totaldata, self.savedata_info['max_batch_size'])

        mesh = iu.cartesian_product2(np.asarray( [range(dim[0] - newdim[0] + 1), \
                                                 range(dim[1] - newdim[1] + 1)]))
        allX = mat['X'].reshape( (dim[0], dim[1],dim[2], ndata), order='F')
        allY = mat['Y'].reshape( (2,-1, ndata), order='F') 
        newlen = iu.prod( newdim )
        res['data'] = np.ndarray([newdim[0], newdim[1], newdim[2], per_size], \
                                 dtype = np.uint8)
        res['labels'] = np.ndarray( [allY.shape[1], per_size], dtype=np.int)
        res['joints8'] = np.ndarray( [8, 2, per_size], dtype=np.float)
        res['oribbox'] = np.ndarray( [4, per_size], dtype = np.float)
        res['indmap'] = np.zeros((nparts, mdim[0], mdim[1], per_size), np.bool)
        res['filenames'] =[str() for x in range(per_size)]
        tmpres = dict()
        cur_id = self.batch_id
        pre_nc = 0
        nc = 0
        for it in range(ndata):
            curX = allX[...,it]
            curY = allY[...,it].transpose()
            curfilename = str(mat['imagepathlist'][0,it][0])
            l = (np.random.permutation(range(len(mesh))))[:sample_num]
            for p in l:
                r,c = mesh[p]
                tmpX = curX
                tmpX = np.roll(tmpX, shift=-int(r), axis = 0)
                tmpX = np.roll(tmpX, shift=-int(c), axis = 1)
                tmpX = tmpX[:newdim[0], :newdim[1],:]
                res['data'][...,nc - pre_nc] = tmpX
                tmpY = curY - 1 + np.asarray([-c,-r])
                res['joints8'][..., nc - pre_nc] = tmpY;
                res['filenames'][nc - pre_nc] = curfilename
                res['oribbox'][...,nc-pre_nc] = res['oribbox'][...,it]
                res['indmap'][...,nc-pre_nc] = self.create_indicatormap(tmpY, part_idx, mdim, rate, filter_size, stride)
				res['jointmasks'] = self.makejointmask(newdim, tmpY)
                nc = nc + 1
                #flip image
                tmpX = tmpX[:,::-1,:]
                res['data'][...,nc - pre_nc] = tmpX
                res['filenames'][nc - pre_nc] = curfilename
				tmmY = flip_pose8_joint(newdim, tmpY)
	            res['joints8'][...,nc -pre_nc] = tmpY
                res['oribbox'][...,nc-pre_nc] = res['oribbox'][...,it]
				res['indmap'][...,nc-pre_nc] = self.create_indicatormap(tmpY, part_idx, mdim, rate, filter_size, stride)
				res['jointmasks'] = self.makejointmask(newdim, tmpY)
				
				
    @classmethod
    def create_single_indicatormap(cls, part, mdim, rate, filter_size, stride):
        """
        part : ((x0,y0),(x1,y1))
        mdim : The size of indicator map
        rate : if lseg > rate * l: then it will be regarded as intersection
        """
        from ipyml import geometry as geo
        resmap = np.zeros(mdim, dtype = np.bool)
        lx = int(min(np.floor(part[0][0]/stride), np.floor(part[1][0]/stride)))
        ly = int(min(np.floor(part[0][1]/stride), np.floor(part[1][1]/stride)))
        mx = int(min(mdim[0], 1 + max(np.floor(part[0][0]/stride), np.floor(part[1][0]/stride))))
        my = int(min(mdim[1], 1 + max(np.floor(part[0][1]/stride), np.floor(part[1][1]/stride))))
        l = np.sqrt((part[0][0] - part[1][0])**2 + (part[0][1] - part[1][1])**2)
        for x in range(lx, mx):
            for y in range(ly, my):
                if x < 0 or y < 0:
                    continue
                rect = [(x * stride, y * stride), (x * stride + filter_size -1, y * stride + filter_size -1)]
                seg = geo.SegmentIntersectRect(Part, rect)
                if len(seg)<2:
                    continue
                lseg = np.sqrt((seg[0][0] - seg[1][0])**2 + (seg[0][1] - seg[1][1])**2)            
                if (lseg > rate * l):
                    # Store in row, col format
                    resmap[y,x] = True            
        return resmap
    def create_indicatormap(cls, joints, partidx, mdim, rate, filter_size, stride):
        """
        This function will return a [npart,  mdim[0],  mdim[1]] indicator map
        In fact res[i,:,:] is a single indicator map for i-th part
        joints is the joints representation of body
        partidx represent parts as pair of joint         
        """        
        nparts = len(partidx)
        joints = joins.reshape((-1,2),order='C')
        tmap = np.zeros((nparts, mdim[0], mdim[1]), dtype=np.bool)
        for p in range(nparts):
            part = (joints[partidx[p][0]], joints[partidx[p][1]])
            tmap[p,:,:] = self.create_single_indicatormap(part, mdim, rate, filter_size, stride )
        return tmap
        
                
    @classmethod
    def get_indmapdim(cls, dim, filter_size, stride):
        """
        
        """    
        inddim = [0,0]
        inddim[0] = iu.get_conv_outputsize(dim[0], filter_size, stride)[1]
        inddim[1] = iu.get_conv_outputsize(dim[1], filter_size, stride)[1]
        return mdim
    
  
