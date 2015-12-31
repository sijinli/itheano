import scipy.io as sio
import myio
import iutils as iu
import numpy as np

class HMLPEError(Exception):
    pass
def flip_pose8_joint(dim, points):
    import copy
    if len(points)!= 8:
        raise HMLPEError(' Invalid length <flip_pose8_joint')
    points = np.asarray([[dim[1] - 1 - p[0], p[1]] for p in points])
    respoints = copy.copy(points)
    respoints[[1,3],:] = respoints[[3,1],:]
    respoints[[4,5],:] = respoints[[5,4],:]
    respoints[[6,7],:] = respoints[[7,6],:]
    return respoints
joint8names = ['head', 'Lshoulder', 'Neck', 'Rshoulder', 'Lelbow', 'Relbow', 'Lwrist', 'Rwrist']
joint8partnames = [ 'head', 'left_shoulder','right_shoulder', 'LUA','LLA', 'RUA', 'RLA']

AHE_names = ['torso', 'LUA', 'RUA', 'LLA', 'RLA', 'Head']
part_idx = [[0,2],[1,2],[2,3],[1,4],[4,6], [3,5],[5,7] ]    
all_dataset_name = set(['buffy', 'ETHZ','FLIC', 'lsp', 'SynchronicActivities', 'WeAreFamily'])
fieldlist = ['data', 'labels', 'filenames', 'joints8', 'oribbox', 'indmap', 'indmap_para']
default_imgdata_info = {}
default_indmap_para = {'filter_size':30.0, 'stride':12.0, 'rate':0.3, \
                       'joint_filter_size':30.0, 'joint_stride':12.0}

default_savedata_info = {'sample_num':16 , \
                         'neg_sample_num':100,\
                         'max_batch_size':4000, \
                         'newdim':(112,112,3), \
                         'start_patch_id':1,\
                         'savename':'data_batch', \
                         'indmap_para':default_indmap_para, \
                         'flipfunc':flip_pose8_joint, \
                         'jointnames': joint8names}  
class HMLPE:
    def __init__(self):
        self.imgdata_info = default_imgdata_info
        self.savedata_info = default_savedata_info
        self.flip_joints = flip_pose8_joint
        self.meta = dict()
    def set_imgdata_info(self,d):
        self.mergedicto(d, self.imgdata_info)
    def set_savedata_info(self,d):
        self.mergedicto(d, self.savedata_info)     
    def check_savedata_info(self):
        savedata_necessary = ['savedir']
        for item in self.savedata_necessary:
            if item not in self.savedata_info:
                raise HMLPEError('%s not found' % item)
    def check_savedata_info(self):
        return True
    def init_meta(self, generate_type):
        """
        init meta
           meta will only have those field
        """
        self.meta = {'imgdata_info':self.imgdata_info,'savedata_info':self.savedata_info}
        self.meta['num_vis'] = iu.prod(self.savedata_info['newdim'])
        self.meta['data_sum'] = 0
        self.meta['ndata'] = 0
        if 'part_idx' in self.savedata_info:
            self.meta['nparts'] = len(self.savedata_info['part_idx'])
        else:
            self.meta['nparts'] = len(part_idx)
            self.meta['savedata_info']['part_idx'] = part_idx
        self.meta['njoints'] = len(self.savedata_info['jointnames'])
        self.meta['generate_type'] = generate_type
        self.meta['matdim'] = None
        self.meta['ind_dim'] = dict()
    def create_augumentation_mesh(self, dim, newdim, generate_type):
        """
        return a list of offset
        [(dr1,dc1),(dr2,dc2), ...,(drn,dcn)]
        (dri,dci) is the lower points for i-th window( in row, col format)
        """
        if generate_type in set(['rt', 'neg_sample']):
            mesh = iu.cartesian_product2(np.asarray( [range(dim[0] - newdim[0] + 1), range(dim[1] - newdim[1] + 1)]))
        elif generate_type == 'ct':
            dx = (dim[0] - newdim[0])/2
            dy = (dim[1] - newdim[1])/2
            mesh =[(dx,dy)]
        return mesh
    def generate_sliding_detection_data_from_image(self, imgpath, scale_pairlist):
        from PIL import Image 
        import iutils as iu
        import imgproc
        self.init_meta('sliding')
        img = Image.open(imgpath)
        ndata = 0
        newdim = self.savedata_info['newdim']
        steps = self.imgdata_info['steps']
        # x1 is the size, x1 - 1 is the last pixel index 
        fx = lambda x1,x2: np.floor((x1  - newdim[1])/x2) + 1
        fy = lambda x1,x2: np.floor((x1  - newdim[0])/x2) + 1

        valid_idx = 0
        for s in scale_pairlist:
            ns = np.floor(np.asarray([img.size[0], img.size[1]]) * np.asarray([s[0], s[1]]))
            if ns[0] < newdim[1] or ns[1] < newdim[0]:
                break
            cur_n = fx(ns[0],steps[0]) * fy(ns[1], steps[1])
            valid_idx = valid_idx + 1
            ndata = ndata + cur_n 
        ndata = int(ndata)
        scale_pairlist = scale_pairlist[:valid_idx]
        print( 'Need to generate %d data' % ndata)
        filter_size = self.savedata_info['indmap_para']['filter_size']
        stride =  self.savedata_info['indmap_para']['stride']
        mdim = self.get_indmapdim(newdim, filter_size, stride)
        joint_filter_size = self.savedata_info['indmap_para']['joint_filter_size']
        joint_stride = self.savedata_info['indmap_para']['joint_stride']
        jtmdim = self.get_indmapdim(newdim, joint_filter_size, joint_stride)
        res = self.prepare_savebuffer({'data':newdim, 'part_indmap':mdim, \
                                       'joint_indmap': jtmdim}, ndata, self.meta['nparts'], self.meta['njoints'])
        dicjtname = 'joints8' if self.meta['njoints'] == 8 else 'joints'
        res[dicjtname][:] = 0
        res['jointmasks'][:] = False
        res['indmap'][:] = False
        res['joint_indmap'][:] = False
        res['is_mirror'][:] = False
        res['is_positive'][:] = False
        res['slide_location'] = np.zeros((2,ndata),dtype=np.float)
        res['scale'] = np.zeros((2,ndata), dtype=np.float)
        res['filenames'] = [imgpath for x in range(ndata)]
        idx = 0
        dimX = iu.prod(newdim)

        for s in scale_pairlist:
            ns = np.floor(np.asarray([img.size[0], img.size[1]]) * np.asarray([s[0], s[1]]))
            ns = (int(ns[0]),int(ns[1]))
            nimg = img.resize((ns[0],ns[1]))
            arrimg = imgproc.ensure_rgb(np.asarray(nimg))
            for x in range(0, ns[0] - newdim[1] + 1, steps[0]):
                for y in range(0, ns[1] - newdim[0] + 1, steps[1]):
                    res['scale'][...,idx] = np.asarray([s[0],s[1]]).reshape((2))
                    res['slide_location'][...,idx] = np.asarray([x,y]).reshape((2))
                    res['data'][...,idx] = arrimg[y:y+newdim[0],x:x+newdim[1],:]
                    idx = idx + 1
        if idx != ndata:
            raise HMLPEError('Number of data is not consistent')
        res['data'] = res['data'].reshape((-1,ndata),order='F')
        return res
    def generate_negative_data_from_image(self, generate_type, allfile=None):
        """
        generate_type = 'neg_sample'
        savedata_info should have 'neg_sample_num':
                      indicating sampling how many negative window per image
        If some image is small, then it will try to generate as much as possible
                      
        """
        import Image
        if allfile is None:
            allfile = iu.getfilelist(self.imgdata_info['imgdatapath'], \
                                     '\w+(\.png|\.jpg|\.pgm|.jpeg)')
        print 'imgdatapath=%s, %d images are found' % (self.imgdata_info['imgdatapath'], len(allfile))
        iu.ensure_dir(self.savedata_info['savedir'])
        savedir = self.savedata_info['savedir']
        self.batch_id = self.savedata_info['start_patch_id']
        self.init_meta(generate_type)
        print(self.meta)
        sample_num = self.savedata_info['neg_sample_num']
        totaldata = len(allfile) * sample_num
        self.meta['ndata'] = 0
        newdim = self.savedata_info['newdim']
        nparts = self.meta['nparts']
        njoints = self.meta['njoints']
        if njoints == 8:
            dicjtname = 'joints8'
        else:
            dicjtname = 'joints'
            #raise HMLPEError('njoints = %d are not supported yet' % njoints)
        filter_size = self.savedata_info['indmap_para']['filter_size']
        stride =  self.savedata_info['indmap_para']['stride']
        #rate = self.savedata_info['indmap_para']['rate']
        mdim = self.get_indmapdim(newdim, filter_size, stride)
        self.meta['ind_dim']['part_indmap'] = mdim
        joint_filter_size = self.savedata_info['indmap_para']['joint_filter_size']
        joint_stride = self.savedata_info['indmap_para']['joint_stride']
        jtmdim = self.get_indmapdim(newdim, joint_filter_size, joint_stride)
        self.meta['ind_dim']['joint_indmap'] = jtmdim
        per_size = min(totaldata, self.savedata_info['max_batch_size'])
        res = self.prepare_savebuffer({'data':newdim, 'part_indmap':mdim, \
                                       'joint_indmap': jtmdim}, per_size, nparts, njoints)
        res[dicjtname][:] = 0
        res['jointmasks'][:] = False
        res['indmap'][:] = False
        res['joint_indmap'][:] = False
        res['is_mirror'][:] = False
        res['is_positive'][:] = False
        pre_nc = 0
        nc = 0
        np.random.seed(7)
        for it, fn in enumerate(allfile):
            print('Processing %s' % fn)
            curimgpath= iu.fullfile(self.imgdata_info['imgdatapath'], fn)
            img = np.asarray(Image.open(curimgpath), dtype=np.uint8)
            imgdim = img.shape
            if imgdim[0] < newdim[0] or imgdim[1] < newdim[1]:
                print('small image, ignored')
                continue
            mesh = self.create_augumentation_mesh(imgdim, newdim, generate_type)
            ts = min(len(mesh), sample_num)
            l = (np.random.permutation(range(len(mesh))))[:ts]
            for p in l:
                r, c = mesh[p]
                timg = img[r:r+newdim[0],c:c+newdim[0],:]
                res['data'][...,nc-pre_nc] = timg
                res['joint_sample_offset'][...,nc-pre_nc] = [c,r]
                res['filenames'][nc-pre_nc] = curimgpath
                res['oribbox'][...,nc-pre_nc] = [c,r,c+newdim[1]-1,r+newdim[0]-1]
                nc = nc + 1
            if sample_num + nc-pre_nc > per_size or it == len(allfile)-1:
                tmpres = self.truncated_copydic(res, nc-pre_nc)
                tmpres['data'] = tmpres['data'].reshape((-1,nc-pre_nc),order='F')
                self.meta['data_sum'] += tmpres['data'].sum(axis=1,dtype=float)
                self.meta['ndata'] += nc - pre_nc
                savepath = iu.fullfile(self.savedata_info['savedir'], \
                                       self.savedata_info['savename'] + \
                                       '_' +  str(self.batch_id))
                myio.pickle(savepath, tmpres)
                self.batch_id = self.batch_id + 1
                pre_nc = nc
        if self.meta['ndata'] > 0:
            self.meta['data_mean'] = self.meta['data_sum'] / self.meta['ndata']
            self.meta['data_mean'] = self.meta['data_mean'].reshape((-1,1),order='F')
        else:
            self.meta['data_mean'] = 0
        del self.meta['data_sum']

        myio.pickle(iu.fullfile(self.savedata_info['savedir'], 'batches.meta'), self.meta)
    def generate_positive_data(self, generate_type, allfile = None):
        """
        generate_type = 'rt': random translation
                        'ct'  center block
        """
        if allfile is None:
            allfile = iu.getfilelist( self.imgdata_info['imgdatapath'], '\w+\.mat')
        print 'imgdatapath=%s, %d files are found' % (self.imgdata_info['imgdatapath'], len(allfile))
        iu.ensure_dir(self.savedata_info['savedir'])
        self.batch_id = self.savedata_info['start_patch_id']
        self.init_meta(generate_type)
        print self.meta
        np.random.seed(7)
        for fn in allfile:
            print 'Processing %s ' % fn
            mpath = iu.fullfile(self.imgdata_info['imgdatapath'], fn)
            self.generate_positive_data_from_mat(generate_type ,iu.fullfile(mpath))
        if self.meta['ndata'] > 0:
            self.meta['data_mean']  = self.meta['data_sum'] / self.meta['ndata']
            self.meta['data_mean'] = self.meta['data_mean'].reshape((-1,1))
        else:
            self.meta['data_mean'] = 0
        del self.meta['data_sum']
        myio.pickle(iu.fullfile(self.savedata_info['savedir'], 'batches.meta'), self.meta)
    def get_fieldpool_for_positive_mat_data(self):
        """
        return a dictionary contained all the necessary field
         for fill_in_positive_mat_data_to_dic
        newdim, dictjname,
                       curfilename,
                        part_idx, mdim, rate,
                        filter_size,
                         stride,
                         jtmdim,
                          joint_filter_size,
                           joint_stride,
                           c,r
        """
        FP = dict()
        njoints = self.meta['njoints']
        FP['Y2dname'] = 'Y'
        if njoints == 8:
            FP['dicjtname'] = 'joints8'
        else:
            #raise HMLPEError('njoints = %d No supported yet' % (njoints))
            FP['dicjtname'] = 'joints'
        FP['newdim'] = self.savedata_info['newdim']
        FP['filter_size'] = self.savedata_info['indmap_para']['filter_size']
        FP['stride'] =  self.savedata_info['indmap_para']['stride']
        FP['rate'] = self.savedata_info['indmap_para']['rate']
        FP['mdim'] = self.get_indmapdim(FP['newdim'], FP['filter_size'], \
                                        FP['stride'])
        FP['part_idx'] = self.meta['savedata_info']['part_idx']
        FP['joint_filter_size'] = self.savedata_info['indmap_para']['joint_filter_size']
        FP['joint_stride'] = self.savedata_info['indmap_para']['joint_stride']
        FP['jtmdim'] = self.get_indmapdim(FP['newdim'], FP['joint_filter_size'], FP['joint_stride'])
        return FP
        
    def fill_in_positive_mat_data_to_dic(self, res,idx,fieldpool, is_mirror):
        """
        Result is the dictionray to hold the data
        fieldpool stores all the necessory variable for use
        Required Filed: curX, Y, newdim, dictjname,
                       curfilename, mat,  part_idx, mdim, rate, filter_size, stride, jtmdim, joint_filter_size, joint_stride,c,r
        """
        FP = fieldpool
        imgX = FP['curX'][:FP['newdim'][0], :FP['newdim'][1],:]
        res['data'][...,idx] = imgX                
        res[FP['dicjtname']][..., idx] = FP['Y']
        res['jointmasks'][...,idx] = self.makejointmask(FP['newdim'], FP['Y'])
        res['filenames'][idx] = FP['curfilename']
        res['oribbox'][...,idx] = FP['mat']['oribbox'][...,FP['matidx']]
        res['indmap'][...,idx] = self.create_part_indicatormap(FP['Y'],self.meta['savedata_info']['part_idx'], FP['mdim'], \
            FP['rate'], FP['filter_size'], FP['stride'])
        res['joint_indmap'][...,idx] = self.create_joint_indicatormap(FP['Y'], FP['jtmdim'], FP['joint_filter_size'], FP['joint_stride'])
        res['joint_sample_offset'][...,idx] = [FP['c'], FP['r']]
        res['is_mirror'][...,idx] = is_mirror
        
    def generate_positive_data_from_mat(self, generate_type, matpath):
        """
        in each mat
                        mat['X'] is image data
                        mat['Y'] is npart x ndata array
        
        """
        mat = sio.loadmat(matpath)
        dim = mat['dim'][0]
        newdim = self.savedata_info['newdim']
        if newdim[0] > dim[0] or newdim[1] > dim[1]:
            raise HMLPEError('Invalid new size ')
        if self.meta['matdim'] is None:
            self.meta['matdim'] = dim # record the dimension before sampling
        else:
            if np.any(self.meta['matdim'] != dim):
                raise HMLPEError('Inconsistent matdim: Previous dim is %s, current mat dim is %s' % (str(self.meta['matdim']), str(dim)))
        ndata = (mat['X'].shape)[1]
        if generate_type in {'rt':1}:
            sample_num = self.savedata_info['sample_num']
            totaldata = sample_num * ndata * 2
            do_mirror = True
        elif generate_type == 'ct':
            sample_num  = 1
            totaldata = sample_num * ndata
            do_mirror = False
        if (dim[0] - newdim[0] + 1) * (dim[1] - newdim[1] + 1) < sample_num:
            raise HMLPEError(' Invalid sample_num')
        
        nparts = self.meta['nparts']
        self.meta['ndata'] += totaldata

        ### BEGIN COMMENT
        # njoints = self.meta['njoints']
        # if njoints == 8:
        #     dicjtname = 'joints8'
        # else:
        #     #raise HMLPEError('njoints = %d No supported yet' % (njoints))
        #     dicjtname = 'joints'
        # newdim = self.savedata_info['newdim']
        # filter_size = self.savedata_info['indmap_para']['filter_size']
        # stride =  self.savedata_info['indmap_para']['stride']
        # rate = self.savedata_info['indmap_para']['rate']
        # mdim = self.get_indmapdim(newdim, filter_size, stride)
        

        # if newdim[0] > dim[0] or newdim[1] > dim[1]:
        #     raise HMLPEError('Invalid new size ')
        # if (dim[0] - newdim[0] + 1) * (dim[1] - newdim[1] + 1) < sample_num:
        #     raise HMLPEError(' Invalid sample_num')
        # joint_filter_size = self.savedata_info['indmap_para']['joint_filter_size']
        # joint_stride = self.savedata_info['indmap_para']['joint_stride']
        # jtmdim = self.get_indmapdim(newdim, joint_filter_size, joint_stride)
        
        ### END COMMENT
        fieldpool = self.get_fieldpool_for_positive_mat_data()
        fieldpool['mat'] = mat
        self.meta['ind_dim']['part_indmap'] = fieldpool['mdim']
        self.meta['ind_dim']['joint_indmap'] = fieldpool['jtmdim']
        res = {}
        per_size = min(totaldata, self.savedata_info['max_batch_size'])
        
        allX = mat['X'].reshape( (dim[0], dim[1],dim[2], ndata), order='F')
        Y2dname = fieldpool['Y2dname']
        allY = mat[Y2dname].reshape( (2,-1, ndata), order='F') 
        newlen = iu.prod( newdim )
        # prepare data buffer
        res = self.prepare_savebuffer({'data':fieldpool['newdim'], 'part_indmap':fieldpool['mdim'], 'joint_indmap': fieldpool['jtmdim']},\
                                       per_size, self.meta['nparts'],\
                                        self.meta['njoints'])
        tmpres = dict()
        pre_nc = 0
        nc = 0
        res['is_positive'][:] = True
        for it in range(ndata):
            curX = allX[...,it]
            curY = allY[...,it].transpose()
            curfilename = str(mat['imagepathlist'][0,it][0]) if 'imagepathlist' in mat else ''
            mesh = self.create_augumentation_mesh(dim, newdim, generate_type)
            l = (np.random.permutation(range(len(mesh))))[:sample_num]
            fieldpool['matidx'] = it
            fieldpool['curfilename'] = curfilename
            for p in l:
                r,c = mesh[p]
                tmpX = curX
                tmpX = np.roll(tmpX, shift=-int(r), axis = 0)
                tmpX = np.roll(tmpX, shift=-int(c), axis = 1)
                tmpY = curY - 1 + np.asarray([-c,-r])
                fieldpool['r'] = r
                fieldpool['c'] = c
                ####
                fieldpool['curX'] = tmpX
                fieldpool['Y'] = tmpY
                
                # tmpX = tmpX[:newdim[0], :newdim[1],:]
                # res['data'][...,nc - pre_nc] = tmpX                
                # res[dicjtname][..., nc - pre_nc] = tmpY
                # res['jointmasks'][...,nc - pre_nc] = self.makejointmask(newdim, tmpY)
                # res['filenames'][nc - pre_nc] = curfilename
                # res['oribbox'][...,nc-pre_nc] = mat['oribbox'][...,it]
                # res['indmap'][...,nc-pre_nc] = self.create_part_indicatormap(tmpY, self.meta['savedata_info']['part_idx'], mdim, rate, filter_size, stride)
                # res['joint_indmap'][...,nc-pre_nc] = self.create_joint_indicatormap(tmpY, jtmdim, joint_filter_size, joint_stride)
                # res['joint_sample_offset'][...,nc-pre_nc] = [c, r]
                # res['is_mirror'][...,nc-pre_nc] = False
                self.fill_in_positive_mat_data_to_dic(res, nc - pre_nc, \
                                                      fieldpool, False)
                nc = nc + 1
                if not do_mirror:
                    continue
                #flip image
                tmpX = tmpX[:,::-1,:]
                tmpY = self.flip_joints(newdim, tmpY)
                fieldpool['curX'] = tmpX
                fieldpool['Y'] = tmpY
                self.fill_in_positive_mat_data_to_dic(res, nc - pre_nc, \
                                                      fieldpool, True)
                # res['data'][...,nc - pre_nc] = tmpX
                # res[dicjtname][...,nc -pre_nc] = tmpY
                # res['jointmasks'][...,nc - pre_nc] = self.makejointmask(newdim, tmpY)
                # res['filenames'][nc - pre_nc] = curfilename
                
                # res['oribbox'][...,nc-pre_nc] = mat['oribbox'][...,it]            
                # res['indmap'][...,nc-pre_nc] = self.create_part_indicatormap(tmpY, part_idx, mdim, rate, filter_size, stride)
                # res['joint_indmap'][...,nc-pre_nc] = self.create_joint_indicatormap(tmpY, jtmdim, joint_filter_size, joint_stride)
                # res['joint_sample_offset'][...,nc-pre_nc] = [c, r]
                # res['is_mirror'][...,nc-pre_nc] = True
                nc = nc + 1
            t = 2 if do_mirror else 1
            if nc - pre_nc + t * sample_num > per_size or nc == totaldata:
                tmpres = self.truncated_copydic(res, nc-pre_nc)
                tmpres['data'] = tmpres['data'].reshape((-1,nc-pre_nc),order='F')
                self.meta['data_sum'] = self.meta['data_sum'] + tmpres['data'].sum(axis=1,dtype=float)
                savepath = iu.fullfile(self.savedata_info['savedir'], \
                                       self.savedata_info['savename'] + \
                                       '_' +  str(self.batch_id))
                myio.pickle( savepath, tmpres)       
                self.batch_id = self.batch_id + 1
                pre_nc = nc
    @classmethod
    def mergedicto(cls, source_d, target_d):
        for item in source_d:
            if source_d[item] != None:
                target_d[item] = source_d[item]
    @classmethod
    def check_imgdata_info(self):
        imgdata_necessary = ['imgdatapath']
        for item in self.imgdata_necessary:
            if item not in self.imgdata_info:
                raise HMLPEError('%s not found' % item)
    @classmethod
    def makejointmask(cls, dim, joints):
        joints = joints.reshape((-1,2),order='C')
        tinvalid = (joints >  np.asarray([dim[0], dim[1]])) | \
               (joints <  np.asarray([0, 0])) | np.isnan(joints)
        invalid = np.tile(tinvalid.max(axis=1).reshape((-1,1)),[1,2])
        return np.invert(invalid)
			
    @classmethod
    def create_single_part_indicatormap(cls, part, mdim, rate, filter_size, stride):
        """
        part : ((x0,y0),(x1,y1))
        mdim : The size of indicator map
        rate : if lseg > rate * l: then it will be regarded as intersection
        """
        from ipyml import geometry as geo
        resmap = np.zeros(mdim, dtype = np.bool)
        if np.any(np.isnan(part)):
            return resmap
        lx = int(min(np.ceil((part[0][0] - filter_size + 1)/stride), \
                     np.ceil((part[1][0] - filter_size + 1)/stride)))
        ly = int(min(np.ceil((part[0][1] - filter_size + 1)/stride), \
                     np.ceil((part[1][1] - filter_size + 1)/stride)))
        mx = int(min(mdim[0], 1 + max(np.floor(part[0][0]/stride), np.floor(part[1][0]/stride))))
        my = int(min(mdim[1], 1 + max(np.floor(part[0][1]/stride), np.floor(part[1][1]/stride))))
        l = np.sqrt((part[0][0] - part[1][0])**2 + (part[0][1] - part[1][1])**2)
        lx = max(0,lx)
        ly = max(0,ly)
        for x in range(lx, mx):
            for y in range(ly, my):
                rect = [(x * stride, y * stride), (x * stride + filter_size -1, y * stride + filter_size -1)]
                seg = geo.SegmentIntersectRect(part, rect)
                if len(seg)<2:
                    continue
                lseg = np.sqrt((seg[0][0] - seg[1][0])**2 + (seg[0][1] - seg[1][1])**2)            
                if (lseg > rate * l):
                    # Store in row, col format
                    resmap[y,x] = True            
        return resmap
    @classmethod
    def create_single_joint_indicatormap(cls, pt, mdim, filter_size, stride):
        """
        pt will be in (x,y) format indicating the position of jonit
        """
        resmap = np.zeros(mdim, dtype=np.bool)
        if np.any(np.isnan(pt)) or np.any(pt < 0):
            return resmap
        ux = int(min(np.floor(pt[0]/stride), mdim[1]-1))
        uy = int(min(np.floor(pt[1]/stride), mdim[0]-1))
        lx = int(max(np.ceil((pt[0]+1-filter_size)/stride), 0))
        ly = int(max(np.ceil((pt[1]+1-filter_size)/stride), 0))
        for x in range( lx, ux + 1):
            for y in range(ly, uy + 1):
                rect = [(x * stride, y * stride), (x * stride + filter_size -1, y * stride + filter_size -1)]
                if rect[0][0] <= pt[0] and pt[0] <= rect[1][0] and \
                   rect[0][1] <= pt[1] and pt[1] <= rect[1][1]:
                    resmap[y,x] = True
        return resmap
    @classmethod
    def create_part_indicatormap(cls, joints, partidx, mdim, rate, filter_size, stride):
        """
        This function will return a [npart,  mdim[0],  mdim[1]] indicator map
        In fact res[i,:,:] is a single indicator map for i-th part
        joints is the joints representation of body
        partidx represent parts as pair of joint         
        """        
        nparts = len(partidx)
        joints = joints.reshape((-1,2),order='C')
        tmap = np.zeros((nparts, mdim[0], mdim[1]), dtype=np.bool)
        for p in range(nparts):
            part = (joints[partidx[p][0]], joints[partidx[p][1]])
            tmap[p,:,:] = cls.create_single_part_indicatormap(part, mdim, rate, filter_size, stride )
        return tmap
    @classmethod
    def create_joint_indicatormap(cls, joints, mdim, filter_size, stride):
        """
        This function will caculate indicator map for each joint point
        """
        joints = joints.reshape((-1,2),order='C')
        nparts = joints.shape[0]
        tmap = np.zeros((nparts, mdim[0], mdim[1]), dtype=np.bool)
        for i,p in enumerate(joints):
            tmap[i,:,:] = cls.create_single_joint_indicatormap(p, mdim, float(filter_size), float(stride))
        return tmap
        
    @classmethod
    def get_indmapdim(cls, dim, filter_size, stride):
        """
        
        """    
        inddim = [0,0]
        inddim[0] = iu.get_conv_outputsize(dim[0], filter_size, stride)[1]
        inddim[1] = iu.get_conv_outputsize(dim[1], filter_size, stride)[1]
        return inddim
    @classmethod
    def adjust_savebuffer_shape(cls, d):
        """
        change all ndarray to (-1,ndata) format
        only support joints8 now
        """
        ndata = d['data'].shape[-1]
        d['data'] = d['data'].reshape((-1,ndata),order='F')
    @classmethod
    def prepare_savebuffer(cls, dimdic, ndata, nparts, njoints,dicjtname=None):
        """
        Allocate buffer for saving
        """
        res = dict()
        dim = dimdic['data']
        mdim = dimdic['part_indmap']
        jtmdim = dimdic['joint_indmap']
        res['data'] = np.zeros([dim[0], dim[1], dim[2], ndata], \
                                 dtype = np.uint8)
        if njoints == 8 and dicjtname is None:
            res['joints8'] = np.zeros( [njoints, 2, ndata], dtype=np.float)
        else: # historical reason. The oldest version called it joints8
            res['joints'] = np.zeros( [njoints, 2, ndata], dtype=np.float)
        res['jointmasks'] = np.zeros( [njoints, 2, ndata], dtype=np.bool) 
        res['oribbox'] = np.zeros( [4, ndata], dtype = np.float)
        res['indmap'] = np.zeros((nparts, mdim[0], mdim[1], ndata), np.bool)
        res['joint_indmap'] = np.zeros((njoints, jtmdim[0], jtmdim[1], ndata), np.bool)
        # record the offset in sampling (data augumentation) 
        res['joint_sample_offset'] = np.zeros([2, ndata], dtype=np.float) 
        res['filenames'] =[str() for x in range(ndata)]
        # indicator whether this is mirror image
        res['is_mirror'] = np.zeros((1,ndata), dtype=np.bool)
        res['is_positive'] = np.zeros((1,ndata),dtype=np.bool)
        return res
    @classmethod
    def truncated_copydic(cls, d, endidx):
        """
        copy range(0,endidx)
        """
        res = dict()
        for item in d:
            if type(d[item]) == list:
                res[item] = d[item][:endidx]
            elif type(d[item]) == np.ndarray:
                res[item] = d[item][...,:endidx]
            else:
                raise HMLPEError('unsupported type%s' % str(type(d[item])))
        return res
    @classmethod    
    def selective_copydic(cls, d_source, d_target, source_list, target_list):
        if len(source_list) != len(target_list):
            raise HMLPEError('The length of two list must be the same')

        for item in d_source:
             # if item not in d_target:
             #     print 'Warning !!!! %s is not in target' % item
             #     continue
            if type(d_source[item]) == list:
                for i in range(len(source_list)):
                    d_target[item][target_list[i]] = d_source[item][source_list[i]]
            elif type(d_source[item]) == np.ndarray:
                d_target[item][...,target_list] = d_source[item][...,source_list]
            elif type(d_source[item]) == str:
                d_target[item] = d_source[item]
            else:
                raise HMLPEError('unsupported type %s' % str(type(d_source[item])))
    @classmethod
    def check_meta_consistancy(cls, meta1, meta2):
        eqlist = ['num_vis', 'nparts', 'njoints']
        ignored = ['matdim', 'generate_type'] # negative sample doesn't have matdim
        for e in eqlist:
            if np.any(meta1[e] != meta2[e]):
                raise HMLPEError('meta filed %s are not consistent: %s != %s' % (e, meta1[e], meta2[e]))

    @classmethod
    def merge_meta(cls, meta1, meta2):
        """
        This function will merge the meta files 
        """
        if meta1 is None:
            return meta2
        if meta2 is None:
            return meta1 
        cls.check_meta_consistancy(meta1, meta2)
        meta = dict()
        HMLPE.mergedicto(meta1, meta)
        HMLPE.mergedicto(meta2, meta)
        meta['data_mean'] = (meta1['data_mean'].reshape((-1,1)) * meta1['ndata'] + \
            meta2['data_mean'].reshape((-1,1)) * meta2['ndata']) / (meta1['ndata'] + meta2['ndata'])
        meta['ndata'] = meta1['ndata'] + meta2['ndata']
        return meta
    @classmethod
    def shuffle_data(cls, source_dir, target_dir, max_per_file = 4000):
        """
        This function will shuflle all the data in source_dir
        and save it to target_dir
        """
        if source_dir == target_dir:
            raise HMLPEError('source dir can not be the same as target dir')
        import shutil
        import sys
        iu.ensure_dir( target_dir)
        shutil.copy(iu.fullfile(source_dir, 'batches.meta'), \
                    iu.fullfile(target_dir, 'batches.meta'))
        meta = myio.unpickle(iu.fullfile(source_dir, 'batches.meta'))
        ndata = meta['ndata']
        nbatch = (ndata  - 1) / max_per_file + 1
        nparts = meta['nparts']
        njoints = meta['njoints']
        newdim = meta['savedata_info']['newdim']
        filter_size = meta['savedata_info']['indmap_para']['filter_size']
        stride = meta['savedata_info']['indmap_para']['stride']
        joint_filter_size = meta['savedata_info']['indmap_para']['joint_filter_size']
        joint_stride = meta['savedata_info']['indmap_para']['joint_stride']
        mdim = cls.get_indmapdim(newdim, filter_size, stride)
        jtmdim = cls.get_indmapdim(newdim, joint_filter_size, joint_stride)
        print('There are %d data in total, I need %d batch to hold it' %(ndata, nbatch))
        print 'Begin creating empty files'
        rest = ndata
        d = cls.prepare_savebuffer({'data':newdim, 'part_indmap':mdim, \
                                       'joint_indmap': jtmdim}, max_per_file, nparts, njoints)
        cls.adjust_savebuffer_shape(d)
        for b in range(nbatch):
            cur_n = min(max_per_file, rest)
            if b != nbatch - 1:
                saved = d
            else:
                saved = cls.prepare_savebuffer({'data':newdim, 'part_indmap':mdim, \
                                       'joint_indmap': jtmdim}, cur_n, nparts, njoints)
                cls.adjust_savebuffer_shape(saved)
            myio.pickle(iu.fullfile(target_dir, 'data_batch_%d' % (b + 1)), saved)
            rest = rest - cur_n
        print 'End creating'
        allbatchfn = iu.getfilelist(source_dir, 'data_batch_\d+')
        np.random.seed(7)
        perm = range(ndata)
        np.random.shuffle(perm)
        buf_cap = 12 # store six batch at most
        nround = (nbatch - 1)/buf_cap + 1
        for rd in range(nround):
            print ('Round %d of %d' % (rd,nround))
            buf = dict()
            offset = 0
            for fn in allbatchfn:
                print( 'Processing %s' % fn )
                d = myio.unpickle(iu.fullfile(source_dir, fn))
                cur_n = d['data'].shape[-1]
                for b in range(rd * buf_cap, min(nbatch, (rd+1)*buf_cap)):
                    sys.stdout.write('\rpadding %d of %d' % (b + 1, nbatch))
                    sys.stdout.flush() 
                    sidx = b * max_per_file
                    eidx = min(ndata, sidx + max_per_file)
                    cur_idx_list = [i for i in range(cur_n) if perm[offset + i] >= sidx and perm[offset + i] < eidx]
                    if len(cur_idx_list) == 0:
                        continue
                    if not b in buf:
                        dsave = myio.unpickle(iu.fullfile(target_dir, 'data_batch_%d' % (b+1)))
                        buf[b] = dsave
                    else:
                        dsave = buf[b]
                    save_idx_list = [perm[ x + offset] - sidx for x in cur_idx_list]
                    HMLPE.selective_copydic(d, dsave, cur_idx_list, save_idx_list)
                    # myio.pickle(iu.fullfile(target_dir, 'data_batch_%d' % (b+1)), dsave)
                print 'Finished %s' % fn
                offset = offset + cur_n
            for b in range(rd * buf_cap, min(nbatch, (rd+1)*buf_cap)):
                myio.pickle(iu.fullfile(target_dir, 'data_batch_%d' % (b+1)), buf[b])
 
def analyze_joints(joints, imgdim, names = None):
    """
    imgdim is the dimension of bounding box with human object
    """
    ndata = joints.shape[-1]
    joints= joints.reshape((-1,2,ndata),order='C')
    nparts = joints.shape[0]    
    cropped = joints < 0
    cropped = (joints > np.tile(np.asarray([imgdim[0],imgdim[1]]).reshape((1,2)), (nparts, 1)).reshape((nparts, 2, 1))) | cropped
    cropped = cropped.max(axis=1).sum(axis=1)/float(ndata) * 100
    nanpart = np.isnan(joints).max(axis=1).sum(axis=1)/float(ndata)*100
    if names is None:
        names = ['head', 'LS', 'Neck', 'RS', 'Lelbow', 'Relbow', 'Lwrist', 'Rwrist']
    print '|Parts \t| occ'
    for i, name in enumerate(names):
        print '|%s \t|  %f%%|' %(name,  nanpart[i])
        #print 'Part(%s) cropped out %f \t  %f' %(name, cropped[i], nanpart[i])
    
     
def add_joint_indicatormap(data_dir, save_dir, mdim, filter_size, stride):
    """
    This function is used for generating joint indicator map for old data
    data_dir is the directory that you put all batch_data
    """
    allfile = iu.getfilelist(data_dir, 'data_batch_\d+')
    meta_path = iu.fullfile(data_dir, 'batches.meta')
    iu.ensure_dir(save_dir)
    if iu.exists(meta_path, 'file'):
        d_meta = myio.unpickle(meta_path)
        if 'savedata_info' not in d_meta:
            d_meta['savedata_info'] = dict()
            d_meta['savedata_info']['indmap_para'] = dict()
        d_meta['savedata_info']['indmap_para']['joint_filter_size'] = filter_size
        d_meta['savedata_info']['indmap_para']['joint_stride'] = stride
        myio.pickle(iu.fullfile(save_dir, 'batches.mata'), d_meta)        
    for fn in allfile:
        print 'Processing %s' % fn
        d = myio.unpickle(iu.fullfile(data_dir, fn))
        ndata = d['data'].shape[-1]
        nparts = 8
        d['joint_indmap'] = np.zeros((nparts, mdim[0], mdim[1], ndata), dtype=np.bool) 
        for i in range(ndata):
            jts = d['joints8'][...,i]
            d['joint_indmap'][...,i] = HMLPE.create_joint_indicatormap(jts, mdim, filter_size, stride)
        myio.pickle(iu.fullfile(save_dir, fn), d)

def add_part_indicatormap(data_dir, save_dir, mdim, rate, filter_size, stride):
    """
    This function is used for generating part indicator map for old data
    data_dir is the directory that you put all batch_datayes
    """
    allfile = iu.getfilelist(data_dir, 'data_batch_\d+')
    meta_path = iu.fullfile(data_dir, 'batches.meta')
    iu.ensure_dir(save_dir)
    if iu.exists(meta_path, 'file'): 
        d_meta = myio.unpickle(meta_path)
        if 'savedata_info' not in d_meta:
            d_meta['savedata_info'] = dict()
            d_meta['savedata_info']['indmap_para'] = dict()
        d_meta['savedata_info']['indmap_para']['filter_size'] = filter_size
        d_meta['savedata_info']['indmap_para']['stride'] = stride
        d_meta['savedata_info']['indmap_para']['rate'] = rate 
        myio.pickle(iu.fullfile(save_dir, 'batches.meta'), d_meta)        
    for fn in allfile:
        print 'Processing %s' % fn
        d = myio.unpickle(iu.fullfile(data_dir, fn))
        ndata = d['data'].shape[-1]
        nparts = 7
        d['indmap'] = np.zeros((nparts, mdim[0], mdim[1], ndata), dtype=np.bool) 
        for i in range(ndata):
            jts = d['joints8'][...,i]
            d['indmap'][...,i] = HMLPE.create_part_indicatormap(jts, part_idx,  mdim, rate, filter_size, stride)
        myio.pickle(iu.fullfile(save_dir, fn), d)
def merge_batch_data(data_dir_list, save_dir, is_symbolic = True, batch_start_num = 1):
    """
    This function will merge all the data_batches in data_dir into one folder
     and rename them accordining.
       Of cause, meta data will be updated 
    """
    import os
    import shutil
    iu.ensure_dir(save_dir)
    meta = None
    for ddir in data_dir_list:
        cur_meta = myio.unpickle(iu.fullfile(ddir, 'batches.meta'))    
        meta = HMLPE.merge_meta(meta, cur_meta)

    myio.pickle(iu.fullfile(save_dir, 'batches.meta'), meta)
    cur_id = batch_start_num
    for ddir in data_dir_list:
        all_file = iu.getfilelist(ddir, 'data_batch_\d+')
        print 'I find %d batches in %s' % (len(all_file), ddir)
        if is_symbolic:
            for fn in all_file:
                sn = iu.fullfile(save_dir, 'data_batch_%d' %  cur_id)
                if iu.exists(sn, 'file'):
                    os.remove(sn)
                os.symlink(iu.fullfile(ddir, fn), sn)
                cur_id = cur_id + 1
        else:
            for fn in all_file:
                shutil.copyfile(iu.fullfile(ddir, fn), iu.fullfile(save_dir, 'data_batch_%d' %  cur_id))
                cur_id = cur_id + 1
def shuffle_data(source_dir, target_dir, max_per_file = 4000):
    """
    This function will shuflle all the data in source_dir
    and save it to target_dir
    """
    if source_dir == target_dir:
        raise HMLPEError('source dir can not be the same as target dir')
    import shutil
    import sys
    iu.ensure_dir( target_dir)
    shutil.copy(iu.fullfile(source_dir, 'batches.meta'), \
                iu.fullfile(target_dir, 'batches.meta'))
    meta = myio.unpickle(iu.fullfile(source_dir, 'batches.meta'))
    ndata = meta['ndata']
    nbatch = (ndata  - 1) / max_per_file + 1
    nparts = meta['nparts']
    njoints = meta['njoints']
    newdim = meta['savedata_info']['newdim']
    filter_size = meta['savedata_info']['indmap_para']['filter_size']
    stride = meta['savedata_info']['indmap_para']['stride']
    joint_filter_size = meta['savedata_info']['indmap_para']['joint_filter_size']
    joint_stride = meta['savedata_info']['indmap_para']['joint_stride']
    mdim = HMLPE.get_indmapdim(newdim, filter_size, stride)
    jtmdim = HMLPE.get_indmapdim(newdim, joint_filter_size, joint_stride)
    print('There are %d data in total, I need %d batch to hold it' %(ndata, nbatch))
    print 'Begin creating empty files'
    rest = ndata
    d = HMLPE.prepare_savebuffer({'data':newdim, 'part_indmap':mdim, \
                                       'joint_indmap': jtmdim}, max_per_file, nparts, njoints)
    HMLPE.adjust_savebuffer_shape(d)
    for b in range(nbatch):
        cur_n = min(max_per_file, rest)
        if b != nbatch - 1:
            saved = d
        else:
            saved = HMLPE.prepare_savebuffer({'data':newdim, 'part_indmap':mdim, \
                                       'joint_indmap': jtmdim}, cur_n, nparts, njoints)
            HMLPE.adjust_savebuffer_shape(saved)
        myio.pickle(iu.fullfile(target_dir, 'data_batch_%d' % (b + 1)), saved)
        rest = rest - cur_n
    print 'End creating'
    allbatchfn = iu.getfilelist(source_dir, 'data_batch_\d+')
    np.random.seed(7)
    perm = range(ndata)
    np.random.shuffle(perm)
    buf_cap = 12 # store six batch at most
    nround = (nbatch - 1)/buf_cap + 1
    for rd in range(nround):
        print ('Round %d of %d' % (rd,nround))
        buf = dict()
        offset = 0
        for fn in allbatchfn:
            print( 'Processing %s' % fn )
            d = myio.unpickle(iu.fullfile(source_dir, fn))
            cur_n = d['data'].shape[-1]
            for b in range(rd * buf_cap, min(nbatch, (rd+1)*buf_cap)):
                sys.stdout.write('\rpadding %d of %d' % (b + 1, nbatch))
                sys.stdout.flush() 
                sidx = b * max_per_file
                eidx = min(ndata, sidx + max_per_file)
                cur_idx_list = [i for i in range(cur_n) if perm[offset + i] >= sidx and perm[offset + i] < eidx]
                if len(cur_idx_list) == 0:
                    continue
                if not b in buf:
                    dsave = myio.unpickle(iu.fullfile(target_dir, 'data_batch_%d' % (b+1)))
                    buf[b] = dsave
                else:
                    dsave = buf[b]
                save_idx_list = [perm[ x + offset] - sidx for x in cur_idx_list]
                HMLPE.selective_copydic(d, dsave, cur_idx_list, save_idx_list)
                # myio.pickle(iu.fullfile(target_dir, 'data_batch_%d' % (b+1)), dsave)
            print 'Finished %s' % fn
            offset = offset + cur_n
        for b in range(rd * buf_cap, min(nbatch, (rd+1)*buf_cap)):
            myio.pickle(iu.fullfile(target_dir, 'data_batch_%d' % (b+1)), buf[b])
                
def test_addmeta(metadir):
    metapath = iu.fullfile(metadir, 'batches.meta')
    d_meta = myio.unpickle(metapath)
    d_meta['ind_dim'] = dict()
    d_meta['ind_dim']['part_indmap'] = (8,8)
    d_meta['ind_dim']['joint_indmap'] = (8,8)
    myio.pickle(metapath, d_meta)    
