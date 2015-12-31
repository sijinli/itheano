import scipy.io as sio
import myio
import iutils as iu
import numpy as np
from hmlpe import HMLPEError
from hmlpe import HMLPE
import hmlpe as hp
def flip_pose_joint(dim, points):
    raise HMLPEError('I haven'' finished it yet')

jointnames = ['root', \
              'Rleg0', 'Rleg1', 'Rleg2', \
              'Lleg0', 'Lleg1', 'Lleg2', \
              'torso', 'neck', 'nose', 'headu', \
              'Larm0', 'Larm1', 'Larm2', \
              'Rarm0', 'Rarm1', 'Rarm2']
nice_jointnames = ['root', \
                   'Rhip', 'Rknee', 'Rankle', \
                   'Lhip', 'Lknee', 'Lankle',\
                   'torso', 'neck', 'nose', 'headu', \
                   'Lshoulder', 'Lelbow', 'Lwrist', \
                   'Rshoulder', 'Relbow', 'Rwrist']                   
jointpartnames = ['unfinished' for x in range(2)]
ch_list = [\
           #0         1    2   3
            [1,4,7],  [2],[3],[], \
           #4         5    6   7 
            [5],     [6], [ ], [8,11,14], \
           #8         9    10  11
            [9],     [10], [], [12],\
           #12        13   14, 15   16
            [13],     [], [15],[16],[]]
matlab_angle_useful_idx = [4,5,6,7,8,9,11, \
                           13,14,15,\
                           17,\
                           19,20,21,\
                           23,\
                           25,26,27,29,\
                           31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,\
                           50,\
                           52,53,54,\
                           61,62,63,64,65,66,\
                           68,\
                           70,71,72];
matlab_body_idx = [1,2,3,4,7,8,9,13,14,15,16,18,19,20,26,27,28]
body_idx = [x -1 for x in matlab_body_idx]
angle_useful_idx = [x - 1 for x in matlab_angle_useful_idx]
part_idx3d = [(0, 1), (1, 2), (2, 3), (0, 6), (6, 7), (7, 8), (0, 12), (12, 13), (13, 14), (14, 15), (13, 17), (17, 18), (18, 19), (13, 25), (25, 26), (26, 27)]
body_list3d =[0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27] #
body_mocap_to_mono_map = dict()
for i,p in enumerate(body_list3d):
    body_mocap_to_mono_map[p] = i
part_idx = [ ( body_mocap_to_mono_map[x[0]],body_mocap_to_mono_map[x[1]]) for x in part_idx3d]
part_name = ['R_Pelvis', 'R_Thigh', 'R_Calf', 'L_Pelvis', 'L_Thigh', 'L_Calf', \
             'torso_lower','torso_upper', 'head_lower', 'head_upper', 'L_shoulder',\
             'L_upper_arm', 'L_forearm', 'R_shoulder', 'R_upper_arm', 'R_forearm']
fieldlist = ['data', 'labels', 'filenames', 'joints', 'oribbox', 'indmap', 'indmap_para', 'mocap_joints3d', 'mono_joints3d'] # NEEDED TO BE CHECED
default_imgdata_info = {}
default_indmap_para = {'filter_size':30.0, 'stride':12.0, 'rate':0.3, \
                       'joint_filter_size':30.0, 'joint_stride':12.0}

default_savedata_info = {'sample_num':1 , \
                         'neg_sample_num':100,\
                         'max_batch_size':4000, \
                         'newdim':(112,112,3), \
                         'start_patch_id':1,\
                         'savename':'data_batch', \
                         'indmap_para':default_indmap_para, \
                         'flipfunc':flip_pose_joint, \
                         'jointnames':jointnames,\
                         'part_idx':part_idx} 
class H36MHMLPE(HMLPE):
    def __init__(self):
        self.imgdata_info = default_imgdata_info
        self.savedata_info = default_savedata_info
        self.flip_joints = flip_pose_joint
        self.meta = dict()
    def get_fieldpool_for_positive_mat_data(self):
        FP = dict()
        njoints = self.meta['njoints']
        FP['Y2dname'] = 'Y2d'
        FP['dicjtname'] = 'joints' # no need to support old version
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
        res['oribbox'][...,idx] = FP['mat']['oribbox'][...,FP['matidx']]
        res['indmap'][...,idx] = self.create_part_indicatormap(FP['Y'],self.meta['savedata_info']['part_idx'], FP['mdim'], \
            FP['rate'], FP['filter_size'], FP['stride'])
        res['joint_indmap'][...,idx] = self.create_joint_indicatormap(FP['Y'], FP['jtmdim'], FP['joint_filter_size'], FP['joint_stride'])
        res['joint_sample_offset'][...,idx] = [FP['c'], FP['r']]
        res['is_mirror'][...,idx] = is_mirror
        ### H36M specific features
        res['camera'][...,idx] = FP['mat']['info']['camera'][0][0][0][0]
        res['subject'][...,idx] = FP['mat']['info']['subject'][0][0][0][0]
        res['action'][...,idx] = FP['mat']['info']['action'][0][0][0][0]
        res['subaction'][...,idx] = FP['mat']['info']['subaction'][0][0][0][0]
        res['mocap_joints3d'][...,idx] = FP['mat']['Y3d'][...,FP['matidx']]
        res['mono_joints3d'][...,idx]  = FP['mat']['Y3d_mono_body'][...,FP['matidx']]
    @classmethod    
    def prepare_savebuffer(cls,dimdic, ndata, nparts, njoints,dicjtname=None):
        res = HMLPE.prepare_savebuffer(dimdic, ndata, nparts, njoints, dicjtname)
        # 'mocap_joints3d', 'mono_joints3d'
        res['mocap_joints3d'] = np.zeros((96, ndata),dtype=np.float)
        res['mono_joints3d'] = np.zeros((njoints*3, ndata), dtype=np.float)
        res['camera'] = np.zeros((1, ndata),dtype=np.int)
        res['subject'] = np.zeros((1,ndata),dtype=np.int)
        res['action'] = np.zeros((1,ndata),dtype=np.int)
        res['subaction'] = np.zeros((1,ndata),dtype=np.int)
        return res

def get_id_from_imagename(s_list):
    #   imgsave_format = 's_%02d_act_%02d_subact_%02d_ca_%02d_%06d.jpg';
    # Note the input is a list
    import re
    def proc(s):
        name = iu.getpath(s,0)
        l = re.split('_|\.',name)
        return [int(l[k]) for k in [1,3,5,7,8]]
    return map(proc, s_list)
