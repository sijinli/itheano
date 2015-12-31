"""
Dont' import matplotlib here
"""
import iutils as iu
import iread.myio as mio
import numpy as np
import time
def show_3d_skeleton(joints, limbs, params = None):
    """
    joints is n x 3 matrix
    """
    from matplotlib import pyplot as plt
    import imgproc
    ax = plt.gca()
    if not params:
        params = {}
    if 'elev' in params and 'azim' in params:
        # ax.view_init(elev=-89, azim=-107.) # for camera one
        ax.view_init(elev=params['elev'], azim=params['azim'])
    n_lim = len(limbs)
    joints = joints.reshape((-1, 3), order='C')
    if 'order' in params:
        tdic = {'x':0,'y':1,'z':2}
        assert(params['order'] in tdic)
        order_idx = tdic[params['order']]
        limb_order = sorted(range(n_lim), key=lambda k:(joints[limbs[k][0]][order_idx] + joints[limbs[k][1]][order_idx]),reverse=True)
    else:
        limb_order = range(n_lim)

    c = params['color'] if 'color' in params else imgproc.get_color_list(n_lim)
    lw = params['linewidth'] if 'linewidth' in params else 8
    for k in limb_order:
        l = limbs[k]
        j1 = joints[l[0]]
        j2 = joints[l[1]]
        x,y,z = [j1[0], j2[0]], [j1[1], j2[1]], [j1[2], j2[2]]
        ax.plot(x, y, z, c = c[k],linewidth=lw)
def show_2d_skeleton(joints, limbs, params = None):
    """
    joints is n x 2 matrix
    
    """
    import imgproc
    ax = plt.gca()
    if not params:
        params = {}
    joints = joints.reshape((-1, 2), order='C')
    n_lim = len(limbs)
    c = params['color'] if 'color' in params else imgproc.get_color_list(n_lim)
    for i,l in enumerate(limbs):
        j1 = joints[l[0]]
        j2 = joints[l[1]]
        x,y = [j1[0], j2[0]], [j1[1], j2[1]]
        ax.plot(x, y, c = c[i])
def extract_batch_num(s):
    """
    s should batch_feature_number_anyotherstring
    """
    p = s.rfind('_')
    if p == -1 or len(s) == 0:
        return -1
    else:
        return int(s[p+1:])
def imerge(X,Y):
    if X is None:
        return Y
    if type(X) is list:
        return X + Y
    elif type(X) is np.ndarray:
        return np.concatenate([X,Y], axis=-1)
def collect_feature_meta(folder, re_exp='batch_feature_\w+$'):
    allfile = sorted(iu.getfilelist(folder, re_exp), key=lambda x:extract_batch_num(x))
    feature_list_lst = []
    feature_dim = None
    indexes_lst = []
    feature_names = None
    if len(allfile) == 0:
        return dict()
    for f in allfile:
        print f
        p =  iu.fullfile(folder, f)
        d = mio.unpickle(p)
        feature_list_lst += [d['feature_list']]
        if feature_dim:
            if feature_dim!= d['feature_dim']:
                raise Exception('feature dim inconsistent')
        else:
            feature_dim = d['feature_dim']
        indexes_lst += [d['info']['indexes']]
    indexes = np.concatenate(indexes_lst)
    n_feature, n_batch = len(feature_dim), len(allfile)
    feature_list = [np.concatenate([feature_list_lst[i][k] for i in range(n_batch)],
                                   axis=-1)
                    for k in range(n_feature)]
    return {'feature_list':feature_list, 'feature_dim':feature_dim, 'info':{'indexes':indexes,
                                                                        'feature_names':d['info']['feature_names']}}
def collect_feature(folder, item, re_exp='batch_feature_\w+$'):
    allfile = sorted(iu.getfilelist(folder, re_exp), key=lambda x:extract_batch_num(x))
    l = []
    for f in allfile:
        p =  iu.fullfile(folder, f)
        d = mio.unpickle(p)
        l = l + [d[item]]
    return np.concatenate(l, axis=1)
def calc_MPJPEfromerr(err,num_joints):
    ndata = err.shape[-1]
    err = err.reshape((-1,num_joints, ndata),order='F')
    t1 = np.sum(np.sqrt(np.sum(err**2,axis=0)).flatten())/num_joints
    return [t1, ndata, t1/ndata]
def calc_all_MPJPEfromerr(err, num_joints):
    ndata = err.shape[-1]
    err = err.reshape((-1,num_joints, ndata),order='F')
    t1 = np.sum(np.sqrt(np.sum(err**2,axis=0)),axis=0)/num_joints
    return t1

def calc_mpjpe_from_residual(x, num_joints):
    """
    return 1 x n array
    """
    ndata = x.shape[-1]
    x = x.reshape((-1,num_joints,ndata),order='F')
    return np.sum(np.sqrt(np.sum(x**2,axis=0)),axis=0).reshape((1,ndata))/num_joints
def calc_jpe_from_residual(x, num_joints):
    """
    return num_joints x n array
    """
    ndata = x.shape[-1]
    x = x.reshape((-1, num_joints, ndata), order='F')
    return np.sqrt(np.sum(x**2,axis=0))
def calc_tanh_score(z, factor, offset):
    def tanh_score(x):
        return 1 - np.tanh(x)
    return tanh_score(np.maximum(z-offset,0)/np.float(factor))
def calc_RBF_score(z, sigma, group_size = 3):
    """
    z is a group_size, num_group, ndata array
    """
    ndata = z.shape[-1]
    z = z.reshape((group_size, -1, ndata),order='F')
    num_group = z.shape[1]
    s2 = sigma**2
    return np.sum(np.exp(-np.sum(z**2,axis=0) / s2), axis=0).reshape((1,ndata),order='F')/num_group
    
def generate_gauss_noise(num_joints, sigma, ndata,dim = 3, ignore_root=True):
    """
    No noise on root prediction
    """
    noise = np.random.normal(0, sigma, ndata * dim * num_joints).reshape((dim*num_joints,ndata))
    if ignore_root:
        noise[0:dim,:] = 0
    return noise

def generate_GMM_noise(num_joints, sigma, prob, ndata, dim=3, ignore_root=True):
    """
    sigma is array with k element
    prob is array iwth k element
    """
    # start_time = time.time()
    noise_list = [generate_gauss_noise(num_joints, s, ndata, dim, ignore_root) for s in sigma]
    res = np.zeros((dim, num_joints, ndata))
    dprob = np.cumsum(np.asarray(prob).flatten()).tolist()
    dprob = [0] + dprob
    sample =  np.random.uniform(0,1,ndata)
    for i,s in enumerate(sigma):
        ind = sample < dprob[i + 1]
        ind = np.logical_and(ind,  sample >= dprob[i])
        if sum(ind)> 1:
            res[..., ind] = noise_list[i][...,ind]
    return res
def smooth_prob(prob, a):
    prob = np.asarray(prob)
    t = prob + a
    return t/(np.sum(t))
def calc_sample_prob(scores, offset=0,reverse=False):
    """
    scores should be sorted
    offset <=np.min(scores)
    """
    scores = np.sort(scores.flatten())
    n = scores.size
    l = [scores[0]-offset  if k == 0 else (scores[k] - scores[k-1]) for k in range(n)]
    s = sum(l)
    if s == 0:
        res= [1.0/n] * n
    else:
        res= [x/s for x in l]
    if reverse:
        res = res[::-1]
    return res
def score2indextable_base(scores, N,reverse=False):
    """
    THe input is a n x 1 array or list
    N should be larger than n

    all elements in score should be > 0
    """
    scores = np.sort(np.asarray(scores).flatten())
    n = scores.size
    if N < n:
        raise Exception('N should be larger than n')
    l = [scores[0]  if k == 0 else (scores[k] - scores[k-1]) for k in range(n)]
    has_neg = sum([ x < -1e-20 for x in l]) > 0
    if has_neg:
        raise Exception('score should be sorted')
    arr_l = np.asarray(l,dtype=np.float)
    s = np.sum(arr_l)
    if s == 0:
        arr_l = np.linspace(0,1, n + 1)[1:]
    else:
        arr_l = arr_l/s

    end_index = np.floor(np.cumsum(arr_l) * N)
    print end_index
    index = np.zeros((1,N),dtype=np.int)
    pre = 0
    for i in range(n):
        index[0,pre:np.int(end_index[i])] = i
        pre = end_index[i]
        if pre == N:
            break
    if end_index[n-1] < N:
        index[0,np.int(end_index[n-1]):] = n - 1
    if reverse:
        index = n - 1 - index
    return index
def score2indextable(scores, N, reverse=False):
    n = scores.size
    if N < n:
        raise Exception('N should be as least larger than n')
    l = list(range(n))
    l1 = score2indextable_base(scores, N - n, reverse).flatten().tolist()
    l_final = sorted(l + l1)
    return np.asarray(l_final)

def calc_relskel(x, num_joints):
    """
    x is dim_coor x num_joints x ndata
    """
    import iread.h36m_hmlpe as h36m
    ndata = x.shape[-1]
    x = x.reshape((-1,num_joints, ndata),order='F')
    r = x.copy()
    part_idx = h36m.part_idx
    r[:,[t[1] for t in part_idx],:] -= x[:,[t[0] for t in part_idx],:]
    r[:,0,:] = 0
    return r.reshape((-1, num_joints,ndata),order='F')


    
# class Cutils(object):  
# 	def __init__(self):
# 		self.model = __import__('cutils')
# 	def convert_angle2pose(self,skel, channel, subject_ids):
# 		ndata = np.int(channel.shape[-1])
# 		#print skel.shape, channel.shape, subject_ids.shape
# 		#print skel.dtype, channel.dtype, subject_ids.dtype
# 		skel = np.require(skel.flatten(order='F'),dtype=np.float64)
# 		channel = np.require(channel.flatten(order='F'),dtype=np.float64) 
# 		subject_ids = np.require(subject_ids.flatten(order='F'),dtype=np.int)
#                 l_skel= [np.float64(x) for x in skel]
#                 l_channel = [np.float64(x) for x in channel]
# 		l_id = [np.int(x) for x in subject_ids]
# 		res = self.model.convert_angle2pose([[l_skel, l_channel, l_id],[np.int(32),np.int(8),ndata,np.int(78)]])
#                 # res = np.concatenate([self.model.convert_angle2pose([[l_skel, l_channel[k*78:(k+1)*78], l_id[k*78:(k+1)*78]],[np.int(32),np.int(8),ndata,np.int(78)]]).reshape((96,1),order='F') for k in range(ndata)], axis=1)
# 		return res.reshape((96,ndata),order='F')
#         def calc_mpjpe_from_residual(self, residuals, num_joints):
#             ndata = residuals.shape[-1]
#             return self.model.calc_mpjpe_from_residual(residuals.reshape((-1,ndata),order='F'), \
#                                                    num_joints)
