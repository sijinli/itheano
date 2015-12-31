import numpy as np
class DHMLPEFeaturesError(Exception):
    pass

class DHMLPEFeatures:
    def __init__(self, input_fields = None, output_fields = None):
        self.input_fields = input_fields
        self.output_fields = output_fields
    def calc_features(self, d):
         pass
    def add_features(self, d):
        pass
class Relative3D(DHMLPEFeatures):
    def __init__(self, input_fields, output_fields):
        DHMLPEFeatures.__init__(self, input_fields, output_fields)
        self.feature_name = 'Relative3D'
    def calc_features(self,d):
        if len(self.input_fields) != 1 or len(self.output_fields)!=1:
            raise DHMLPEFeaturesError('The number of input is invalid for Feature %s' % self.feature_name)
        if not self.input_fields[0] in d:
            raise DHMLPEFeaturesError('Required field %s missing' % self.input_fields)
        inputs = d[self.input_fields[0]]
        ndata = inputs.shape[-1]
        inputs = inputs.reshape((3,-1,ndata),order='F')
        outputs = inputs.copy()
        root_data = inputs[:,0,:].reshape((3,1,ndata),order='F')
        outputs = (inputs - root_data).reshape((-1,ndata),order='F')
        return [outputs]
    def add_features(self,d):
        r = self.calc_features(d)
        d[self.output_fields[0]] = r[0]
        
class RelativeSkel3D(DHMLPEFeatures):
    def __init__(self, input_fields, output_fields):
        DHMLPEFeatures.__init__(self, input_fields, output_fields)
        import iread.h36m_hmlpe as h36m
        self.body_idx = h36m.part_idx
        self.feature_name = 'RelativeSkel3D'
    def calc(self,x):
        """
        x is a 3 x n matrix
        """
        r = x.copy()
        # for p in self.body_idx:
        #     r[...,p[1]] -= x[...,p[0]]
        # be carefull with -=, need to copy first 
        r[...,[t[1] for t in self.body_idx]] -= x[...,[t[0] for t in self.body_idx]]
        r[...,0] = 0
        return r.reshape((3,-1,1),order='F')
    def calc_features(self,d):
        if len(self.input_fields) != 1 or len(self.output_fields)!=1:
            raise DHMLPEFeaturesError('The number of input is invalid for Feature %s') % self.feature_name
        if not self.input_fields[0] in d:
            raise DHMLPEFeaturesError('Required field %s missing' % self.input_fields)
        inputs = d[self.input_fields[0]]
        ndata = inputs.shape[-1]
        inputs = inputs.reshape((3,-1,ndata),order='F')
        outputs = np.concatenate(map(lambda k:self.calc(inputs[...,k]),\
                                     range(ndata)),axis=2)
        return [outputs]
    def add_features(self,d):
        r = self.calc_features(d)
        d[self.output_fields[0]] = r[0]

class RelativeSkel3DEva(RelativeSkel3D):
    def __init__(self, input_fields, output_fields):
        RelativeSkel3D.__init__(self, input_fields, output_fields)
        import humaneva_meta as hm
        self.body_idx = hm.limbconnection
        self.feature_name = 'RelativeSkel3DEva'
    
class MaximumAbs(DHMLPEFeatures):
    def __init__(self, input_fields, output_fields, params = None):
        DHMLPEFeatures.__init__(self, input_fields, output_fields)
        self.feature_name = 'MaximumAbs'
        if params is None:
            self.scale = 1
        else:
            self.scale = params[0]
    def calc_features(self,d):
        if len(self.input_fields) != 1 or len(self.output_fields)!=1:
            raise DHMLPEFeaturesError('The number of input is invalid for Feature %s' % self.feature_name)
        r = np.max(np.abs(d[self.input_fields[0]].flatten())) * self.scale
        return [r]
    def add_features(self,d):
        l = self.calc_features(d)
        d[self.output_fields[0]] = l[0]

class LimbLength3D(DHMLPEFeatures):
    def __init__(self, input_fields, output_fields):
        DHMLPEFeatures.__init__(self, input_fields, output_fields)
        import iread.h36m_hmlpe as h36m
        self.body_idx = h36m.part_idx
        self.feature_name = 'LimbLength3D'
        self.feature_dim = len(self.body_idx)
    def norm(self,x):
        return np.sqrt(np.sum(x[:]**2))
    def calc(self,x):
        """
        x is a [3 , num_points] matrix
        return r is a [num_points , 1] matrix
        """
        r = [self.norm(x[...,p[0]] - x[...,p[1]]) for p in self.body_idx]
        return np.asarray(r).reshape((self.feature_dim, 1))
    def calc_features(self,d):
        if len(self.input_fields) != 1 or len(self.output_fields)!=1:
            raise DHMLPEFeaturesError('The number of input is invalid for Feature %s') % self.feature_name
        if not self.input_fields[0] in d:
            raise DHMLPEFeaturesError('Required field %s missing' % self.input_fields)
        ndata = d[self.input_fields[0]].shape[-1]
        inputs = d[self.input_fields[0]].reshape((3,-1,ndata),order='F')
        outputs = np.concatenate(map(lambda k:self.calc(inputs[...,k]), \
                                     range(ndata)), axis=1)
        return [outputs]
    def add_features(self,d):
        l = self.calc_features(d)
        d[self.output_fields[0]] = l[0]

def convert_relskel2rel_base(relskel_features, part_idx, root_coor = 0, root_idx = 0):
    ndata = relskel_features.shape[-1]
    # print 'There are %d data' % ndata
    # print 'The shape is ', relskel_features.shape
    res = relskel_features.copy().reshape((3,-1,ndata),order='F')
    if root_idx >=0:
        res[:,root_idx,:] = 0
    ## Here I am pretty sure part_idx is sorted by topology
    for p in part_idx:
        res[:,p[1],:] = res[:,p[1],:] + res[:,p[0],:]
    if type(root_coor) == np.ndarray:
        res = res + root_coor.reshape((3,1,ndata),order='F')
    else:
        res = res + root_coor
    return res

def convert_relskel2rel(relskel_features, root_coor = 0):
    """
    root_coor = 0 or a np.ndarray with shape[-1] = ndata
    """
    import iread.h36m_hmlpe as h36m
    return convert_relskel2rel_base(relskel_features, h36m.part_idx, root_coor)


def calc_pairwise_diff(mat, group=1):
    """
    mat is a group * num_group by n matrix
    The result will be a group * num_group * num_group  by n matrix
    The points with a group will be treated as one point
    """
    if len(mat.shape)!=2:
        raise DHMLPEFeaturesError('The shape of math')
    ndata = mat.shape[-1]
    feature_dim = mat.shape[0]
    if feature_dim % group != 0:
        raise DHMLPEFeaturesError('Feature dimension %d is not divisible by group %d ' % (feature_dim, group))
    num_group = feature_dim / group
    return np.tile(mat, [num_group,1]) - \
       np.tile(mat.reshape((group, num_group, ndata), order='F'), [num_group,1,1]).reshape((-1,ndata),order='F')    

def convert_pairwise2rel_simple(mat, group):
    """
    This function only crop the data and return the first several 
    """
    dim, ndata = mat.shape
    n_joints = np.int(np.sqrt(dim/group))
    if (n_joints **2) * group != dim:
        raise DHMLPEFeaturesError('The first dimension/group should be a square number')
    return mat[:n_joints * group, :]

def extract_pairwise_dims(sp, group):
    if len(sp) != 2:
        raise DHMLPEFeaturesError('The shape should contains only two elements')
    ndata = sp[-1]
    pairwise_feature_dim = sp[0]
    number_group = np.int(np.sqrt(pairwise_feature_dim / group))
    if (number_group ** 2) * group != pairwise_feature_dim:
        raise DHMLPEFeaturesError('The dimension doesn''t match')
    return ndata, pairwise_feature_dim, number_group
def pairwise_rel_recover(mat, group=1):
    """
    Assume mat is the
    group * num_group * num_group, ndata matrix
    """
    ndata, pairwise_feature_dim, num_group = extract_pairwise_dims(mat.shape, group)
    index_arr = np.asarray(range(group) * num_group, dtype=np.int) \
      + (np.asarray(range(group * num_group),dtype=np.int)/group )* (group * num_group)
    tmp = np.tile(mat[index_arr,:].reshape((group,num_group,ndata),order='F'), [num_group, 1,1]).reshape((group*num_group*num_group,ndata),order='F')
    tmp[:group*num_group,:] = 0
    return mat - tmp
    
def pairwise_order2recover(mat, group=1):
    """
    Note: This is not real order 2 recover.
    In fact,
    all the results will be
    rel2_i_(J_k) = rel1_root(i) + rel1_i(J_k) 
    """
    ndata, pairwise_feature_dim, num_group = extract_pairwise_dims(mat.shape, group)
    tmp = np.tile(mat[:group*num_group, :].reshape((group, num_group, ndata),order='F'),\
                  [num_group, 1, 1]).reshape((-1, ndata),order='F')
    tmp[:num_group*group,:] = 0
    return mat + tmp
    
    

feature_dic = {'Relative3D': Relative3D, 'MaximumAbs':MaximumAbs, 'RelativeSkel3D':RelativeSkel3D, 'LimbLength3D': LimbLength3D, 'RelativeSkel3DEva':RelativeSkel3DEva}
