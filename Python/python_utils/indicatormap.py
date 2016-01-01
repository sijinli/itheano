import iutils as iu
import numpy as np
class BasicIndicatorMap:
    def __init__(self):
        pass
    @classmethod
    def generate_ticks(cls, l, fsize, stride):
        ## Add boundary, so that it is able to deal with "missing points"
        allticks = range(0, l,stride) + \
          [x + fsize for x in range(0,l,stride) if x + fsize < l]
        allticks += [-1, l]
        return sorted(list(set(allticks)))
    @classmethod
    def locate_index(cls, x, ticks):
        n = len(ticks)
        while (n >= 1):
            if ticks[n-1] <= x:
                return n-1
            n = n - 1
        return 0
    def get_joints_indicatormap(pts):
        """
        pts = n x 2 matrix
        Assume it is python 0-index
        """
        pass
    @classmethod
    def get_indmapdim(cls, dim, filter_size, stride):
        """
        
        """
        inddim = [0,0]
        inddim[0] = iu.get_conv_outputsize(dim[0], filter_size, stride)[1]
        inddim[1] = iu.get_conv_outputsize(dim[1], filter_size, stride)[1]
        return inddim
class IndicatorMap(BasicIndicatorMap):
    def __init__(self, imgdim, filter_size, stride, create_lookup_table = True):
        """
        In this indicator map
        All the map value is map[r,c] = True or False
        For the interface, it accept pt = (x,y) for query
        """
        self.imgdim = imgdim
        self.mdim = self.get_indmapdim(imgdim, filter_size, stride)
        self.filter_size = filter_size
        self.stride = stride
        self.r_ticks = self.generate_ticks(imgdim[0], filter_size, stride) 
        self.c_ticks = self.generate_ticks(imgdim[1], filter_size, stride)
        self.num_index = len(self.r_ticks) * len(self.c_ticks)
        self.create_lookup_table = create_lookup_table
        if create_lookup_table:
            ## lookup table are always 0-index
            self.lookup_table = self.create_indicator_map_table(self.r_ticks, \
                                                                self.c_ticks, \
                                                                self.mdim, \
                                                                filter_size, \
                                                               stride)
        else:
            self.lookup_table = None
    def idisp(self):
        l = ['mdim', 'num_index', 'r_ticks', 'c_ticks']
        for x in l:
            print '%s is '  % x
            print getattr(self, x)
    def pt2index(self, pt):
        # pt is n x 2 matrix
        # index generation will be in F order
        nr = len(self.r_ticks)
        f = lambda x: self.locate_index(x[0], self.c_ticks) * nr+  \
          self.locate_index(x[1], self.r_ticks)
        return np.asarray(map(f, pt))
    def get_joints_indicatormap(self, pts):
        """
        pts = n x 2 matrix
        """
        """
        Assume it is python 0-index
        """
        pts = np.asarray(pts)
        pts = pts.reshape((-1,2),order='C')
        njoints = pts.shape[0]
        if self.lookup_table is None:
            # I haven't finish this part;
            # I think everyone should build the lookup table
            return None
        else:
            idx = self.pt2index(pts)
            return self.lookup_table[...,idx]
    @classmethod
    def create_joint_indicatormap(cls, pt, mdim, filter_size, stride):
        """
        pt: (x,y) coordinates of the bounding box
        mdim: (nr, nc) 
        """
        # Always assume the index starts at 0
        # Just leave all those preprocess outside this code
        resmap = np.zeros(mdim, dtype=np.bool)
        if np.any(np.isnan(pt)):
            return resmap
        stride = np.float(stride)
        ux = int(min(np.floor(pt[0]/stride), mdim[1]-1))
        uy = int(min(np.floor(pt[1]/stride), mdim[0]-1))
        lx = int(max(np.ceil((pt[0]+1-filter_size)/stride), 0))
        ly = int(max(np.ceil((pt[1]+1-filter_size)/stride), 0))
        for x in range( lx, ux + 1):
            for y in range(ly, uy + 1):
                rect = [(x * stride, y * stride), (x * stride + filter_size -1, y * stride + filter_size -1)]
                # Only count open range [l, u)
                if rect[0][0] <= pt[0] and pt[0] < rect[1][0] and \
                   rect[0][1] <= pt[1] and pt[1] < rect[1][1]:
                    resmap[y,x] = True
        return resmap
    @classmethod
    def get_nice_indicatormap(cls, ind, per_dim, margin=(1,1)):
        """
        ind is a np.ndarray
        """
        outdim = (ind.shape[0] * (per_dim[0] + margin[0]) - margin[0], \
                ind.shape[0] * (per_dim[0] + margin[0]) - margin[0])
        res = np.zeros(outdim, dtype=np.float)
        for r in range(ind.shape[0]):
            for c in range(ind.shape[1]):
                if ind[r,c]:
                    sr = r * (per_dim[0] + margin[0])
                    sc = c * (per_dim[1] + margin[1])
                    res[sr:sr+per_dim[0], sc:sc+per_dim[1]] = 1
        return res

    @classmethod
    def create_indicator_map_table(cls, r_ticks, c_ticks, mdim, filter_size, stride):
        """
        This function will return a table for lookup the indicator map
        
        """
        k = 0
        n = len(c_ticks) * len(r_ticks)
        dim = list(mdim) + [n]
        map_table = np.zeros(dim,dtype=np.bool) 
        for x in c_ticks:
            for y in r_ticks:
                cur_pt = np.asarray((x,y))
                map_table[...,k] = cls.create_joint_indicatormap(cur_pt, \
                                                                 mdim, \
                                                                 filter_size, \
                                                                 stride)
                k = k + 1
        return map_table
class  MaxIndicatorMap(BasicIndicatorMap):
    def __init__(self, imgdim, filter_size, stride, create_lookup_table = True):
        """
        In this indicator map
        All the map value is map[r,c] = True or False
        For the interface, it accept pt = (x,y) for query
        For this dataprovider, every input will activate the cloest filter
        """
        self.imgdim = imgdim
        self.mdim = self.get_indmapdim(imgdim, filter_size, stride)
        self.filter_size = filter_size
        self.stride = stride
        self.r_ticks = self.generate_ticks(imgdim[0], filter_size, stride) 
        self.c_ticks = self.generate_ticks(imgdim[1], filter_size, stride)
        self.num_index = (len(self.r_ticks) + 1) * (len(self.c_ticks)+1)
        self.create_lookup_table = create_lookup_table
        if create_lookup_table:
            ## lookup table are always 0-index
            self.lookup_table = self.create_indicator_map_table(self.r_ticks, \
                                                                self.c_ticks, \
                                                                self.mdim, \
                                                                filter_size, \
                                                               stride)
        else:
            self.lookup_table = None
    @classmethod
    def create_indicator_map_table(cls, r_ticks, c_ticks, mdim, filter_size, stride):
        """
            This function will create a table for lookup the indicatormap
            The map is still in map order
            map[r,c] = True
            mdim = [nr-1, nc-1] 
        """
        nr = len(r_ticks)
        nc = len(c_ticks)
        dim = list(mdim) + [nr * nc]
        if mdim[0] != nr or mdim[1] != nc:
            raise Exception('InConsistent mdim %d %d vs nr(%d), nc%d' % (mdim[0], mdim[1], \
                                                                             nr, nc))
        resmap = np.zeros(dim, dtype=np.bool)
        for r in range(0,nr):
            for c in range(0,nc):
                i = c * nr + r
                resmap[r, c,i] = True
        return resmap
    @classmethod
    def generate_ticks(cls, l, fsize, stride):
        hf = (fsize - 1) / 2.0
        max_n = np.ceil((l - fsize + 0.0)/stride)
        allticks = [((k +0.5)*stride + hf)  for k in range(max_n+1)]
        allticks[-1] = l-1
        return allticks
    def pt2index(self, pt):
        # pt is n x 2 matrix
        # or a list of n pair
        def f_locate(x):
            return self.locate_index(x[0], self.c_ticks) * len(self.r_ticks)  + \
                self.locate_index(x[1], self.r_ticks)
        return np.asarray(map(f_locate, pt))
    def get_joints_indicatormap(self, pts):
        """
        pts = n x 2 matrix
        """
        pts = np.asarray(pts)
        if type(pts) is np.ndarray and pts.shape[1] != 2:
            raise Exception('The pts should be a n x 2 matrix')
        if self.lookup_table is None:
                return None
        else:
                idx = self.pt2index(pts)
        return self.lookup_table[...,idx]
    @classmethod
    def locate_index(cls, x, ticks):
        n = len(ticks)
        while (n >= 1):
                if x > ticks[n-1]:
                        break
                n = n - 1
        return min(len(ticks)-1, n)
class DepthIndicatorMap(BasicIndicatorMap):
    def __init__(self, depthdim, filter_size2d, stride2d, win_z, stride_z, create_lookup_table = True):
        """
        depthdim: decribe the dimension in x, y, z respectively

        Please pay attention upon creating the maps
        """
        self.depthdim = depthdim
        self.mdim = self.get_indmapdim(depthdim, filter_size2d, \
                                       stride2d, win_z, stride_z)
        self.x_ticks = self.generate_ticks(depthdim[0], filter_size2d, stride2d)
        self.y_ticks = self.generate_ticks(depthdim[1], filter_size2d, stride2d)
        self.z_ticks = self.generate_ticks(depthdim[2], win_z, stride_z)
        self.filter_size2d = filter_size2d
        self.stride2d = stride2d
        self.win_z = win_z
        self.stride_z = stride_z
        self.num_index = len(self.x_ticks) * len(self.y_ticks) * len(self.z_ticks)
        self.create_lookup_table = create_lookup_table
        # offset_z will be only used for interpredating the pt2index
        self.offset_z = (self.depthdim[2]-1)/2 
        if create_lookup_table:
            self.lookup_table = self.create_indicator_map_table(self.x_ticks, \
                                                                self.y_ticks,\
                                                                self.z_ticks, \
                                                                self.mdim,\
                                                                filter_size2d,\
                                                                stride2d,\
                                                                win_z,\
                                                                stride_z)
    def idisp(self):
        print '===DepthIndicatorMap==='
        l = ['depthdim', 'num_index', 'x_ticks', 'y_ticks', 'z_ticks', \
             'lookup_table']
        for x in l:
            print '%s is '  % x
            s =  getattr(self, x)
            if type(s)== np.ndarray:
                print '[Shape = ', s.shape, ']'
            else:
                print s
    def pt2index(self,pt):
        # pt is n x 3 matrix
        #
        
        nx = len(self.x_ticks)
        nxy = nx * len(self.y_ticks)
        f = lambda a: self.locate_index(a[0], self.x_ticks) + self.locate_index(a[1], self.y_ticks) * nx + self.locate_index(a[2] + self.offset_z, self.z_ticks) * nxy
        return np.asarray(map(f, pt))
    def get_joints_indicatormap(self, pts):
        """
        pts: n x 3 matrix of 0-index  
        
        """
        pts = np.asarray(pts)
        if pts.shape[1] != 3:
            raise Exception('The pts should be a n x 3 matrix')
        njoints = pts.shape[0]
        if self.lookup_table is None:
            return None
        else:
            idx = self.pt2index(pts)
            return self.lookup_table[...,idx]
    @classmethod
    def get_indmapdim(cls, depthdim, filter_size2d, stride2d, win_z, stride_z):
        """
        depthdim should be array of int
        """
        inddim = [0,0,0]
        inddim[0] = iu.get_conv_outputsize(depthdim[0], filter_size2d, stride2d)[1]
        inddim[1] = iu.get_conv_outputsize(depthdim[1], filter_size2d, stride2d)[1]
        inddim[2] = iu.get_conv_outputsize(depthdim[2], win_z, stride_z)[1]
        return inddim
    @classmethod
    def create_joint_indicatormap(cls, pt, mdim, filter_size2d, stride2d, \
                                  win_z, stride_z):
        """
         pt:(x,y,z) x,y are the coordinates of bounding box, z is usually the mono            depth value
         Please note that, This is the version without offset.
         Pay attention if you want to call it directly
        """
        pt = np.asarray(pt).flatten()
        resmap = np.zeros(mdim, dtype=np.bool)
        if np.any(np.isnan(pt)):
            return resmaps
        ux = int(min(np.floor(pt[0]/stride2d), mdim[0]-1))
        uy = int(min(np.floor(pt[1]/stride2d), mdim[1]-1))
        uz = int(min(np.floor(pt[2]/stride_z), mdim[2]-1))
        lx = int(max(np.floor((pt[0]+1-filter_size2d)/stride2d) + 1, 0))
        ly = int(max(np.floor((pt[1]+1-filter_size2d)/stride2d) + 1, 0))
        lz = int(max(np.floor((pt[2]+1-win_z)/stride_z) + 1, 0))
        resmap[lx:ux+1,ly:uy+1,lz:uz+1] = True
        return resmap

    def create_indicator_map_table(self, x_ticks, y_ticks, z_ticks, mdim, \
                                                                filter_size2d,\
                                                                stride2d,\
                                                                win_z,\
                                                                stride_z):
        """
        Generate 3d indicator map table
        """
        k = 0
        n = len(x_ticks) * len(y_ticks) * len(z_ticks)
        table_dim = list(mdim) + [n]
        map_table = np.zeros(table_dim, dtype=np.bool)
        for z in z_ticks:
            for y in y_ticks:
                for x in x_ticks:
                    cur_pt = np.asarray((x,y,z))
                    map_table[...,k] = self.create_joint_indicatormap(cur_pt, \
                                                                 mdim, \
                                                                 filter_size2d, \
                                                                 stride2d,\
                                                                 win_z, \
                                                                 stride_z)
                    cur_pt[2] -= self.offset_z
                    # print k, self.pt2index(cur_pt.reshape((1,3)))
                    k = k + 1 

        return map_table
IndMapDic = {'indmap':IndicatorMap, 'maxindmap':MaxIndicatorMap}
# IndMapDic = {'indmap':IndicatorMap}
