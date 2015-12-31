import os
import options
from time import time
from operator import iadd
class itimer:
    def __init__(self):
        self.start_time = 0
        self.stop_time = 0
        self.dur = 0
        self.time_list = []
        self.tag_list = []
        self.tic_time = 0
        self.tic_st = None
    def start(self):
        self.start_time= time()
        self.dur = 0
    def pause(self):
        self.dur = self.dur + time() - self.start_time
    def resume(self):
        self.start_time = time()
    def stop(self):
        self.dur = self.dur + time() - self.start_time
        return self.dur
    def restart(self):
        self.time_list = []
        self.tag_list = []
        self.start_time = time()
    def addtag(self,st):
        self.time_list.append(time())
        self.tag_list.append(st)
    def print_all(self):
        pre = self.start_time
        for t, tag in zip(self.time_list, self.tag_list):
            print('    {}:\t cost {} seconds'.format(tag, t - pre))
            pre = t
        print '    All cost {} seconds'.format(self.time_list[-1] - self.start_time)
    def tic(self, st=None):
        self.tic_time = time()
        self.tic_st = st
    def toc(self):
        print 'Time elapase [{}]{} seconds: '.format(self.tic_st if self.tic_st else '', time() - self.tic_time)
                
def iprint_dic(d, curlevel=0, maxlevel = 0):
    import numpy as np
    space = ' ' * curlevel
    print('=======%d=======' % curlevel)
    for k in d.keys():
        print('----------------------')
        if type(d[k]) == np.ndarray:
            print('%s%s: ndarray size=%s' % (space, k, str(d[k].shape)))
        elif type(d[k]) == dict:
            if maxlevel == curlevel:
                print('%s%s:dict{%s}' % (space, k, d[k].keys()))
            else:
                iprint_dic(d,curlevel+1, maxlevel)
        elif type(d[k]) == list:
            print('%s%s: list length =%d ' % (space, k, len(d[k])))
        else:
            print('%s%s:%s type=%s' % (space, k, str(d[k]), str(type(d[k]))))
    
def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)
def exists(p, check_type = 'file'):
    if check_type == 'file':
        return os.path.isfile(p)
    elif check_type == 'dir':
        return os.path.exists(p)
    else:
        return False 
def enter_dir(dirname):
    import os
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    os.chdir(dirname)


def removelastslash(st):
    return st[:-1] if st[-1] == os.sep else st 
def removefirstslash(st):
    return st[1:] if st[0] == os.sep else st
def fullfile(dirpath, *argumentlist):
    dirpath = removelastslash(dirpath)
    ff = dirpath
    for arg in argumentlist:
        ff = ff + os.sep + removefirstslash(removelastslash( arg ))
    return ff
def getfilelist(searchdir, regexp='.*'):
    """
    Find all the file that match regexp where regexp is the regular expression.
    """
    import os
    import re
    if not os.path.exists(searchdir):
        return []
    allfile = os.listdir( searchdir )
    pattern = re.compile( regexp )
    filelist = []
    for f in allfile:
        tmp = pattern.match( f )
        if tmp is not None:
            filelist.append( f )
    return filelist

def get_conv_outputsize(imgsize, filtersize, stride, filternum = -1):
    import math
    # sride <= filtersize and imgsize >= filtersize
    # The idea is to cover all the pixel with filtersize 
    t = math.ceil((imgsize - filtersize + 0.0 )/ stride) + 1
    return (filternum, t, t)
def concatenate_list(lst):
    return reduce(iadd, lst)
def back_track_filter_range(conv_list, curp):
    """
    curp is (x,y,x1,y1) representing the rectangle of interest in the top layer
    conv_list is the list of (filter_size, stride, [start]) from bottow to top
    x1>= x and y1 >=y
    
    """
    curp = list(curp)
    for v in reversed( conv_list):
        curp[0] = curp[0] * v[1] 
        curp[1] = curp[1] * v[1]
        curp[2] = curp[2] * v[1] + v[0] - 1
        curp[3] = curp[3] * v[1] + v[0] - 1
        if len(v) == 2:
            offset = 0
        else:
            offset = v[2]
        curp = [ x + offset for x in curp] 
    return tuple( curp )
def get_conv_fs(conv_list):
    """
    This function will produce (filter_size, stride) so that
    applying convolution with (filter_size, stride) can have the "same" effect as the
    conv_list in terms of size
    """
    a = back_track_filter_range(conv_list, (0,0,0,0))
    b = back_track_filter_range(conv_list, (1,1,1,1))
    return (a[2]-a[0] + 1, b[0]-a[0])
    
def cartesian_product2(arrays):
    import numpy as np
    la = len(arrays)
    arr = np.empty([len(a) for a in arrays] + [la])
    # Now arr is len1 x len2 x len array
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
        # print a
        # print i
        # print arr[...,i].shape
        # print a.shape
        # print '========='
    return arr.reshape(-1, la)


def fileparts(filepath):
    p = filepath.rfind(os.sep)
    if p == -1:
        return '.' + os.sep, filepath
    else:
        return filepath[:p], filepath[p+1:]

def get_list_from_str(st, sep = ',', interpret_as=str):
    st = st.strip('[]')
    try:
        return [interpret_as(x.strip()) for x in st.split(',')]
    except Exception as err:
        raise Exception(err)
def get_int_list_from_str(st, sep=','):
    return get_list_from_str(st, sep, interpret_as=int)
def get_float_list_from_str(st, sep=','):
    return get_list_from_str(st, sep, interpret_as=float)

def my_base_slice(x, indexes):
    if type(x) is list:
        return [ x[k] for k in indexes]
    else:
        return x[..., indexes]
def myslice(param, indexes):
    if type(indexes) is not list:
        indexes = indexes.flatten(order='F')
    return [my_base_slice(x, indexes) for x in param]
def prod(l):
    r = 1
    for x in l:
        r*=x
    return r
def print_common_statistics(X, name=None):
    import numpy as np
    if name:
        print '----%s------' % name
    else:
        print '------------'
    print 'Shape: ', X.shape
    X = X.flatten(order='F')
    print 'Max = %.6e \t median = %.6e \t min = %.6e' % (np.max(X), np.median(X), np.min(X))
    print 'avg = %.6e , std = %.6e' % (np.mean(X), np.std(X))
    support_rate = np.sum(np.abs(X).flatten() > 0) * 100.0/ X.size
    print '''
    Max = {:.6e}\t media = {:.6e} \t min = {:.6e}\n
    avg = {:.6e}\t std = {:.6e}
    support rate = {:.2f}%
    '''.format(np.max(X), np.median(X), np.min(X), np.mean(X), np.std(X), support_rate)