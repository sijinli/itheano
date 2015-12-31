
"""
Written by Li Sijin

Some common image processing functions
"""
import numpy as np
import matplotlib.pyplot as plt

class BigImagePlot:
    """
    single_size is the maximum image size to put into a single cell
    tile = nrow, ncol
    """
    def __init__(self, patch_size, tile, nchannel=3,  \
                 margin=3, margin_color=(0,0,0)):
        imgr, imgc = patch_size[0], patch_size[1]
        if len(patch_size)>2:
            nchannel = patch_size[2]
        rsize = (imgr + margin)* tile[0] - margin
        csize = (imgc + margin)* tile[1] - margin
        self.imgdata = np.zeros((rsize, csize, nchannel), dtype=np.single)
        self.tile = tile
        self.margin = margin
        self.nchannel = nchannel
        self.patch_size = patch_size
        if self.nchannel == 3 and len(margin_color) == 3:
            self.color = margin_color
        else:
            self.color = 1
        # In defalut, all the image share teh same size
        self.same_size = True
    def set_same_size(self, flag):
        self.same_size = flag is True
    def resize_img(self, img_arr, tosize):
        import Image
        tosize = (tosize[0], tosize[1])
        if len(img_arr.shape) == 2 and self.nchannel == 3:
            # sp = img_arr.shape
            # img_arr = np.tile(img_arr.reshape((sp[0],sp[1],1)),[1,1,3])
            img_arr = graytoheapimage(img_arr)
        return np.asarray(Image.fromarray((img_arr * 255.0).astype(np.uint8)).resize(tosize), dtype=np.float)/255
    def subplot(self, img, pos_r, pos_c = None):
        if pos_c is None:
            r = pos_r /self.tile[1]
            c = pos_r - self.tile[1] * r
        else:
            r,c = pos_r, pos_c 
            if r <0 or r >= self.tile[0] or c < 0 or c >= self.tile[1]:
                return
        r = r * (self.patch_size[0] + self.margin)
        c = c * (self.patch_size[1] + self.margin)
        if not self.same_size:
            img = self.resize_img(img, self.patch_size)
        r_end = min(self.patch_size[0], img.shape[0]) + r
        c_end = min(self.patch_size[1], img.shape[1]) + c
        if self.nchannel == 1:
            self.imgdata[r:r_end,c:c_end,0] = img[:r_end-r,:c_end-c]
        else:
            self.imgdata[r:r_end,c:c_end,:] = img[:r_end-r,:c_end-c,:]
    def getdata(self):
        return self.imgdata
    def getplotdata(self):
        import Image
        if self.nchannel == 1:
            sp = self.imgdata.shape
            res = self.imgdata.reshape((sp[0], sp[1]))
        else:
            res = self.imgdata          
        return res
    def save(self,savepath):
        import Image
        if self.nchannel == 1:
            sp = self.imgdata.shape
            res = (self.imgdata.reshape((sp[0], sp[1])) * 255.0).astype(np.uint8)
        else:
            res = (self.imgdata*255.00).astype(np.uint8)
        img = Image.fromarray(res)
        img.save(savepath)
def graytoheapimage(img):
    cmap = plt.get_cmap('jet')
    return np.delete(cmap(img), 3, 2)
def ensure_rgb(img):
    if len(img.shape) == 2 or len(img.shape)==3 and img.shape[-1]==1:
        img = img.reshape((img.shape[0],img.shape[1],-1))
        img = np.concatenate((img,img,img),axis=-1)
    elif len(img.shape)==3 and img.shape[-1]==3:
        pass
    elif len(img.shape)==3:
        img = img[...,0:3]
    return img
# def drawrec(img, box,color=(1,1,1), linewidth=3):
#     """
#     box is 4 x n
#     n is the number of boxes
#     """
#     n = box.shape[-1]
#     sp = img.shape
#     m = linewidth
#     for i in range(n):
#         x,y,x1,y1 = bbox[...,i]
#         x = max(0,x)
#         y = max(0,y)
#         x1 = min(sp[1],x1)
#         y1 = min(sp[0],y1)
#         if x+m>=x1 or y + m >=y1:
#             continue
#         img[x:,y:,:]

def maptorange(img, to_range, ori_range = None):
    resimg = img.copy()
    if ori_range is None:
        t1 = resimg.min()
        t2 = resimg.max()
        ori_range= [t1,t2]
    else:
        resimg = np.minimum(ori_range[1], np.maximum(ori_range[0], resimg))
    if (ori_range[1] == ori_range[0]):
        resimg[:] = to_range[0]
    else:
        resimg = (img - ori_range[0]) / (ori_range[1] - ori_range[0]) \
            * (to_range[1] - to_range[0]) + to_range[0]
    return resimg
def show_mat_as_surface(mat):
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    import numpy as np
    ny = mat.shape[0]
    nx = mat.shape[1]

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X = np.arange(0,nx,1) 
    Y = np.arange(0,ny,1)
    X, Y = np.meshgrid(X, Y)
    Z = mat
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
            linewidth=0, antialiased=False)
    # ax.set_zlim(-1.01, 1.01)
    # ax.zaxis.set_major_locator(LinearLocator(10))
    # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    # fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()
    
def imgeq(img, nbin = 255):
    flat_img = img.reshape((-1,1),order='F')
    flat_idx = np.require(sorted(range(flat_img.size), key=lambda k:flat_img[k]))
    last_idx = 0
    per = (flat_img.size - 1) / nbin + 1
    lbin = 0
    for i in range(nbin):
        r_start = per * i
        r_end = min(r_start + per, flat_idx.size)
        if lbin > r_end:
            continue
        s = max(r_start, lbin)
        v = flat_img[flat_idx[r_end-1]]
        lbin = r_end
        while (lbin < flat_idx.size and flat_img[flat_idx[lbin]] == v):
               lbin += 1
        flat_img[flat_idx[s:lbin]] = i
    # print 'last bin is ' + str( lbin)
    # print flat_img[flat_idx[range(10)]].transpose()
    # print ' '
    return flat_img.reshape(img.shape, order='F')

def turn_off_axis(a = None, is3d=False):
    if a is None:
        a = plt.gca()
    a.set_axis_off()
    a.axes.get_xaxis().set_visible(False)
    a.axes.get_yaxis().set_visible(False)
    if is3d:
        a.axes.get_zaxis().set_visible(False)
def turn_on_axis(a = None):
    if a is None:
        a = plt.gca()
    a.set_axis_on()
    a.axes.get_xaxis().set_visible(True)
    a.axes.get_yaxis().set_visible(True)
def CalcEigenFaces(imgdata):
    from ipyml.dimreduce import pca
    import time
    ori_shape = imgdata.shape
    ndata = imgdata.shape[-1]
    img_shape = ori_shape[:-1]
    flatdata = imgdata.reshape((-1,ndata),order='F')
    start_time = time.time()
    pcamodel = pca.PCA(flatdata)
    end_time = time.time()
    print 'Cost %f seconds' % (end_time - start_time)
    ushape = tuple(list(img_shape) + [pcamodel.U.shape[-1]])
    return pcamodel.U.reshape(ushape,order='F'), pcamodel.l, pcamodel.mean
def CalcKPCAEigenFaces(imgdata, kpca_type='GKPCA', kpca_para=None):
    """
    In this function, the returned value is the embedding and KPCA model
    No reconstruction is conducted
    """
    from ipyml import dimreduce
    from ipyml.dimreduce import kpca
    import time
    ori_shape = imgdata.shape
    ndata = imgdata.shape[-1]
    img_shape = ori_shape[:-1]
    flatdata = imgdata.reshape((-1,ndata),order='F')
    start_time = time.time()
    kpca_model = kpca.KPCA(flatdata, kpca_type, kpca_para)
    end_time = time.time()
    print 'Cost %f seconds for model construction' %(end_time - start_time)
    embedding = kpca_model.apply(flatdata)
    return embedding, kpca_model
def makeindmap(indmap, blocksize=10, margin=1):
    """
    create indmap for showing
    """
    wx,wy = indmap.shape[0], indmap.shape[1]
    t = 10
    res = np.zeros((wx * t,wy * t), dtype=np.float32)
    for i in range(wx):
        for j in range(wy):
            if indmap[i,j] != 0:
                res[i*t:i*t+t-margin,j*t:j*t+t-margin]=1
    return res
def get_rel_pos_in_figure(rect, xlim, ylim, ax_pos):
    """
    rect is (x,y,x1,y1) indicating the position of box in data coordinate
    xlim
    ylim are the limits of x,y axis
    ax_pos is the position of axis in the figure
        in format (sx, sy, ex, ey)

    return rect_pos is (sx1, sy1, ex1, ey1) is the region relative positon in figure
    
    """
    xr = xlim[1] - xlim[0]
    yr = ylim[1] - ylim[0]
    rng_rel = (rect[0] - xlim[0])/xr, (rect[1] - ylim[0])/yr, \
        (rect[2] - xlim[0])/xr, (rect[3] - ylim[0])/yr
    xar, yar = ax_pos[2] - ax_pos[0], ax_pos[3] - ax_pos[1], 
    rng_rel = (rng_rel[0] * xar, rng_rel[1] * yar, \
               rng_rel[2] * xar, rng_rel[3] * yar)
    rng_rel = (rng_rel[0] + ax_pos[0], rng_rel[1] + ax_pos[1], \
               rng_rel[2] + ax_pos[0], rng_rel[3] + ax_pos[1])
    return rng_rel

def draw_rect(draw, rect, fill=(255,0,0), width=1):
    rect = np.floor(np.asarray(rect).copy()).flatten()
    rect[2],rect[3] = rect[2]-1,rect[3]-1
    draw.line((rect[0], rect[1], rect[0], rect[3]),fill=fill,width=width)
    draw.line((rect[0], rect[3], rect[2], rect[3]),fill=fill,width=width)
    draw.line((rect[2], rect[3], rect[2], rect[1]),fill=fill,width=width)
    draw.line((rect[2], rect[1], rect[0], rect[1]),fill=fill,width=width)

def maximum_supression(scores, loc, scales, winwidth, max_keep):
    # input score : 1 x ndata
    #       loc   : 2 x ndata
    #       scales: 2 x ndata
    #      winwidth: [wx, wy]
    # output: indexed list (len <= max_keep)
    #
    import ipyml.geometry as igeo
    if max_keep <= 0:
        return []
    scores = scores.flatten() 
    ndata = scores.size    
    take = np.ones((ndata), dtype=np.bool)
    sindex = sorted(range(ndata), key=lambda x:(scores[x], 1/scales[0,x]/scales[1,x]), reverse=True)
    nloc = loc / scales
    wlist = np.tile(np.asarray(winwidth).reshape((2,1)), (1,ndata))/scales
    f_area = lambda x:(x[1][0] - x[0][0])*(x[1][1] - x[0][1]) if len(x) == 2 else 0
    res = [0 for x in range(max_keep)]
    nc = 0
    rej = 0
    for i in range(0,ndata):
        if take[ sindex[i]  ] == False:
            continue
        idx = sindex[i]
        res[nc] = idx
        nc = nc + 1
        if nc == max_keep:
            break
        recti = ((nloc[0,idx],nloc[1,idx]),\
                (nloc[0,idx]+wlist[0,idx]-1,\
                 nloc[1,idx]+wlist[1,idx]-1))
        areai = f_area(recti)
        for j in range(i+1, ndata):
            jdx =  sindex[j]
            if take[ jdx ] == False:
                continue
            rectj = ((nloc[0,jdx],nloc[1,jdx]),\
                    (nloc[0,jdx]+wlist[0,jdx]-1,\
                    nloc[1,jdx]+wlist[1,jdx]-1))
            areaj = f_area(rectj)
            rect_int = igeo.RectIntersectRect(recti,rectj)
            if f_area(rect_int) > min( areai, areaj) * 0.5:
                take[jdx] = False
                rej += 1
    #print 'reject %d ' % rej
    return res[:nc]
def generate_colorbar_cax(ax, loc = 'right', size='5%', pad=0.05):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    return divider.append_axes(loc, size=size,pad=pad)
def generate_line_seg_colormap(seq):
    """
    The input for map should be
    """
    import matplotlib.colors as mcolors
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap('CustomMap', cdict)
def get_color_list(N):
    """
    This function will reburn a list of n different colors
    each element is represented by (r,g,b) tuple
    """
    import colorsys
    HSV_tuples = [(x*1.0/N, 0.9, 0.9) for x in range(N)]
    RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
    return RGB_tuples
def set_figure_size(width, height):
    from pylab import rcParams
    rcParams['figure.figsize'] = width, height
def imsave_tight(imgpath):
    plt.savefig(imgpath, bbox_inches='tight')
