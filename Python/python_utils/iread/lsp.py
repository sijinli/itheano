"""
For generating lsp-hmlpe consistent data
"""
import numpy as np
def GetJoints8(jt):
    # % http://www.comp.leeds.ac.uk/mat4saj/lsp.html
    # note: This is python format
    # % Right ankle  0
    # % Right knee 1
    # % Right hip 2
    # % Left hip 3
    # % Left knee 4
    # % Left ankle 5
    # % Right wrist 6
    # % Right elbow 7
    # % Right shoulder 8 
    # % Left shoulder 9
    # % Left elbow 10
    # % Left wrist 11
    # % Neck 12
    # % Head top 13
    r = np.zeros((8,2), dtype=np.float32)
    v = np.zeros((8,1), dtype=np.bool)
    vl = ~np.require(jt[2,:], dtype=np.bool)
    pts = jt[0:2,:].transpose()
    r[0] = (pts[12] + pts[13])/2.0
    r[1] = pts[9]
    r[2] = pts[12]
    r[3] = pts[8]
    r[4] = pts[10]
    r[5] = pts[7]
    r[6] = pts[11]
    r[7] = pts[6]
    v[0] = vl[12] and vl[13]
    v[1] = vl[9]
    v[2] = vl[12]
    v[3] = vl[8]
    v[4] = vl[10]
    v[5] = vl[7]
    v[6] = vl[11]
    v[7] = vl[6]
    v = np.tile(v, (1,2))
    return r - 1,v # convert to python 0-index
def GetUpperBodyBox(imgsize):
    """
    imgsize  = (width, height)
                wx     wy
    """
    rate = 1.5 # defined in matlab file test_create_lsp_regression_data.m
    rbox = np.zeros((4,1),np.int)
    rbox[2] = imgsize[0] - 1
    rbox[3] = int(min( imgsize[1] - 1, rbox[2] * rate))
    return rbox
    
    
def ReadCropImageToHMLPEDic(dataset_dir, save_dir, istrain = False, isOC = True):
    """
    This function will be used for generating testing data
    Because training and testing data has different format in oribbox

    For generating trainign samples,
    please use
    create_lsp_regression_data.m
    (dataset_dir, type=3, opt)
      opt.OC = ?
      
    and hmlpe.py
    """
    import iutils as iu
    import iread.hmlpe as hmlpe
    import iread.myio as mio
    import scipy.io as sio
    from PIL import Image
    ndata = 1000
    if istrain:
        s_idx = 0
    else:
        s_idx = 1000
    imgdir = iu.fullfile(dataset_dir, 'images-crop')
    if isOC:
        dmat = sio.loadmat(iu.fullfile(dataset_dir, 'jointsOC.mat'))
    else:
        dmat = sio.loadmat(iu.fullfile(dataset_dir, 'joints-crop.mat'))
    lsp_jt = dmat['joints']
    dimdic = {'data':(112,112,3), 'part_indmap':(8,8), 'joint_indmap': (8,8)}
    nparts = 7
    njoints = 8
    d = hmlpe.HMLPE.prepare_savebuffer(dimdic, ndata, nparts, njoints)
    d['data'] = d['data'].reshape((-1,ndata),order='F')
    d['is_mirror'][:] = False
    d['is_positive'][:] = True
    for idx in range(s_idx, s_idx + ndata):
        imgpath = iu.fullfile(imgdir, 'im%04d.jpg' % (idx + 1))
        img = Image.open(imgpath)
        i = idx - s_idx
        orijoints8, isvisible = GetJoints8(lsp_jt[...,idx]) 
        bbox = GetUpperBodyBox(img.size)
        img_arr = np.asarray(img)[bbox[1]:bbox[3], bbox[0]:bbox[2],:]
        s = np.asarray([(dimdic['data'][1]-1.0)/(bbox[2] - bbox[0]),(dimdic['data'][0]-1.0)/(bbox[3]-bbox[1])]).reshape((1,2))
        tjoints = (orijoints8 - bbox[0:2,:].reshape((1,2)))*s
        masks = hmlpe.HMLPE.makejointmask(dimdic['data'], tjoints)
        d['data'][...,i] = np.asarray(Image.fromarray(img_arr).resize((dimdic['data'][0], dimdic['data'][1]))).reshape((-1,1),order='F').flatten()
        d['joints8'][...,i] =  tjoints
        d['jointmasks'][...,i] = np.logical_and(masks, isvisible)
        d['filenames'][i] = imgpath
        d['oribbox'][...,i] = bbox.flatten()
        d['indmap'][...,i] = hmlpe.HMLPE.create_part_indicatormap(tjoints, hmlpe.part_idx, dimdic['part_indmap'], 0.3, 30.0, 12.0)
        d['joint_indmap'][...,i] = hmlpe.HMLPE.create_joint_indicatormap(tjoints, dimdic['joint_indmap'], 30.0, 12.0)
    mio.pickle(iu.fullfile(save_dir, 'data_batch_1'), d)
        
