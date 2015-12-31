import numpy as np
class ModecError(Exception):
    pass
def GetTorsobox(examples):
    """
    return numpy.ndarray([4, ndata]
    each column will be [x,y, x1, y1]
    """
    ndata = examples.shape[-1]
    return np.concatenate([ examples['torsobox'][0,x].transpose() for x in range(ndata)])
def ExtractSubExample(examples, data_category):
    """
    data_category can be
    istrain,
    isbad,
    isunchecked,
    istest
    """
    all_set = set(['istrain', 'isbad', 'isunchecked', 'istest'])
    if not data_category in all_set:
        raise ModecError('Can''t find class %s ' % data_category)
    ndata = examples.shape[-1]
    idx = np.require([examples[data_category][0, x][0][0] for x in range(ndata)], dtype=np.bool)
    return examples[0,idx]
def ExtendBndbox(bbox, imgsize, sx = 3.5, sy = 4.5, yup=0.5):
    """
    bbox = (x,y,x1,y1)
    imgsize = (width, height) 
    """
    bbox = np.require(bbox, dtype=np.int)
    cx = (bbox[0] + bbox[2])/2
    ly = bbox[1]
    hlenx = (bbox[2] - bbox[0])/2
    leny = (bbox[3] - bbox[1])/2
    x = max(0, cx - hlenx * sx)
    y = max(0, ly - leny * (sy - 1) * yup)
    x1 = min(imgsize[0] - 1, cx + hlenx * sx)  
    y1 = min(imgsize[1] -1, bbox[3] + leny*(sy-1)*(1-yup))
    return np.require([x,y,x1,y1],dtype=np.int)
def CvtCoordsToJoints(coords):
    """
    %% please check FLIC-full/lookupPart.m file if confused
    """
    coords = coords.transpose()
    joints8 = np.zeros((8,2),dtype=np.float32)
    joints8[0] = coords[16]
    joints8[1] = coords[0]
    joints8[2] = (coords[0] + coords[3])/2;## generate neck pos
    joints8[3] =  coords[3];
    joints8[4] =  coords[1];
    joints8[5] =  coords[4];
    joints8[6] =  coords[2];
    joints8[7] =  coords[5];
    return joints8.reshape((16),order='C')
def ReadDataToHMLPEDic(imgdir,example_path, data_category, max_per_batch,save_dir):
    """
    Read all data in 'data_category'
    into HMLPE dictionary
    There is no need to generating training data, since they can be generated in
    hmlpe.py 
    """
    import scipy.io as sio
    import iutils as iu
    import iread.myio as mio
    import iread.hmlpe as hmlpe
    import imgproc
    from PIL import Image
    if data_category != 'istest':
        print 'Warn: The correctness of data type %s is not guaranteed' % data_category
    all_example = sio.loadmat(example_path)['examples']
    examples = ExtractSubExample(all_example, data_category)
    ndata = examples.shape[-1]
    iu.ensure_dir(save_dir)
    buf_size = min(ndata, max_per_batch)
    dimdic = {'data':(112,112,3), 'part_indmap':(8,8), 'joint_indmap':(8,8)} 
    nparts  = 7
    njoints = 8
    d = hmlpe.HMLPE.prepare_savebuffer(dimdic, buf_size, nparts, njoints)
    d['oridet'] = np.zeros((4,buf_size), dtype=np.int)
    d['coords'] = np.ndarray((2,29, buf_size), dtype=np.float32)
    tdsize = dimdic['data'][0]
    dsize = dimdic['data'][0] * dimdic['data'][1] * dimdic['data'][2]
    d['data'] = d['data'].reshape((dsize, -1),order='F')
    d['is_positive'][:] = True
    d['is_mirror'][:] = False
    bid = 1
    j = 0
    for i in range(ndata):
        if j == max_per_batch:
           mio.pickle(iu.fullfile(save_dir, 'data_batch_%d' % bid), d)
           bid = bid + 1
           if ndata - i < max_per_batch:
               d = hmlpe.HMLPE.prepare_savebuffer(dimdic, buf_size, nparts, njoints)
        fp = iu.fullfile(imgdir, str(examples[i]['filepath'][0]))
        img = Image.open(fp)
        tbox = examples[i]['torsobox'][0].reshape((4))
        d['filenames'][j] = fp
        d['coords'][...,j] = examples[i]['coords']
        d['oribbox'][...,j] = bbox = ExtendBndbox(tbox, img.size) 
        orijoints8 = CvtCoordsToJoints(examples[i]['coords']).reshape((8,2),order='C') - 1 # to python stype 0-idx
        d['joints8'][...,j] = TransformPoints(orijoints8, bbox, dimdic['data']).reshape((8,2),order='C')
        imgarr = imgproc.ensure_rgb(np.asarray(img))
        sub_img = Image.fromarray(imgarr[bbox[1]:bbox[3], bbox[0]:bbox[2],:])
        data_img = np.asarray(sub_img.resize((dimdic['data'][0], dimdic['data'][1]))).reshape((dsize),order='F') 
        d['data'][...,j] = data_img
        d['indmap'][...,j] = hmlpe.HMLPE.create_part_indicatormap(d['joints8'][...,j], hmlpe.part_idx, dimdic['part_indmap'], 0.3, 30.0,  12.0)
        d['joint_indmap'][...,j] = hmlpe.HMLPE.create_joint_indicatormap(d['joints8'][...,j], dimdic['joint_indmap'], 30.0, 12.0)
        d['jointmasks'][...,j] = hmlpe.HMLPE.makejointmask(dimdic['data'], d['joints8'][...,j])
        j = j + 1
    mio.pickle(iu.fullfile(save_dir, 'data_batch_%d' % bid), d)
                 
def ReadDataToCifarDic(imgdir,example_path, data_category, max_per_batch,save_dir):
    """
        read all data in 'data_category'
        into cifar style dictionary                
    """
    import scipy.io as sio
    import iutils as iu
    import cifar
    import iconvnet_datacvt as icvt
    from iutils import imgproc as imgproc
    from PIL import Image
    if data_category != 'istest':
        print 'I haven''t implement joints8 part '
        #raise ModecError('I haven''t implement joints8 part ')
    all_examples = sio.loadmat(example_path)['examples']
    examples = ExtractSubExample(all_examples, data_category)

    ndata = examples.shape[-1]
    iu.ensure_dir(save_dir)
    s_first = min(ndata, max_per_batch)
    d = cifar.PrepareData(s_first)
    d['oridet'] = np.ndarray((4,s_first),dtype=np.int)
    d['filepath'] = [str() for x in range(s_first)]
    d['coords'] = np.ndarray((2,29,s_first),dtype=np.float32)
    tdsize= cifar.img_size[0] # make sure img_size[0] == img_size[1]
    
    j = 0
    bid = 1
    for i in range(ndata):
        if j == max_per_batch:
            icvt.ut.pickle(iu.fullfile(save_dir, 'data_batch_' + str(bid)), \
                           d)
            bid = bid + 1
            j = 0
            if ndata - i < max_per_batch:
                d = cifar.PrepareData(ndata-i)                
        fn = str(examples[i]['filepath'][0])
        fp = iu.fullfile(imgdir, fn)
        img = Image.open(fp)
        tbox = examples[i]['torsobox'][0].reshape((4))
        d['filepath'][j] = fp
        d['oridet'][...,j] = tbox
        d['oribbox'][...,j] = bbox = ExtendBndbox(tbox,img.size)
        d['coords'][...,j] = examples[i]['coords']
        orijoints8 = CvtCoordsToJoints(examples[i]['coords']).reshape((8,2),order='C')
        d['joints8'][...,j] = TransformPoints(orijoints8, bbox,cifar.img_size).reshape((16),order='C')
        img = imgproc.ensure_rgb(np.asarray(img))
        sub_img = Image.fromarray(img[bbox[1]:bbox[3], bbox[0]:bbox[2],:])
        data_img = np.asarray(sub_img.resize((cifar.img_size[0],\
                                               cifar.img_size[1]))).reshape((cifar.dim_data),order='F')
        d['data'][...,j] = data_img
        j = j + 1
    icvt.ut.pickle(iu.fullfile(save_dir, 'data_batch_' + str(bid)),d)
    

def TransformPoints(points,bbox, tosize, inv=False):
    # Note that bbox is in (x,y, x1, y1) format
    # and *each row* in points is a point
    # Now I didn't use fromsize
    # since I will keep record of invalid estimation
    # Note that tosize is image width and height(not rows and cols)
    res_points = np.zeros(points.shape, dtype=np.float32)
    sx = (tosize[0])/(bbox[2] - bbox[0] + 1e-9)
    sy = (tosize[1])/(bbox[3] - bbox[1] + 1e-9)
    for i, p in enumerate(points):
        if not inv:
            x = (p[0] - bbox[0]) * sx
            y = (p[1] - bbox[1]) * sy
        else:
            x = (p[0]/sx) + bbox[0]
            y = (p[1]/sy) + bbox[1]
        res_points[i,:] = [x,y]
    return res_points
