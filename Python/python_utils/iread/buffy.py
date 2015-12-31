import numpy as np

class BuffyError(Exception):
    pass
def ReadBuffyTxt(txtdir):
    d = dict()
    f = open(txtdir)
    lines = f.readlines()
    ndata = len(lines)/7
    for i in range(ndata):
        number = lines[i * 7]
        coor = np.zeros((4,6), order='F')
        for j in range(6):
            l = lines[i * 7 + j + 1].split(' ')[1:-1]
            #print lines[i * 7 + j + 1]
            coor[:,j] = np.asarray([ float(x) for x in l])
        d[int(number)] = coor
    return d


def ExtractAnnotation(annot_dir):
    import iutils as iu
    eplist = [2,3,4,5,6]
    d = dict()
    for e in eplist:
        txtname = iu.fullfile(annot_dir, 'buffy_s5e' + str(e) + '_sticks.txt')
        ed = ReadBuffyTxt(txtname)
        d[e] = ed
    return d
def ExtendBndbox(bbox, imgsize, sx = 1.7, sy=4.2):
    # imgsize = (width, height)
    # bbox = (x,y, x1, y1)
    #
    bbox = np.require(bbox, dtype=np.int)
    cx = (bbox[0] + bbox[2])/2
    ly = bbox[1]
    lhx = (bbox[2] - bbox[0])/2
    lhy = (bbox[3] - bbox[1])/2
    x = max(0, cx - int(lhx * sx))
    y = max(0, ly)
    x1 = min(imgsize[0]-1, cx + int(lhx * sx))
    y1 = min(imgsize[1]-1, ly + int(lhy * sy))
    return np.require([x,y,x1,y1], dtype=np.int)
def GetImagePath(imgdir, ep, fr):
    import iutils as iu
    imgpath = iu.fullfile(imgdir, 'buffy_s5e'+str(ep)+'_original', \
                               ('%06d' % fr) + '.jpg')
    return imgpath


def convert_AHEcoor_to_joints8(coor):
    joints8 = np.ndarray([2,8], dtype=np.single, order = 'F')
    if coor.size == 0:
        return None
    else:
        coor = coor.reshape( (2,12), order ='F')
    joints8[:,0] = (coor[:,10] + coor[:,11])/2
    joints8[:,1] = coor[:,4]
    joints8[:,2] = coor[:,0]
    joints8[:,3] = coor[:,2]
    joints8[:,4] = (coor[:,5] + coor[:,8])/2
    joints8[:,5] = (coor[:,3] + coor[:,6])/2
    joints8[:,6] = coor[:,9]
    joints8[:,7] = coor[:,7]
    return joints8.transpose()

def add_buffy_specific_field(data_dic, ndet, imgdir):
      data_dic['ep'] = np.ndarray([ndet], dtype=np.int)
      data_dic['fr'] = np.ndarray([ndet], dtype=np.int)
      data_dic['oridet'] = np.zeros([4,ndet],dtype=np.float32)
      data_dic['imgdir'] = imgdir
      data_dic['annotation'] = np.ndarray([4,6,ndet], dtype=np.float32)
      data_dic['gt_det'] = np.zeros([4,ndet],dtype=np.float32) # ground truth detection

def ReadTestDataToHMLPEDic(imgdir, annot_dir, tech_report_path, save_path=None, iou_thresh=0.5):
    """
    This function will use detection window provided by the author
     Also, erro detection window are not included
     For image without detection, all the joint points will be zero
     
    """
    import Stickmen
    import scipy.io as sio
    import PIL
    import iutils as iu
    from PIL import Image
    import matplotlib.pyplot as plt
    import iread.hmlpe as hp
    import iread.myio as mio
    import ipyml.geometry as igeo
    rp = sio.loadmat(tech_report_path)['techrep2010_buffy']
    d_annot = ExtractAnnotation(annot_dir)
    ndata = rp.shape[1]
    data = []
    tsize = 112 # the size of test image
    part_ind_dim = (8,8)
    joint_ind_dim = (8,8)
    filter_size = 32.0
    stride = 12.0
    indrate = 0.3
    joint_filter_size = 32.0
    joint_stride = 12.0
    njoints = 8
    nparts = 7
    ndet = ndata # One test point for one image
    newdim = (tsize,tsize,3)
    data = hp.HMLPE.prepare_savebuffer({'data':newdim,\
                                        'part_indmap': part_ind_dim,\
                                        'joint_indmap':joint_ind_dim },\
                                        ndet, nparts, njoints)
    # add Buffy specific field
    add_buffy_specific_field(data, ndet, imgdir)
    idx = 0
    f_calc_area = lambda rec: (rec[1][1]-rec[0][1]) * (rec[1][0]-rec[0][0]) if len(rec)==2 else 0
    f_mk_rec = lambda det: ((det[0],det[1]), (det[2], det[3]))
    for i in range(ndata):
        ep = rp[0,i][1][0][0]
        fr = rp[0,i][2][0][0]
        imgpath = iu.fullfile(imgdir, 'buffy_s5e'+str(ep)+'_original', \
                               ('%06d' % fr) + '.jpg')
        img = Image.open(imgpath)
        data['ep'][...,i] = ep
        data['fr'][...,i] = fr
        data['filenames'][i] = imgpath
        data['is_positive'][...,i] = True
        data['annotation'][...,i] = Stickmen.ReorderStickmen(d_annot[ep][fr])
        data['gt_det'][...,i] = np.asarray(Stickmen.EstDetFromStickmen( data['annotation'][...,i]))
        gt_rec = f_mk_rec(data['gt_det'][...,i]) #
        gt_area = f_calc_area(gt_rec) 
        if rp[0,i][0].size == 0: # No detection are found in this image
            # imgdata will also be all zero
            # oridet will be all zero
            # oribbox will be all zero
            # joints8 will be all zero
            # jointmasks wiil be all zero
            # indmap will be all zero
            # joint_indmap will be all zero
            # nothing need to be done, since they are default value
            pass
        else:
            m = -1
            for j in range(rp[0,i][0].size):
                det = rp[0,i][0][0,j]['det'][0] # det = (x,y,x1,y1)
                cur_rec = f_mk_rec(det)
                int_area = f_calc_area(igeo.RectIntersectRect(cur_rec, gt_rec))
                if int_area > ( gt_area - int_area ) * iou_thresh:
                  m = j
                  break
            if m != -1:
                oribbox = ExtendBndbox(det, img.size)
                arr_img = np.asarray(img)[oribbox[1]:oribbox[3]+1, \
                                          oribbox[0]:oribbox[2]+1,:]
                res_img = Image.fromarray(arr_img).resize((tsize,tsize))
                data['data'][...,i] =np.asarray(res_img)
                tmppoints = convert_AHEcoor_to_joints8(data['annotation'][...,i])
                data['joints8'][...,i] = TransformPoints(tmppoints, oribbox, np.asarray(res_img.size) -1)
                data['jointmasks'][...,i] = hp.HMLPE.makejointmask(newdim, data['joints8'][...,i])
                data['indmap'][...,i] = hp.HMLPE.create_part_indicatormap(data['joints8'][...,i], hp.part_idx, part_ind_dim, indrate, filter_size, stride)
                data['joint_indmap'][...,i] = hp.HMLPE.create_joint_indicatormap(data['joints8'][...,i], joint_ind_dim, joint_filter_size, joint_stride)
                data['oribbox'][...,i] = oribbox
    data['data'] = data['data'].reshape((-1,ndet),order='F')
    if save_path is not None:
        mio.pickle(save_path, data)
    return data
    



def ReadTestDataToCifarDic(imgdir, annot_dir, tech_report_path, save_path=None):
    """
    This program will use the detection window by the author
    image data are extracted from the extended detection window
    return type: dictionary with the following keys
                    ep
                    fr
                    annotation
                    oriimg  
                    bbox
                    joints8          ! (coordinate w.r.t bounding box)
                    data             !
                    indmap           ! (all zeros)
    This is the previous version, latest version is
      ReadTestDataToHMLPEDic
    """ 
    import Stickmen
    import scipy.io as sio
    import PIL
    import iutils as iu
    from PIL import Image
    import matplotlib.pyplot as plt
    import iconvnet_datacvt as icvt
    rp = sio.loadmat(tech_report_path)['techrep2010_buffy']
    d_annot = ExtractAnnotation(annot_dir)
    ndata = rp.shape[1]
    data = []
    tsize = 112 # the size of test image
    indsize = 8
    data = dict()
    ndet = sum([ (1 if rp[0,i][0].size==0 else rp[0,i][0].size) for i in range(ndata)]) 
    data['ep'] = np.ndarray([ndet], dtype=np.int)
    data['fr'] = np.ndarray([ndet], dtype=np.int)
    data['annotation'] = np.ndarray([4,6,ndet], dtype=np.float32)
    data['indmap'] = np.zeros([7,indsize,indsize, ndet], dtype=np.bool,order='F')
    data['indmap_para'] = (30,12)
    data['oribbox'] = np.ndarray([4, ndet], dtype=np.float32)
    data['data'] = np.ndarray([tsize, tsize, 3, ndet], order='F', dtype=np.uint8)
    data['joints8'] = np.ndarray([8,2,ndet], dtype=np.float32)
    data['oridet'] = np.ndarray([4,ndet],dtype=np.float32)
    data['imgdir'] = imgdir
    idx = 0   
    for i in range(ndata):
        ep = rp[0,i][1][0][0]
        fr = rp[0,i][2][0][0]
        imgpath = iu.fullfile(imgdir, 'buffy_s5e'+str(ep)+'_original', \
                               ('%06d' % fr) + '.jpg')
        img = Image.open(imgpath)
       
        if rp[0,i][0].size==0: # Not detection found by detector
            data['ep'][...,idx] = ep
            data['fr'][...,idx] = fr
            data['annotation'][...,idx] = Stickmen.ReorderStickmen(d_annot[ep][fr])           
            data['oribbox'][...,idx] = np.zeros([4],dtype=np.float32)
            data['joints8'][...,idx] = np.ones([8,2], dtype=np.float32) * tsize * 2
            data['data'][...,idx] = np.zeros([tsize,tsize,3], dtype=np.uint8)
            data['oridet'][...,idx] = np.zeros([4], dtype=np.float32)
            idx  = idx + 1
            continue
        for j in range(rp[0,i][0].size): 
            det = rp[0,i][0][0,j]['det'][0] # det = (x,y,x1,y1)
            data['oridet'][...,idx] = det 
            det = ExtendBndbox(det, img.size)
            arr_img = np.asarray(img)[det[1]:det[3]+1, det[0]:det[2]+1,:]
            res_img = Image.fromarray(arr_img).resize((tsize,tsize))
            data['ep'][...,idx] = ep
            data['fr'][...,idx] = fr
            data['annotation'][...,idx] = Stickmen.ReorderStickmen(d_annot[ep][fr])
            data['data'][...,idx] = np.asarray(res_img)
            tmppoints = icvt.convert_AHEcoor_to_joints8(data['annotation'][...,idx])
            
            data['joints8'][...,idx] = TransformPoints(tmppoints, det, np.asarray(res_img.size) - 1)
            data['oribbox'][...,idx] = det
            idx = idx + 1
            
    data['data'] = data['data'].reshape((-1, ndet), order='F')
    if save_path is not None:
        icvt.ut.pickle(save_path, data) 
    return data

def MergeAHEBuffyDetection(est_joints8, est_det, est_bbox, gt_annot, ep_arr, fr_arr):
    """
    ep_arr.shape[-1] = fr_arr.shape[-1] = est_joints8.shape[-1]
    est_joints8: the estimated pose in joints8 format
                   the value is relative to the bounding box
    est_det: is the detection window in the original image
    est_bbox: is the bounding box in the original image
    gt_annot: is the annotation for pose in AHEcoor format
               the value is under the original image
    ep_arr, fr_arr: represent the ep and frame number in buffy data set
    """
    import iconvnet_datacvt as icvt
    import Stickmen
    if ep_arr.shape[-1] != fr_arr.shape[-1] or \
      fr_arr.shape[-1] != est_bbox.shape[-1] or \
      fr_arr.shape[-1] != est_det.shape[-1] or \
      fr_arr.shape[-1] != est_joints8.shape[-1]:
        raise BuffyError('ep_arr,  fr_arr, est_det, est_bbox,  est_joints8 should have the same number of last dimension')
    nsample = est_joints8.shape[-1]
    pre_m = (ep_arr[0], fr_arr[0])
    d = dict()
    d['est_coor'] = icvt.convert_joints8_to_AHEcoor(est_joints8[...,0])
    d['est_det'] = est_det[...,0]
    d['est_bbox'] = est_bbox[...,0]
    d['gt_coor'] = gt_annot[...,0]
    d['gt_det'] = np.asarray(Stickmen.EstDetFromStickmen(d['gt_coor']))
    
    l = [d]
    #print est_det[...,42]
    
    for i in range(1, nsample):
        m = (ep_arr[i], fr_arr[i])
        ecoor = icvt.convert_joints8_to_AHEcoor(est_joints8[...,i])
        edet = est_det[...,i]
        ebbox = est_bbox[...,i]
        if (m == pre_m):                  
            l[-1]['est_coor'] = np.concatenate((l[-1]['est_coor'], ecoor), axis=-1)
            l[-1]['est_det'] = np.concatenate((l[-1]['est_det'],edet), axis=-1)
            l[-1]['est_bbox'] = np.concatenate((l[-1]['est_bbox'],ebbox), axis=-1)
        else:
            d = dict()
            d['est_coor'] = ecoor
            d['est_det'] = edet
            d['est_bbox'] = ebbox
            d['gt_coor'] = gt_annot[...,i]
            d['gt_det'] = np.asarray(Stickmen.EstDetFromStickmen(d['gt_coor']))
            l += [d]
        pre_m = m
    return l
def EvaluatePCPFromMergedResult(Res_list,  data_dim, verbose=False, pcp_alpha=0.5,iou_thresh=0.5):
    """
    Res_list is returned by MergeAHEBuffyDetection
    data_dim is the size of resized data of bounding box , say (112,112,3)
    
    """
    import ipyml.geometry as igeo
    import iconvnet_datacvt as icvt
    Mkrec = lambda det: ((det[0],det[1]), (det[2], det[3]))
    CalcA = lambda rec: (rec[1][1]-rec[0][1]) * (rec[1][0]-rec[0][0]) if len(rec)==2 else 0
    pcp_stat = np.zeros((6, len(Res_list)), dtype=np.single)
    matched = -1 * np.ones((len(Res_list)), dtype=np.int)
    det_id = 0    
    for i,d in enumerate(Res_list):
        est_coor = d['est_coor'].reshape((4,6,-1), order='F')
        est_det = d['est_det'].reshape((4,-1), order='F')

        est_bbox = d['est_bbox'].reshape((4,-1),order='F')
        gt_coor = d['gt_coor'].reshape((4,6),order='F')
        gt_det=  d['gt_det'].reshape((4),order='F')
        rec_det = Mkrec(gt_det)
        area = CalcA(rec_det)
        matched[i] = -(det_id + 1)
        if area == 0:
            print 'Mis Detection occur'
            det_id = det_id + est_det.shape[-1]
            # Mis detection occur, This code will never be called, since area!=0
            continue
        
        for j in range(est_det.shape[-1]):
            est_rec = Mkrec(est_det[...,j])
            area_int = CalcA(igeo.RectIntersectRect(est_rec, rec_det))
            ebbox = est_bbox[...,j]
            if area_int > (area - area_int) * iou_thresh :
                
                t_est_coor = TransformPoints(est_coor[...,j].reshape((2,-1),order='F').transpose(), ebbox, np.asarray([data_dim[1],data_dim[0]])-1, inv=True).transpose().reshape((4,6),order='F')
                pcp = icvt.calc_PCP_from_AHEcoor(gt_coor, t_est_coor, pcp_alpha)
                pcp_stat[...,i] = pcp.reshape((len(icvt.AHE_names)))
                matched[i] = det_id + j 
                break
        det_id = det_id + est_det.shape[-1]
    if verbose:
        pcp_stat = pcp_stat.sum(axis=1)/pcp_stat.shape[-1]*100
        print ('PCP(%.6f)' %  pcp_alpha)
        for i in range(6):
            print('| %s \t | %.3f %%' % (icvt.AHE_names[i], pcp_stat[i])) 
            #print 'PCP('+ icvt.AHE_names[i] +   ')=' + str(pcp_stat[i]) + '%'
        print '======='
    return pcp_stat, matched
            
def TransformPoints(points,bbox, tosize, inv=False):
    # Note that bbox is in (x,y, x1, y1) format
    # and *each row* in points is a point
    # 
    # note that bbox can contain negative value
    #   in that case, zero padding are added
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
def TransformPtMatrix(points_matrix, bbox, tosize, inv=False):
    """
    Each colomn in points_matrix is a piont
    """
    sx = (tosize[0])/(bbox[2] - bbox[0]) if (bbox[2]!=bbox[0]) else 1e9
    sy = (tosize[1])/(bbox[3] - bbox[1]) if (bbox[3]!=bbox[1]) else 1e9
    res_matrix = np.zeros(points_matrix.shape, dtype=np.float32)
    bbox = np.require(bbox, dtype=np.single).reshape((4,1))
    S = np.require([sx,sy],dtype=np.single).reshape((2,1))
    if not inv:
        # print points_matrix
        # print bbox[[0,1]]
        # print S
        res_matrix = (points_matrix - bbox[[0,1]]) * S
    else:
        res_matrix = (points_matrix/S) + bbox[[0,1]] 
    return res_matrix
def TransformJoints8Data(jts8, bbox, tosize, inv=False):
    """
    jts8 is 16 x ndata
    bbox is 4 x ndata
    tosize is (width and height)
    """
    if jts8.shape[-1] != bbox.shape[-1]:
        raise BuffyError('jts8 and bbox should have the same size ' + \
                     str(jts8.shape) + 'and' + str(bbox.shape))
    ndata = jts8.shape[-1]
    
    res_jts8 = np.ndarray(jts8.shape, dtype=np.float32)
    for i in range(ndata):
        res_jts8[...,i] = TransformPtMatrix(jts8[...,i].reshape((2,-1),order='F'), bbox[...,i], tosize, inv).reshape((16),order='F')
    
    
    return res_jts8
