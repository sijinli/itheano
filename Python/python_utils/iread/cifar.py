import numpy as np
class CifarError(Exception):
    pass
img_size = (112,112,3)
dim_data = img_size[0] * img_size[1] * 3

def PrepareData(ndata):
    """
    This function will create empty data of cifar format
    ndata is the number of data
    """
    ind_map_shape = (7,8,8,ndata)
    d = dict()
    d['data'] = np.ndarray((dim_data, ndata), dtype=np.uint8,order='F')
    d['label'] = None
    d['joints8'] = np.zeros((16,ndata),dtype=np.float32)
    d['oribbox'] = np.zeros((4,ndata), dtype=np.int)
    d['indmap'] = np.zeros(ind_map_shape,dtype=np.bool)
    d['ind_para'] = None
    return d
def GetNumofData(d):
    return d['data'].shape[-1]
def MakeDataFromImages(imgdir, max_per_batch , save_dir = None, save_name=None):
    import iutils as iu
    import iconvnet_datacvt as icvt
    from PIL import Image
    if max_per_batch == 0:
        raise CifarError('max_per_batch can''t not be zero')
    allfiles = iu.getfilelist(imgdir, '.*jpg|.*bmp|.*png$')
    ndata = len(allfiles)
    iu.ensure_dir(save_dir)
    d = PrepareData(min(max_per_batch, ndata))
    j = 0
    if save_name is None:
        save_name = 'data_batch'
    bid = 1
    for i,fn in enumerate(allfiles):
        if j == max_per_batch:
            j = 0
            if not save_dir is None:
                icvt.ut.pickle(iu.fullfile(save_dir, save_name + '_' + str(bid)), d)
                bid = bid + 1 
            if ndata - i < max_per_batch:
                d = PrepareData(ndata-i)
        fp = iu.fullfile(imgdir, fn)
        
        img = iu.imgproc.ensure_rgb(np.asarray(Image.open(fp)))
        img = Image.fromarray(img).resize((img_size[0],img_size[1]))
        arr_img = np.asarray(img).reshape((dim_data), order='F')
        d['data'][...,j] = arr_img
        j = j + 1
    if not save_dir is None:
        icvt.ut.pickle(iu.fullfile(save_dir, save_name + '_' + str(bid)), d)         
