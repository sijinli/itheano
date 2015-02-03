"""
Test whether I can put all the images into the memory
"""
from init_test import *
from time import time
import Image
def memory_test():
    print 'Start'
    start_time = time()
    ndata = 170000
    res = np.zeros((112*112*3, ndata),dtype=np.float32)
    print 'Allocation cost {} seconds'.format(time()- start_time)
    print 'Finished res shape is {}'.format(res.shape)
def test_load():
    start_time = time()
    data_path = '/opt/visal/tmp/for_sijin/Data/H36M/H36MExp/folder_ASM_act_14_exp_3/batches.meta'
    meta = mio.unpickle(data_path)
    t_load_test = 10000
    images_path = meta['images_path'][:t_load_test]
    # res = np.zeros((ndata, 128*128*3),dtype=np.float32)
    # print 'Allocating memeory cost {} seconds'.format(time()- start_time)
    start_time = time()
    res_list = [np.array(Image.open(p),dtype=np.float32).reshape((-1,1),order='F') for p in images_path]
    res = np.concatenate(res_list, axis=1)
    print 'Loading {} images cost {} seconds'.format(t_load_test, time()- start_time)
def main():
    # memory_test()
    test_load()

if __name__ == '__main__':
    main()
    