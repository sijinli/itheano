from init_test import *
from ilayer import *
from igraphparser import *
import Image
import pylab as pl
def test_dropout():
    ndata, ndim = 100, 200
    a = np.ones((256, 1000), dtype=theano.config.floatX)

    file_path  = '/home/grads/sijinli2/Projects/Itheano/doc/netdef/graph_def_0006.cfg'
    g = GraphParser(file_path)
    net = Network(g.layers, g.network_config['network'])
    f = theano.function(inputs=net.inputs,
                        outputs=net.outputs
    )

    dlay = g.layers['drop0'][2]
    print 'keep = {}'.format(dlay.keep)
    for i in range(5):
        net.set_dropout_on(True)
        res = f(a)[0]
        s = a.sum()
        keep = dlay.keep
        print 'sum = {}, exp sum ={} dtype={}'.format(res.sum(),s * keep, res.dtype)
        net.set_dropout_on(False)
        res= np.require(f(a)[0],dtype=np.float)
        print 'sum = {}, exp sum ={} dtype={}'.format(res.sum(), s * keep, res.dtype)
        
def test_conv():
    file_path = '/home/grads/sijinli2/Projects/Itheano/doc/netdef/graph_def_0005.cfg'
    g = GraphParser(file_path)
    layers = g.layers
    GraphParser.print_graph_connections(layers)
    net_config = g.network_config['network']
    net = Network(layers, net_config)
    img = Image.open('/public/sijinli2/tmp_test/pic_test/cat.jpg')
    nh = 100
    nw = 100
    img = img.resize((nh, nw))
    img_arr = np.array(img, dtype=theano.config.floatX)
    pl.subplot(2,1,1)
    pl.imshow(np.array(img_arr,dtype=np.uint8))
    img_arr = img_arr.transpose([2,0,1])
    img_arr = img_arr.reshape((1, 3, nh, nw), order='F')
    print net.outputs
    l = net.outputs[0]
    f= theano.function(inputs=net.inputs,
                       outputs=net.outputs
    )
    res_img = f(img_arr)[0]
    print 'OK'
    print 'shape is {}'.format(res_img[0].shape)
    print 'The desired outpts shape is {}'.format(layers['pool1'][2].param_dic['output_dims'])
    pl.subplot(2,1,2)
    print 'res_img of type {}'.format(type(res_img))
    nc, sx, sy = layers['pool1'][2].param_dic['output_dims'][0]
    show_img = np.array(res_img, dtype=np.uint8).reshape((nc,sx,sy)).transpose([1,2,0])
    pl.imshow(show_img)
    pl.show()
    
def main():
    # test_conv()
    test_dropout()
    
if __name__ == '__main__':
    main()
