"""
Basic rules for data format

All the data will be a matrix
   [ndata, dim]
All the weights will be n_input x n_output matrix
"""
import theano
import theano.tensor as tensor
import numpy as np
import sys
from theano.tensor.signal import downsample
#
from actfunc import *
import iread.myio as mio

dnn_available = False

if theano.config.device.startswith("gpu"):
    from theano.sandbox.cuda import dnn
    if dnn.dnn_available():
        dnn_available = True

class LayerException(Exception):
    pass
class LayerParser(object):
    def __init__(self, name, inputs, mcp, input_dims, advanced_params=None):
        self.inputs = inputs
        self.mcp = mcp
        self.name = name
        self.input_dims = input_dims
        self.advanced_params = advanced_params
    def str_to_list(self, s,interpret_as=float):
        s_list = s.split(',')
        return [interpret_as(x) for x in s_list]
    def parse_neuron_params(self, s):
        p = s.find('[')
        nname, nparams = [s, None] if p == -1 else [s[:p], self.str_to_list(s[p+1:s.rfind(']')])]
        return nname, nparams
    def parse_layer_params(self):
        """
        Return the Layer parameber dic
        It will be called by parse
        """
        # print self.input_dims, '================>>>>>>'
        dic = {'input_dims':self.input_dims,'name':self.name}
        # self.mcp.check_options(self.name, ['input_dims'])
        if self.mcp.has_option(self.name, 'neuron'):
            s = self.mcp.safe_get(self.name, 'neuron')
            nname, nparams = self.parse_neuron_params(s)
            dic['actfunc'] = {'name':nname, 'act_params':nparams}
        return dic
    def parse(self):
        raise LayerException('Can not instantiate a Layer base class')

class LayerWithWeightsParser(LayerParser):
    @classmethod
    def parse_external_func(cls, st):
        p0 = st.find('.')
        p1 = st.find('(',p0)
        p2 = st.rfind(')')
        module_name, func_name, tmp_params = st[:p0].strip(), st[p0+1:p1].strip(), st[p1+1:p2]
        func_params = [p.strip() for p in tmp_params.split(',')]
        module = __import__(module_name, globals(), locals(), [func_name])
        f = eval('module.{}'.format(func_name))
        def func(name, sp, params=None):
            return f(name, sp, func_params)
        return func, func_params
    def parse_layer_params(self):
        try:
            dic = LayerParser.parse_layer_params(self)
            self.mcp.check_options_with_excuse(self.name, ['initW', 'initb'],
                                               {'initW':['initWfunc', 'weightSource'],
                                                'initb':['initBfunc', 'biasSource']})
            n = len(self.inputs)
            dic['initW'] = self.mcp.safe_get_float_list(self.name, 'initW',default=[0] * n)
            dic['initb'] = self.mcp.safe_get_float_list(self.name, 'initb',default=[0] * n)
            dic['wd'] = self.mcp.safe_get_float_list(self.name, 'wd')
            # epsW, epsB is the (relative) learning rate for weights and biases
            dic['epsW'] = self.mcp.safe_get_float_list(self.name, 'epsW', default=[1.0] * n)
            dic['epsB'] = self.mcp.safe_get_float_list(self.name, 'epsB', default= [1.0])
            dic['momW'] = self.mcp.safe_get_float_list(self.name, 'momW', default=[0.9] * n)
            dic['momB'] = self.mcp.safe_get_float_list(self.name, 'momB', default=[0.9])

            for e in ['initWfunc', 'initBfunc']:
                if self.mcp.has_option(self.name, e):
                    st = self.mcp.safe_get(self.name, e)
                    func, func_params = self.parse_external_func(st)
                    dic[e] = func
            if len(dic['epsW']) != len(dic['wd']):
                raise LayerException('len (epsW) != len (wd)')
            if self.advanced_params:
                for e in ['weights', 'biases', 'weights_inc', 'biases_inc']:
                    if e in self.advanced_params:
                        dic[e] = self.advanced_params[e]
            if len(dic['initW']) != len(self.inputs):
                raise LayerException('{}:#initW {}!=#inputs {}'.format(self.name,
                                                                       len(dic['initW']),
                                                                       len(self.inputs)))
        except Exception as err:
            print err, 'LayerWithWeightsParser'
            sys.exit(1)
        assert(len(dic['initb']) == 1)
        return dic

        # if len(self.inputs) != 
class Layer(object):
    """
    inputs will be the list of all the inputs
    Also, outputs will be a list
    """
    def __init__(self, inputs, param_dic=None):
        self.inputs = inputs
        self.outputs = None
        self.activation_func = None
        self.params = None
        self.param_dic = dict()
        self.required_field = ['input_dims', 'name']
    @classmethod
    def get_var_name(cls, name, var_name):
        return '{}.{}'.format(name, var_name)
    def set_output_names(cls, name, outputs):
        for i,e in enumerate(outputs):
            e.name = cls.get_var_name(name, 'outputs[{}]'.format(i))
    def copy_from_saved_layer(self, saved_layers):
        # print self.param_dic.keys()
        # print saved_layers.keys()
        # print '<<<<<<<<<<<<<<<<<<'
        assert(saved_layers['type'] == self.param_dic['type'])
    def parse_param_dic(self, param_dic):
        try:
            if not param_dic:
                raise LayerException('Empty parameters')
            for e in self.required_field:
                if not e in param_dic:
                    print param_dic
                    raise LayerException('Required field {} not in paramers'.format(e))
                else:
                    self.param_dic[e] = param_dic[e]
            if 'actfunc' in param_dic:
                self.activation_func = make_actfunc(param_dic['actfunc']['name'],
                                                    param_dic['actfunc']['act_params'])
                self.param_dic['actfunc'] = param_dic['actfunc']
            self.name = self.param_dic['name']
        except LayerException,err:
            print err
            sys.exit(1)
    def save_to_dic(self):
        return self.param_dic.copy()

class LayerWithWeightsLayer(Layer):
    def __init__(self, inputs, param_dic=None):
        Layer.__init__(self, inputs, param_dic)
        self.required_field += ['output_dims', 'wd', 'epsW', 'epsB', 'momW', 'momB']
    def copy_from_saved_layer(self, sl):
        Layer.copy_from_saved_layer(self, sl)
        for W,W_value in zip(self.W_list, sl['weights']):
            W.set_value(np.array(W_value,dtype=theano.config.floatX))
        self.b.set_value(np.array(sl['biases'][0],dtype=theano.config.floatX))
        print '{}: copy weights|biases from {} layer successfully'.format(self.param_dic['name'], sl['name'])
        if 'weights_inc' in sl:
            for W_inc,W_inc_value in zip(self.W_inc_list, sl['weights_inc']):
                W_inc.set_value(np.array(W_inc_value,dtype=theano.config.floatX))
            self.b_inc.set_value(np.array(sl['biases_inc'][0], dtype=theano.config.floatX))
    def initW_inc(self):
        
        self.W_inc_list = [theano.shared(np.asarray(e.get_value() * 0.0, dtype=theano.config.floatX)) for e in self.W_list]
        for idx, e in enumerate(self.W_inc_list):
            e.name = '%s_weights_inc_%d' % (self.param_dic['name'], idx)
    def initb_inc(self):
        self.b_inc = theano.shared(np.asarray(self.b.get_value() * 0.0,
                                              dtype=theano.config.floatX))
        self.b_inc.name = '%s_biasinc_%d' % (self.param_dic['name'], 0)
    @classmethod
    def cvt2sharedfloatX(cls, X, var_base_name = None):
        res = [theano.shared(np.cast[theano.config.floatX](x)) for x in X]
        if var_base_name is not None:
            for idx, r in enumerate(res):
                r.name = '%s_%d' % (var_base_name, idx)
        return res
    def get_w_shape(self, param_dic):
        raise Exception('get w shape in base class')
    def get_b_shape(self, param_dic):
        raise Exception('get_b_shape:in base class')
    def initW(self, param_dic):
        W_list = []
        sp_list = self.get_w_shape(param_dic)
        for i, sp in enumerate(sp_list):
            W_value = np.require(np.random.randn(*sp) * param_dic['initW'][i],
                                 dtype=theano.config.floatX)
            W_list += [theano.shared(W_value,borrow=False, name='%s_weights_%d' % (param_dic['name'],i))]
        self.W_list = W_list
    def initb(self, param_dic):
        s = param_dic['initb']
        sp = self.get_b_shape(param_dic)
        b_value = np.ones(sp,dtype=theano.config.floatX) * np.cast[theano.config.floatX](s) 
        self.b = theano.shared(b_value,borrow=False)
        self.b.name = '%s_biases_%d' % (param_dic['name'], 0)
    def parse_param_dic(self, param_dic):
        Layer.parse_param_dic(self, param_dic)
        if not 'weights' in param_dic:
            if 'initWfunc' in param_dic:
                sp_list = self.get_w_shape(param_dic)
                W_value_list = param_dic['initWfunc'](param_dic['name'], sp_list)
                self.W_list = [theano.shared(np.cast[theano.config.floatX](W_value),
                                             borrow=False,
                                             name='%s_weights_%d' % (param_dic['name'],i))
                               for i, W_value in enumerate(W_value_list)]
            else:
                self.initW(param_dic)
            self.initW_inc()
        else:
            self.W_list = param_dic['weights']
            self.W_inc_list = param_dic['weights_inc']
            print '    Init weights from outside<' + ','.join([w.name for w in self.W_list]) + '>'
        if not 'biases' in param_dic:
            if 'initBfunc' in param_dic:
                sp = self.get_b_shape(param_dic)
                b_value = param_dic['initBfunc'](param_dic['name'], sp)
                self.b= theano.shared(np.cast[theano.config.floatX](b_value),
                                      name='%s_biases_%d' % (param_dic['name'], 0),
                                      borrow=False)
            else:
                self.initb(param_dic)
            self.initb_inc()
        else:
            self.b = param_dic['biases']
            self.b_inc = param_dic['biases_inc']
            print '    Init biase from outside {}'.format(self.b.name)
        if len(param_dic['wd']) != len(self.W_list):
            raise LayerException('weight decay list has {} elements != {}'.format(len(param_dic['wd']), len(self.W_list)))
        self.wd = self.cvt2sharedfloatX(self.param_dic['wd'], '%s_wd' % param_dic['name'])
        self.epsW = self.cvt2sharedfloatX(self.param_dic['epsW'],'%s_epsW' % param_dic['name'])
        self.epsB = self.cvt2sharedfloatX(self.param_dic['epsB'],'%s_epsB' % param_dic['name'])
        self.momW = self.cvt2sharedfloatX(self.param_dic['momW'],'%s_momW' % param_dic['name'])
        self.momB = self.cvt2sharedfloatX(self.param_dic['momB'],'%s_momB' % param_dic['name'])

        self.params = self.W_list + [self.b]
        self.params_eps= self.epsW + self.epsB
        self.params_inc = self.W_inc_list + [self.b_inc]
        self.params_mom = self.momW + self.momB
    def save_to_dic(self):
        save_dic = self.param_dic.copy()
        # Those value might be saved in gpu or it is of function type
        for e in ['weights', 'biases', 'weights_inc', 'biases_inc', 'initWfunc', 'initBfunc']:
            if e in save_dic:
                del save_dic[e]
        save_dic['weights'] = [w.get_value() for w in self.W_list]
        save_dic['biases'] = [self.b.get_value()]
        save_dic['weights_inc'] = [w_inc.get_value() for w_inc in self.W_inc_list]
        save_dic['biases_inc'] = [self.b_inc.get_value()]
        save_dic['epsW'] = [e.get_value() for e in self.epsW]
        save_dic['epsB'] = [e.get_value() for e in self.epsB]
        return save_dic
    def get_regularizations(self):
        return sum([(W**2).sum() * decay for W, decay in zip(self.W_list, self.wd)])


class DropoutParser(LayerParser):
    def parse_layer_params(self):
        dic = LayerParser.parse_layer_params(self)
        dic['keep'] = self.mcp.safe_get_float(self.name, 'keep')
        if dic['keep'] > 1 or dic['keep'] < 0:
            raise LayerException('Invalid value {} for keep'.format(dic['keep']))
        return dic
    def parse(self):
        assert(len(self.inputs) == 1)
        dic = self.parse_layer_params()
        return DropoutLayer(self.inputs, dic)
class DropoutLayer(Layer):
    ### use to generate the seed for each dropout layer
    __dropout_seed_srng = np.random.RandomState(0)
    def __init__(self, inputs, param_dic=None):
        Layer.__init__(self, inputs, param_dic)
        self.required_field += ['keep']
        if param_dic:
            self.parse_param_dic(param_dic)
        self.train_mode = True
        flag = 1.0 if self.train_mode else 0.0
        self.seed = DropoutLayer.__dropout_seed_srng.randint(0, sys.maxint)
        self.srng = tensor.shared_randomstreams.RandomStreams(self.seed)
        self.dropout_on = theano.shared(np.cast[theano.config.floatX](flag), \
                                     borrow=True)
        self.mask = self.srng.binomial(n=1, p=self.keep, size=self.inputs[0].shape)
        self.param_dic['output_dims'] = self.param_dic['input_dims']
        one = theano.shared(np.cast[theano.config.floatX](1.0))
        raw_outputs = self.dropout_on * self.inputs[0] * tensor.cast(self.mask, theano.config.floatX) + (one - self.dropout_on) * self.keep * self.inputs[0]
        if self.activation_func:
            self.outputs = [self.activation_func(raw_outputs)]
        else:
            self.outputs = [raw_outputs]
        self.set_output_names(self.param_dic['name'], self.outputs)
        self.param_dic['type'] = 'dropout'
    def parse_param_dic(self, param_dic):
        Layer.parse_param_dic(self, param_dic)
        self.keep = param_dic['keep']
    def set_dropout_on(self, train=True):
        if train != self.train_mode:
            flag = 1.0 if train else 0.0
            self.dropout_on.set_value(flag)
            self.train_mode = train
        
class DataParser(LayerParser):
    def parse(self):
        param_dic = self.parse_layer_params()
        print 'Parse Layer {} \n \tComplete {}'.format(param_dic['name'],param_dic)
        return DataLayer(self.inputs, param_dic)
        
class DataLayer(Layer):
    def __init__(self, inputs, param_dic=None):
        assert(len(inputs) == 1)
        Layer.__init__(self, inputs, param_dic)
        if param_dic:
            self.parse_param_dic(param_dic)
        if self.activation_func:
            self.inputs = inputs
            self.outputs = [self.activation_func(e) for e in self.inputs]
        else:
            self.inputs = self.outputs = inputs
        self.param_dic['type'] = 'data'
        self.param_dic['input_dims'] = param_dic['input_dims']
        self.param_dic['output_dims'] = self.param_dic['input_dims']
        # for e1,e2 in zip(self.inputs, self.outputs):
        #     print '{} ----> {}\n\n'.format(e1,e2)
        for idx,e in enumerate(self.outputs):
            e.name = self.get_var_name(self.param_dic['name'], 'outputs[{}]'.format(idx))
class ElementwiseSumParser(LayerParser):
    def parse(self):
        dic = LayerParser.parse_layer_params(self)
        if self.mcp.has_option(self.name, 'coeffs'):
            dic['coeffs'] = self.mcp.safe_get_float_list(self.name, 'coeffs')
        print 'Parse Layer {} \n \tComplete {}'.format(dic['name'],dic)
        return ElementwiseSumLayer(self.inputs, dic)
class ElementwiseSumLayer(Layer):
    def __init__(self, inputs, param_dic=None):
        Layer.__init__(self, inputs, param_dic)
        self.inputs = inputs
        if param_dic:
            self.parse_param_dic(param_dic)
        assert(len(inputs)>1)
        if self.coeffs:
            raw_outputs = [sum([e * c for e,c in zip(inputs, self.coeffs)])]
        else:
            raw_outputs = [sum(self.inputs)]
        if self.activation_func:
            self.outputs = [self.activation_func(e) for e in raw_outputs]
        else:
            self.outputs = raw_outputs
        self.param_dic['type'] = 'fc'
        self.param_dic['output_dims'] = [param_dic['input_dims'][0]]
        for idx,e in enumerate(self.outputs):
            e.name = self.get_var_name(self.param_dic['name'], 'outputs[{}]'.format(idx))
    def parse_param_dic(self, param_dic):
        Layer.parse_param_dic(self, param_dic)
        if 'coeffs' in param_dic:
            n_input = len(self.inputs)
            if len(param_dic['coeffs']) != n_input:
                raise LayerException('The number of input mat {} vs {} coeffs'.format(n_input, len(param_dic['coefffs'])))
            self.coeffs = [theano.shared(x) for x in param_dic['coeffs']]
        else:
            self.coeffs = None
class FCParser(LayerWithWeightsParser):
    def parse(self):
        dic = LayerWithWeightsParser.parse_layer_params(self)
        if self.mcp.has_option(self.name, 'output_dims'):
            dic['output_dims'] = self.mcp.safe_get_int_list(self.name, 'output_dims')
        elif self.mcp.has_option(self.name, 'outputs'): # to support convnet files
            dic['output_dims'] = self.mcp.safe_get_int_list(self.name, 'outputs')
        else:
            raise LayerException('Fully connected layer missing outputs or output_dims')
        print 'Parse Layer {} \n \tComplete {}'.format(self.name, dic)
        return FCLayer(self.inputs, dic)

class FCLayer(LayerWithWeightsLayer):
    """
    Fully connected layer
    """
    def __init__(self, inputs, param_dic = None):
        LayerWithWeightsLayer.__init__(self, inputs, param_dic)
        self.parse_param_dic(param_dic)
        for W,din in zip(self.W_list, param_dic['input_dims']):
            din_r = din if type(din) is int else int(np.prod(din))
            self.check_shape(W, (din_r, self.param_dic['output_dims'][0]))
        self.check_shape(self.b, (self.param_dic['output_dims'][0],))
        has_tuple = np.any([type(x) is tuple for x in self.param_dic['input_dims']])
        if has_tuple:
            lin_outputs = sum([tensor.dot(elem.flatten(ndim=2), W) for elem, W in zip(inputs, self.W_list)]) + self.b
        else:
            lin_outputs = sum([tensor.dot(elem, W) for elem, W in zip(inputs, self.W_list)]) \
                          + self.b
        if self.activation_func:
            self.outputs = [self.activation_func(lin_outputs)]
        else:
            self.outputs = [lin_outputs]
        # self.eps_parmas = self.eps
        self.param_dic['type'] = 'fc'
        self.set_output_names(self.param_dic['name'], self.outputs)
    def check_shape(self, tensor_a, shape):
        if not tensor_a.get_value().shape == shape:
            raise LayerException('Tensor {} shape is not correct {} vs {}'.format(tensor_a.name, tensor_a.get_value().shape, shape))
    def get_w_shape(self, param_dic):
        return [(int(np.prod(d)), param_dic['output_dims'][0]) for d in param_dic['input_dims']]
    def get_b_shape(self, param_dic):
        return (param_dic['output_dims'][0],)


class ConvParser(LayerWithWeightsParser):
    def __init__(self, name, inputs, mcp, input_dims, advanced_params=None):
        #batch size, stack size, nb row, nb col
        # input_dims = [ [s1,nr,nc], [s2,nr,nc],...]
        LayerWithWeightsParser.__init__(self, name, inputs, mcp, input_dims, advanced_params)
    @classmethod
    def get_output_conv_dims(cls, L, F_size, stride, pad):
        """
        Assume it is valid mode
        """
        # print 'pad is {}, type of pad is {}, stride is {}'.format(pad, type(pad), stride)
        return (L +  pad * 2 - F_size) // stride + 1
    def parse_layer_params(self):
        dic = LayerWithWeightsParser.parse_layer_params(self)
        mcp, name = self.mcp, self.name
        dic['sharebias'] = mcp.safe_get_int(name, 'sharebias') if mcp.has_option(name, 'sharebias') else 1
        dic['sizeX'] = mcp.safe_get_int_list(name, 'sizeX')
        dic['sizeY'] = mcp.safe_get_int_list(name, 'sizeY')
        dic['filters'] = mcp.safe_get_int(name, 'filters')
        dic['strideX'] = mcp.safe_get_int_list(name, 'strideX')
        dic['strideY'] = mcp.safe_get_int_list(name, 'strideY')
        dic['board_mode'] = mcp.safe_get(name, 'board_mode') if mcp.has_option(name, 'board_mode') else None
        dic['pad'] = mcp.safe_get_int_list(name, 'pad') if mcp.has_option(name, 'pad') else None
        if dic['board_mode'] and dic['pad']:
            raise LayerException('You are advised to specifly either pad or mode (not both)')
        if (not dic['board_mode']) and (not dic['pad']):
            raise LayerException('Either mode or pad should be specified')
        dic['conv_mode'] = mcp.safe_get(name, 'conv_mode') if mcp.has_option(name, 'conv_mode') else 'conv'
        if dic['board_mode']:
            if dic['board_mode'] not in ['valid', 'full']:
                raise LayerException('Invalid board_mode {}'.format(dic['board_mode']))
            dic['pad'] = [int(0),int(0)] if dic['board_mode'] == 'valid' else [dic['sizeX'] - 1,
                                                                               dic['sizeY'] - 1]
        if len(dic['pad']) == 1:
            dic['pad'] *= 2
        dic['pad'] = tuple(dic['pad'])
        assert(len(dic['sizeX']) == len(dic['sizeY']))

        def get_dims(i):
            Lx,Ly = dic['input_dims'][i][2], dic['input_dims'][i][1]
            sx,sy = dic['sizeX'][i], dic['sizeY'][i]
            strideX, strideY = dic['strideX'][i], dic['strideY'][i]
            return (self.get_output_conv_dims(Lx, sx, strideX, dic['pad'][0]), self.get_output_conv_dims(Ly, sy, strideY, dic['pad'][1]))
        output_dims = [get_dims(k) for k in range(len(dic['input_dims']))]
        if len(set(output_dims)) != 1:
            raise LayerException('The output dimension is not consistent {}'.format(output_dims))
        dim = output_dims[0]
        dic['output_dims'] = [(dic['filters'], dim[1], dim[0])]
        return dic
    def parse(self):
        dic = self.parse_layer_params()
        print 'Parse Layer {} \n \tComplete {}'.format(dic['name'],dic)
        return ConvDNNLayer(self.inputs, dic)
class ConvDNNLayer(LayerWithWeightsLayer):
    def __init__(self, inputs, param_dic=None):
        assert(dnn_available)
        LayerWithWeightsLayer.__init__(self, inputs, param_dic)
        self.required_field += ['sizeX', 'sizeY', 'filters', 'strideX', 'strideY',
                                'pad', 'board_mode', 'conv_mode', 'sharebias']
        if param_dic:
            self.parse_param_dic(param_dic)
        self.param_dic['type'] = 'convdnn'
        board_mode = self.param_dic['pad']
        conv_mode = self.param_dic['conv_mode']
        strides = zip(self.param_dic['strideX'], self.param_dic['strideY'])
        # lin_outputs = sum([dnn.dnn_conv(img=I,
        #                                 kerns=W,
        #                                 subsample=s,
        #                                 border_mode=board_mode,
        #                                 conv_mode=conv_mode
        #                             ) for I,W,s in zip(inputs, self.W_list, strides)]) + self.b
        if self.param_dic['sharebias']:
            biases = self.b.dimshuffle('x', 0, 'x', 'x')
        else:
            biases = self.b.dimshuffle('x', 0,1,2)
        lin_outputs = sum([dnn.dnn_conv(img=img,
                                       kerns=W,
                                       subsample=s,
                                       border_mode=board_mode,
                                       conv_mode=conv_mode)
                           for img,W,s in zip(inputs, self.W_list, strides)]) + biases
                                   
        if self.activation_func:
            self.outputs = [self.activation_func(lin_outputs)]
        else:
            self.outputs = [lin_outputs]
        self.set_output_names(self.param_dic['name'], self.outputs)
    def get_w_shape(self,  param_dic):
        filters= param_dic['filters']
        return [(filters, d[0], sy, sx) for d, sy, sx in zip(param_dic['input_dims'],
                                                             param_dic['sizeY'],
                                                             param_dic['sizeX'])]
    def get_b_shape(self, param_dic):
        flag, dims = param_dic['sharebias'], param_dic['output_dims'][0]
        return (dims[0],) if flag == 1 else (dims[0], dims[1], dims[2])
            
class PoolParser(LayerParser):
    def parse_layer_params(self):
        dic = LayerParser.parse_layer_params(self)
        try:
            # int_var = ['sizeX', 'sizeY', 'strideX', 'strideY']
            dic['sizeX'] = self.mcp.safe_get_int(self.name, 'sizeX')
            dic['sizeY'] = self.mcp.safe_get_int(self.name, 'sizeY')
            dic['strideX'] = self.mcp.safe_get_int(self.name, 'strideX')
            dic['strideY'] = self.mcp.safe_get_int(self.name, 'strideY')
            dic['pooling_type'] = self.mcp.safe_get(self.name, 'pooling_type')
            dic['ignore_boarder'] = self.mcp.safe_get_bool(self.name, 'ignore_boarder',
                                                           default=False)
            dic['pooling_method'] = self.mcp.safe_get(self.name, 'pooling_method', default='convnet')
        except Exception as err:
            print err
            sys.exit(1)
        return dic
    def parse(self):
        dic = self.parse_layer_params()
        # Decide which class to call as this stage
        if dic['pooling_method'] == 'convnet':
            return ConvNetPool2dLayer(self.inputs, dic)
        else:
            raise LayerException('Only support convnet for pooling')


class ConvNetPool2dLayer(Layer):
    """
    Use pylearn2 convnet pooling for max pooling
    requirements: the input should be square image, and of course gpu continuous
    """
    from pylearn2.sandbox.cuda_convnet.pool import MaxPool
    from theano.sandbox.cuda.basic_ops import gpu_contiguous
    def __init__(self, inputs, param_dic):
        Layer.__init__(self, inputs, param_dic)
        self.required_field += ['strideX', 'strideY', 'pooling_type', 'sizeX', 'sizeY']
        self.parse_param_dic(param_dic)
        assert(len(self.param_dic['input_dims']) == 1)
        assert(param_dic['strideX'] == param_dic['strideY'])
        assert(param_dic['sizeX'] == param_dic['sizeY'])
        dim, sX, stride = param_dic['input_dims'][0], param_dic['sizeX'], param_dic['strideX']
        # ignore_boarder = param_dic['ignore_boarder']
        self.param_dic['output_dims'] = [self.get_output_pool_dims(dim,sX, stride)]
        self.pooling_type = self.param_dic['pooling_type']
        self.ignore_boarder = False
        self.param_dic['type'] = 'convnetpool'
        if self.pooling_type == 'max':
            # convert to convnet style
            shuffled_input = self.inputs[0].dimshuffle(1,2,3, 0)
            c_input = self.gpu_contiguous(shuffled_input)
            pool_op = self.MaxPool(ds=sX, stride=stride)
            raw_outputs = pool_op(c_input)
            self.outputs = [raw_outputs.dimshuffle(3,0,1,2)]
        else:
            raise LayerException('Un supported pooling type {}'.format(self.pooling_type))
        self.set_output_names(self.param_dic['name'], self.outputs)
    @classmethod
    def get_output_pool_dims(cls, input_dim, sX, stride):
        """
        input_dims = [filters, Height, Width]
        """
        if input_dim[1] != input_dim[2]:
            raise Exception('In: dim[1] != dim[2]: {}'.format(input_dim))
        imgsize = input_dim[1]
        s = int(np.ceil(float(imgsize - sX) / stride)) + 1
        return (input_dim[0], s, s)
        
class NeuronParser(LayerParser):
    def parse(self):
        assert(len(self.inputs) == 1)
        dic = self.parse_layer_params()
        print 'Parse Layer {} \n \tComplete {}'.format(dic['name'],dic)
        return NeuronLayer(self.inputs, dic)
 
class NeuronLayer(Layer):
    def __init__(self, inputs, param_dic):
        Layer.__init__(self, inputs, param_dic)
        self.required_field += ['actfunc']
        self.parse_param_dic(param_dic)
        self.outputs = [self.activation_func(self.inputs[0])]
        self.set_output_names(self.param_dic['name'], self.outputs)
        self.param_dic['type'] = 'neuron'
        self.param_dic['output_dims'] = self.param_dic['input_dims']
        
class BatchNormParser(LayerParser):
    def parse(self):
        dic = self.parse_layer_params()
        print 'Parse Layer {} \n \tComplete {}'.format(dic['name'],dic)
        assert(len(self.inputs) == 1)
        return BatchNormLayer(self.inputs, dic)
    def parse_layer_params(self):
        dic = LayerParser.parse_layer_params(self)
        dic['epsilon'] = self.mcp.safe_get_float(self.name, 'epsilon', default=1e-6)
        return dic
class BatchNormLayer(Layer):
    def __init__(self, inputs, param_dic):
        Layer.__init__(self, inputs, param_dic)
        self.required_field += ['epsilon']
        self.parse_param_dic(param_dic)
        m = inputs[0].mean(axis=0,keepdims=True)
        centered_inputs = inputs[0] - m
        var = (centered_inputs**2).mean(axis=0, keepdims=True) + param_dic['epsilon']
        raw_outputs = centered_inputs/tensor.sqrt(var)
        if self.activation_func:
            self.outputs = [self.activation_func(raw_outputs)]
        else:
            self.outputs = [raw_outputs]
        self.set_output_names(self.param_dic['name'], self.outputs)
        self.param_dic['output_dims'] = self.param_dic['input_dims']
        self.param_dic['type'] = 'batchnorm'

class ElementwiseScaleParser(LayerWithWeightsParser):
    def parse(self):
        assert(len(self.inputs) == 1)
        dic = LayerWithWeightsParser.parse_layer_params(self)
        dic['output_dims'] = dic['input_dims']
        print 'Parse Layer {} \n \tComplete {}'.format(dic['name'],dic)
        return ElementwiseScaleLayer(self.inputs, dic)

class ElementwiseScaleLayer(LayerWithWeightsLayer):
    def __init__(self, inputs, param_dic):
        LayerWithWeightsLayer.__init__(self, inputs, param_dic)
        self.parse_param_dic(param_dic)
        raw_outputs = self.W_list[0] * inputs[0] + self.b
        if self.activation_func:
            self.outputs = [self.activation_func(raw_outputs)]
        else:
            self.outputs = [raw_outputs]
        self.set_output_names(self.param_dic['name'], self.outputs)
        self.param_dic['output_dims'] = self.param_dic['input_dims']
        self.param_dic['type'] = 'elemscale'

    def get_w_shape(self, param_dic):
        return [(np.prod(d),) for d in param_dic['input_dims']]
    def get_b_shape(self, param_dic):
        return (param_dic['input_dims'][0],)
        
class CostParser(LayerParser):
    def parse_layer_params(self):
        dic = LayerParser.parse_layer_params(self)
        dic['coeff'] = self.mcp.safe_get_float(self.name, 'coeff', default=1.0)
        return dic
        
class CostLayer(Layer):
    """
    self.cost        : shared scalar variable, for  gradient calculation
    self.cost_list   : list of shared scalar variable  
    """
    def __init__(self,inputs, param_dic=None):
        self.params = None
        self.eps_params = None                                                   
        Layer.__init__(self, inputs, param_dic)
        self.required_field += ['coeff']
    def parse_param_dic(self, param_dic):
        Layer.parse_param_dic(self, param_dic)
        self.coeff = self.param_dic['coeff']

class MaxMarginCostParser(CostParser):
    def parse(self):
        dic = self.parse_layer_params()
        print 'Parse Layer {} \n \tComplete {}'.format(dic['name'],dic)
        return MaxMarginCostLayer(self.inputs, dic)
class BinaryCrossEntropyCostParser(CostParser):
    def parse(self):
        dic = self.parse_layer_params()
        assert(len(self.inputs) == 2)
        return BinaryCrossEntropyCostLayer(self.inputs, dic)
class SquareDiffCostParser(CostParser):
    def parse(self):
        dic = self.parse_layer_params()
        assert(len(self.inputs) == 2)
        return SquareDiffCostLayer(self.inputs, dic)
class CosineCostParser(CostParser):
    def parse_layer_params(self):
        dic = CostParser.parse_layer_params(self)
        dic['norm'] = self.mcp.safe_get_float(self.name, 'coeff', default=False)
        if dic['norm']:
            print 'Do normalization on inputs'
        return dic
    def parse(self):
        dic = self.parse_layer_params()
        assert(len(self.inputs) == 2)
        return CosineCostLayer(self.inputs, dic)
class MaxMarginCostLayer(CostLayer):
    """
    Here it will take two inputs and compare the difference of those two inputs
    Assume the inputs[0] is the ground truth scoure
               inputs[1] is the score for most violated pose
    """
    def __init__(self, inputs, param_dic=None):
        CostLayer.__init__(self, inputs, param_dic)
        self.parse_param_dic(param_dic)
        if ('actfunc' in param_dic) and (not param_dic['actfunc']['name'] in ['relu', 'relu2']):
            raise LayerException('CostLayer only can use relu2 or relu as neuron')
        else:
            self.param_dic['actfunc'] = {'name':'relu', 'act_params':None}
            print 'Use Default Relu as Neuron'
            self.activation_func = make_actfunc(self.param_dic['actfunc']['name'],
                                                self.param_dic['actfunc']['act_params'])
        diff = inputs[1] - inputs[0]
        relu_diff = self.activation_func(diff)
        self.outputs = [relu_diff]
        self.cost = relu_diff.sum() * theano.shared(np.cast[theano.config.floatX](self.coeff)) 
        self.param_dic['type'] = 'cost.maxmargin'
        self.set_output_names(self.param_dic['name'], self.outputs)
        err = (relu_diff > 0).sum(acc_dtype=theano.config.floatX)/inputs[0].shape[0]
        self.cost_list= [self.cost, err]
        
class BinaryCrossEntropyCostLayer(CostLayer):
    """
    inputs=[ground_truth label, prediction]
    """
    def __init__(self, inputs, param_dic=None):
        CostLayer.__init__(self, inputs, param_dic)
        self.parse_param_dic(param_dic)
        if ('actfunc' in param_dic) and (not param_dic['actfunc']['name'] in ['binary_crossentropy']):
            raise LayerException('CostLayer only can use crossentropy')
        else:
            self.param_dic['actfunc'] = {'name':'binary_crossentropy', 'act_params':None}
            print 'Use Default binary_crossentropy as Neuron'
            self.activation_func = make_actfunc(self.param_dic['actfunc']['name'],
                                                self.param_dic['actfunc']['act_params'])
        self.outputs = [self.activation_func(inputs[1], inputs[0])]

        self.cost = self.outputs[0].sum() * theano.shared(np.cast[theano.config.floatX](self.coeff))
        self.param_dic['type'] = 'cost.binary_crossentropy'
                                                    
        self.set_output_names(self.param_dic['name'], self.outputs)
                
        self.cost_list = [self.cost]
class SquareDiffCostLayer(CostLayer):
    """
    # the order doesn't matter in fact
    inputs=[ground_ruth label, prediction] 
    """
    def __init__(self, inputs, param_dic=None):
        CostLayer.__init__(self, inputs, param_dic)
        self.parse_param_dic(param_dic)
        self.outputs = [tensor.sqr(inputs[1] - inputs[0])]
        self.cost = self.outputs[0].sum() * theano.shared(np.cast[theano.config.floatX](self.coeff))
        self.param_dic['type'] = 'cost.sqdiff'
        self.set_output_names(self.param_dic['name'], self.outputs)
        self.cost_list = [self.cost]
class CosineCostLayer(CostLayer):
    """
    This cost layer will calculate
       dot product of inputs[0] and inputs[1], d = <inputs[0], inputs[1]>
    if norm is true, then it will be d/n1/n2
    Please use negative coeff
    """
    def __init__(self, inputs, param_dic=None):
        CostLayer.__init__(self, inputs, param_dic)
        self.parse_param_dic(param_dic)
        raw_outputs = (inputs[0] * inputs[1]).sum(axis=1, keepdims=True)
        if param_dic['norm']:
            n1 = tensor.sqrt((inputs[0]**2).sum(axis=1, keepdims=True))
            n2 = tensor.sqrt((inputs[1]**2).sum(axis=1, keepdims=True))
            tmp = n1 * n2
            nf = theano.tensor.switch(tmp > 0, tmp, 1) 
            self.outputs = [raw_outputs/nf]
        else:
            self.outputs = [raw_outputs]
        if self.coeff > 0:
            print 'Warn----: Use Positive CosineCostLayer\n\n----'
        self.cost = self.outputs[0].sum() * theano.shared(np.cast[theano.config.floatX](self.coeff))
        self.param_dic['type'] = 'cost.cosine'
        self.set_output_names(self.param_dic['name'], self.outputs)
        self.cost_list = [self.cost]
class Network(object):
    """
    For the basic interface to the layers
    """
    def __init__(self, layers, config_dic):
        """
        layers is a dic
          each element of which is a triple
          ([input_layers], [output_layers], layer_instance)
        It is not necessary for the network to make use of all the layers
        The inputs and outputs should be specified manually
        """
        self.load(layers, config_dic)
    def __str__(self):
        s = ''
        if self.inputs:
            s = s +  'inputs:\t(' + ','.join([s.name for s in self.inputs]) + ')\n'
        if self.outputs:
            s = s + 'outputs:\t(' + ','.join([s.name for s in self.inputs]) + ')\n'
        if self.dropout_layers:
            s = s + 'dropout layers:\t(' + ','.join([s.name for s in self.dropout_layers]) + ')\n'
        return s
    def load(self, layers, config_dic):
        self.layers = layers
        self.config_dic = config_dic
        cost_layers = self.get_layer_by_names(config_dic['cost_layer_names'])
        self.all_cost  = [l.cost for l in cost_layers]
        self.cost_names = config_dic['cost_layer_names']
        self.cost_list = sum([l.cost_list for l in cost_layers],[])
        self.cost_list_idx = np.cumsum([0] + [len(l.cost_list) for l in cost_layers])
        
        # only usefull for calculate the gradients
        if self.all_cost:
            self.costs = [sum(self.all_cost)  + self.get_regularizations()]
        else:
            self.costs = None
        data_layers = self.get_layer_by_names(config_dic['data_layer_names'])
        self.inputs = sum([l.inputs for l in data_layers],[])
        output_layers = self.get_layer_by_names(config_dic['output_layer_names'])
        if output_layers:
            self.outputs = sum([l.outputs for l in output_layers], [])
        else:
            self.outputs= []
        layer_with_weights = self.get_layer_by_names(config_dic['layer_with_weights'])
        if layer_with_weights:
            self.params = sum([l.params for l in layer_with_weights],[])
            self.params_eps = sum([l.params_eps for l in layer_with_weights],[])
            self.params_inc =  sum([l.params_inc for l in layer_with_weights],[])
            self.params_mom = sum([l.params_mom for l in layer_with_weights],[])
        else:
            self.params = None
            self.eps_params = None
            self.params_inc = None
            self.params_mom = None
        if config_dic['dropout_layer_names'] is None:
            config_dic['dropout_layer_names'] = [lname for lname in layers if layers[lname][2].param_dic['type'] == 'dropout']
            # for lay in layers:
            #     print '{}:\t {}'.format(layer[2].name, layer[2].param_dic['type'])
        self.data_idx = config_dic['data_idx'] if config_dic['data_idx'] else None
        print config_dic
        self.dropout_layers = self.get_layer_by_names(config_dic['dropout_layer_names'])
    def set_dropout_on(self, train=True):
        for lay in self.dropout_layers:
            lay.set_dropout_on(train)
    def load_net_from_disk(net_saved_path):
        d = mio.unpickle(net_saved_path)
        self.load(d['layers'], d['config_dic'])
    def get_layer_by_names(self, name_list):
        return [self.layers[l_name][2] for l_name in name_list ]
    def get_regularizations(self):
        config_dic = self.config_dic
        layer_with_weights = [self.layers[l_name][2] for l_name in config_dic['layer_with_weights']]
        if layer_with_weights:
            return sum([l.get_regularizations() for l in layer_with_weights])
        else:
            return theano.shared(np.cast[theano.config.floatX](0.0))
    def save_layers_to_dic(self, layers):
        res_dic = dict()
        for e in layers:
            l = layers[e]
            res_dic[e] = [l[0], l[1], l[2].save_to_dic()]
        return res_dic
    def set_train_mode(self, train):
        self.set_dropout_on(train)
    def save_to_dic(self,ignore=None):
        if ignore and 'layers' in ignore:
            return {'layers':None, 'config_dic': self.config_dic}
        else:
            return {'layers':self.save_layers_to_dic(self.layers), 'config_dic': self.config_dic}
        
layer_parser_dic={'fc':FCParser,'data':DataParser, 'cost.maxmargin':MaxMarginCostParser,
                  'cost.bce':BinaryCrossEntropyCostParser, 'cost.sqdiff':SquareDiffCostParser,
                  'eltsum':ElementwiseSumParser, 'conv':ConvParser, 'pool':PoolParser,
                  'dropout':DropoutParser, 'cost.cosine':CosineCostParser,
                  'neuron':NeuronParser, 'batchnorm':BatchNormParser,
                  'elemscale':ElementwiseScaleParser
}