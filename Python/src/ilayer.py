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
import iutils as iu
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
            dic['wd'] = self.mcp.safe_get_float_list(self.name, 'wd', default=[0]*n)
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
                for e in ['weights', 'biases', 'weights_inc', 'biases_inc',
                          'running_mean', 'std']:
                    if e in self.advanced_params:
                        dic[e] = self.advanced_params[e]
            if len(dic['initW']) != len(self.inputs):
                raise LayerException('{}:#initW {}!=#inputs {}'.format(self.name,
                                                                       len(dic['initW']),
                                                                       len(self.inputs)))
        except Exception as err:
            print 'LayerWithWeightsParser: error info[''{}'']'.format(err)
            sys.exit(1)
        #assert(len(dic['initb']) == 1)
        return dic

        # if len(self.inputs) != 
class Layer(object):
    """
    inputs will be the list of all the inputs
    Also, outputs will be a list
    """
    _required_field =  ['input_dims', 'name']
    def __init__(self, inputs, param_dic=None):
        self.inputs = inputs
        self.outputs = None
        self.activation_func = None
        self.params = None
        self.param_dic = dict()
        self.required_field = list(self._required_field)
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
        # assert(saved_layers['type'] == self.param_dic['type'])
        if saved_layers['type'] != self.param_dic['type']:
            raise Exception('saved layer type [{}] ! = param_type [{}]'.format(saved_layers['type'], self.param_dic['type']))
    def parse_param_dic(self, param_dic):
        try:
            if not param_dic:
                raise LayerException('Empty parameters')
            for e in self.required_field:
                if not e in param_dic:
                    print '    {}'.format(param_dic)
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
    def get_additional_updates(self):
        """
        Any layer parameter
        """
        return []
    def save_to_dic(self):
        return self.param_dic.copy()

class LayerWithWeightsLayer(Layer):
    _required_field = Layer._required_field +  ['output_dims', 'wd', 'epsW', 'epsB', 'momW', 'momB']
    def __init__(self, inputs, param_dic=None):
        Layer.__init__(self, inputs, param_dic)
        self.required_field = self._required_field
    def copy_from_saved_layer(self, sl):
        Layer.copy_from_saved_layer(self, sl)
        for W,W_value in zip(self.W_list, sl['weights']):
            W.set_value(np.array(W_value,dtype=theano.config.floatX))
        for b,b_value in zip(self.b_list, sl['biases']):
            b.set_value(np.array(b_value,dtype=theano.config.floatX))
        print '    {}: copy weights|biases from {} layer successfully'.format(self.param_dic['name'], sl['name'])
        if 'weights_inc' in sl:
            for W_inc,W_inc_value in zip(self.W_inc_list, sl['weights_inc']):
                W_inc.set_value(np.array(W_inc_value,dtype=theano.config.floatX))
            for b_inc, b_inc_value in zip(self.b_inc_list, sl['biases_inc']):
                b_inc.set_value(np.array(b_inc_value, dtype=theano.config.floatX))
    def initW_inc(self):
        self.W_inc_list = [theano.shared(np.asarray(e.get_value() * 0.0, dtype=theano.config.floatX)) for e in self.W_list]
        for idx, e in enumerate(self.W_inc_list):
            e.name = '%s_weights_inc_%d' % (self.param_dic['name'], idx)
    def initb_inc(self):
        self.b_inc_list = [theano.shared(np.asarray(e.get_value() * 0.0, dtype=theano.config.floatX)) for e in self.b_list]
        for idx, e in enumerate(self.b_inc_list):
            e.name = '%s_biasinc_%d' % (self.param_dic['name'], idx)
        #self.b_inc.name = '%s_biasinc_%d' % (self.param_dic['name'], 0)
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
        assert(type(s) is list)
        sp_list = self.get_b_shape(param_dic)
        b_list = []
        for i, sp in enumerate(sp_list):
            b_value = np.ones(sp,dtype=theano.config.floatX) * np.cast[theano.config.floatX](s[i])
            b_list.append(theano.shared(b_value,borrow=False))
            b_list[-1].name = '%s_biases_%d' % (param_dic['name'], i)
        self.b_list = b_list
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
            print '        Init weights from outside<' + ','.join([w.name for w in self.W_list]) + '>'
        if not 'biases' in param_dic:
            if 'initBfunc' in param_dic:
                sp_list = self.get_b_shape(param_dic)
                b_value_list = param_dic['initBfunc'](param_dic['name'], sp_list)
                self.b_list = [theano.shared(np.cast[theano.config.floatX](b_value),
                                             borrow=False,
                                             name='%s_biases_%d' % (param_dic['name'],i))
                               for i, b_value in enumerate(b_value_list)]
            else:
                self.initb(param_dic)
            self.initb_inc()
        else:
            self.b_list = param_dic['biases']
            self.b_inc_list = param_dic['biases_inc']
            print '        Init biase from outside < {}>'.format(','.join([b.name for b in self.b_list]))
        if len(param_dic['wd']) != len(self.W_list):
            raise LayerException('weight decay list has {} elements != {}'.format(len(param_dic['wd']), len(self.W_list)))
        self.wd = self.cvt2sharedfloatX(self.param_dic['wd'], '%s_wd' % param_dic['name'])
        self.epsW = self.cvt2sharedfloatX(self.param_dic['epsW'],'%s_epsW' % param_dic['name'])
        self.epsB = self.cvt2sharedfloatX(self.param_dic['epsB'],'%s_epsB' % param_dic['name'])
        self.momW = self.cvt2sharedfloatX(self.param_dic['momW'],'%s_momW' % param_dic['name'])
        self.momB = self.cvt2sharedfloatX(self.param_dic['momB'],'%s_momB' % param_dic['name'])

        self.params = self.W_list + self.b_list
        self.params_eps= self.epsW + self.epsB
        self.params_inc = self.W_inc_list + self.b_inc_list
        self.params_mom = self.momW + self.momB
    def save_to_dic(self):
        save_dic = self.param_dic.copy()
        # Those value might be saved in gpu or it is of function type
        for e in ['weights', 'biases', 'weights_inc', 'biases_inc', 'initWfunc', 'initBfunc']:
            if e in save_dic:
                del save_dic[e]
        save_dic['weights'] = [w.get_value() for w in self.W_list]
        save_dic['biases'] = [b.get_value() for b in self.b_list]
        save_dic['weights_inc'] = [w_inc.get_value() for w_inc in self.W_inc_list]
        save_dic['biases_inc'] = [b_inc.get_value() for b_inc in self.b_inc_list]
        save_dic['epsW'] = [e.get_value() for e in self.epsW]
        save_dic['epsB'] = [e.get_value() for e in self.epsB]
        # In priciple, momW.get_value() = self.param_dic['momW']
        # save_dic['momW'] = [e.get_value() for e in self.momW] 
        # save_dic['momB'] = [e.get_value() for e in self.momB]
        return save_dic
    def get_regularizations(self):
        return sum([(W**2).sum() * decay for W, decay in zip(self.W_list, self.wd)])


class SlackVarParser(LayerWithWeightsParser):
    def parse_layer_params(self):
        dic = LayerParser.parse_layer_params(self)
        self.mcp.check_options(self.name, ['initS'])
        assert(len(self.inputs)==1)
        n = 1
        dic['initS'] = self.mcp.safe_get_float_list(self.name, 'initS', default=[0.0] * n)
        dic['slacktype'] = self.mcp.safe_get(self.name, 'slacktype')
        dic['coeff'] = self.mcp.safe_get_float_list(self.name, 'coeff', default=[1.0] * n)
        dic['momS'] = self.mcp.safe_get_float_list(self.name, 'momS', default=[1.0] * n)
        dic['epsS'] = self.mcp.safe_get_float_list(self.name, 'epsS', default=[1.0] * n)
        dic['output_dims'] = dic['input_dims']
        _slack_type_list = ['softrelu', 'exp']
        if dic['slacktype'] not in _slack_type_list:
            raise Exception('slacktype can only be one of {}'.format(_slack_type_list))
        if self.advanced_params:
            for e in ['slackvar']:
                if e in self.advanced_params:
                    dic[e] = self.advanced_params[e]
        return dic
    def parse(self):
        dic = self.parse_layer_params()
        print '    Parse Layer {} \n \tComplete {}'.format(dic['name'], dic)
        return SlackVarLayer(self.inputs, dic)
        
class SlackVarLayer(LayerWithWeightsLayer):
    def __init__(self, inputs, param_dic=None):
        LayerWithWeightsLayer.__init__(self, inputs, param_dic)
        field_to_remove = set(['epsW', 'momW', 'epsB', 'momB','wd'])
        self.required_field = self.required_field + ['epsS','momS', 'coeff', 'slacktype','initS']
        self.required_field = [e for e in self.required_field if e not in field_to_remove]
        self.parse_param_dic(param_dic)
        self.pos_slack  = None
        if self.param_dic['slacktype'] == 'exp':
            self.pos_slack = tensor.exp(self.s)
        elif self.param_dic['slacktype'] == 'softrelu':
            self.pos_slack = tensor.log(tensor.exp(self.s) + np.cast[theano.config.floatX](1))
        lin_outputs = self.inputs[0] + self.pos_slack 
        if self.activation_func:
            raise Exception('are you sure what you are doing')
            self.outputs = [self.activation_func(lin_outputs)]
        else:
            self.outputs = [lin_outputs]
        self.set_output_names(self.param_dic['name'], self.outputs)
        self.param_dic['type'] = 'slackvar'
        self.cost = (self.pos_slack * self.coeff[0]).sum()
        self.cost_list = [self.cost]
    def copy_from_saved_layer(self, sl):
        self.s.set_value(np.cast[theano.config.floatX](sl['slackvar'][0]))
        print '    {}: copy slackvar from {} layer successfully'.format(self.param_dic['name'], sl['name'])
        if 'slackvar_inc' in sl:
            self.s_inc.set_value(np.cast[theano.config.floatX](sl['slackvar_inc'][0]))
    def inits_inc(self):
        self.s_inc = theano.shared(np.cast[theano.config.floatX](self.s.get_value() * 0.0))
        self.s_inc.name = '%s_slackvarinc_%d' % (self.param_dic['name'], 0)
    def inits(self, param_dic):
        s_value = np.cast[theano.config.floatX](param_dic['initS'][0])
        self.s = theano.shared(s_value,borrow=False)
        self.s.name = '%s_slackvar_%d' % (param_dic['name'], 0)
    def parse_param_dic(self, param_dic):
        Layer.parse_param_dic(self, param_dic)
        if not 'slackvar' in param_dic:
            self.inits(param_dic)
            self.inits_inc()
        else:
            self.s = param_dic['slackvar']
            self.s_inc = param_dic['slackvar_inc']
            print '        Init slackvar from outside {}'.format(self.b.name)
        self.coeff = self.cvt2sharedfloatX([self.param_dic['coeff']], '%s_coeff' % param_dic['name'])        
        self.momS = self.cvt2sharedfloatX(self.param_dic['momS'], '%s_momS' % param_dic['name'])
        self.epsS = self.cvt2sharedfloatX(self.param_dic['epsS'], '%s_epsS' % param_dic['name'])
        
        self.params = [self.s]
        self.params_eps = self.epsS
        self.params_inc = [self.s_inc]
        self.params_mom = self.momS
    def save_to_dic(self):
        save_dic = self.param_dic.copy()
        for e in ['slackvar', 'slackvar_inc']:
            if e in save_dic:
                del save_dic[e]
        save_dic['slackvar'] = [self.s.get_value()]
        save_dic['slackvar_inc'] = [self.s_inc.get_value()]
        save_dic['epsS'] = [e.get_value() for e in self.epsS]
        return save_dic
    def get_regularizations(self):
        return 0
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
    __dropout_seed_srng = np.random.RandomState(7)
    def __init__(self, inputs, param_dic=None):
        Layer.__init__(self, inputs, param_dic)
        self.required_field = self.required_field +  ['keep']
        if param_dic:
            self.parse_param_dic(param_dic)
        self.train_mode = True
        flag = 1.0 if self.train_mode else 0.0
        # I don't know why. After updating to Numpy 1.9. Then I have to do that.
        self.seed = (DropoutLayer.__dropout_seed_srng.randint(0, sys.maxint)) % 4294967295
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
        print '    Parse Layer {} \n \tComplete {}'.format(param_dic['name'],param_dic)
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
        print '    Parse Layer {} \n \tComplete {}'.format(dic['name'],dic)
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
        self.param_dic['type'] = 'eltsum'
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
            self.param_dic['coeffs'] = param_dic['coeffs']
        else:
            self.coeffs = None
            self.param_dic['coeffs'] = [1] * len(self.inputs)
class ElementwiseMulParser(LayerParser):
    def parse(self):
        dic = LayerParser.parse_layer_params(self)
        print '    Parse Layer {} \n \tComplete {}'.format(dic['name'], dic)
        return ElementwiseMulLayer(self.inputs, dic)
class ElementwiseMulLayer(Layer):
    def __init__(self, inputs, param_dic=None):
        Layer.__init__(self, inputs, param_dic)
        self.inputs = inputs
        if param_dic:
            self.parse_param_dic(param_dic)
        raw_outputs = inputs[0]
        for e in inputs[1:]:
            raw_outputs = raw_outputs * e
        if self.activation_func:
            self.outputs = [self.activation_func(raw_outputs)]
        else:
            self.outputs = [raw_outputs]
        self.param_dic['type'] = 'eltmul'
        self.param_dic['output_dims'] = [param_dic['input_dims'][0]]
        self.set_output_names(self.param_dic['name'], self.outputs)
class ElementwiseMaxParser(LayerParser):
    def parse(self):
        dic = LayerParser.parse_layer_params(self)
        return ElementwiseMaxLayer(self.inputs, dic)
class ElementwiseMaxLayer(Layer):
    def __init__(self, inputs, param_dic=None):
        Layer.__init__(self, inputs, param_dic)
        self.inputs = inputs
        if param_dic:
            self.parse_param_dic(param_dic)
        assert(len(inputs)  ==  2)
        raw_outputs = tensor.maximum(inputs[0], inputs[1]) 
        if self.activation_func:
            self.outputs = [self.activation_func(raw_outputs)]
        else:
            self.outputs = [raw_outputs]
        self.param_dic['type'] = 'eltmax'
        self.param_dic['output_dims'] = [param_dic['input_dims'][0]]
        self.set_output_names(self.param_dic['name'], self.outputs)
class ReduceParser(LayerParser):
    """
    Note that the first dimension is data dimension by default
    
    """
    def parse(self):
        dic = self.parse_layer_params()
        print '    Parse Layer {} \n \tComplete {}'.format(dic['name'],dic)
        return ReduceLayer(self.inputs, dic)
    def parse_layer_params(self):
        dic = LayerParser.parse_layer_params(self)
        dic['reduce_method'] = self.mcp.safe_get(self.name, 'reduce_method')
        dic['axis'] = self.mcp.safe_get_int_list(self.name, 'axis')
        return dic
class ReduceLayer(Layer):
    _required_field = Layer._required_field + ['reduce_method', 'axis']
    def __init__(self, inputs, param_dic=None):
        Layer.__init__(self, inputs, param_dic)
        self.inputs = inputs
        self.parse_param_dic(param_dic)
        assert(len(inputs)==1)
        raw_outputs = self.do_reduce(inputs)
        if self.activation_func:
            self.outputs = [self.activation_func(e) for e in raw_outputs]
        else:
            self.outputs = raw_outputs
        self.param_dic['type'] = 'reduce'
        self.set_output_names(self.param_dic['name'], self.outputs)
    def do_reduce(self, inputs):
        reduce_method = self.param_dic['reduce_method']
        axis = self.param_dic['axis']
        if reduce_method == 'max':
            raw_outputs = [e.max(axis=axis[0],keepdims=True) for e in inputs]
        elif reduce_method == 'min':
            raw_outputs = [e.min(axis=axis[0],keepdims=True) for e in inputs]
        elif reduce_method == 'average':
            raw_outputs = [e.mean(axis=axis[0],keepdims=True) for e in inputs]
        else:
            raise Exception('Unsupported method {}'.format(reduce_method))
        return raw_outputs
    def parse_param_dic(self, param_dic):
        Layer.parse_param_dic(self, param_dic)
        dims = param_dic['input_dims'][0]
        assert(param_dic['reduce_method'] in ['max', 'min', 'average'])
        if type(dims) is int:
            dims = 1
        else:
            axis = param_dic['axis']
            assert(axis[0] > 0)
            dims[axis[0] - 1] = 1
        self.param_dic['output_dims'] = [dims]
        
class ConcatenateParser(LayerParser):
    def parse(self):
        dic = LayerParser.parse_layer_params(self)
        return ConcatenateLayer(self.inputs, dic)
class ConcatenateLayer(Layer):
    def __init__(self, inputs, param_dic=None):
        Layer.__init__(self, inputs, param_dic)
        if param_dic:
            self.parse_param_dic(param_dic)
        assert( len(inputs) > 1)
        raw_outputs = tensor.concatenate(inputs, axis=1)
        if self.activation_func:
            self.outputs = [self.activation_func(raw_outputs)]
        else:
            self.outputs = [raw_outputs]
        self.param_dic['type'] = 'concat'
        self.param_dic['output_dims'] = [sum(param_dic['input_dims'])]
        self.set_output_names(self.param_dic['name'], self.outputs)
        
class StackParser(LayerParser):
    def parse(self):
        dic = LayerParser.parse_layer_params(self)
        if (len(set(dic['input_dims'])) != 1):
            raise Exception('The layers to be stacked does not have the same dimension \n {}'.format(dic['input_dims']))
        return StackLayer(self.inputs, dic)
class StackLayer(Layer):
    def __init__(self,inputs, param_dic=None):
        Layer.__init__(self, inputs, param_dic)
        if param_dic:
            self.parse_param_dic(param_dic)
        assert( len(inputs) > 1)
        raw_outputs = tensor.concatenate(inputs, axis=0)
        if self.activation_func:
            self.outputs = [self.activation_func(raw_outputs)]
        else:
            self.outputs = [raw_outputs]
        self.param_dic['type'] = 'stack'
        self.param_dic['output_dims'] = [param_dic['input_dims'][0]]
        self.set_output_names(self.param_dic['name'], self.outputs)
        
class DotProdParser(LayerParser):
    def parse(self):
        dic = LayerParser.parse_layer_params(self)
        if len(dic['input_dims']) != 2:
            raise Exception('The input of layer {} should be 2'.format(dic['name']))
        if dic['input_dims'][0] != dic['input_dims'][1]:
            raise Exception('The input for dot product layer should have the same dimension \n {}'.format(dic['input_dims']))
        return DotProdLayer(self.inputs, dic)
class DotProdLayer(Layer):
    def __init__(self, inputs, param_dic=None):
        Layer.__init__(self, inputs, param_dic)
        if param_dic:
            self.parse_param_dic(param_dic)
        assert( len(inputs) == 2)
        raw_outputs = (inputs[0] * inputs[1]).sum(axis=1, keepdims=True)
        if self.activation_func:
            self.outputs = [self.activation_func(raw_outputs)]
        else:
            self.outputs = [raw_outputs]
        self.param_dic['type'] = 'dotprod'
        self.param_dic['output_dims'] = [1]
        self.set_output_names(self.param_dic['name'], self.outputs)
        
        
class FCParser(LayerWithWeightsParser):
    def parse(self):
        dic = LayerWithWeightsParser.parse_layer_params(self)
        if self.mcp.has_option(self.name, 'output_dims'):
            dic['output_dims'] = self.mcp.safe_get_int_list(self.name, 'output_dims')
        elif self.mcp.has_option(self.name, 'outputs'): # to support convnet files
            dic['output_dims'] = self.mcp.safe_get_int_list(self.name, 'outputs')
        else:
            raise LayerException('Fully connected layer missing outputs or output_dims')
        print '    Parse Layer {} \n \tComplete {}'.format(self.name, dic)
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
        self.check_shape(self.b_list[0], (self.param_dic['output_dims'][0],))
        has_tuple = np.any([type(x) is tuple for x in self.param_dic['input_dims']])
        if has_tuple:
            lin_outputs = sum([tensor.dot(elem.flatten(ndim=2), W) for elem, W in zip(inputs, self.W_list)]) + self.b_list[0]
        else:
            lin_outputs = sum([tensor.dot(elem, W) for elem, W in zip(inputs, self.W_list)]) \
                          + self.b_list[0]
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
        return [(param_dic['output_dims'][0],)]


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
        print '    Parse Layer {} \n \tComplete {}'.format(dic['name'],dic)
        return ConvDNNLayer(self.inputs, dic)
###Under Construction
class SkipConvParser(LayerWithWeightsParser):
    def __init__(self, name, inputs, mcp, input_dims, advanced_params=None):
        LayerWithWeightsParser.__init__(self, name, inputs, mcp, input_dims, advanced_params)
    @classmethod
    def get_output_skip_conv_dims(cls, L, F_size, skip, pad):
        """
        Note:skip is not stride
             pad is the padding in the inner shape (not the out shape)
             padding for each inner convolution
        """
        n_part = L // skip
        return ((n_part - F_size + pad * 2) + 1)* skip
    def parse_layer_params(self):
        dic = LayerWithWeightsParser.parse_layer_params(self)
        mcp, name = self.mcp, self.name
        dic['sharebias'] = mcp.safe_get_int(name, 'sharebias') if mcp.has_option(name, 'sharebias') else 1
        dic['sizeX'] = mcp.safe_get_int_list(name, 'sizeX')
        dic['sizeY'] = mcp.safe_get_int_list(name, 'sizeY')
        dic['filters'] = mcp.safe_get_int(name, 'filters')
        dic['skipX'] = mcp.safe_get_int_list(name, 'skipX')
        dic['skipY'] = mcp.safe_get_int_list(name, 'skipY')
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
            Lx, Ly = dic['input_dims'][i][2], dic['input_dims'][i][1]
            sx,sy = dic['sizeX'][i], dic['sizeY'][i]
            skipx, skipy = dic['skipX'][i], dic['skipY'][i]
            return (self.get_output_skip_conv_dims(Lx,sx, skipx, dic['pad'][0]),
                    self.get_output_skip_conv_dims(Ly,sy, skipy, dic['pad'][1]))
        output_dims= [get_dims(k) for k in range(len(dic['input_dims']))]
        if len(set(output_dims)) != 1:
            raise LayerException('The output dimension is not consistent {}'.format(output_dims))
        dim = output_dims[0]
        dic['output_dims'] = [(dic['filters'], dim[1], dim[0])]
        return dic
    def parse(self):
        dic = self.parse_layer_params()
        print '    Parse Layer {} \n \tComplete {}'.format(dic['name'],dic)
        return SkipConvDNNLayer(self.inputs, dic)
class SkipConvDNNLayer(LayerWithWeightsLayer):
    def __init__(self, inputs, param_dic=None):
        assert(dnn_available)
        LayerWithWeightsLayer.__init__(self, inputs, param_dic)
        self.required_field = self.required_field +  ['sizeX', 'sizeY', 'filters', 'skipX',
                                                      'skipY',   'pad', 'board_mode', 'conv_mode', 'sharebias']
        if param_dic:
            self.parse_param_dic(param_dic)
        self.param_dic['type'] = 'skipconvdnn'
        board_mode = self.param_dic['pad']
        conv_mode = self.param_dic['conv_mode']
        if self.param_dic['sharebias']:
            biases = self.b_list[0].dimshuffle('x', 0, 'x', 'x')
        else:
            biases = self.b_list[0].dimshuffle('x', 0,1,2)
        def skip_conv(img, kerns, border_mode, conv_mode, input_dim, output_dim, skipx, skipy):
            nc, H,W =input_dim[0],input_dim[1], input_dim[2]
            nh, nw = H // skipy, W // skipx
            res_H, res_W = nh * skipy, nw * skipx
            cropped_img = img[:,:,:res_H, :res_W]
            reshaped_img = cropped_img.reshape((img.shape[0], nc, nh, skipy, nw, skipx))
            shuffled_img = reshaped_img.dimshuffle((0,3,5,1, 2,4))
            shuffled_img = shuffled_img.reshape((img.shape[0]  * skipy * skipx, nc, nh, nw))
            res_img = dnn.dnn_conv(img = shuffled_img, 
                                   kerns=kerns,
                                   border_mode=border_mode,
                                   conv_mode=conv_mode)
            output_h, output_w = output_dim[1], output_dim[2]
            assert(output_h % skipy == 0)
            assert(output_w % skipx == 0)
            res_img = res_img.reshape((img.shape[0], skipy, skipx, nc, output_h//skipy, output_w//skipx))
            res_img = res_img.dimshuffle((0, 3, 4, 1, 5, 2))
            res_img = res_img.reshape((img.shape[0], nc,
                                       output_h,
                                       output_w))
            return res_img
        lin_outputs = sum([skip_conv(img=img,
                                     kerns=W,
                                     border_mode=board_mode,
                                     conv_mode=conv_mode,
                                     input_dim=input_dim,
                                     output_dim=output_dim,
                                     skipx=skipx, skipy=skipy
                                  )
        for img,W,input_dim, output_dim, skipx, skipy in zip(inputs, self.W_list,
                                                             self.param_dic['input_dims'],
                                                             self.param_dic['output_dims'],
                                                             self.param_dic['skipX'],
                                                             self.param_dic['skipY'])]) + biases
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
        return [(dims[0],)] if flag == 1 else [(dims[0], dims[1], dims[2])]

#######################
class ConvDNNLayer(LayerWithWeightsLayer):
    def __init__(self, inputs, param_dic=None):
        assert(dnn_available)
        LayerWithWeightsLayer.__init__(self, inputs, param_dic)
        self.required_field = self.required_field +  ['sizeX', 'sizeY', 'filters', 'strideX', 'strideY', 'pad', 'board_mode', 'conv_mode', 'sharebias']
        if param_dic:
            self.parse_param_dic(param_dic)
        self.param_dic['type'] = 'convdnn'
        board_mode = self.param_dic['pad']
        conv_mode = self.param_dic['conv_mode']
        # Modified in 2015/07/30 strides are for windows H, W
        strides = zip(self.param_dic['strideY'], self.param_dic['strideX'])
        if self.param_dic['sharebias']:
            biases = self.b_list[0].dimshuffle('x', 0, 'x', 'x')
        else:
            biases = self.b_list[0].dimshuffle('x', 0,1,2)
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
        return [(dims[0],)] if flag == 1 else [(dims[0], dims[1], dims[2])]

class LRNParser(LayerParser):
    def __init__(self, name, inputs, mcp, input_dims, advanced_params=None):
        #batch size, stack size, nb row, nb col
        # input_dims = [ [s1,nr,nc], [s2,nr,nc],...]
        LayerParser.__init__(self, name, inputs, mcp, input_dims, advanced_params)
    def parse_layer_params(self):
        dic = LayerParser.parse_layer_params(self)
        mcp, name = self.mcp, self.name
        dic['sizeX'] = mcp.safe_get_int_list(name, 'sizeX')
        dic['sizeY'] = mcp.safe_get_int_list(name, 'sizeY')
        dic['alpha'] = mcp.safe_get_float_list(name, 'alpha')
        dic['beta'] = mcp.safe_get_float_list(name, 'beta')
        dic['norm_mode'] = mcp.safe_get(name, 'norm_mode')
        if dic['norm_mode'] == 'cross_channel':
            dic['size'] = mcp.safe_get_int_list(name, 'size')
            assert(dic['sizeX'][0] == 1)
            assert(dic['sizeY'][0] == 1)
            assert(dic['size'][0] % 2 == 1)
            dic['pad'] = (dic['size'][0] // 2,0)
        elif dic['norm_mode'] in ['local_response', 'local_contrast']:
            dic['size'] = [1]
            assert(dic['sizeX'][0] % 2 == 1)
            assert(dic['sizeY'][0] % 2 == 1)
            dic['pad'] = (dic['sizeX'][0] // 2, dic['sizeY'][0] // 2)
            # raise Exception('Not supported yet')
        dic['output_dims'] = dic['input_dims']
        return dic
    def parse(self):
        dic = self.parse_layer_params()
        print '    Parse Layer {} \n \tComplete {}'.format(dic['name'],dic)
        return LRNLayer(self.inputs, dic)

class LRNLayer(Layer):
    def __init__(self,inputs, param_dic=None):
        assert(dnn_available)
        Layer.__init__(self, inputs, param_dic)
        self.required_field = self.required_field +  ['sizeX', 'sizeY', 'alpha', 'beta', 'norm_mode', 'size',  'pad', 'output_dims']

        if param_dic:
            self.parse_param_dic(param_dic)

        self.param_dic['type'] = 'lrn'
        assert(len(inputs) == 1)
        nchannel_const = self.param_dic['input_dims'][0][0]
        print 'nchannel_const = {}'.format(nchannel_const)
        if self.param_dic['norm_mode'] == 'cross_channel':
            sq_input = theano.tensor.sqr(inputs[0])
            nb,nchannel,nr,nc = inputs[0].shape
            hf_size = self.param_dic['size'][0]//2
            extra_channels = tensor.alloc(np.cast[theano.config.floatX](0.),
                                                  nb, nchannel + int(hf_size) * 2, nr, nc)
            sq_input = tensor.set_subtensor(extra_channels[:,hf_size:hf_size + nchannel,:,:], sq_input)
            raw_lin_outputs = np.cast[theano.config.floatX](0)
            raw_lin_outputs = sum([sq_input[:,k:k+nchannel_const,:,:] for k in range(self.param_dic['size'][0])])            
        else:
            nb,nchannel,nr,nc = inputs[0].shape
            input_dim = self.param_dic['input_dims'][0]
            border_mode = self.param_dic['pad']
            w_shape = [1, 1, self.param_dic['sizeY'][0],
                       self.param_dic['sizeX'][0]]
            w = theano.shared(np.ones(w_shape, dtype=theano.config.floatX))
            if self.param_dic['norm_mode'] == 'local_response':
                sq_input = theano.tensor.sqr(inputs[0])
            else:
                # This is slow. But I have no choice
                assert(self.param_dic['norm_mode'] == 'local_contrast')
                # one_tmp = theano.tensor.ones_like(inputs[0])
                # sum_region = tensor.concatenate([dnn.dnn_conv(img=inputs[0][:,[k],:,:],
                #                                               kerns=w,
                #                                               subsample=(1,1),
                #                                               border_mode=border_mode,
                #                                               conv_mode = 'conv') for k in range(input_dim[0])],axis=1)
                # count_region= tensor.concatenate([dnn.dnn_conv(img=one_tmp[:,[k],:,:],
                #                                                kerns=w,
                #                                                subsample=(1,1),
                #                                                border_mode=border_mode,
                #                                                conv_mode = 'conv') for k in range(input_dim[0])],axis=1)
                # mean_region = sum_region / count_region
                # Change the order of sizeY, sizeX in 2015/7/29
                mean_region = dnn.dnn_pool(self.inputs[0], ws=(self.param_dic['sizeY'][0],
                                                               self.param_dic['sizeX'][0]),
                                           stride=(1,1), pad=self.param_dic['pad'],
                                           mode='average')
                sq_input = tensor.sqr(inputs[0] - mean_region)
            raw_lin_outputs = tensor.concatenate([dnn.dnn_conv(img=sq_input[:,[k],:,:],
                                                               kerns=w,
                                                               subsample=(1,1),
                                                               border_mode=border_mode,
                                                               conv_mode = 'conv') for k in range(input_dim[0])],axis=1)
        a = np.cast[theano.config.floatX](self.param_dic['alpha'][0] / (self.param_dic['size'][0]* self.param_dic['sizeX'][0] * self.param_dic['sizeY'][0]))
        scale = (raw_lin_outputs * a + np.cast[theano.config.floatX](1.0))**(self.param_dic['beta'][0])
        lin_outputs = inputs[0]/scale
        # lin_outputs = inputs[0]/raw_lin_outputs
        
        if self.activation_func:
            self.outputs = [self.activation_func(lin_outputs)]
        else:
            self.outputs = [lin_outputs]
        
       
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
            if dic['pooling_method'] == 'cudnn':
                dic['padX'] = self.mcp.safe_get_int(self.name, 'padX', default=0)
                dic['padY'] = self.mcp.safe_get_int(self.name, 'padY', default=0)
                dic['ignore_boarder'] = True
        except Exception as err:
            print err
            sys.exit(1)
        return dic
    def parse(self):
        dic = self.parse_layer_params()
        # Decide which class to call as this stage
        if dic['pooling_method'] == 'convnet':
            return ConvNetPool2dLayer(self.inputs, dic)
        elif dic['pooling_method'] == 'theano':
            return TheanoPoolLayer(self.inputs, dic)
        elif dic['pooling_method'] == 'cudnn':
            return CUDNNPoolLayer(self.inputs, dic)
        else:
            raise LayerException('Only support convnet for pooling')

class CUDNNPoolLayer(Layer):
    """
    Trust Nvidia!!!
    """
    def __init__(self, inputs, param_dic):
        assert(dnn_available)
        Layer.__init__(self, inputs, param_dic)
        # print '<<<<<<<<<<<<<<<<<<<< {}'.format(param_dic)
        self.required_field = self.required_field +  ['strideX', 'strideY', 'pooling_type', 'sizeX', 'sizeY', 'padX', 'padY']
        self.parse_param_dic(param_dic)
        assert(len(self.param_dic['input_dims']) == 1)
        dim = param_dic['input_dims'][0]
        sX, strideX, sY,strideY, padX, padY = param_dic['sizeX'], param_dic['strideX'], param_dic['sizeY'], param_dic['strideY'], param_dic['padX'], param_dic['padY']
        self.param_dic['output_dims'] = [self.get_output_pool_dims(dim,sX, strideX,
                                                                   sY,strideY, (padX,padY))]
        self.pooling_type = self.param_dic['pooling_type']
        self.ignore_boarder = True
        self.param_dic['type'] = 'cudnnpool'
        pad = (param_dic['padX'], param_dic['padY'])
        # print 'sX = {} sY = {}||||||||||||||||||'.format(sX, sY)
        if self.pooling_type in ['max','average']:
            # Change to sY,sX in 2015/07/16
            pool_out = dnn.dnn_pool(self.inputs[0],
                                    ws=(sY,sX), stride=(strideY, strideX), pad=pad,
                                    mode=self.pooling_type)
            self.outputs = [pool_out]
        else:
            raise LayerException('Un supported pooling type {}'.format(self.pooling_type))
        self.set_output_names(self.param_dic['name'], self.outputs)
    @classmethod
    def get_output_pool_dims(cls, input_dim, sX, strideX, sY, strideY, paddings):
        """
        input_dims = [filters, Height, Width]
        # Because it always ignore the boarder
        """
        s1 = int(np.floor(float((input_dim[1] + 2 * paddings[1]) - sY) / strideY)) + 1
        s2 = int(np.floor(float((input_dim[2] + 2 * paddings[0]) - sX) / strideX)) + 1
        return (input_dim[0], s1, s2)


            
class TheanoPoolLayer(Layer):
    """
    Use downsample for pooling.
    It should be slower than ConvNetPool2dLayer
    However, it support non-square image
    """
    from theano.tensor.signal import downsample
    def __init__(self, inputs, param_dic):
        Layer.__init__(self, inputs, param_dic)
        self.required_field = self.required_field +  ['strideX', 'strideY', 'pooling_type', 'sizeX', 'sizeY']
        self.parse_param_dic(param_dic)
        assert(len(self.param_dic['input_dims']) == 1)
        # assert(param_dic['strideX'] == param_dic['strideY'])
        # assert(param_dic['sizeX'] == param_dic['sizeY'])
        dim, sX, strideX, sY,strideY = param_dic['input_dims'][0], param_dic['sizeX'], param_dic['strideX'], param_dic['sizeY'], param_dic['strideY']
        self.param_dic['output_dims'] = [self.get_output_pool_dims(dim,sX, strideX,
                                                                   sY,strideY)]
        self.pooling_type = self.param_dic['pooling_type']
        self.ignore_boarder = False
        self.param_dic['type'] = 'theanopool'
        if self.pooling_type in ['max','sum','average_inc_pad', 'average_exc_pad']:
            #       st=(strideY, strideX),  update theano version
            #       mode=self.pooling_type
            # assert(strideY == sY and strideX == sX)
            # assert(self.pooling_type=='max')
            pool_out = downsample.max_pool_2d(self.inputs[0], ds=(sY,sX), st=(strideY, strideX),
                                              ignore_border=self.ignore_boarder
            )
            self.outputs = [pool_out]
        else:
            raise LayerException('Un supported pooling type {}'.format(self.pooling_type))
        self.set_output_names(self.param_dic['name'], self.outputs)
    @classmethod
    def get_output_pool_dims(cls, input_dim, sX, strideX, sY, strideY, paddings=None):
        """
        input_dims = [filters, Height, Width]
        """
        assert(paddings is None)
        s1 = int(np.ceil(float(input_dim[1] - sY) / strideY)) + 1
        s2 = int(np.ceil(float(input_dim[2] - sX) / strideX)) + 1
        return (input_dim[0], s1, s2)

class ConvNetPool2dLayer(Layer):
    """
    Use pylearn2 convnet pooling for max pooling
    requirements: the input should be square image, and of course gpu continuous
    """
    from pylearn2.sandbox.cuda_convnet.pool import MaxPool
    from theano.sandbox.cuda.basic_ops import gpu_contiguous
    def __init__(self, inputs, param_dic):
        Layer.__init__(self, inputs, param_dic)
        self.required_field = self.required_field + ['strideX', 'strideY', 'pooling_type', 'sizeX', 'sizeY']
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
class ReshapeParser(LayerParser):
    def parse(self):
        assert(len(self.inputs) == 1)
        dic = self.parse_layer_params()
        print '    Parse Layer {} \n \tComplete {}'.format(dic['name'],dic)
        return ReshapeLayer(self.inputs, dic)
    def parse_layer_params(self):
        dic = LayerParser.parse_layer_params(self)
        str_dim = self.mcp.safe_get(self.name, 'output_dims')
        output_dims = self.mcp.safe_get_int_list(self.name, 'output_dims') if str_dim.find('(') == -1 else self.mcp.safe_get_tuple_int_list(self.name, 'output_dims')
        print 'outputdim = {}'.format(output_dims)
        dic['output_dims'] = output_dims
        return dic
class ReshapeLayer(Layer):
    def __init__(self, inputs, param_dic):
        Layer.__init__(self, inputs, param_dic)
        self.required_field = self.required_field +  ['output_dims']
        self.parse_param_dic(param_dic)
        output_dim = list(self.param_dic['output_dims'][0]) if type(self.param_dic['output_dims'][0]) is not int else [self.param_dic['output_dims'][0]]
        assert(np.prod(output_dim) == np.prod(self.param_dic['input_dims'][0]))
        # self.outputs = [self.inputs[0].reshape([-1] + output_dim)]
        self.outputs = [self.inputs[0].reshape([self.inputs[0].shape[0]] + output_dim)]
        self.set_output_names(self.param_dic['name'], self.outputs)
        self.param_dic['type'] = 'reshape'
class NeuronParser(LayerParser):
    def parse(self):
        assert(len(self.inputs) == 1)
        dic = self.parse_layer_params()
        print '    Parse Layer {} \n \tComplete {}'.format(dic['name'],dic)
        return NeuronLayer(self.inputs, dic)
 
class NeuronLayer(Layer):
    def __init__(self, inputs, param_dic):
        Layer.__init__(self, inputs, param_dic)
        self.required_field = self.required_field +  ['actfunc']
        self.parse_param_dic(param_dic)
        self.outputs = [self.activation_func(self.inputs[0])]
        self.set_output_names(self.param_dic['name'], self.outputs)
        self.param_dic['type'] = 'neuron'
        self.param_dic['output_dims'] = self.param_dic['input_dims']
        
class BatchNormParser(LayerWithWeightsParser):
    def parse(self):
        dic = self.parse_layer_params()
        print '    Parse Layer {} \n \tComplete {}'.format(dic['name'],dic)
        assert(len(self.inputs) == 1)
        return BatchNormLayer(self.inputs, dic)
    def parse_layer_params(self):
        dic = LayerWithWeightsParser.parse_layer_params(self)
        dic['epsilon'] = self.mcp.safe_get_float(self.name, 'epsilon', default=1e-6)
        dic['running_mom'] = self.mcp.safe_get_float(self.name, 'running_mom', default=0.9)
        return dic
class BatchNormLayer(LayerWithWeightsLayer):
    def __init__(self, inputs, param_dic):
        LayerWithWeightsLayer.__init__(self, inputs, param_dic)
        self.required_field = self.required_field +  ['epsilon', 'running_mom']
        self.required_field = [x for x in self.required_field if x != 'output_dims']
        self.parse_param_dic(param_dic)
        self.train_mode_flag = theano.shared(np.cast[theano.config.floatX](1))
        m = inputs[0].mean(axis=0,keepdims=True)
        centered_inputs_train = inputs[0] - m
        std_train = (centered_inputs_train**2).mean(axis=0, keepdims=True)
        epsilon = theano.shared(np.cast[theano.config.floatX](param_dic['epsilon']))
        raw_outputs_train = centered_inputs_train/tensor.sqrt(std_train +epsilon)
        one = theano.shared(np.cast[theano.config.floatX](1))
        centered_inputs_test = inputs[0] - self.running_mean
        raw_outputs_test = centered_inputs_test / tensor.sqrt(self.running_std + epsilon)
        # raw_outputs = theano.ifelse.ifelse(self.train_mode_flag, raw_outputs_train, raw_outputs_test)
        raw_outputs = self.train_mode_flag * raw_outputs_train + (1 - self.train_mode_flag) * raw_outputs_test
        rms = theano.shared(np.cast[theano.config.floatX](param_dic['running_mom']))

        # self.running_mean_update = theano.ifelse.ifelse(self.train_mode_flag, self.running_mean * rms + m * ( one - rms), self.running_mean)
        self.running_mean_update = self.train_mode_flag * (self.running_mean * rms + m * (one - rms)) + (1 - self.train_mode_flag) * self.running_mean
        # self.running_std_update = theano.ifelse.ifelse(self.train_mode_flag, self.running_std * rms + std_train * (one - rms), self.running_std)
        self.running_std_update = self.train_mode_flag * (self.running_std * rms + std_train * (one - rms)) + (1 - self.train_mode_flag) * self.running_std
        input_dim = self.get_regular_input_dim(param_dic)
        sp = ['x'] + range(len(input_dim) - 1)
        W = self.W_list[0].dimshuffle(*sp)
        b = self.b_list[0].dimshuffle(*sp)
        W_raw_outputs = W * raw_outputs + b
        if self.activation_func:
            self.outputs = [self.activation_func(W_raw_outputs)]
        else:
            self.outputs = [W_raw_outputs]
        self.set_output_names(self.param_dic['name'], self.outputs)
        self.param_dic['output_dims'] = self.param_dic['input_dims']
        self.param_dic['type'] = 'batchnorm'
    def get_additional_updates(self):
        return [(self.running_mean, self.running_mean_update), (self.running_std, self.running_std_update)]
    def get_regular_input_dim(self, param_dic):
        input_dim = param_dic['input_dims'][0]
        if type(input_dim) is list or type(input_dim) is tuple:
            input_dim = [1] + list(input_dim)
        else:
            input_dim = [1, input_dim]
        return input_dim
    def get_w_shape(self, param_dic):
        input_dim = self.get_regular_input_dim(param_dic)
        return [input_dim[1:]]
    def get_b_shape(self, param_dic):
        input_dim = self.get_regular_input_dim(param_dic)
        return [input_dim[1:]]
        
    def parse_param_dic(self, param_dic):
        LayerWithWeightsLayer.parse_param_dic(self, param_dic)
        input_dim = self.get_regular_input_dim(param_dic)
        bc = [False] * len(input_dim)
        bc[0] = True
        if 'running_mean' in param_dic:
            raise Exception('Not supported yet. Batch Norm')
            self.running_mean = param_dic['running_mean']
        else:
            value = np.zeros(input_dim, dtype=theano.config.floatX)
            self.running_mean = theano.shared(value, broadcastable=bc)
        if 'running_std' in param_dic:
            raise Exception('Not supported yet. Batch Norm')
            self.running_std = param_dic['running_std']
        else:
            value = np.ones(input_dim, dtype=theano.config.floatX)
            self.running_std = theano.shared(value, broadcastable=bc)
    def copy_from_saved_layer(self, sl):
        LayerWithWeightsLayer.copy_from_saved_layer(self, sl)
        self.running_mean.set_value(sl['running_mean'])
        self.running_std.set_value(sl['running_std'])
    def save_to_dic(self):
        save_dic = LayerWithWeightsLayer.save_to_dic(self)
        save_dic['running_mean'] = self.running_mean.get_value()
        save_dic['running_std'] = self.running_std.get_value()
        return save_dic
    def set_train_mode(self, train):
        # print '\n\nSetting training mode in batchnorm \n\n train is {}'.format(train)
        if train:
            self.train_mode_flag.set_value(np.cast[theano.config.floatX](1.0))
        else:
            self.train_mode_flag.set_value(np.cast[theano.config.floatX](0.0))
class ElementwiseScaleParser(LayerWithWeightsParser):
    def parse(self):
        assert(len(self.inputs) == 1)
        dic = LayerWithWeightsParser.parse_layer_params(self)
        dic['output_dims'] = dic['input_dims']
        print '    Parse Layer {} \n \tComplete {}'.format(dic['name'],dic)
        return ElementwiseScaleLayer(self.inputs, dic)

class ElementwiseScaleLayer(LayerWithWeightsLayer):
    def __init__(self, inputs, param_dic):
        LayerWithWeightsLayer.__init__(self, inputs, param_dic)
        self.parse_param_dic(param_dic)
        w_shape, b_shape = self.get_w_shape(param_dic), self.get_b_shape(param_dic)
        w_new_dim = ['x'] + range(len(w_shape[0]))
        b_new_dim = ['x'] + range(len(b_shape[0]))
        raw_outputs = self.W_list[0].dimshuffle(*w_new_dim) * inputs[0] + self.b_list[0].dimshuffle(*b_new_dim)
        if self.activation_func:
            self.outputs = [self.activation_func(raw_outputs)]
        else:
            self.outputs = [raw_outputs]
        self.set_output_names(self.param_dic['name'], self.outputs)
        self.param_dic['output_dims'] = self.param_dic['input_dims']
        self.param_dic['type'] = 'elemscale'

    def get_w_shape(self, param_dic):
        d = (param_dic['input_dims'][0],) if type(param_dic['input_dims'][0]) is int else param_dic['input_dims'][0]
        return [ d ]
    def get_b_shape(self, param_dic):
        d = (param_dic['input_dims'][0],) if type(param_dic['input_dims'][0]) is int else param_dic['input_dims'][0]
        return [d]

class LSTMParser(LayerWithWeightsParser):
    def parse(self):
        dic = self.parse_layer_params()
        print '    Parse Layer {} \n \tComplete {}'.format(dic['name'], dic)
        return LSTMLayer(self.inputs, dic)
    def parse_layer_params(self):
        # Need to check whether I need all !!!
        dic = LayerParser.parse_layer_params(self)

        dic['wd'] = self.mcp.safe_get_float_list(self.name, 'wd', default=[0]*8)
        dic['epsW'] = self.mcp.safe_get_float_list(self.name, 'epsW', default=[1.0] * 8)
        dic['epsB'] = self.mcp.safe_get_float_list(self.name, 'epsB', default= [1.0] * 4)
        dic['momW'] = self.mcp.safe_get_float_list(self.name, 'momW', default=[0.9] * 8)
        dic['momB'] = self.mcp.safe_get_float_list(self.name, 'momB', default=[0.9] * 4)
        
        for e in ['initWfunc', 'initBfunc']:
            if self.mcp.has_option(self.name, e):
                st = self.mcp.safe_get(self.name, e)
                func, func_params = self.parse_external_func(st)
                dic[e] = func
        if self.advanced_params:
            for e in ['weights', 'biases', 'weights_inc', 'biases_inc']:
                if e in self.advanced_params:
                    dic[e] = self.advanced_params[e]
        if not 'actfunc' in dic:
            dic['actfunc'] = {'name':'tanh', 'act_params':[1,1]}
        if self.mcp.has_option(self.name, 'inner_neuron'):
            s = self.mcp.safe_get(self.name, 'inner_neuron')
            nname, nparams = self.parse_neuron_params(s)
            dic['inner_actfunc'] = {'name':nname, 'act_params': nparams}
        else:
            dic['inner_actfunc'] = {'name':'hard_sigmoid', 'act_params': None}
        dic['truncate_gradient'] = self.mcp.safe_get_int(self.name,
                                                         'truncate_gradient', default=-1)
        dic['return_sequences'] = self.mcp.safe_get_bool(self.name,
                                                         'return_sequences',default=False)
        str_dims = self.mcp.safe_get(self.name, 'output_dims')
        dic['output_dims'] = self.mcp.safe_get_int_list(self.name, 'output_dims') if str_dims.find('(') == -1 else self.mcp.safe_get_tuple_int_list(self.name, 'output_dims')
        assert(len(dic['output_dims']) == 1)
        if dic['return_sequences']:
            assert(dic['output_dims'][0] is tuple)
            assert(dic['output_dims'][0][0] == dic['input_dims'][0][0])
            dic['n_output'] = dic['output_dims'][0][-1]
        else:
            assert(type(dic['output_dims'][0]) is int)
            dic['n_output'] = dic['output_dims'][0]
        return dic
        
class LSTMLayer(LayerWithWeightsLayer):
    def __init__(self, inputs, param_dic):
        super(LSTMLayer, self).__init__(inputs, param_dic)
        self.required_field = self.required_field +  ['truncate_gradient', 'output_dims', 'n_output']
        self.parse_param_dic(param_dic)
        self.outputs = self.get_outputs(inputs, param_dic)
        self.set_output_names(param_dic['name'], self.outputs)
        self.param_dic['type'] = 'lstm'

    def parse_param_dic(self, param_dic):
        super(LSTMLayer, self).parse_param_dic(param_dic)
        self.truncate_gradient = param_dic['truncate_gradient']
        self.inner_activation_func = make_actfunc(param_dic['inner_actfunc']['name'],
                                                  param_dic['inner_actfunc']['act_params'])
        
    def _step(self,
              X_t,
              h_t_pre, c_t_pre,
              W_i, W_f, W_c, W_o, 
              u_i, u_f, u_c, u_o,
              b_i, b_f, b_c, b_o
    ):
        xi_t = tensor.dot(X_t, W_i) + b_i
        xf_t = tensor.dot(X_t, W_f) + b_f
        xc_t = tensor.dot(X_t, W_c) + b_c
        xo_t = tensor.dot(X_t, W_o) + b_o
 
        i_t = self.inner_activation_func(xi_t + tensor.dot(h_t_pre, u_i))
        f_t = self.inner_activation_func(xf_t + tensor.dot(h_t_pre, u_f))
        c_t = f_t * c_t_pre + i_t * self.activation_func(xc_t + tensor.dot(h_t_pre, u_c))
        o_t = self.inner_activation_func(xo_t + tensor.dot(h_t_pre, u_o))
        h_t = o_t * self.activation_func(c_t)
        return h_t, c_t
        
    def get_outputs(self, inputs, param_dic):
        X = inputs[0]
        X = X.dimshuffle((1,0,2))


        W_i, U_i, W_f, U_f, W_c, U_c, W_o, U_o = self.W_list
        b_i, b_f, b_c, b_o = self.b_list
        # for W in self.W_list:
        #     print W.get_value().shape, W.name
        # for b in self.b_list:
        #     print b.get_value().shape, b.name
       
        def alloc_zeros_matrix(*dims):
            return tensor.alloc(np.cast[theano.config.floatX](0.)[()], *dims)
        n_output = param_dic['n_output']
        [outputs, memories], updates = theano.scan(
            self._step, 
            sequences=[X],
            outputs_info=[alloc_zeros_matrix(X.shape[1], n_output),
                          alloc_zeros_matrix(X.shape[1], n_output)],
            non_sequences=[W_i, W_f, W_c, W_o,
                           U_i, U_f, U_c, U_o,
                           b_i, b_f, b_c, b_o
                       ], 
            truncate_gradient=self.truncate_gradient 
        )
        if param_dic['return_sequences']:
            return [outputs.dimshuffle((1,0,2))]
        print 'ndim of outputs[-1] is {}'.format(outputs[-1].ndim)
        return [outputs[-1]]
        
    def get_w_shape(self, param_dic):
        # W_i, U_i, W_f, U_f, W_c, U_c, W_o, U_o
        idim = param_dic['input_dims'][0][1]
        odim = param_dic['output_dims'][0][1] if param_dic['return_sequences'] else param_dic['output_dims'][0]
        return [(idim, odim), (odim,odim)] * 4
    def get_b_shape(self, param_dic):
        # b_i, b_f, b_c, b_o
        return [(param_dic['n_output'],)] * 4


class GradParser(LayerParser):
    def parse_layer_params(self):
        dic = LayerParser.parse_layer_params(self)
        dic['target_mode'] = self.mcp.safe_get(self.name, 'target_mode', default='mean')
        return dic
    def parse(self):
        dic = self.parse_layer_params()
        print '    Parse Layer {} \n \t Complete {}'.format(self.name, dic)
        return GradLayer(self.inputs, dic)
class GradLayer(Layer):
    def __init__(self, inputs, param_dic):
        super(GradLayer, self).__init__(inputs, param_dic)
        self.required_field = self.required_field + ['target_mode']
        self.parse_param_dic(param_dic)
        self.outputs = self.get_outputs(inputs, param_dic)
        self.set_output_names(param_dic['name'], self.outputs)
        self.param_dic['output_dims'] = self.param_dic['input_dims'][1:]
        self.param_dic['type'] = 'grad'
    def get_outputs(self, inputs, param_dic):
        if param_dic['target_mode'] == 'mean':
            cost = self.inputs[0].sum(axis=1).mean()
        else:
            raise Exception('Invalid target mode')
        var = inputs[1:]
        assert(len(var) == 1)
        grad = theano.tensor.grad(cost, var)
        if self.activation_func:
            grad = [self.activation_func(x) for x in grad]
        return grad
        

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
        self.required_field = self.required_field +  ['coeff']
    def parse_param_dic(self, param_dic):
        Layer.parse_param_dic(self, param_dic)
        self.coeff = self.param_dic['coeff']
  
class MaxMarginCostParser(CostParser):
    def parse(self):
        dic = self.parse_layer_params()
        print '    Parse Layer {} \n \tComplete {}'.format(dic['name'],dic)
        return MaxMarginCostLayer(self.inputs, dic)
class BinaryCrossEntropyCostParser(CostParser):
    def parse(self):
        dic = self.parse_layer_params()
        assert(len(self.inputs) == 2)
        dic['label_weight'] = self.mcp.safe_get_float_list(self.name, 'label_weight', default=[1,1])
        print '    Parse Layer {} \n \tComplete {}'.format(dic['name'],dic)
        return BinaryCrossEntropyCostLayer(self.inputs, dic)

class MaskedBinaryCrossEntropyCostParser(CostParser):
    def parse(self):
        dic = self.parse_layer_params()
        assert(len(self.inputs) == 3)
        return MaskedBinaryCrossEntropyCostLayer(self.inputs, dic)
class SquareDiffCostParser(CostParser):
    def parse(self):
        dic = self.parse_layer_params()
        assert(len(self.inputs) == 2)
        return SquareDiffCostLayer(self.inputs, dic)
class AbsRatioCostParser(CostParser):
    def parse(self):
        dic = self.parse_layer_params()
        assert(len(self.inputs) == 2)
        return AbsRatioCostLayer(self.inputs, dic)
class CosineCostParser(CostParser):
    def parse_layer_params(self):
        dic = CostParser.parse_layer_params(self)
        dic['norm'] = self.mcp.safe_get_int(self.name, 'norm', default=True)
        dic['sign'] = self.mcp.safe_get_int(self.name, 'sign', default=-1)
        assert(dic['sign'] in [-1,1])
        if dic['norm']:
            print '    Do normalization on inputs'
        return dic
    def parse(self):
        dic = self.parse_layer_params()
        assert(len(self.inputs) == 2)
        return CosineCostLayer(self.inputs, dic)
        
class BinarySVMCostParser(CostParser):
    def parse_layer_params(self):
        dic = CostParser.parse_layer_params(self)
        return dic
    def parse(self):
        dic = self.parse_layer_params()
        assert(len(self.inputs) == 2)
        return BinarySVMCostLayer(self.inputs, dic)
class TripleSVMCostParser(CostParser):
    def parse_layer_params(self):
        dic = CostParser.parse_layer_params(self)
        dic['method'] = self.mcp.safe_get(self.name, 'method', default='sum')
        return dic
    def parse(self):
        dic = self.parse_layer_params()
        assert(len(self.inputs) == 2)
        print '    Parse Layer {} \n \tComplete {}'.format(dic['name'],dic)
        return TripleSVMCostLayer(self.inputs, dic)
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
            print '    Use Default Relu as Neuron'
            self.activation_func = make_actfunc(self.param_dic['actfunc']['name'],
                                                self.param_dic['actfunc']['act_params'])
        diff = inputs[1] - inputs[0]
        relu_diff = self.activation_func(diff)
        self.outputs = [relu_diff]
        tot_mean = relu_diff.sum(axis=1, keepdims=True).mean()
        self.cost = tot_mean * theano.shared(np.cast[theano.config.floatX](self.coeff))
        self.param_dic['type'] = 'cost.maxmargin'
        self.set_output_names(self.param_dic['name'], self.outputs)
        err = (relu_diff.sum(axis=1,keepdims=True) > 0).mean(acc_dtype=theano.config.floatX) * 1.0 
        self.cost_list= [tot_mean, err]
    def parse_param_dic(self, param_dic):
        CostLayer.parse_param_dic(self, param_dic)
        self.param_dic['margin_dim'] = param_dic['input_dims'][0]
        print """
             The margin_dim is {}
        """.format(self.param_dic['margin_dim'])
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
            print '    Use Default binary_crossentropy as Neuron'
            self.activation_func = make_actfunc(self.param_dic['actfunc']['name'],
                                                self.param_dic['actfunc']['act_params'])
        self.outputs = [self.activation_func(inputs[1], inputs[0])]
        label_weight = self.param_dic['label_weight']
        if not (label_weight[0] == label_weight[1] and label_weight[0] == 1):
            mask = inputs[0] * (label_weight[1]- label_weight[0]) + label_weight[0]
            self.outputs[0] = self.outputs[0] * mask
        tot_mean = self.outputs[0].sum(axis=1,keepdims=True).mean() 
        self.cost = tot_mean * theano.shared(np.cast[theano.config.floatX](self.coeff))
        self.param_dic['type'] = 'cost.binary_crossentropy'
        acc = tensor.eq((inputs[1] > 0.5), inputs[0]).mean(axis=1).mean()
        self.set_output_names(self.param_dic['name'], self.outputs)
        self.cost_list = [tot_mean, acc]
    def parse_param_dic(self, param_dic):
        CostLayer.parse_param_dic(self, param_dic)
        if 'label_weight' in param_dic:
            self.param_dic['label_weight'] = param_dic['label_weight']
        else:
            self.param_dic['label_weight'] = [1,1]
class MaskedBinaryCrossEntropyCostLayer(CostLayer):
    """
    inputs=[ground_truth label, prediction, masked]
    """
    def __init__(self, inputs, param_dic=None):
        CostLayer.__init__(self, inputs, param_dic)
        self.parse_param_dic(param_dic)
        if ('actfunc' in param_dic) and (not param_dic['actfunc']['name'] in ['binary_crossentropy']):
            raise LayerException('CostLayer only can use crossentropy')
        else:
            self.param_dic['actfunc'] = {'name':'binary_crossentropy', 'act_params':None}
            print '    Use Default binary_crossentropy as Neuron'
            self.activation_func = make_actfunc(self.param_dic['actfunc']['name'],
                                                self.param_dic['actfunc']['act_params'])
        masked_pred = theano.tensor.switch(masked > 0, inputs[0], inputs[1])
        self.outputs = [self.activation_func(masked_pred, inputs[0])]
        tot_mean = self.outputs[0].sum(axis=1,keepdims=True).mean() 
        self.cost = tot_mean * theano.shared(np.cast[theano.config.floatX](self.coeff))
        self.param_dic['type'] = 'cost.masked_binary_crossentropy'
        self.set_output_names(self.param_dic['name'], self.outputs)
        self.cost_list = [tot_mean]
class SquareDiffCostLayer(CostLayer):
    """
    # the order doesn't matter in fact
    inputs=[ground_ruth label, prediction] 
    """
    def __init__(self, inputs, param_dic=None):
        CostLayer.__init__(self, inputs, param_dic)
        self.parse_param_dic(param_dic)
        self.outputs = [tensor.sqr(inputs[1] - inputs[0])]
        tot_mean = self.outputs[0].sum(axis=1,keepdims=True).mean() 
        self.cost = tot_mean * theano.shared(np.cast[theano.config.floatX](self.coeff))
        self.param_dic['type'] = 'cost.sqdiff'
        self.set_output_names(self.param_dic['name'], self.outputs)
        self.cost_list = [tot_mean]
class AbsRatioCostLayer(CostLayer):
    """
    The cost is
    inputs= [ground_truth, prediction]
    cost = | gt - pred | / gt
    """
    def __init__(self, inputs, param_dic=None):
        CostLayer.__init__(self, inputs, param_dic)
        self.parse_param_dic(param_dic)
        raw_outputs = abs(inputs[0] - inputs[1]) / inputs[0]
        if self.activation_func:
            self.outputs = [self.activation_func(raw_outputs)]
        else:
            self.outputs = [raw_outputs]
        # self.outputs = [abs(inputs[0] - inputs[1]) / inputs[0]]
        tot_mean = self.outputs[0].sum(axis=1, keepdims=True).mean()
        self.cost = tot_mean * theano.shared(np.cast[theano.config.floatX](self.coeff))
        self.param_dic['type'] = 'cost.absratio'
        self.set_output_names(self.param_dic['name'], self.outputs)
        self.cost_list = [tot_mean]
class CosineCostLayer(CostLayer):
    """
    This cost layer will calculate
       dot product of inputs[0] and inputs[1], d = <inputs[0], inputs[1]>
    if norm is true, then it will be d/n1/n2
    Please use negative coeff
    """ 
    def __init__(self, inputs, param_dic=None):
        CostLayer.__init__(self, inputs, param_dic)
        self.required_field = self.required_field + ['norm', 'sign']
        self.parse_param_dic(param_dic)
        raw_outputs = (inputs[0] * inputs[1]).sum(axis=1, keepdims=True)
        sign = param_dic['sign']
        if param_dic['norm']:
            n1 = tensor.sqrt((inputs[0]**2).sum(axis=1, keepdims=True))
            n2 = tensor.sqrt((inputs[1]**2).sum(axis=1, keepdims=True))
            tmp = n1 * n2
            nf = theano.tensor.switch(tmp > 0, tmp, 1) 
            self.outputs = [sign * raw_outputs/nf]
        else:
            self.outputs = [sign * raw_outputs]
        if sign > 0:
            print '    Warn----: Use Positive CosineCostLayer\n\n----'
        tot_mean = self.outputs[0].sum(axis=1,keepdims=True).mean() 
        self.cost = tot_mean * theano.shared(np.cast[theano.config.floatX](self.coeff))
        self.param_dic['type'] = 'cost.cosine'
        self.set_output_names(self.param_dic['name'], self.outputs)
        self.cost_list = [tot_mean]
class BinarySVMCostLayer(CostLayer):
    """
    This layer will use SVM cost, i.e.,
    cost(y,pred) = max(0, 1 - (2 * y- 1) * pred)
    Assume y will either be 1 (positive) or 0 (negative)
    inputs[0]: The ground-truth label
    inputs[1]: The predicted value
    """
    def __init__(self, inputs, param_dic=None):
        CostLayer.__init__(self, inputs, param_dic)
        self.parse_param_dic(param_dic)
        if 'actfunc' in param_dic:
            raise Exception('actfunc are not supported for BinarySVMCostLayer')
        flag = (2 * inputs[0] - 1) * inputs[1]
        res =  theano.tensor.maximum(1 - flag, 0)
        self.outputs =[res]
        tot_mean = self.outputs[0].sum(axis=1,keepdims=True).mean()
        self.cost = tot_mean * theano.shared(np.cast[theano.config.floatX](self.coeff))
        self.param_dic['type'] = 'cost.binary_svm'
        acc = (flag > 0).mean(axis=1).mean()
        self.set_output_names(self.param_dic['name'], self.outputs)
        self.cost_list = [tot_mean, acc]        
class TripleSVMCostLayer(CostLayer):
    """
    This layer will do three class classification.
    The "second class" is the class between the first and last class
    1, 0, -1. The input can only take those 3 value for the three class respectively.
    cost = |y| * max(0, 1 - y * pred) + (1 - |y|) * |pred|
    inputs[0]: The ground-truth label
    inputs[1]: The predicted value
    """
    _required_field = CostLayer._required_field + ['method']
    def __init__(self, inputs, param_dic=None):
        CostLayer.__init__(self, inputs, param_dic)
        self.parse_param_dic(param_dic)
        if 'actfunc' in param_dic:
            raise Exception('actfunc are not supported for BinarySVMCostLayer')
        self.outputs = self.get_outputs(inputs, param_dic)
        tot_mean = self.outputs[0].sum(axis=1,keepdims=True).mean()
        self.cost = tot_mean * theano.shared(np.cast[theano.config.floatX](self.coeff))
        self.param_dic['type'] = 'cost.triple_svm'
        self.set_output_names(self.param_dic['name'], self.outputs)
        self.cost_list = [tot_mean]
    def get_outputs(self, inputs, param_dic):
        method = param_dic['method']
        abs_gt = abs(inputs[0])
        if method == 'sum':
            abs_pred = abs(inputs[1])
            res = abs_gt * theano.tensor.maximum(0, 1 - (inputs[0] * inputs[1])) + (1 - abs_gt) * abs_pred
        elif method == 'avgmaxcontrast':
            avg_high = (abs_gt * inputs[0] * inputs[1]).sum(axis=1,keepdims=True) / theano.tensor.maximum(1, abs_gt.sum(axis=1,keepdims=True))
            max_low = ((1 - abs_gt) * abs(inputs[1])).max(axis=1,keepdims=True)
            res = max_low - avg_high
        else:
            raise Exception('{}:Method [{}] are not supported '.format(self.name, method))
        return [res]
        
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
        if config_dic['dual_mode_layer_names'] is None:
            config_dic['dual_mode_layer_names'] = [lname for lname in layers if layers[lname][2].param_dic['type'] in ['batchnorm']]

            # for lay in layers:
            #     print '{}:\t {}'.format(layer[2].name, layer[2].param_dic['type'])
        self.data_idx = config_dic['data_idx'] if config_dic['data_idx'] else None
        print '    config_dic:\t{}'.format(config_dic)
        self.dropout_layers = self.get_layer_by_names(config_dic['dropout_layer_names'])
        self.dual_mode_layers = self.get_layer_by_names(config_dic['dual_mode_layer_names'])
        if config_dic['additional_update_layer_names']:
            self.additional_updates = self.get_additional_updates(config_dic)
        else:
            self.additional_updates = []
    def get_additional_updates(self, config_dic):
        additional_update_layers = self.get_layer_by_names(config_dic['additional_update_layer_names'])
        return iu.concatenate_list([lay.get_additional_updates() for lay in additional_update_layers])
    def set_dropout_on(self, train=True):
        for lay in self.dropout_layers:
            lay.set_dropout_on(train)
    def set_dual_mode_layers(self, train=True):
        for lay in self.dual_mode_layers:
            lay.set_train_mode(train)
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
    @classmethod
    def save_layers_to_dic(cls, layers):
        res_dic = dict()
        for e in layers:
            l = layers[e]
            res_dic[e] = [l[0], l[1], l[2].save_to_dic()]
        return res_dic
    def set_train_mode(self, train):
        self.set_dropout_on(train)
        self.set_dual_mode_layers(train)
    def save_to_dic(self,ignore=None):
        if ignore and 'layers' in ignore:
            return {'layers':None, 'config_dic': self.config_dic}
        else:
            return {'layers':self.save_layers_to_dic(self.layers), 'config_dic': self.config_dic}
        
layer_parser_dic={'fc':FCParser,'data':DataParser, 'cost.maxmargin':MaxMarginCostParser,
                  'cost.bce':BinaryCrossEntropyCostParser,
                  'cost.mbce':MaskedBinaryCrossEntropyCostParser,
                  'cost.sqdiff':SquareDiffCostParser,
                  'cost.absratio':AbsRatioCostParser,
                  'cost.bsvm':BinarySVMCostParser,
                  'cost.cosine':CosineCostParser,
                  'cost.trisvm':TripleSVMCostParser,
                  'lrn':LRNParser,
                  'eltsum':ElementwiseSumParser, 'eltmul':ElementwiseMulParser,
                  'eltmax': ElementwiseMaxParser,
                  'reduce':ReduceParser,
                  'stack':StackParser, 'dotprod':DotProdParser,
                  'concat': ConcatenateParser,
                  'lstm':LSTMParser,
                  'grad':GradParser,
                  'conv':ConvParser, 'pool':PoolParser,
                  'skipconv':SkipConvParser,
                  'reshape':ReshapeParser,
                  'dropout':DropoutParser, 'cost.cosine':CosineCostParser,
                  'neuron':NeuronParser, 'batchnorm':BatchNormParser,
                  'elemscale':ElementwiseScaleParser,
                  'slackvar':SlackVarParser,
}
