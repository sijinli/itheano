from init_src import *
from icfg import IConfigParser
from collections import OrderedDict
import theano
from theano import tensor as tensor
from ilayer import *
class GraphParserError(Exception):
    pass

class GraphParser(object):
    """
    Read the cfg and build the network graph
    """
    def __init__(self, file_path, graph_params = None):
        self.mcp = IConfigParser(dict_type=OrderedDict)
        self.mcp.readfp(open(file_path))
        self.file_path = file_path
        self.layers = dict()
        self.network_config = dict()
        self.parse(self.mcp)
    def get_input_data(self, name, mcp, input_dims):
        """
        """
        input_names = mcp.safe_get_list(name, 'inputs')
        l = []
        for e,dim in zip(input_names, input_dims):
            if e in self.symbolic_var_dic:
                print('    use symbolic variable {}'.format(e))
                l += [self.symbolic_var_dic[e]]
            else:
                try:
                    if isinstance(dim, int):
                        sym, descr = tensor.fmatrix(e), 'fmatrix'
                    elif isinstance(dim, tuple):
                        t = len(dim)
                        if t != 2 and t != 3:
                            raise Exception('Unsupported dimension {}'.format(t))
                        if t == 2:
                            sym, descr = tensor.ftensor3(e), 'tensor3'
                        else:
                            sym, descr = tensor.ftensor4(e), 'tensor4'
                    print('    create symbolic variable {} as {}'.format(e, descr))
                    self.symbolic_var_dic[e] = sym 
                    l += [sym]
                except Exception as err:
                    print err
                    sys.exit(1)
        return input_names, l
    @classmethod
    def parse_name_idx(self,s):
        p = s.find('[')
        return [s,0] if p == -1 else [s[:p],int(s[p+1:s.rfind(']')])]
    def get_shared_weights(self, w_s, layers, dic):
        l = [self.parse_name_idx(s) for s in w_s]
        dic['weights'] = [layers[name][2].W_list[idx] for name, idx in l]
        dic['weights_inc'] = [layers[name][2].W_inc_list[idx] for name, idx in l]
    def get_shared_biases(self, b_s, layers, dic):
        l = [self.parse_name_idx(s) for s in b_s]
        # assert(len(l) == 1)
        # assert(l[0][1] == 0)
        dic['biases'] = [layers[name][2].b_list[idx] for name, idx in l]
        dic['biases_inc'] = [layers[name][2].b_inc_list[idx] for name, idx in l]
    def add_network_config(self, name, mcp):
        # Move to Network Parser latter
        net_config = {'layer_def_path':self.file_path}
        default_config = {'dropout_layer_names': None, 'data_idx':None,
                          'dual_mode_layer_names':None, 'additional_update_layer':None}
        for e in ['cost_layer_names', 'data_layer_names', 'output_layer_names',
                  'layer_with_weights', 'dropout_layer_names', 'data_idx',
                  'dual_mode_layer_names', 'additional_update_layer_names']:
            if mcp.has_option(name, e):
                if e in ['data_idx']:
                    net_config[e] = mcp.safe_get_int_list(name, e)
                else:
                    t = mcp.safe_get_list(name, e)
                    net_config[e] = t if (len(t)>1 or t[0]!='') else []
                print '{}:\t{}\t{}'.format(name, e, net_config[e])
            elif e in default_config:
                net_config[e] = default_config[e]
            else:
                net_config[e] = []
        self.network_config[name] = net_config
    @classmethod
    def print_graph_connections(cls, layers):
        for name in layers:
            l = layers[name]
            cur_l = l[2]
            print('Layer: {} :input layer {} \n \t output layers {}'.format(l[2].param_dic['name'], l[0], l[1]))
            print ('    input_var {}'.format(l[2].inputs))
    def parse(self,mcp):
        self.layers = OrderedDict()
        self.network_config = OrderedDict()
        self.symbolic_var_dic = OrderedDict()
        for name in mcp.sections(): # Ensure the iterator is ordered
            mcp.check_options(name, ['type'])
            layer_type= mcp.safe_get(name, 'type')
            advanced_params = OrderedDict()
            if layer_type == 'network':
                self.add_network_config(name, mcp)
                continue
            elif layer_type == 'data':
                mcp.check_options(name, ['inputs','input_dims'])
                str_dim = mcp.safe_get(name, 'input_dims')
                input_dims = mcp.safe_get_int_list(name, 'input_dims') if str_dim.find('(') == -1 else mcp.safe_get_tuple_int_list(name, 'input_dims')
                input_names,input_var = self.get_input_data(name, mcp, input_dims)
            else:
                mcp.check_options(name, ['inputs'])
                input_names = mcp.safe_get_list(name, 'inputs')
                input_var = sum([self.layers[lname][2].outputs for lname in input_names], [])
                input_dims = sum([self.layers[lname][2].param_dic['output_dims'] for lname in input_names],[])
                if mcp.has_option(name, 'weightSource'):
                    w_s = mcp.safe_get_list(name, 'weightSource')
                    self.get_shared_weights(w_s, self.layers, advanced_params)
                if mcp.has_option(name, 'biasSource'):
                    b_s = mcp.safe_get_list(name, 'biasSource')
                    self.get_shared_biases(b_s, self.layers,advanced_params)
            if name in self.layers:
                raise Exception('Layer [{}] has been defined'.format(name))
            lp = layer_parser_dic[layer_type](name, input_var, mcp, input_dims, advanced_params)
            self.layers[name] = [input_names, [], lp.parse()]
            for lname in input_names:
                if lname in self.layers:
                    self.layers[lname][1] += [name]
            out_dims = self.layers[name][2].param_dic['output_dims'] if ('output_dims' in self.layers[name][2].param_dic) else None
            print 'Init {} finished: input_dims{}\t output_dims {}'.format(name,self.layers[name][2].param_dic['input_dims'], out_dims)
# class NetworkParser(object):
#     def __init__(self, file_path, graph_params = None):
#         self.mcp = IConfigParser(dict_type=OrderedDict)
#         self.mcp.readfp(open(file_path))
#         self.file_path = file_path
   

            