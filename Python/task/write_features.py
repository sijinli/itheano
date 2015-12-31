"""

python ~/Projects/Itheano/Python/task/write_features.py --load-file=/opt/visal/tmp/for_sijin/Data/Pedestrian/Models/CRP_2015_07_22_graph_0004_1 --write-layer-names=pred,pool3 --res-save-path=/opt/visal/tmp/for_sijin/tmp/tmp_test --mode=write_feature

python ~/Projects/Itheano/Python/task/write_features.py  --res-save-path=/opt/visal/tmp/for_sijin/tmp/tmp_test --mode=collect_feature
"""
from init_task import *
import options
from isolver import *
import idata
sys.path.append('/home/grads/sijinli2/Projects/DJI_Pedestrian/Python/src/')
# sys.path.append('/home/grads/sijinli2/Projects/PaperCode/ijcv2015deepstruct/Python/src/')
import pedestrian_data as peddata

def add_extra_op(op):
    op.add_option('mode', 'mode', options.StringOptionParser, 'the mode for testing',default='write_feature',excuses=options.OptionsParser.EXCLUDE_ALL)
    op.add_option('res-save-path', 'res_save_path', options.StringOptionParser, 'The folder for saving results')
    op.add_option('write-layer-names', 'write_layer_names', options.StringOptionParser, 'The names of the features to write')
    op.add_option('start-batch-idx', 'start_batch_idx', options.IntegerOptionParser, 'the index to start writing feature', default=0)
def add_duplicate_op(op):
    op.add_option('load-file', 'load_file', options.StringOptionParser, 'the file to loade', default='')
    op.add_option('test-range', 'test_range', options.RangeOptionParser, 'The range of test range', default=[-1,-1])
    op.add_option('train-range', 'train_range', options.RangeOptionParser, 'The range of test range', default=[-1,-1])
    op.add_option('data-path', 'data_path', options.StringOptionParser, 'The data path ')
    op.add_option('data-provider', 'data_provider', options.StringOptionParser, 'The data provider')
    op.add_option('batch-size', 'batch_size', options.IntegerOptionParser, 'The batch size for processing')
    op.add_option('solver-type', 'solver_type', options.StringOptionParser, 'Solver type')
    op.add_option('save-path', 'save_path', options.StringOptionParser, 'Save path')
    op.add_option('num-epoch', 'num_epoch', options.IntegerOptionParser, 'num epoch')
    op.add_option('force-collect-type', 'force_collect_type', options.StringOptionParser, 'float32', default='')
    # op.add_option('as-bp', 'as_bp', options.IntegerOptionParser, 'Force to use bp loader to load the network')
def bp_write_features(op):
    loader = BasicBPLoader()
    op = loader.op
    add_extra_op(op)
    solver = loader.parse()
    net = solver.net
    save_folder = op.get_value('res_save_path')
    write_layer_names = [s.strip(' ') for s in op.get_value('write_layer_names').split(',')]
    print 'Begin to write layers {}'.format(write_layer_names)
    print 'Save folder = {}'.format(save_folder)
    start_batch_idx = op.get_value('start_batch_idx')
    print 'Start batch idx = {}'.format(start_batch_idx)
    ex_params = {'start_batch_idx':start_batch_idx}
    solver.write_features(solver.test_dp, net, save_folder, write_layer_names, test_one=False, inputs=None, dataidx=None, ex_params=ex_params)
def general_write_features(op, solver_type):
    if solver_type == 'imgdpmm':
        loader = ImageDotProdMMSolverLoader()
    else:
        loader = BasicBPLoader
    op = loader.op
    add_extra_op(op)
    solver = loader.parse()
    net = solver.net_dic.items()[0][1]
    write_layer_names = [s.strip(' ') for s in op.get_value('write_layer_names').split(',')]
    input_layer_names =  op.get_value('input_layer_names')
    if len(input_layer_names):
        input_layer_names =[s.strip(' ') for s in input_layer_names.split(',')]
        inputs = net.get_layer_by_names(input_layer_names)
    else:
        inputs = None
    start_batch_idx = op.get_value('start_batch_idx')
    print 'Start batch idx = {}'.format(start_batch_idx)
    ex_params = {'start_batch_idx':start_batch_idx}
    solver.write_feature(solver.test_dp, net, save_folder, write_layer_names, test_one=False, inputs=inputs, dataidx=None, ex_params=ex_params)
def collect_feature(op):
    save_path = op.get_value('res_save_path')
    import dhmlpe_utils as dutils
    meta = dutils.collect_feature_meta(save_path)
    meta_save_path =iu.fullfile(save_path, 'imgfeatures.meta')
    force_collect_type = op.get_value('force_collect_type')
    if force_collect_type == 'float32':
        print 'Force the feature dtype=float32'
        meta['feature_list'] = [np.array(x,dtype=np.float32) for x in meta['feature_list']]
    mio.pickle(meta_save_path, meta)
def main():
    idata.dp_dic['pedmem'] = peddata.PedestrianMemoryData
    op = options.OptionsParser()
    add_extra_op(op)
    add_duplicate_op(op)
    op.parse()
    op.eval_expr_defaults()
    mode = op.get_value('mode')
    if mode == 'write_feature':
        load_file_path = op.get_value('load_file')
        save_model = Solver.get_saved_model(load_file_path)
        if 'solver_type' in  save_model['solver_params']:
            solver_type = save_model['solver_params']['solver_type']
        else:
            solver_type = 'basicbp'
        if solver_type == 'basicbp':
            bp_write_features(op)
        else:
            general_write_features(op)
    elif mode == 'collect_feature':
        collect_feature(op)



if __name__ == '__main__':
    main()