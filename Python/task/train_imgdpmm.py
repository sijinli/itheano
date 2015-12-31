"""
Examples:

python train_imgdpmm.py --data-path=$(path_of_batch_data) --num-epoch=200 --data-provider=croppedjt --layer-def=/home/grads/sijinli2/Projects/Itheano/doc/netdef/graph_def_ICCV_structnet.cfg --testing-freq=15 --batch-size=128 --train-range=0-132743 --test-range=132744-162007 --save-path=/opt/visal/tmp/for_sijin/Data/saved/theano_models/$(save_name) --K-candidate=2000 --K-most-violated=10 --max-num=500 --solver-type=imgdpmm --force-shuffle=1 --opt-method=ls  --candidate-mode=random2 --opt-params=0

"""
from init_task import *
import options
import iread.myio as mio
import options
from ilayer import *
from idata import *
from isolver import *
from igraphparser import *
from isolver_ext import *
import sys

def main():
    loader = ImageDotProdMMSolverLoader()
    solver = loader.parse()
    solver.train()
if __name__ == '__main__':
    main()
    # main_test()