"""
This is the basic script for training a feedforward network
Examples:

Usage:

python ~/Projects/Itheano/Python/task/train_basicbp.py --data-path=/opt/visal/tmp/for_sijin/Data/H36M/H36MExp/folder_ASM_act_14_exp_2 --num-epoch=200 --data-provider=croppedjt --layer-def=/home/grads/sijinli2/Projects/Itheano/doc/netdef/graph_def_0008.cfg --testing-freq=15 --batch-size=256 --train-range=0-132743 --test-range=132744-162007 --save-path=/opt/visal/tmp/for_sijin/tmp/test_basic_bp --solver-type=basicbp --load-file=/opt/visal/tmp/for_sijin/tmp/test_basic_bp

"""
from init_task import *
import options
import iread.myio as mio
import options
from ilayer import *
from idata import *
from isolver import *
from igraphparser import *

import sys
               
def main():
    ## TODO NEED TO CHANGE.
    ## SolverLoader should split the process into several steps
    solver_loader = SolverLoader()
    solver = solver_loader.parse()
    # print len(solver.train_error)
    solver.train()
if __name__ == '__main__':
    main()