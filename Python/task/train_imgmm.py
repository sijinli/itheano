"""
Examples:
EXE=/home/grads/sijinli2/pkg/anaconda/bin/python
${EXE} /home/grads/sijinli2/Projects/Itheano/Python/task/train_mmls.py --data-path=/opt/visal/tmp/for_sijin/Data/H36M/H36MExp/folder_${exp_name} --num-epoch=${EP} --data-provider=${DP} --layer-def=/home/grads/sijinli2/Projects/Itheano/doc/netdef/graph_def_${JT}.cfg --testing-freq=1 --batch-size=${BSIZE} --train-range=${TrainRange} --test-range=${TestRange} --save-path=/opt/visal/tmp/for_sijin/Data/saved/Test/${save_name} --K-candidate=${KC} --K-most-violated=${KMV} --max-num=${MAXNUM} --solver-type=imgmm 
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
    loader = ImageMMSolverLoader()
    solver = loader.parse()
    solver.train()
if __name__ == '__main__':
    main()
    # main_test()