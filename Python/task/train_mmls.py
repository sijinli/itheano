"""
Examples:
python --data-path=/opt/visal/tmp/for_sijin/Data/H36M/H36MExp/folder_SP_t004_act_14 --num-epoch=200 --data-provider=mem --layer-def=/home/grads/sijinli2/Projects/Itheano/doc/netdef/graph_def_0002.cfg --testing-freq=1 --batch-size=1024 --train-range=0-132743 --test-range=132744-162007 --save-path=/public/sijinli2/ibuffer/2015-01-16/net2_test_for_stat --K-candidate=200

python ~/Projects/Itheano/Python/task/train_mmls.py --data-path=/opt/visal/tmp/for_sijin/Data/H36M/H36MExp/folder_SP_t004_act_14 --num-epoch=200 --data-provider=mem --layer-def=/home/grads/sijinli2/Projects/Itheano/doc/netdef/graph_def_0003.cfg --testing-freq=1 --batch-size=1024 --train-range=0-132743 --test-range=132744-162007 --save-path=/public/sijinli2/ibuffer/2015-01-16/net3_test_for_stat --K-candidate=200

python ~/Projects/Itheano/Python/task/train_mmls.py --data-path=/opt/visal/tmp/for_sijin/Data/H36M/H36MExp/folder_SP_t004_act_14 --num-epoch=200 --data-provider=mem --layer-def=/home/grads/sijinli2/Projects/Itheano/doc/netdef/graph_def_0009.cfg --testing-freq=1 --batch-size=1024 --train-range=0-132743 --test-range=132744-162007 --save-path=/opt/visal/tmp/for_sijin/tmp/tmp_theano --K-candidate=200 --K-most-violated=100 --margin-func=rbf,0.0416666667 --max-num=20

python ~/Projects/Itheano/Python/task/train_mmls.py --data-path=/opt/visal/tmp/for_sijin/Data/H36M/H36MExp/folder_SP_t004_act_14 --num-epoch=200 --data-provider=mem --layer-def=/home/grads/sijinli2/Projects/Itheano/doc/netdef/graph_def_0009.cfg --testing-freq=1 --batch-size=1024 --train-range=0-132743 --test-range=132744-162007 --save-path=/opt/visal/tmp/for_sijin/tmp/tmp_theano --K-candidate=200 --K-most-violated=100 --max-num=20

python ~/Projects/Itheano/Python/task/train_mmls.py --data-path=/opt/visal/tmp/for_sijin/Data/H36M/H36MExp/folder_Raw_SP_t004_act_14 --num-epoch=200 --data-provider=mem --layer-def=/home/grads/sijinli2/Projects/Itheano/doc/netdef/graph_def_0014.cfg --testing-freq=1 --batch-size=1024 --train-range=0-132743 --test-range=132744-162007 --save-path=/opt/visal/tmp/for_sijin/tmp/tmp_theano_t --K-candidate=200 --K-most-violated=100 --max-num=20 --solver-type=mmls

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
    mmlsloader = MMSolverLoader()
    solver = mmlsloader.parse()
    solver.train()
if __name__ == '__main__':
    main()
    # main_test()