import sys
sys.path.append('/home/grads/sijinli2/Projects/DHMLPE/Python/src')
sys.path.append('/home/grads/sijinli2/I_ProgramFile/I_Python/Project/I_utils')
sys.path.append('/home/grads/sijinli2/I_ProgramFile/I_Python/Project')
import iutils as iu
import os
ppath = iu.getparentpath(os.path.realpath(__file__), 2)
# sys.path.append(ppath)
# sys.path.append(iu.fullfile(ppath, 'task'))
sys.path.append(iu.fullfile(ppath, 'src'))
from options import *
