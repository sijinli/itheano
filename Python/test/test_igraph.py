from init_test import *
from igraphparser import *

file_path = '/home/grads/sijinli2/Projects/Itheano/doc/netdef/graph_def_0001.cfg'
g = GraphParser(file_path)
layers = g.layers
