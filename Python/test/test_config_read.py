from init_test import *

from icfg import IConfigParser
from collections import OrderedDict
def test1():
    file_path = '/home/grads/sijinli2/Projects/DHMLPE/doc/netdef/dhmlpe-layer-def-t0.cfg'
    a = IConfigParser(dict_type=OrderedDict)
    a.readfp(open(file_path))
    for name in a.sections():
        print 'Section name {}'.format(name)
        if a.has_option(name, 'type'):
            e = a.safe_get(name, 'type')
            print '    type is {}'.format(e)
def test2():
    file_path = '/opt/visal/tmp/for_sijin/tmp/t.cfg'
    a = IConfigParser(dict_type=OrderedDict)
    a.readfp(open(file_path))
    for name in a.sections():
        print 'Section name {}'.format(name)
        e = a.safe_get_tuple_list(name, 'L')
        print e
        print '----------'
        e1 = a.safe_get_tuple_int_list(name, 'L')
        print e1
if __name__ == '__main__':
    test2()
        

    

