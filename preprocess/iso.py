#!/usr/bin/python
# -*- coding: utf-8 -*-
from collections import OrderedDict
import re
import sys
reload(sys)
sys.setdefaultencoding("utf-8")

f = open("iso9", 'rb')
lines = [line for line in f]
bigru = lines[::4]
smallru = lines[1::4]
bigen = lines[2::4]
smallen = lines[3::4]
iso = OrderedDict()

for br, sr, be, se in zip(bigru, smallru, bigen, smallen):
    iso[br.replace("\n", "")] = be.replace("\n", "")
    iso[sr.replace("\n", "")] = se.replace("\n", "")

def rep(a):
    #aa = a.decode('utf-8')
    aa = a
    for k,v in iso.iteritems():
        aa = aa.replace(k,v)
        #aa = aa.replace(k.decode('utf-8'),v.decode('utf-8'))
    #return aa.encode('utf-8')
    return aa

if __name__ == '__main__':
    filename = sys.argv[1]
    rr = open(filename, 'rb')
    txt = rr.read()
    txt = rep(txt)
    ww = open(filename+".iso9", "w")
    ww.write(txt)
    rr.close()
    ww.close()
