import pprint
import numpy as np

def pbatch(source, dic):
    ss = np.transpose(source)
    for line in ss[:10]:
        for word in line:
            a = dic[word]
            b = a
            if a == "eos":
                b = "_"
            elif a == "UNK":
                b = "|"
            print b,
        print " "
    print ""
