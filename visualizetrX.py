###############################################################
#                        TEST AUTOENCODER                      #
###############################################################

import numpy as np
import random
import subprocess
from matplotlib import pyplot as plot
import matplotlib.cm as cm
from PIL import Image
import pickle
from numpy import genfromtxt
import csv
import os

###########################################
# PICKLE WRAPPERS ...oh my
def save(var,name):
    f = open(name, 'w')
    pickle.dump(var, f)
    f.close()

def load(name):
    f = open(name,'rb')
    var = pickle.load(f)
    f.close()
    return var

def main():
    trX = load('trYsmall.pckl')
        
    S1 = 100
    S2 = 60*60

    s1 = int(np.ceil(np.sqrt(S1)))
    s2 = int(np.sqrt(S2))
        
    for i in range(0,S1):
        a = trX[i]
        print np.var(a)
        plot.subplot(s1,s1,i)
        plot.imshow(a.reshape((s2,s2,3)))
        frame = plot.gca()
        frame.axes.get_xaxis().set_visible(False)
        frame.axes.get_yaxis().set_visible(False)
    plot.show()
    




if __name__ == '__main__':
    main()

    
