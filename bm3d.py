from PIL import Image
import numpy as np
import pickle
from random import shuffle
from scipy import signal
from scipy import ndimage
import copy

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
###########################################

# SUB-REGION THAT WILL BE SEARCHED
def sampleRange(x,y,w,h,size):
    if x-size < 0:
        start_x = 0
    else:
        start_x = x-size

    if y-size < 0:
        start_y = 0
    else:
        start_y = y-size

    if x+size > (w-1):
        end_x = w-1
    else:
        end_x = x+size

    if y+size > (h-1):
        end_y = h-1
    else:
        end_y = y+size

    return start_x,start_y,end_x,end_y


def getCorrespondingHighPassSection(x,y,w,h,step,search_size,sample,search_space):
    min_diff = 100000
    min_i = 0
    min_j = 0

    i_len = search_space.shape[0]-step
    j_len = search_space.shape[1]-step
    for i in range(0,i_len,step):
        for j in range(0,j_len,step):
            diff = np.sum(np.square(sample - search_space[i:i+step,j:j+step,:]))
            if diff < min_diff:
                min_diff = diff
                min_i = i
                min_j = j
    return min_i,min_j


def bm3dSearch(step,search_size,upscale,low_pass,high_pass):
    high_pass_upscale = copy.copy(upscale)
    w,h,z = high_pass.shape

    i_len = upscale.shape[0]-step
    j_len = upscale.shape[1]-step
    for i in range(0,i_len,step):
        print 'I:',i,'of',i_len
        for j in range(0,j_len,step):
            sample = upscale[i:i+step,i:i+step,:]
            start_x,start_y,end_x,end_y = sampleRange(i,j,w,h,search_size)
            search_space = low_pass[start_x:end_x,start_y:end_y,:]
            min_i,min_j = getCorrespondingHighPassSection(i,j,w,h,step,search_size,sample,search_space)
            high_pass_upscale[i:i+step,j:j+step,:] += high_pass[min_i:min_i+step,min_j:min_j+step,:]
    return high_pass_upscale

    



path  = 'kp.jpg'

im = Image.open(path)
h,w,z = np.asarray(im).shape
d = im.resize((w/2,h/2),Image.ANTIALIAS)
data = np.asarray(d.resize((w,h),Image.BILINEAR))
low_pass = np.zeros(data.shape)
for i in range(3):
    low_pass[:,:,i] = ndimage.gaussian_filter(data[:,:,i], 2)
high_pass  = np.asarray(im) - np.asarray(low_pass)

##data = np.asarray(im)
##low_pass = np.zeros(data.shape)
##for i in range(3):
##    low_pass[:,:,i] = ndimage.gaussian_filter(data[:,:,i], 5)
##low_pass = ndimage.gaussian_filter(data, 3)
##high_pass = data - low_pass

upscale = np.asarray(im.resize((2*w,2*h),Image.BILINEAR))


step = 2
search_size = 30

bm3d_array = bm3dSearch(step,search_size,upscale,low_pass,high_pass)

bm3d_image = Image.fromarray(bm3d_array)
bm3d_image.save('bm3d.jpg')


lp_trans = low_pass.astype('uint8')
l = Image.fromarray(lp_trans)
h = Image.fromarray(high_pass.astype('uint8'))
l.save('low_pass.jpg')
h.save('high_pass.jpg')

