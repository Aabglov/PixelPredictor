import numpy as np
import pickle

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


# 2D Downscale
def downscale(img):
  w,h = img.shape
  c = np.zeros((w/2,h/2))
  for i in range(0,w):
    for j in range(0,h):
        c[i/2,j/2] = np.sum(img[i:i+2,j:j+2])/4.
  return c


trX = load('trXimg.pckl')
ind = int(len(trX)*0.8)

l= len(trX)
# CREATE TRAINING/TESTING X and Y value for each color RED,BLUE,GREEN

trXr = [] # 0
trXg = [] # 1
trXb = [] # 2

trYr = [] # 0
trYg = [] # 1
trYb = [] # 2

for i in range(ind):
    y = trX[0].T
    trYr.append(y[0])
    trYg.append(y[1])
    trYb.append(y[2])

    trXr.append(downscale(y[0]))
    trXg.append(downscale(y[1]))
    trXb.append(downscale(y[2]))
    print str(i),' of ',str(l),'processed'

# SAVE NEW X    
save(trXr,'trXr.pckl')
save(trXg,'trXg.pckl')
save(trXb,'trXb.pckl')

# SAVE NEW Y
save(trYr,'trYr.pckl')
save(trYg,'trYg.pckl')
save(trYb,'trYb.pckl')

teXr = [] # 0
teXg = [] # 1
teXb = [] # 2

teYr = [] # 0
teYg = [] # 1
teYb = [] # 2

for i in range(ind,len(trX)):
    y = trX[0].T
    teYr.append(y[0])
    teYg.append(y[1])
    teYb.append(y[2])

    teXr.append(downscale(y[0]))
    teXg.append(downscale(y[1]))
    teXb.append(downscale(y[2]))
    print str(i),' of ',str(l),'processed'

# SAVE NEW X    
save(teXr,'teXr.pckl')
save(teXg,'teXg.pckl')
save(teXb,'teXb.pckl')

# SAVE NEW Y
save(teYr,'teYr.pckl')
save(teYg,'teYg.pckl')
save(teYb,'teYb.pckl')
