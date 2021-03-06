import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
from load import mnist
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.downsample import max_pool_2d
import pickle

srng = RandomStreams()

size = 3
mid = (size-1)/2
found = False
# IMAGE PROCESSING/ACCESS UTILITY FUNCTIONS
def getTrX():
    i = Image.open('/Users/keganrabil/Desktop/Theano/kp.jpg')
    #i = Image.open('/Users/keganrabil/Desktop/cage_sm.jpg')
    a = np.asarray(i)
    trX = a/255.
    return trX

trX = getTrX()

def getRandPixel(w,h):
    x = np.random.randint(0,w-size)
    y = np.random.randint(0,h-size)
    z = np.random.randint(0,3)
    return x,y,z

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.01))

def rectify(X):
    return T.maximum(X, 0.)

def softmax(X):
    e_x = T.exp(X - X.max(axis=1).dimshuffle(0, 'x'))
    return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')

def dropout(X, p=0.):
    if p > 0:
        retain_prob = 1 - p
        X *= srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
        X /= retain_prob
    return X

def RMSprop(cost, params, lr=0.001, rho=0.9, epsilon=1e-6):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - lr * g))
    return updates

def model(x, w, w2, w3, w4, p_drop_conv, p_drop_hidden):
    X.type.dtype = 'float64'
    l1a = rectify(conv2d(X, w, border_mode='full'))
    l1 = max_pool_2d(l1a, (2, 2))
    l1 = dropout(l1, p_drop_conv)

    l2a = rectify(conv2d(l1, w2))
    l2 = max_pool_2d(l2a, (2, 2))
    l2 = dropout(l2, p_drop_conv)

    l3a = rectify(conv2d(l2, w3))
    l3b = max_pool_2d(l3a, (2, 2))
    l3 = T.flatten(l3b, outdim=2)
    l3 = dropout(l3, p_drop_conv)

    l4 = rectify(T.dot(l3, w4))
    l4 = dropout(l4, p_drop_hidden)

    pyx = softmax(T.dot(l4, w_o))
    return l1, l2, l3, l4, pyx


############## Value extractors ####################
def get_w_h(w_h):
    return w_h.get_value()

def get_w_h2(w_h2):
    return w_h2.get_value()

# PICKLE WRAPPERS ... bowchickawowow
def save(var,name):
    f = open(name, 'w')
    pickle.dump(var, f)
    f.close()

def load(name):
    f = open(name,'rb')
    var = pickle.load(f)
    f.close()
    return var

####################################################

def getTrCoords(i,j,k,pred=None):
    coords = []
    coords.append(trX[i,j,k])
    return np.asarray(coords).reshape((1,1))

def getYCoords(i,j,k):
    coords = []
    coords.append(trX[i-1,j,k])
    coords.append(trX[i-1,j-1,k])
    coords.append(trX[i-1,j+1,k])
    coords.append(trX[i,j-1,k])
    coords.append(trX[i,j+1,k])
    coords.append(trX[i+1,j-1,k])
    coords.append(trX[i+1,j,k])
    coords.append(trX[i+1,j+1,k])
    return np.asarray(coords).reshape((1,8))

def evalAcc():
    pred = []
    y = []
    for a in range(20):
        for b in range(20):
            i,j,k = getRandPixel(w,h)
            trainX = getTrCoords(i,j,k)
            y.append(getYCoords(i,j,k))
            p = predict(trainX)
            pred.append(p)
    err = np.mean(np.square(p-y))
    print "Error: ",err
    return err

#############################################

trX, teX, trY, teY = mnist(onehot=True)

trX = trX.reshape(-1, 1, 28, 28)
teX = teX.reshape(-1, 1, 28, 28)

X = T.ftensor4()
Y = T.fmatrix()

w = init_weights((32, 1, 3, 3))
w2 = init_weights((64, 32, 3, 3))
w3 = init_weights((128, 64, 3, 3))
w4 = init_weights((128 * 3 * 3, 625))
w_o = init_weights((625, 10))

noise_l1, noise_l2, noise_l3, noise_l4, noise_py_x = model(X, w, w2, w3, w4, 0.2, 0.5)
l1, l2, l3, l4, py_x = model(X, w, w2, w3, w4, 0., 0.)
y_x = T.argmax(py_x, axis=1)


cost = T.mean(T.nnet.categorical_crossentropy(noise_py_x, Y))
params = [w, w2, w3, w4, w_o]
updates = RMSprop(cost, params, lr=0.001)

train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
predict = theano.function(inputs=[X], outputs=y_x, allow_input_downcast=True)



print len(trX)
for i in range(100):
    print 'begin'
    for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
        cost = train(trX[start:end], trY[start:end])
        #print cost
    print 'Completed iteration ',i,', Cost: ',cost
    print np.mean(np.argmax(teY, axis=1) == predict(teX))

weights = [get_w(w),get_w(w2),get_w(w3),get_w(w4),get_w(w_o)]

f = open('convolution_weights.pckl', 'w')
pickle.dump(weights, f)
f.close()
