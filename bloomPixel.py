import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.downsample import max_pool_2d
import pickle
from random import shuffle

srng = RandomStreams()

# TESTING/DEBUGGING
#theano.config.compute_test_value = 'warn'

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

def model(X, w, w2, w3, w4, w_o, p_drop_conv, p_drop_hidden):
    l1a = rectify(conv2d(X, w, border_mode='valid'))
    l1 = dropout(l1a, p_drop_conv)

    l2a = rectify(conv2d(l1, w2, border_mode='valid'))
    l2 = dropout(l2a, p_drop_conv)

    l3a = rectify(conv2d(l2, w3, border_mode='valid'))
    l3 = dropout(l3a, p_drop_conv)

    l4a = rectify(conv2d(l3, w4, border_mode='valid'))
    l4a = l4a.reshape((-1,30,30))#T.flatten(l4a,outdim=2)
    l4 = dropout(l4a, p_drop_hidden)

    pyx = T.dot(l4, w_o)
    return l1, l2, l3, l4, pyx


############## Value extractors ####################
def get_w(w):
    return w.get_value()


####################################################
# LOAD DATA
print 'loading data...'
##allX = np.asarray(load('trXimg.pckl')).reshape((-1,1,30,30))
##allY = np.asarray(load('trYimg.pckl')).reshape((-1,1*60*60))
##
##ind = int(len(allX) * 0.8)
##
##trX = allX[0:ind]
##trY = allY[0:ind]
##
##teX = allX[ind:]
##teY = allY[ind:]
##print 'data loaded'

trX = np.array(load('trXr.pckl')).reshape((-1,1,15,15))
trY = np.array(load('trYr.pckl')).reshape((-1,30,30))

teX = np.array(load('teXr.pckl')).reshape((-1,1,15,15))
teY = np.array(load('teYr.pckl')).reshape((-1,30,30))
#####################################################


X = T.ftensor4()
Y = T.ftensor3()
X.type.dtype = 'float64'
Y.type.dtype = 'float64'

# TESTING
#X.tag.test_value = np.random.rand(2,1,30,30)
#Y.tag.test_value = np.random.rand(2,60*60)

w  = init_weights((100, 1, 5, 5))
w2 = init_weights((200, 100, 5, 5))
w3 = init_weights((500, 200, 5, 5))
w4 = init_weights((900, 500, 3, 3))
w_o = init_weights((30,30))


noise_l1, noise_l2, noise_l3, noise_l4, noise_py_x = model(X, w, w2, w3, w4, w_o, 0.2, 0.5)
l1, l2, l3, l4, py_x = model(X, w, w2, w3, w4, w_o, 0., 0.)
y_x = py_x


cost = T.mean(T.sqr(noise_py_x - Y))
params = [w, w2, w3, w4, w_o]
updates = RMSprop(cost, params, lr=0.001)

train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
predict = theano.function(inputs=[X], outputs=y_x, allow_input_downcast=True)


#step = 2631
step = 10
try:
    ind = range(len(trX))
    for i in range(100):
        print 'begin'
        shuffle(ind)
        for start, end in zip(range(0, len(trX), step), range(step, len(trX), step)):
            rand_ind = ind[start:end]
            cost = train(trX[rand_ind],trY[rand_ind])
            print cost
        print "ERROR: ",T.mean(T.sqr(predict(teX) - teY)).eval(),' ,Iteration: ',str(i)

    weights = [get_w(w),get_w(w2),get_w(w3),get_w(w4),get_w(w_o)]
    save(weights,'bloom_weights.pckl')
    
except KeyboardInterrupt:
    print 'keyboard interrupt, saving weights...'
    weights = [get_w(w),get_w(w2),get_w(w3),get_w(w4),get_w(w_o)]
    save(weights,'bloom_weights.pckl')
