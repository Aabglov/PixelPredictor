import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
from load import mnist
from PIL import Image
from numpy.linalg import svd
from matplotlib import pyplot as plot
import pickle

srng = RandomStreams()

size = 3
mid = (size-1)/2
found = False

def grayscale(a):
    b = (0.2989 * a[:,:,0]) + (0.587 * a[:,:,1]) + (0.114 * a[:,:,2])
    return b

def getRandPixel(w,h):
    x = np.random.randint(mid,w-mid)
    y = np.random.randint(mid,h-mid)
    z = np.random.randint(0,3)
    return x,y,z

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.01))

def rectify(X):
    return T.maximum(X, 0.)

def inv_rectify(X):
    return T.minimum(X, 1.)

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

def dropout(X, p=0.):
    if p > 0:
        retain_prob = 1 - p
        X *= srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
        X /= retain_prob
    return X

def model(X, w_h, w_h2, p_drop_input):
    X = dropout(X, p_drop_input)
    h = rectify(T.dot(X, w_h))

    h = dropout(h, p_drop_input)
    py_x = rectify(T.dot(h, w_h2))
    return h, py_x

def invModel(Y,inv_w_h,inv_w_h2):
    h = inv_rectify(T.dot(Y,inv_w_h2))
    i_x = inv_rectify(T.dot(h,inv_w_h))
    return i_x

def get_w_h(w_h):
    return w_h.get_value()

def get_w_h2(w_h2):
    return w_h2.get_value()


X = T.fmatrix()
Y = T.fmatrix()


try:
    file = open('w_h.pckl','rb')
    w_h_raw = pickle.load(file)
    file.close()
    file = open('w_h2.pckl','rb')
    w_h2_raw = pickle.load(file)
    file.close()
    w_h = theano.shared(floatX(w_h_raw))
    w_h2= theano.shared(floatX(w_h2_raw))
    found = True
except:
    w_h = init_weights((3,100))
    w_h2 = init_weights((100, 1))

noise_h, noise_py_x = model(X, w_h, w_h2, 0.0)
h, py_x = model(X, w_h, w_h2, 0.0)

cost = T.mean(T.sqr(noise_py_x - Y))#T.mean(T.nnet.categorical_crossentropy(noise_py_x, Y))
params = [w_h, w_h2]
updates = RMSprop(cost, params, lr=0.000003)

# INVERSE FUNCTIONS
inv_w_h = theano.shared(floatX(np.linalg.pinv(get_w_h(w_h))))
inv_w_h2 = theano.shared(floatX(np.linalg.pinv(get_w_h2(w_h2))))
i_x = invModel(Y,inv_w_h,inv_w_h2)
imagine = theano.function(inputs=[Y], outputs=i_x, allow_input_downcast=True)

train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
predict = theano.function(inputs=[X], outputs=py_x, allow_input_downcast=True)

i = Image.open('/Users/keganrabil/Desktop/Theano/kp.jpg')
#i = Image.open('/Users/keganrabil/Desktop/cage_sm.jpg')
a = np.asarray(i)
g = grayscale(a)
trX = a/255.#Normalize(g)

w,h,d = trX.shape

def getTrCoords(i,j,k):
    coords = []
    coords.append(trX[i,j,k])
    coords.append(trX[i+1,j,k])
    coords.append(trX[i,j+1,k])
    return np.asarray(coords).reshape((1,3))

def getImCoords(i,j,k):
    coords = []
    coords.append(trX[i,j,k])
    coords.append(trX[i+1,j,k])
    coords.append(trX[i,j+1,k])
    return np.asarray(coords).reshape((3,1))

def getYCoords(i,j,k):
    coords = []
    coords.append(trX[i+1,j+1,k])
    return np.asarray(coords).reshape((1,1))

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

        
err = 1.0
mid = 1#(size-1)/2

if not found:
    print 'preparing training set...'
    trainX = getTrCoords(1,1,0)
    trainY = getYCoords(1,1,0)
    for i in range(2,w-size):
        for j in range(2,h-size):
            for k in range(0,3):
                    x = getTrCoords(i,j,k)
                    y = getYCoords(i,j,k)
                    np.concatenate((trainX,x))
                    np.concatenate((trainY,y))

    for m in range(100000):
        cost = train(trainX,trainY)
        if m % 1000 == 0:
            err = evalAcc()

    f = open('w_h.pckl', 'w')
    pickle.dump(get_w_h(w_h), f)
    f.close()

    f = open('w_h2.pckl', 'w')
    pickle.dump(get_w_h2(w_h2), f)
    f.close()
        
print 'creating new image...'
new_image = np.zeros((2*w,2*h,3))
for i in range(0,w-size):
    for j in range(0,h-size):
        for k in range(0,3):
            new_image[2*i,2*j,k] = trX[i,j,k]
            p = imagine(getImCoords(i,j,k))[0]
            #print "P: ",str(p)
            new_image[(2*i)+1,(2*j),k] = p[0]
            new_image[(2*i),(2*j)+1,k] = p[1]
            new_image[(2*i)+1,(2*j)+1,k] = p[2]


k = new_image.shape[0] * new_image.shape[1] * new_image.shape[2]
l = len(new_image[new_image == 0.]) * 1.

plot.imshow(new_image)#b)
#plot.gray()
frame = plot.gca()
frame.axes.get_xaxis().set_visible(False)
frame.axes.get_yaxis().set_visible(False)
plot.show()


