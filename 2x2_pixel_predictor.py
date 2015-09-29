import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
from load import mnist
from PIL import Image
from numpy.linalg import svd
from matplotlib import pyplot as plot

srng = RandomStreams()

def grayscale(a):
    b = (0.2989 * a[:,:,0]) + (0.587 * a[:,:,1]) + (0.114 * a[:,:,2])
    return b

def getRandPixel(w,h):
    x = np.random.randint(0,w-2)
    y = np.random.randint(0,h-2)
    z = np.random.randint(0,3)
    return x,y,z

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.01))

def rectify(X):
    return T.maximum(X, 0.)
    #return T.nnet.sigmoid(X)

def softmax(X):
    e_x = T.exp(X - X.max(axis=1).dimshuffle(0, 'x'))
    return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')

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

def getTrCoords(i,j,k):
    coords = []
    for a in range(0,4,2):
        for b in range(0,4,2):
            coords.append(trX[i+a,j+b,k])
    return np.asarray(coords).reshape((1,4))

def getPredCoords(i,j,k):
    coords = []
    for a in range(0,2):
        for b in range(0,2):
            coords.append(trX[i+a,j+b,k])
    return np.asarray(coords).reshape((1,4))

def evalAcc():
    pred = []
    y = []
    for a in range(20):
        for b in range(20):
            i,j,k = getRandPixel(w,h)
            trainX = getTrCoords(i,j,k)#np.asarray([trX[i,j,k],trX[i,j+2,k],trX[i+2,j,k],trX[i+2,j+2,k]]).reshape((1,4))
            trainY = np.asarray(trX[i+1,j+1,k]).reshape((1,1))
            y.append(trX[i+1,j+1,k])
            p = predict(trainX)
            pred.append(p)
    err = np.mean(np.square(p-y))
    print "Error: ",err
    return err

X = T.fmatrix()
Y = T.fmatrix()

w_h = init_weights((4,25))
w_h2 = init_weights((25, 1))

noise_h, noise_py_x = model(X, w_h, w_h2, 0.0)
h, py_x = model(X, w_h, w_h2, 0.)
y_x = py_x

cost = T.mean(T.sqr(noise_py_x - Y))#T.mean(T.nnet.categorical_crossentropy(noise_py_x, Y))
params = [w_h, w_h2]
updates = RMSprop(cost, params, lr=0.000003)

train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
predict = theano.function(inputs=[X], outputs=y_x, allow_input_downcast=True)

i = Image.open('/Users/keganrabil/Desktop/Theano/kp.jpg')
a = np.asarray(i)
g = grayscale(a)
trX = a/255.#Normalize(g)
w,h,d = trX.shape

       
err = 1.0
#i = 0
for i in range(100000):
#while err > 0.001:
    #for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
    #    cost = train(trX[start:end], trY[start:end])
    #print np.mean(np.argmax(teY, axis=1) == predict(teX))
    indx,indy,z = getRandPixel(w,h)
    trainX = getTrCoords(indx,indy,z)#np.asarray([trX[indx,indy,z],trX[indx,indy+2,z],trX[indx+2,indy,z],trX[indx+2,indy+2,z]]).reshape((1,4))
    trainY = np.asarray(trX[indx+1,indy+1,z]).reshape((1,1))
    cost = train(trainX,trainY)
    if i % 1000 == 0:
        err = evalAcc()
    #i += 1

        
print 'calculating new image'
new_image = np.zeros((2*w,2*h,3))
for i in range(1,w-1):
    for j in range(1,h-1):
        for k in range(0,3):
            new_image[2*i,2*j,k] = trX[i,j,k]
            p1 = predict(np.asarray(getPredCoords(i,j,k)))
            p2 = predict(np.asarray(getPredCoords(i-1,j,k)))
            p3 = predict(np.asarray(getPredCoords(i,j-1,k)))
            #p1 = predict(np.asarray([trX[i,j,k],trX[i,j+1,k],trX[i+1,j,k],trX[i+1,j+1,k]]).reshape((1,4)))
            #p2 = predict(np.asarray([trX[i-1,j,k],trX[i-1,j+1,k],trX[i,j,k],trX[i,j+1,k]]).reshape((1,4)))
            #p3 = predict(np.asarray([trX[i,j-1,k],trX[i,j,k],trX[i+1,j-1,k],trX[i+1,j,k]]).reshape((1,4)))
            
            new_image[(2*i)+1,(2*j)+1,k] = p1
            new_image[(2*i),(2*j)+1,k] = p2
            new_image[(2*i)+1,(2*j),k] = p3

plot.imshow(new_image)#b)
#plot.gray()
frame = plot.gca()
frame.axes.get_xaxis().set_visible(False)
frame.axes.get_yaxis().set_visible(False)
plot.show()


