import jax.numpy as np
from jax import grad
from jax.scipy.signal import convolve2d
from numpy import random as rd
import numpy
from optimizadores import *

# activadores

class Relu:

    def activar(self, inputs):
        return np.where(inputs > 0.0, inputs, -0.1 * inputs)

class Softmax:

    def aplicar_softmax(self, inputs):
        temp = np.exp(inputs)
        return temp / np.sum(temp)

    def activar(self, inputs):
        return np.array([self.aplicar_softmax(item) for item in inputs])

class Tanh:

    def activar(self, inputs):
        return np.tanh(inputs)

class Sigmoid:

    def activar(self, inputs):
        return 1.0 / (1.0 + np.exp(-inputs))

class Lineal:

    def activar(self, inputs):
        return inputs


relu = 'relu'
softmax = 'softmax'
sigmoid = 'sigmoid'
lineal = 'lineal'
tanh = 'tanh'

activadores = {relu: Relu(), softmax: Softmax(), sigmoid: Sigmoid(), lineal: Lineal(), '': Lineal(), tanh: Tanh() }

# /////////////////////////////////////

class Model:
    def __init__(self, layers, optimizer=None):
        self.layers = layers
        self.N = None
        self.init_weights()
        self.optimizer = optimizer
        self.g_history = []

    def init_weights(self):
        self.N = 0
        i_aux = {}
        for layer in self.layers:
            i_aux[layer] = self.N
            self.N += layer.N

        #INICIALIZACION DE LOS PESOS
        self.w = rd.rand(self.N) - 0.5
        self.slices = {}
        for layer in self.layers:
            i = i_aux[layer]
            self.slices[layer] = (slice(i, i+layer.N_w), slice(i+layer.N_w, i+layer.N))

    def call(self, inputs, w):
        for layer in self.layers:
            cur_W = w[self.slices[layer][0]]
            cur_b = w[self.slices[layer][1]]

            inputs = layer(inputs, cur_W, cur_b)
        return inputs

    def __call__(self, inputs):
        return self.call(inputs, self.w)

    def predict(self, inputs):
        return self(inputs)

    def loss_function(self, w, inputs, y_targets):
        y_targets = y_targets.reshape((len(y_targets), 1))
        return np.linalg.norm(self.call(inputs,  w) - y_targets)

    def sparse_categorical_crossentropy_loss(self, w, inputs, y_targets):
        y_pred = self.call(inputs, w)
        tmp = np.array([np.log(pred[i]) for pred, i in zip(y_pred, y_targets)])
        return -np.mean(tmp)

    def train(self, X, y, epochs=100, batch_size=50):
        n = len(y)
        for t in range(epochs):
            if batch_size*t >= n:
                print("Data is out.")
                break

            slc = slice(batch_size*t, batch_size*(t+1))

            err = self.sparse_categorical_crossentropy_loss(self.w, X[slc], y[slc])

            y_pred = self(X[slc])
            y_pred_n = numpy.argmax(y_pred, axis=1)
            acc = numpy.sum(y_pred_n==y[slc])/len(y[slc])

            loss_grad = grad(lambda w:self.sparse_categorical_crossentropy_loss(w, X[slc], y[slc]))
            g = loss_grad(self.w)

            self.g_history.append(g)
            self.optimizer.apply_gradients(self, t)

            # print('Epoch: {}              |          Error: {}           |            Accuracy: {} '.format(t+1, err, 0. ))
            str_epoch = 'Epoch {}'.format(t+1)
            str_error = 'Error {}'.format(err)
            str_acc = 'Accuracy {}'.format(acc)

            print( str_epoch + (11- len(str_epoch))*' ' + ' | ' + str_error + (26-len(str_error))*' ' + ' | ' + str_acc + (15-len(str_acc))*' '  )


class Layer:
    def __init__(self, input_dim=None, output_dim=None, activador = ''):
        self.input_dim = input_dim
        self.output_dim = output_dim

        if type(activador) == type(''):
            self.activador = activadores[activador]
        else:
            self.activador = activador

class Dense(Layer):
    def __init__(self, input_dim = None, output_dim=None, activador=''):
        super(Dense, self).__init__(input_dim, output_dim, activador)
        self.N_w = input_dim * output_dim
        self.N_b = output_dim
        self.N = self.N_w + self.N_b

    def __call__(self, inputs, w_vec, b_vec):
        w = np.reshape(w_vec, (self.input_dim, self.output_dim))

        return self.activador.activar(np.dot(inputs, w) + b_vec)


class Conv2d:
    def __init__(self, input_shape=None, kernel_shape = None,  filters=None, activador='', mode = 'valid'):
        self.input_shape = input_shape
        self.filters = filters
        self.mode = mode
        self.kernel_shape = kernel_shape

        if type(activador) == type(''):
            self.activador = activadores[activador]
        else:
            self.activador = activador

        self.calculate_shapes()

    def calculate_shapes(self):
        if self.mode == 'full':
            self.b_shape = (self.filters, self.input_shape[0] + self.kernel_shape[0] - 1, self.input_shape[1] + self.kernel_shape[1] - 1)

        elif self.mode == 'valid':
            self.b_shape = (self.filters, self.input_shape[0] - self.kernel_shape[0] + 1, self.input_shape[1] - self.kernel_shape[1] + 1)

        elif self.mode == 'same':
            self.b_shape = (self.filters, self.input_shape[0], self.input_shape[1])

        self.w_shape = (self.kernel_shape[0], self.kernel_shape[1], self.filters)

        self.N_w =  np.prod(self.w_shape)
        self.N_b = np.prod(self.b_shape)
        self.N = self.N_w + self.N_b

    def __call__(self, inputs, w_vec, b_vec):
        W = np.reshape(w_vec, self.w_shape)
        b = np.reshape(b_vec, self.b_shape)

        def apply(item):
            nrns = np.array([convolve2d(item, W[:,:,i], mode=self.mode) for i in range(self.filters)]) + b
            return self.activador.activar(nrns)

        return np.array(list(map(apply, inputs)))

class Dropout:
    def __init__(self, percent, output_dim):
        self.output_dim = output_dim
        self.N = 0
        self.N_w = 0
        self.N_b = 0
        self.percent = percent
        self.init_ind_c()

    def init_ind_c(self):
        n = int(self.percent * self.output_dim)
        ind_c = numpy.zeros(self.output_dim)

        ind = rd.choice(range(self.output_dim), n, replace=False)
        ind_c[ind] = 1

        self.ind_c = ind_c

    def __call__(self, inputs, w_vec, b_vec):
        tmp = np.array([np.where(self.ind_c, 0., item) for item in inputs])
        return tmp

class Flatten:
    def __init__(self):
        self.N = 0
        self.N_w = 0
        self.N_b = 0

    def __call__(self, inputs, w_vec, b_vec):
        return inputs.reshape((len(inputs), -1))

if __name__ == '__main__':
    pass
    # # ESTE INTENTO FUE MUY EXITOSO. LO HICE EN UN CUADERNO DE JUPYTER
    # from tensorflow.keras.datasets.mnist import load_data
    #
    # (x_train, y_train), (x_test, y_test) = load_data()
    #
    # x_train = x_train/255.0 - 0.5
    #
    # fl = Flatten()
    # d1 = Dense(784, 128, activador='relu')
    # dr = Dropout(0.2, 128)
    # d2 = Dense(128, 10, activador='softmax')
    #
    # opt = Adam(0.005)s
    # m = Model([fl, d1, dr, d2], optimizer=opt)
    #
    # m.train(x_train, y_train, epochs=100, batch_size=100)
