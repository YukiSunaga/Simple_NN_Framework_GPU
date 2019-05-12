from np import *
from layers import *
from functions import *
import sys, os
import pickle
import datetime



class Model:
    def __init__(self, epochs=20, batch_size=32):
        self.layers = []
        self.epochs = epochs
        self.batch_size = batch_size
        self.history = {}
        self.final_result = None

    def predict(self, x, train_flg=False):
        for i, layer in enumerate(self.layers):
            x = layer.forward(x, train_flg)
        return x

    def classify(self, x, train_flg=False):
        y = self.predict(x, train_flg)
        y = np.argmax(y ,axis=1)

        return y

    def classify_all(self, x, train_flg=False, psize=64):
        size = int(x.shape[0]/psize)
        rem = x.shape[0] - size*psize
        y_pred = np.zeros((x.shape[0], ), dtype=int)
        now_iter = 0

        for i in range(size):
            y_pred[psize*i:psize*(i+1)] = self.classify(x[psize*i:psize*(i+1)])
            now_iter += 1
            sys.stdout.write("\r %d / %d" %(now_iter, size+2))
            sys.stdout.flush()

        if not rem == 0:
            y_pred[psize*size:] = self.classify(x[psize*size:])
            now_iter += 1
            sys.stdout.write("\r %d / %d" %(now_iter, size+2))
            sys.stdout.flush()

        return y_pred

    def loss(self, x, t, train_flg=False):
        y = self.predict(x, train_flg)
        y = cross_entropy(y, t)
        return y

    def accuracy(self, x, t):
        y = self.classify(x, train_flg=False)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        acc = np.sum(y == t) / float(x.shape[0])
        return acc

    def loss_acc(self, x, t):
        y = self.predict(x, train_flg=False)
        loss = cross_entropy(y, t)

        y = np.argmax(y ,axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        acc = np.sum(y == t) / float(x.shape[0])
        return loss, acc

    def loss_acc_all(self, x, t, psize=64):
        size = int(x.shape[0]/psize)
        rem = x.shape[0] - size*psize
        loss = 0
        acc = 0
        now_iter = 0

        for i in range(size):
            l, a = self.loss_acc(x[psize*i:psize*(i+1)], t[psize*i:psize*(i+1)])
            loss += l / size
            acc += a / size
            now_iter += 1
            sys.stdout.write("\r %d / %d" %(now_iter, size+2))
            sys.stdout.flush()

        if not rem == 0:
            l, a = self.loss_acc(x[psize*size:], t[psize*size:])
            loss += l / size
            acc += a / size
            now_iter += 1
            sys.stdout.write("\r %d / %d" %(now_iter, size+2))
            sys.stdout.flush()

        return loss, acc

    def _fit(self, x, t):
        loss = self.loss(x, t, train_flg=True)
        layers_r = self.layers.copy()
        layers_r.reverse()
        for i, layer in enumerate(layers_r):
            t = layer.backward(t)
        return loss

    def fit(self, x_train, y_train, x_test, y_test):
        iter_per_epoch = int(max(x_train.shape[0] / self.batch_size, 1))
        max_iter = int(self.epochs * iter_per_epoch)
        iter = 0
        piter = 1
        epoch = 0
        loss = 0
        tr_loss_list = []
        ts_loss_list = []
        tr_acc_list = []
        ts_acc_list = []

        for i in range(max_iter):
            batch_mask = np.random.choice(x_train.shape[0], self.batch_size)
            x_batch = x_train[batch_mask]
            y_batch = y_train[batch_mask]

            loss += self._fit(x_batch, y_batch) / iter_per_epoch
            '''
            ma = 0.
            for l in self.layers:
                maxx = np.max(np.abs(l.W))
                if maxx > ma:
                    ma = maxx
            '''

            #sys.stdout.write("\repoch %d | %d / %d | train_loss %f | %f" %(epoch, piter, iter_per_epoch, loss*iter_per_epoch/piter, ma))
            sys.stdout.write("\repoch %d | %d / %d | train_loss %f" %(epoch, piter, iter_per_epoch, loss*iter_per_epoch/piter))
            sys.stdout.flush()

            if iter % iter_per_epoch == 0:
                epoch += 1

                size = int(x_test.shape[0]/32)
                t_loss = 0
                t_acc = 0
                for i in range(size):
                    loss, acc = self.loss_acc(x_test[32*i:32*(i+1)], y_test[32*i:32*(i+1)])
                    t_loss += loss / size
                    t_acc += acc / size

                print('\n ', 'test_loss', t_loss, 'test_acc', t_acc)

                tr_loss_list.append(loss/piter)
                ts_loss_list.append(t_loss)
                ts_acc_list.append(t_acc)

                piter = 0
                loss = 0
            iter += 1
            piter += 1


        size = int(x_train.shape[0]/32)
        tr_loss = 0
        tr_acc = 0
        for i in range(size):
            loss, acc = self.loss_acc(x_train[32*i:32*(i+1)], y_train[32*i:32*(i+1)])
            tr_loss += loss / size
            tr_acc += acc / size

        size = int(x_test.shape[0]/32)
        ts_loss = 0
        ts_acc = 0
        for i in range(size):
            loss, acc = self.loss_acc(x_test[32*i:32*(i+1)], y_test[32*i:32*(i+1)])
            ts_loss += loss / size
            ts_acc += acc / size

        tr_loss_list.append(tr_loss)
        ts_loss_list.append(ts_loss)
        tr_acc_list.append(tr_acc)
        ts_acc_list.append(ts_acc)

        self.final_result = 'train_loss ' + str(tr_loss) + ', train_acc ' + str(tr_acc) +  ', test_loss ' + str(ts_loss) +  ', test_acc ' + str(ts_acc)
        print('\n\n', 'train_loss', tr_loss, 'train_acc', tr_acc, 'test_loss', ts_loss, 'test_acc', ts_acc)

        self.history['tr_loss'] = tr_loss_list
        self.history['ts_loss'] = ts_loss_list
        self.history['tr_acc'] = tr_acc_list
        self.history['ts_acc'] = ts_acc_list

    def save_history(self, filepath='history.pkl'):
        with open(filepath, 'wb') as f:
            pickle.dump(self.history, f, -1)

    def save_models(self, filepath='model.pkl'):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f, -1)

    def save_txt(self, filepath='hparams.txt'):
        model_name = self.__class__.__name__
        with open(filepath, "a") as f:
            f.write('model : ' + model_name + '\n')
            if not self.final_result is None:
                f.write(self.final_result)

    def save_all(self, savedir=None):
        if savedir is None:
            today = datetime.datetime.today()
            savedir = "result/" + today.strftime("%Y%m%d%H%M") + "/"
            os.makedirs(savedir)
        self.save_txt(savedir + 'hparams.txt')
        self.save_models(savedir + 'model.pkl')
        self.save_history(savedir + 'history.pkl')



class Basic_Model(Model):
    def __init__(self, epochs=10, batch_size=32):
        super().__init__(epochs, batch_size)
        self.layers.append(Dense(activation='Softmax'))

class Hid_Model(Model):
    def __init__(self, epochs=20, batch_size=32, hid_size=128):
        super().__init__(epochs, batch_size)
        self.layers.append(Dense(input_shape=(784, ), output_shape=(hid_size, ), activation='Relu'))
        self.layers.append(Dense(input_shape=(hid_size, ), output_shape=(10, ), activation='Softmax'))

class Conv_Model(Model):
    def __init__(self, epochs=10, batch_size=64):
        super().__init__(epochs, batch_size)
        self.layers.append(Conv(kernels=8, input_shape=(1,28,28), conv_shape=(5,5), conv_pad=0, pool_shape=(2,2), activation='Relu', optimizer='Adam', eps=0.001))
        self.layers.append(Dense(input_shape=(8*12*12, ), output_shape=(32, ), activation='Relu', optimizer='Adam', eps=0.001))
        self.layers.append(Dense(input_shape=(32, ), output_shape=(10, ), activation='Softmax', optimizer='Adam', eps=0.001))

class Conv_Model3(Model):
    def __init__(self, epochs=20, batch_size=128):
        super().__init__(epochs, batch_size)
        self.layers.append(Conv(kernels=32, input_shape=(1,28,28), conv_shape=(5,5), conv_pad=2, pool_shape=(2,2), activation='Relu', batchnorm=True, pos_of_bn=1, optimizer='Nesterov', eps=0.015))
        self.layers.append(Conv(kernels=32, input_shape=(32,14,14), conv_shape=(5,5), conv_pad=2, pool_shape=(2,2), activation='Relu', batchnorm=True, pos_of_bn=1, optimizer='Nesterov', eps=0.015))
        self.layers.append(Dense(input_shape=(32*7*7, ), output_shape=(32*7*7, ), activation='Relu', optimizer='Nesterov', eps=0.015))
        self.layers.append(Dense(input_shape=(32*7*7, ), output_shape=(10, ), activation='Softmax', optimizer='Nesterov', eps=0.015))

class Conv_Model32(Model):
    def __init__(self, epochs=20, batch_size=128, weight_decay=0.0001, batchnorm=True, dropout=True, dropout_ratio=0.2):
        super().__init__(epochs, batch_size)
        self.layers.append(Conv(kernels=32, input_shape=(1,28,28), conv_shape=(5,5), conv_pad=2, pool_shape=(2,2), activation='Relu', batchnorm=True, pos_of_bn=1, weight_decay=weight_decay, dropout=dropout, dropout_ratio=dropout_ratio, optimizer='Nesterov', eps=0.015))
        self.layers.append(Conv(kernels=32, input_shape=(32,14,14), conv_shape=(5,5), conv_pad=2, pool_shape=(2,2), activation='Relu', batchnorm=True, pos_of_bn=1, weight_decay=weight_decay, dropout=dropout, dropout_ratio=dropout_ratio, optimizer='Nesterov', eps=0.015))
        self.layers.append(Dense(input_shape=(32*7*7, ), output_shape=(32*7*7, ), activation='Relu', weight_decay=weight_decay, dropout=dropout, dropout_ratio=dropout_ratio, optimizer='Nesterov', eps=0.015))
        self.layers.append(Dense(input_shape=(32*7*7, ), output_shape=(10, ), activation='Softmax', weight_decay=weight_decay, optimizer='Nesterov', eps=0.015))


class Conv_Model_WDDOBN(Model):
    def __init__(self, epochs=20, batch_size=128, kernels=10, hid_size=32, weight_decay=0.0, batchnorm=True, dropout=False, dropout_ratio=0.25):
        super().__init__(epochs, batch_size)
        self.layers.append(Conv(kernels=kernels, input_shape=(1,28,28), conv_shape=(5,5), conv_pad=0, pool_shape=(2,2), activation='Relu', weight_decay=weight_decay, dropout=dropout, dropout_ratio=dropout_ratio, batchnorm=batchnorm, optimizer='Adam', eps=0.01))
        self.layers.append(Dense(input_shape=(kernels*12*12, ), output_shape=(hid_size, ), activation='Relu', weight_decay=weight_decay, dropout=dropout, dropout_ratio=dropout_ratio, batchnorm=batchnorm, optimizer='Adam', eps=0.01))
        self.layers.append(Dense(input_shape=(hid_size, ), output_shape=(10, ), activation='Softmax', optimizer='Adam', weight_decay=weight_decay, batchnorm=batchnorm, eps=0.001))


class Conv_Model2_WDDOBN(Model):
    def __init__(self, epochs=20, batch_size=128, kernels=32, hid_size=64, weight_decay=0.0001, batchnorm=True, dropout=True, dropout_ratio=0.25, optimizer='Adam', eps=0.01):
        super().__init__(epochs, batch_size)
        self.layers.append(Conv(kernels=kernels, input_shape=(1,28,28), conv_shape=(3,3), conv_pad=0, pool_shape=(2,2), activation='Relu', weight_decay=weight_decay, dropout=dropout, dropout_ratio=dropout_ratio, batchnorm=batchnorm, optimizer=optimizer, eps=eps))
        self.layers.append(Conv(kernels=kernels, input_shape=(kernels,13,13), conv_shape=(4,4), conv_pad=0, pool_shape=(2,2), activation='Relu', weight_decay=weight_decay, dropout=dropout, dropout_ratio=dropout_ratio, batchnorm=batchnorm, optimizer=optimizer, eps=eps))
        self.layers.append(Conv(kernels=kernels, input_shape=(kernels,5,5), conv_shape=(4,4), conv_pad=0, pool_shape=(2,2), activation='Relu', weight_decay=weight_decay, dropout=dropout, dropout_ratio=dropout_ratio, batchnorm=batchnorm, optimizer=optimizer, eps=eps))
        self.layers.append(Dense(input_shape=(kernels*1*1, ), output_shape=(hid_size, ), activation='Relu', weight_decay=weight_decay, dropout=dropout, dropout_ratio=dropout_ratio, batchnorm=batchnorm, optimizer=optimizer, eps=eps))
        self.layers.append(Dense(input_shape=(hid_size, ), output_shape=(10, ), activation='Softmax', weight_decay=weight_decay, batchnorm=batchnorm, optimizer=optimizer, eps=eps))



class Res_Model(Model):
    def __init__(self, epochs=100, batch_size=128, kernels=32, hid_size=256, weight_decay=0.0001, batchnorm=True, dropout=True, dropout_ratio=0.5, optimizer='Nesterov', eps=0.1):
        super().__init__(epochs, batch_size)
        self.layers.append(Conv(kernels=kernels, input_shape=(1,28,28), conv_shape=(5,5), conv_pad=0, pool_shape=(2,2), activation='Relu', weight_decay=weight_decay, dropout=dropout, dropout_ratio=dropout_ratio, batchnorm=batchnorm, optimizer=optimizer, eps=eps))
        self.layers.append(Residual_Block(kernels=kernels, input_shape=(kernels,12,12), conv_shape=(5,5), activation='Relu', weight_decay=weight_decay, dropout=dropout, dropout_ratio=dropout_ratio, batchnorm=batchnorm, optimizer=optimizer, eps=eps))
        self.layers.append(Residual_Block(kernels=kernels, input_shape=(kernels,12,12), conv_shape=(5,5), activation='Relu', weight_decay=weight_decay, dropout=dropout, dropout_ratio=dropout_ratio, batchnorm=batchnorm, optimizer=optimizer, eps=eps))
        self.layers.append(Dense(input_shape=(kernels*12*12, ), output_shape=(hid_size, ), activation='Relu', weight_decay=weight_decay, dropout=dropout, dropout_ratio=dropout_ratio, batchnorm=batchnorm, optimizer=optimizer, eps=eps))
        self.layers.append(Dense(input_shape=(hid_size, ), output_shape=(10, ), activation='Softmax', weight_decay=weight_decay, batchnorm=batchnorm, optimizer=optimizer, eps=eps))

class W_Res_Model(Model):
    def __init__(self, epochs=200, batch_size=128, kernels=32, hid_size=256, weight_decay=0.0005, batchnorm=True, dropout=True, dropout_ratio=0.5, optimizer='Nesterov', eps=0.1):
        super().__init__(epochs, batch_size)
        self.layers.append(Conv(kernels=kernels, input_shape=(1,28,28), conv_shape=(5,5), conv_pad=0, pool_shape=(2,2), activation='Relu', weight_decay=weight_decay, dropout=dropout, dropout_ratio=dropout_ratio, batchnorm=batchnorm, optimizer=optimizer, eps=eps))
        self.layers.append(Residual_Block(kernels=kernels, input_shape=(kernels,12,12), conv_shape=(5,5), activation='Relu', weight_decay=weight_decay, dropout=dropout, dropout_ratio=dropout_ratio, batchnorm=batchnorm, optimizer=optimizer, eps=eps))
        self.layers.append(Residual_Block(kernels=kernels, input_shape=(kernels,12,12), conv_shape=(5,5), activation='Relu', weight_decay=weight_decay, dropout=dropout, dropout_ratio=dropout_ratio, batchnorm=batchnorm, optimizer=optimizer, eps=eps))
        self.layers.append(Residual_Block(kernels=2*kernels, input_shape=(kernels,12,12), conv_shape=(5,5), activation='Relu', weight_decay=weight_decay, dropout=dropout, dropout_ratio=dropout_ratio, batchnorm=batchnorm, optimizer=optimizer, eps=eps))
        self.layers.append(Residual_Block(kernels=2*kernels, input_shape=(2*kernels,12,12), conv_shape=(5,5), activation='Relu', weight_decay=weight_decay, dropout=dropout, dropout_ratio=dropout_ratio, batchnorm=batchnorm, optimizer=optimizer, eps=eps))
        self.layers.append(Dense(input_shape=(2*kernels*12*12, ), output_shape=(hid_size, ), activation='Relu', weight_decay=weight_decay, dropout=dropout, dropout_ratio=dropout_ratio, batchnorm=batchnorm, optimizer=optimizer, eps=eps))
        self.layers.append(Dense(input_shape=(hid_size, ), output_shape=(10, ), activation='Softmax', weight_decay=weight_decay, batchnorm=batchnorm, optimizer=optimizer, eps=eps))


class Conv_Model2(Model):
    def __init__(self, epochs=20, batch_size=32):
        super().__init__(epochs, batch_size)
        self.layers.append(Conv(kernels=32, input_shape=(1,28,28), conv_shape=(5,5), conv_pad=2, pool_shape=(0,0), optimizer='Adam', eps=0.001))
        self.layers.append(Conv(kernels=32, input_shape=(32,28,28), conv_shape=(5,5), conv_pad=2, pool_shape=(2,2), optimizer='Adam', eps=0.001))
        self.layers.append(Conv(kernels=32, input_shape=(32,14,14), conv_shape=(5,5), conv_pad=2, pool_shape=(2,2), optimizer='Adam', eps=0.001))
        self.layers.append(Dense(input_shape=(32*7*7, ), output_shape=(128, ), activation='Relu', optimizer='Adam', eps=0.001))
        self.layers.append(Dense(input_shape=(128, ), output_shape=(10, ), activation='Softmax', optimizer='Adam', eps=0.001))
