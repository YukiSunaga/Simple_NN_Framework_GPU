from np import *
from activations import *
from optimizers import *
from functions import *



class Dense:
    def __init__(self, input_shape=(784, ), output_shape=(10, ), activation='Relu',
                batchnorm=False, pos_of_bn=1, dropout=False, dropout_ratio=0.5, weight_decay=0.0,
                optimizer='SGD', eps=0.01):
        self.W = np.sqrt(2.0 / input_shape[0]) * np.random.randn( input_shape[0], output_shape[0] )
        self.b = np.zeros(output_shape[0])
        self.x_shape = None
        self.x = None
        self.u = None
        self.activation = act[activation]()
        self.optimizer = opt[optimizer](eps=eps)
        self.weight_decay = weight_decay
        if batchnorm:
            self.batchnorm = BatchNormalization()
        else:
            self.batchnorm = None
        self.posofbn = pos_of_bn
        if dropout:
            self.dropout = Dropout(dropout_ratio=dropout_ratio)
        else:
            self.dropout = None

    def forward(self, x, train_flg=False):
        if self.batchnorm is not None and self.posofbn == 0:
            x = self.batchnorm.forward(x, train_flg)

        self.x_shape = x.shape
        self.x = x.reshape(x.shape[0], -1)

        u = np.dot(self.x, self.W) + self.b
        self.u = u

        if self.batchnorm is not None and self.posofbn == 1:
            u = self.batchnorm.forward(u, train_flg)

        y = self.activation.forward(u)

        if not self.dropout is None:
            y = self.dropout.forward(y, train_flg)

        return y

    def backward(self, delta):
        if not self.dropout is None:
            delta = self.dropout.backward(delta)
        dx = self.activation.backward(delta)

        if self.batchnorm is not None and self.posofbn == 1:
            dx = self.batchnorm.backward(dx)

        self.dW = np.dot(self.x.T, dx) + self.weight_decay * self.W
        self.db = np.sum(dx, axis=0)

        dx = np.dot(dx, self.W.T)
        dx = dx.reshape(*self.x_shape)

        if self.batchnorm is not None and self.posofbn == 0:
            dx = self.batchnorm.backward(dx)

        self.optimizer.update([self.W, self.b], [self.dW, self.db])

        return dx


class Conv:
    def __init__(self, kernels=8, input_shape=(1,28,28), conv_shape=(5,5), conv_pad=0, conv_stride=1,
                pool_shape=(2,2), pool_pad=0, pool_stride=2,
                batchnorm=False, pos_of_bn=1, dropout=False, dropout_ratio=0.25, weight_decay=0.0,
                activation='Relu', optimizer='SGD', eps=0.01):
        self.W = np.sqrt(2.0/ (conv_shape[0] * conv_shape[1]*input_shape[0])) * np.random.randn( kernels, input_shape[0], conv_shape[0], conv_shape[1] )
        self.b = np.zeros(kernels)
        self.W_col = None
        self.conv_pad = conv_pad
        self.conv_stride = conv_stride
        self.pool_shape = pool_shape
        self.pool_pad = pool_pad
        self.pool_stride = pool_stride
        self.x_shape = None
        self.x = None
        self.x_col = None
        self.u = None
        self.u_col = None
        self.conv_y = None
        self.conv_y_col = None
        self.activation = act[activation]()
        self.optimizer = opt[optimizer](eps=eps)
        self.weight_decay = weight_decay
        if batchnorm:
            self.batchnorm = BatchNormalization()
        else:
            self.batchnorm = None
        self.posofbn = pos_of_bn
        if dropout:
            self.dropout = Dropout(dropout_ratio=dropout_ratio)
        else:
            self.dropout = None

    def forward(self, x, train_flg=False):
        if self.batchnorm is not None and self.posofbn == 0:
            x = self.batchnorm.forward(x, train_flg)

        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = 1 + int((H + 2*self.conv_pad - FH) / self.conv_stride)
        out_w = 1 + int((W + 2*self.conv_pad - FW) / self.conv_stride)

        col = im2col(x, FH, FW, self.conv_stride, self.conv_pad)
        col_W = self.W.reshape(FN, -1).T

        self.u_col = np.dot(col, col_W) + self.b
        u = self.u_col.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)
        self.u = u

        self.x = x
        self.x_col = col
        self.W_col = col_W


        if self.batchnorm is not None and self.posofbn == 1:
            u = self.batchnorm.forward(u, train_flg)


        self.conv_y = self.activation.forward(u)
        y = self.conv_y

        if self.batchnorm is not None and self.posofbn == 2:
            y = self.batchnorm.forward(y, train_flg)

        if self.pool_shape[0] != 0 and self.pool_shape[1] != 0:
            N, C, H, W = self.conv_y.shape
            out_h = 1 + int((H + 2*self.pool_pad - self.pool_shape[0]) / self.pool_stride)
            out_w = 1 + int((W + 2*self.pool_pad - self.pool_shape[1]) / self.pool_stride)

            self.conv_y_col = im2col(self.conv_y, self.pool_shape[0], self.pool_shape[1], self.pool_stride, self.pool_pad)
            self.conv_y_col = self.conv_y_col.reshape(-1, self.pool_shape[0]*self.pool_shape[1])

            arg_max = np.argmax(self.conv_y_col, axis=1)
            self.pool_y = np.max(self.conv_y_col, axis=1)
            self.pool_y = self.pool_y.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)
            y = self.pool_y

            self.arg_max = arg_max

        if not self.dropout is None:
            y = self.dropout.forward(y, train_flg)

        return y

    def backward(self, delta):
        if not self.dropout is None:
            delta = self.dropout.backward(delta)

        if self.pool_shape[0] != 0 and self.pool_shape[1] != 0:
            delta = delta.transpose(0, 2, 3, 1)
            pool_size = self.pool_shape[0] * self.pool_shape[1]
            dmax = np.zeros((delta.size, pool_size))
            dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = delta.flatten()
            dmax = dmax.reshape(delta.shape + (pool_size,))

            dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
            delta = col2im(dcol, self.conv_y.shape, self.pool_shape[0], self.pool_shape[1], self.pool_stride, self.pool_pad)

        if self.batchnorm is not None and self.posofbn == 2:
            delta = self.batchnorm.backward(delta)

        FN, C, FH, FW = self.W.shape

        dx = self.activation.backward(delta)
        if self.batchnorm is not None and self.posofbn == 1:
            dx = self.batchnorm.backward(dx)

        dx = dx.transpose(0,2,3,1).reshape(-1, FN)

        self.db = np.sum(dx, axis=0)
        self.dW = np.dot(self.x_col.T, dx)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW) + self.weight_decay * self.W

        dx = np.dot(dx, self.W_col.T)
        dx = col2im(dx, self.x.shape, FH, FW, self.conv_stride, self.conv_pad)

        if self.batchnorm is not None and self.posofbn == 0:
            dx = self.batchnorm.backward(dx)

        self.optimizer.update([self.W, self.b], [self.dW, self.db])

        return dx


class Pool:
    def __init__(self, pool_shape=(2,2), pool_pad=0, pool_stride=2):
        self.pool_shape = pool_shape
        self.pool_pad = pool_pad
        self.pool_stride = pool_stride
        self.arg_max = None
        self.x_shape = None


    def forward(self, x, train_flg=False):
        self.x_shape = x.shape
        y = x
        if self.pool_shape[0] != 0 and self.pool_shape[1] != 0:
            N, C, H, W = y.shape
            out_h = 1 + int((H + 2*self.pool_pad - self.pool_shape[0]) / self.pool_stride)
            out_w = 1 + int((W + 2*self.pool_pad - self.pool_shape[1]) / self.pool_stride)

            y = im2col(y, self.pool_shape[0], self.pool_shape[1], self.pool_stride, self.pool_pad)
            y = y.reshape(-1, self.pool_shape[0]*self.pool_shape[1])

            arg_max = np.argmax(y, axis=1)
            y = np.max(y, axis=1)
            y= y.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

            self.arg_max = arg_max

        return y

    def backward(self, delta):
        if self.pool_shape[0] != 0 and self.pool_shape[1] != 0:
            delta = delta.transpose(0, 2, 3, 1)
            pool_size = self.pool_shape[0] * self.pool_shape[1]
            dmax = np.zeros((delta.size, pool_size))
            dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = delta.flatten()
            dmax = dmax.reshape(delta.shape + (pool_size,))

            dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
            delta = col2im(dcol, self.x_shape, self.pool_shape[0], self.pool_shape[1], self.pool_stride, self.pool_pad)

        return delta



class Dropout:
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_flg=True):
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        return dout * self.mask


class BatchNormalization:
    def __init__(self, momentum=0.9, running_mean=None, running_var=None, optimizer='Adam', eps=0.001):
        self.gamma = 1
        self.beta = 0
        self.flg = True
        self.momentum = momentum
        self.input_shape = None

        self.running_mean = running_mean
        self.running_var = running_var

        self.batch_size = None
        self.xc = None
        self.xn = None
        self.std = None
        self.dgamma = None
        self.dbeta = None

        self.optimizer = opt[optimizer](eps=eps)

    def forward(self, x, train_flg=True):
        self.input_shape = x.shape
        if x.ndim != 2:
            N, C, H, W = x.shape
            D = C*H*W
            x = x.reshape(N, -1)
        else:
            D = x.shape[1]

        if self.flg:
            self.gamma = np.ones((D))
            self.beta = np.zeros((D))
            self.flg = False

        out = self.__forward(x, train_flg)

        return out.reshape(*self.input_shape)

    def __forward(self, x, train_flg):
        if self.running_mean is None:
            N, D = x.shape
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)

        if train_flg:
            mu = x.mean(axis=0)
            xc = x - mu
            var = np.mean(xc**2, axis=0)
            std = np.sqrt(var + 10e-7)
            xn = xc / std

            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std
            self.running_mean = self.momentum * self.running_mean + (1-self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1-self.momentum) * var
        else:
            xc = x - self.running_mean
            xn = xc / ((np.sqrt(self.running_var + 10e-7)))

        out = self.gamma * xn + self.beta
        return out

    def backward(self, dout):
        if dout.ndim != 2:
            N, C, H, W = dout.shape
            dout = dout.reshape(N, -1)

        dx = self.__backward(dout)

        dx = dx.reshape(*self.input_shape)

        self.optimizer.update([self.gamma, self.beta], [self.dgamma, self.dbeta])

        return dx

    def __backward(self, dout):
        dbeta = dout.sum(axis=0)
        dgamma = np.sum(self.xn * dout, axis=0)
        dxn = self.gamma * dout
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size

        self.dgamma = dgamma
        self.dbeta = dbeta

        return dx

class Residual_Block:
    def __init__(self, input_shape, kernels=32, conv_shape=(3,3), pool_shape=(0,0),
                batchnorm=False, dropout=False, dropout_ratio=0.25, weight_decay=0.0,
                activation='Relu', optimizer='Adam', eps=0.001):
        self.filters = kernels
        self.input_shape = input_shape

        channels = kernels
        pad = int((conv_shape[0] - 1)/2)

        self.conv1 = Conv(input_shape=input_shape, kernels=kernels, conv_shape=conv_shape, conv_pad=pad, pool_shape=(0,0),
                            optimizer=optimizer, batchnorm=batchnorm, weight_decay=weight_decay, dropout=dropout, dropout_ratio=dropout_ratio,
                            activation=activation, eps=eps)
        x_shape2 = (channels, ) + input_shape[1:]

        self.conv2 = Conv(input_shape=x_shape2, kernels=kernels, conv_shape=conv_shape, conv_pad=pad, pool_shape=(0,0),
                            optimizer=optimizer, batchnorm=batchnorm, weight_decay=weight_decay, dropout=False, dropout_ratio=dropout_ratio,
                            activation=activation, eps=eps)

        self.pool = None
        if pool_shape[0] != 0 and pool_shape[1] != 0:
            self.pool = Pool(pool_shape=pool_shape)

    def forward(self, x, train_flg=False):
        y = self.conv1.forward(x, train_flg)
        y = self.conv2.forward(y, train_flg)

        if x.shape[1] < self.filters:
            shortcut = np.zeros((x.shape[0], self.filters, x.shape[2], x.shape[3]))
            shortcut[:, :x.shape[1]] = x
        else:
            shortcut = x
        y = y + shortcut

        if not self.pool is None:
            y = self.pool.forward(x, train_flg)

        return y

    def backward(self, delta):
        if not self.pool is None:
            delta = self.pool.backward(delta)
        dx = self.conv2.backward(delta)
        dx = self.conv1.backward(dx)
        if self.input_shape[0] < self.filters:
            delta = delta[:,:self.input_shape[0]]
        dx = delta + dx
        return dx


class DataAugmentation:
    def __init__(self, hflip=True, vflip=True, shift=True, max_shift_px=2):
        self.hflip = hflip
        self.vflip = vflip
        self.shift = shift
        self.auglist = [self.id]
        if hflip:
            self.auglist.append(self.horizontal_flip)
        if vflip:
            self.auglist.append(self.vertical_flip)
        if shift:
            self.auglist.append(self.hv_shift)

        self.max_shift_px = max_shift_px

    def id(self, images):
        return images

    def horizontal_flip(self, images): # image.shape = (N, Channels, Height, Width)
        out = images.copy()
        out = out[:, :, :, ::-1]
        return out

    def vertical_flip(self, images): # image.shape = (N, Channels, Height, Width)
        out = images.copy()
        out = out[:, :, ::-1]
        return out

    def hv_shift0(self, images):
        size = images.shape[0]
        hs = np.random.randint(0, self.max_shift_px+1, size)
        vs = np.clip(np.random.randint(0, self.max_shift_px+1, size) - hs, 0, None)
        hs *= np.random.choice([-1,1], size)
        vs *= np.random.choice([-1,1], size)
        out = images.copy()
        for i in range(size):
            out[i] = np.roll(out[i], hs[i], axis=2)
            if hs[i] > 0:
                out[i,:,:,:hs[i]] = 0
            elif hs[i] < 0:
                out[i,:,:,hs[i]:] = 0
        for i in range(size):
            out[i] = np.roll(out[i], vs[i], axis=1)
            if vs[i] > 0:
                out[i,:,:vs[i]] = 0
            elif vs[i] < 0:
                out[i,:,vs[i]:] = 0

        return out

    def hv_shift(self, images):
        import random
        size = images.shape[0]
        hs = np.random.randint(0, self.max_shift_px+1)
        vs = np.clip(np.random.randint(0, self.max_shift_px+1) - hs, 0, None)
        hs *= random.choice([-1,1])
        vs *= random.choice([-1,1])
        out = images.copy()

        out = np.roll(out, hs, axis=3)
        if hs > 0:
            out[:,:,:,:hs] = 0
        elif hs < 0:
            out[:,:,:,hs:] = 0

        out = np.roll(out, vs, axis=2)
        if vs > 0:
            out[:,:,:vs] = 0
        elif vs < 0:
            out[:,:,vs:] = 0

        return out

    def forward1(self, x, train_flg=False):
        if (not train_flg) or len(self.auglist) == 1:
            return x

        y = x.copy()
        size = x.shape[0]
        aug = np.random.randint(0, len(self.auglist), size)
        for i in range(size):
            y[i] = self.auglist[int(aug[i])](x[i].reshape( (1,) + x[i].shape )).reshape(x[i].shape)

        return y


    def forward(self, x, train_flg=False):
        if (not train_flg) or len(self.auglist) == 1:
            return x

        size = x.shape[0]
        aug = np.random.randint(0, len(self.auglist))
        y = self.auglist[int(aug)](x)

        return y

    def backward(self, delta):
        return delta
