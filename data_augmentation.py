from np import *

def horizontal_flip(images, labels): # image.shape = (N, Channels, Height, Width)
    out = images.copy()
    out = out[:, :, :, ::-1]
    return out, labels.copy()

def vertical_flip(images, labels): # image.shape = (N, Channels, Height, Width)
    out = images.copy()
    out = out[:, :, ::-1]
    return out, labels.copy()

def augmentation(images, labels, hflip=True, vflip=True):
    oimage = images.copy()
    olabel = labels.copy()

    if hflip:
        out, hlabel = horizontal_flip(images, labels)
        oimage = np.concatenate((oimage, out),axis=0)
        olabel = np.concatenate((olabel, hlabel),axis=0)

    if vflip:
        out, vlabel = vertical_flip(images, labels)
        oimage = np.concatenate((oimage, out),axis=0)
        olabel = np.concatenate((olabel, vlabel),axis=0)

    return oimage, olabel
