from np import *

def horizontal_flip(images, labels): # image.shape = (N, Channels, Height, Width)
    out = images.copy()
    out = out[:, :, :, ::-1]
    return out, labels.copy()

def vertical_flip(images, labels): # image.shape = (N, Channels, Height, Width)
    out = images.copy()
    out = out[:, :, ::-1]
    return out, labels.copy()

def horizontal_shift(images, labels, shift_pixels=2):
    rs = np.roll(images, shift_pixels, axis=3)
    rs[:,:,:,:shift_pixels] = 0
    ls = np.roll(images, -1*shift_pixels, axis=3)
    ls[:,:,:,-1*shift_pixels:] = 0
    out = np.concatenate((rs, ls), axis=0)
    olabels = np.concatenate((labels, labels), axis=0)
    return out,olabels

def vertical_shift(images, labels, shift_pixels=2):
    ds = np.roll(images, shift_pixels, axis=2)
    ds[:,:,:shift_pixels] = 0
    us = np.roll(images, -1*shift_pixels, axis=2)
    us[:,:,-1*shift_pixels:] = 0
    out = np.concatenate((ds, us), axis=0)
    olabels = np.concatenate((labels, labels), axis=0)
    return out,olabels


def augmentation(images, labels, hflip=True, vflip=True, hshift=True, vshift=True):
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

    if hshift:
        out, vlabel = horizontal_shift(images, labels)
        oimage = np.concatenate((oimage, out),axis=0)
        olabel = np.concatenate((olabel, vlabel),axis=0)

    if vshift:
        out, vlabel = vertical_shift(images, labels)
        oimage = np.concatenate((oimage, out),axis=0)
        olabel = np.concatenate((olabel, vlabel),axis=0)

    return oimage, olabel
