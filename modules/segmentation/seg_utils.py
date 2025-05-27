import numpy as np
import keras.backend as K


def weighted_dice_coef(y_true, y_pred, smooth=1e-7):
    y_true_0 = K.flatten(K.one_hot(K.cast(y_true, 'int64'), num_classes=15)[...,0])
    y_pred_0 = K.flatten(y_pred[...,0])
    intersect_0 = K.sum(y_true_0 * y_pred_0, axis=-1)
    denom_0 = K.sum(y_true_0 + y_pred_0, axis=-1)
    y_true_1 = K.flatten(K.one_hot(K.cast(y_true, 'int64'), num_classes=15)[...,1:])
    y_pred_1 = K.flatten(y_pred[...,1:])
    intersect_1 = K.sum(y_true_1 * y_pred_1, axis=-1)*10000
    denom_1 = K.sum(y_true_1 + y_pred_1, axis=-1)*10000
    intersect = intersect_0 + intersect_1
    denom = denom_0 + denom_1
    return K.mean((2. * intersect / (denom + smooth)))

def weighted_dice_coef_loss(y_true, y_pred):
    '''
    Dice loss to minimize. Pass to model as loss during compile statement
    '''
    return 1 - weighted_dice_coef(y_true, y_pred)

def dice_coef(y_true, y_pred, smooth=1e-7):
    y_true_f = K.flatten(K.one_hot(K.cast(y_true, 'int64'), num_classes=15)[...,1:])
    y_pred_f = K.flatten(y_pred[...,1:])
    intersect = K.sum(y_true_f * y_pred_f, axis=-1)
    denom = K.sum(y_true_f + y_pred_f, axis=-1)
    return K.mean((2. * intersect / (denom + smooth)))

def dice_coef_loss(y_true, y_pred):
    '''
    Dice loss to minimize. Pass to model as loss during compile statement
    '''
    return 1 - dice_coef(y_true, y_pred)

def standardize(img):
    '''
    Standardize the img value to near -1~1
    '''
    N = np.shape(img)[0] * np.shape(img)[1]
    s = np.maximum(np.std(img), 1.0 / np.sqrt(N))
    m = np.mean(img)
    img = (img - m) / s
    del m, s, N
    #
    if np.ndim(img) == 2:
        img = np.dstack((img, img, img))
    return img