import os
import gc
import PIL
import numpy as np
import pandas as pd
import tensorflow as tf
import warnings
warnings.filterwarnings(action='ignore')

from .seg_utils import *

_MODEL_PATH = "./trained_models/OpenEarthMap_9class_RGB_512_v4_fullmodel.h5"
_CLASSES = ["nodata", "bareland", "rangeland", "development", "road", "tree", "water", "agricultural", "building"]
_IMG_SIZE = 512


def _model_predict_batchwise(model, img_path_list, batch_size=512):    
    return_dict = {}
    for class_name in _CLASSES:
        return_dict[class_name] = {}
        return_dict[class_name]['val'] = 0
        return_dict[class_name]['desc'] = f'Land cover ratio of {class_name}'
        return_dict[class_name]['type'] = 'ratio'
        return_dict[class_name]['weight'] = None
            
    total_length = len(img_path_list)    
    for i in range((total_length-1) // batch_size + 1):
        target_img_path_list = img_path_list[i * (batch_size): (i+1) * (batch_size)]
        img_list = []
        for ImPath in target_img_path_list:
            img = tf.keras.preprocessing.image.load_img(ImPath, target_size = (_IMG_SIZE, _IMG_SIZE))
            img_arr = tf.keras.preprocessing.image.img_to_array(img)    
            TestIm = standardize(img_arr)
            img_list.append(TestIm)

        batched_dataset = np.stack(img_list, axis=0)  
        pred = model.predict(batched_dataset).squeeze()
        gc.collect()
        
        flip = np.stack([np.flipud(im) for im in batched_dataset], axis=0)
        pred += np.stack([np.flipud(result) for result in model.predict(flip).squeeze()], axis=0) 
        gc.collect()
        
        flip = np.stack([np.fliplr(im) for im in batched_dataset], axis=0)
        pred += np.stack([np.fliplr(result) for result in model.predict(flip).squeeze()], axis=0) 
        gc.collect()
        
        flip = np.stack([np.flipud(np.fliplr(im)) for im in batched_dataset], axis=0)
        pred += np.stack([np.flipud(np.fliplr(result)) for result in model.predict(flip).squeeze()], axis=0)
        gc.collect()

        est_label = pred / 4
        if len(est_label.shape) < 4:
            est_label = np.expand_dims(est_label, axis=0)
        
        TTAlab = np.argmax(est_label, -1)
        for i in range(TTAlab.shape[0]):
            key, val = np.unique(TTAlab[i], return_counts=True)
            for i, k in enumerate(key):
                return_dict[_CLASSES[k]]['val'] += (val[i] / (_IMG_SIZE * _IMG_SIZE * total_length))

    return return_dict
    

def get_segments(img_path_list):
    """
    Perform image segmentation on given image lists.
    
    Args: 
        img_path_list: list of string path to satellite image
    
    Return:
        Dictionary that includes the ratio of each segment, computed from entire image set.
    """
    # model = tf.keras.models.load_model(_MODEL_PATH, custom_objects={
    #        'weighted_MC_dice_coef_loss':weighted_dice_coef_loss,
    #        'mean_iou':tf.compat.v1.metrics.mean_iou,
    #        'dice_coef':dice_coef
    #     }
    # )
        # 모델 로드만 scope 안에서
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = tf.keras.models.load_model(
            _MODEL_PATH,
            custom_objects={
                'weighted_MC_dice_coef_loss': weighted_dice_coef_loss,
                'mean_iou': tf.compat.v1.metrics.mean_iou,
                'dice_coef': dice_coef
            }
        )
    return_dict = _model_predict_batchwise(model, img_path_list)
    
    return return_dict
    