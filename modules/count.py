import utils
import json
from pathlib import Path

from .helper import _point2adm2imgpathlist, point2adm2meta, get_ccode_loc
from .segmentation.segmentation import get_segments

_CONFIG = utils.read_config()

def get_landcover_ratio(loc, target_class, zoom_level=None, proxy_dir=None):
    """
    Count the pixels in the area of the target class in the location image
    
    Args:
        target_class: The class of target object
        loc: The location of the interest
        zoom_level: Zoom level of the image
        proxy_dir: The proxy dir to save the intermediate results
    
    Return:
        Number of pixels in the area of target class objects in the location image
    """
    # timeline =32645
    config = utils.read_config()
    timeline = config.arcgis.timeline
    if proxy_dir is None:
        proxy_dir = _CONFIG.path.proxy_dir
    ccode, adm1_name, adm2_name, areaid, _ = point2adm2meta(loc)
    proxy_dir = Path(proxy_dir)
    seg_path = proxy_dir / f'{ccode}_segmentation_{timeline}.json'
    loc_data = get_ccode_loc(ccode)
    assert len(loc_data)>0
    loc_weight = loc_data[areaid]['weight']

    if seg_path.exists():
        with open(seg_path, 'r') as file:
            data = json.load(file)
           
        if areaid in data.keys() and 'segmentation' in data[areaid].keys():
            target_area_info = data[areaid]['segmentation'][target_class]
            result_dict = {'segmentation': target_area_info}
            return result_dict
    else:
        data = {}
    
    data[areaid] = {}
    img_path_list = _point2adm2imgpathlist(loc, zoom_level, timeline=timeline)
    area_dict = get_segments(img_path_list)

    for each_class in area_dict.keys():
        area_dict[each_class]['type'] = 'ratio'
        area_dict[each_class]['weight'] = loc_weight
    
    data[areaid]["ADM0"] = ccode
    data[areaid]["ADM1"] = adm1_name
    if adm2_name is not None:
        data[areaid]["ADM2"] = adm2_name
    data[areaid]['segmentation'] = area_dict
        
    with open(seg_path, 'w') as file:
        json.dump(data, file, indent=4)
    
    result_dict = {'segmentation': area_dict[target_class]}
    return result_dict

def get_landuse_sum(loc, target_class, zoom_level=None, proxy_dir=None):
    """
    GeoChat module calculates the total sum of each classes.
    
    Args:
        target_class: The class of target object
        loc: The location of the interest
        zoom_level: Zoom level of the image
        proxy_dir: The proxy dir to save the intermediate results
    
    Return:
        sum of probabilities of each classes within the target location
    """
    # timeline =32645
    config = utils.read_config()
    timeline = config.arcgis.timeline
    if proxy_dir is None:
        proxy_dir = _CONFIG.path.proxy_dir
    ccode, adm1_name, adm2_name, areaid, _ = point2adm2meta(loc)
    proxy_dir = Path(proxy_dir)
    seg_path = proxy_dir / f'{ccode}_landuse_{timeline}.json'
    loc_data = get_ccode_loc(ccode)
    assert len(loc_data)>0

    if seg_path.exists():
        with open(seg_path, 'r') as file:
            data = json.load(file)
           
        if areaid in data.keys() and 'landuse' in data[areaid].keys():
            target_area_info ={}
            target_area_val = data[areaid]['landuse']['Landuse_Sum']['val'][target_class]
            target_area_desc= f'{target_class} satelite image count'
            target_area_type= data[areaid]['landuse']['Landuse_Sum']['type']
            target_area_weight= data[areaid]['landuse']['Landuse_Sum']['weight']
            target_area_info['val'] = target_area_val
            target_area_info['desc'] = target_area_desc
            target_area_info['type'] = target_area_type
            target_area_info['weight'] = target_area_weight
            result_dict = {'segmentation': target_area_info}
            return result_dict
    else:
        data = {}
    

    return result_dict

