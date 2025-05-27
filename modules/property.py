import json
import pickle
import arcgis
import requests
import rasterio
from arcgis.geocoding import reverse_geocode
from arcgis.geoenrichment import Country
from arcgis.geometry import Geometry
from rasterio.mask import mask
from pathlib import Path

import pandas as pd
import geopandas as gpd
from pyproj import Geod
from shapely.geometry import Point, MultiPolygon, mapping
from shapely.geometry.polygon import Polygon
from shapely.ops import transform
import rasterio
from rasterio.mask import mask
import os, glob
import utils
from PIL import Image

from .helper import get_loc_geometry, get_ring_contained_loc, get_country, _get_y_x
from .helper import _point2address, point2adm2meta, _point2adm2area, _point2adm2imgpathlist
from .helper import deg2num, num2deg
from .helper import get_temp_func_str, set_temp_func_str, get_func_repr_dict, set_func_repr_dict, clear_func_repr_dict, get_ccode_loc

_CONFIG = utils.read_config()

Image.MAX_IMAGE_PIXELS = None

def get_address(loc):
    """
    Get address string of the given location
    
    Args:
        loc: The location of the interest
    
    Return:
        The address string of the given location
        Return None for error case.
    """

    loc_info = {
        'val': _point2address(loc),
        'desc': 'full address of the given location',
        'type':'str',
        'weight':None
    }
    result_dict = {'address': loc_info}
    return result_dict

def get_adm2meta(loc):
    return point2adm2meta(loc)

def get_adm2imgpathlist(loc, zoom_level=None, timeline=None):
    return _point2adm2imgpathlist(loc, zoom_level, timeline)

def point2enrichment(
    loc,
    enrich_v,
    proxy_dir=None
):
    """
    Get enrichment information of the given location
    
    Args:
        loc: The location of the interest
        enrich_v: The enrichment column of the interest
        proxy_dir: The proxy dir to save the intermediate results
    
    Return:
        The enrichment information of the given location
        Return None for error case.
    """
    if proxy_dir is None:
        proxy_dir = _CONFIG.path.proxy_dir
    ccode, adm1_name, adm2_name, areaid, loc_geom = get_adm2meta(loc)
    country = get_country(ccode)
    proxy_dir = Path(proxy_dir)
    
    enrich_v = enrich_v.lower()
    ev_df = country.enrich_variables
    target_ev_df = ev_df[(ev_df.data_collection.str.lower().str.contains(enrich_v))]
    varname_list = list(target_ev_df['name'])
    vardesc_list = list(target_ev_df['description'])
    var_dict = dict(zip(varname_list, vardesc_list))

    #result_dict = {'<varname>':{'stat':<float>, 'desc':<str>}}
    #json -- {'areaid':{'ADM0':<str>, 'ADM1':<str>, 'ADM2':<str>, 'enrichment':{'<enrich_v>':{'<varname>':{'val':<float>, 'desc':<str>}}}}, ...}
    enrich_path = proxy_dir / f'{ccode}_{enrich_v}_enrich.json'
    result_dict = None
    download = False
    
    if enrich_path.exists():
        with open(enrich_path, 'r') as file:
            data = json.load(file)
            
        if areaid not in data.keys():
            data[areaid] = {
                'ADM0':ccode,
                'ADM1':adm1_name,
                'ADM2':adm2_name,
                'enrichment':{}
            }
            download = True
        elif enrich_v in data[areaid]['enrichment'].keys():
            result_dict = data[areaid]['enrichment'][enrich_v]
            download = False
        else:
            download = True
    else:
        data = {}
        data[areaid] = {
            'ADM0':ccode,
            'ADM1':adm1_name,
            'ADM2':adm2_name,
            'enrichment':{}
        }
        download = True
    
    if download:
        geom_enrich = country.enrich(study_areas= [loc_geom],enrich_variables=target_ev_df)
        result_dict={}
        print(target_ev_df)

        for varname in varname_list:
            varname_low = varname.lower()
            if varname_low in geom_enrich.columns:
                result_dict[varname]={'val':geom_enrich.iloc[0][varname_low], 'desc':var_dict[varname], 'type':'default','weight':None}

        if result_dict != {}:
            data[areaid]['enrichment'][enrich_v] = result_dict

        with open(enrich_path, 'w') as file:           
            json.dump(data, file, indent=4)
        
    return result_dict

def get_area(loc):
    val_desc = {
        'val':_point2adm2area(loc),
        'desc': 'area of district (km^2) of the given location',
        'type': 'default',
        'weight': None
    }
    result_dict = {'area': val_desc}
    return result_dict

def get_night_light(
    loc=(float, float),
    tif_path = None,
    proxy_dir = None
) -> float:
    lat, lng = loc
    curr_interest = 'nightlight'
    
    if tif_path is None:
        tif_path = getattr(_CONFIG.res_global_geotiff, curr_interest)
    if proxy_dir is None:
        proxy_dir = _CONFIG.path.proxy_dir

    ccode, adm1_name, adm2_name, areaid, loc_geom = get_adm2meta(loc)
    proxy_dir = Path(proxy_dir)
    seg_path = proxy_dir / f'{ccode}_{curr_interest}.json'
    loc_data = get_ccode_loc(ccode)
    assert len(loc_data)>0
    loc_weight = loc_data[areaid]['weight']

    if seg_path.exists():
        with open(seg_path, 'r') as file:
            data = json.load(file)
           
        if areaid in data.keys() and curr_interest in data[areaid].keys():
            return data[areaid][curr_interest]
    else:
        data = {}

    data[areaid] = {} 

    geometry = [mapping(loc_geom)]
    src = rasterio.open(tif_path)
    out_image, out_transform = mask(src, geometry, crop=True, filled=False)
    nl_sum = float(out_image[0].sum())
    nl_sum_desc = 'Sum of nightlight intensity'
    nl_avg = float(out_image[0].filled(0).sum() / out_image[0].mask.sum())
    nl_avg_desc = 'Average nightlight intensity'
    result_dict = {'Nightlight_Sum':{'val':nl_sum,'desc':nl_sum_desc,'type':'default','weight':None}, 'Nightlight_Average':{'val':nl_avg,'desc':nl_avg_desc,'type':'ratio','weight':loc_weight}}

    data[areaid]["ADM0"] = ccode
    data[areaid]["ADM1"] = adm1_name
    data[areaid]["ADM2"] = adm2_name
    data[areaid][curr_interest] = result_dict

    with open(seg_path, 'w') as file:
        json.dump(data, file, indent=4)

    return result_dict

def get_co2_emission(
    loc=(float, float),
    tif_path = None,
    proxy_dir = None
) -> float:
    lat, lng = loc
    curr_interest = 'co2_emission'
    
    if tif_path is None:
        tif_path = getattr(_CONFIG.res_global_geotiff, curr_interest)
    if proxy_dir is None:
        proxy_dir = _CONFIG.path.proxy_dir

    ccode, adm1_name, adm2_name, areaid, loc_geom = get_adm2meta(loc)
    proxy_dir = Path(proxy_dir)
    seg_path = proxy_dir / f'{ccode}_{curr_interest}.json'
    loc_data = get_ccode_loc(ccode)
    assert len(loc_data)>0
    loc_weight = loc_data[areaid]['weight']

    if seg_path.exists():
        with open(seg_path, 'r') as file:
            data = json.load(file)
           
        if areaid in data.keys() and curr_interest in data[areaid].keys():
            return data[areaid][curr_interest]
    else:
        data = {}

    data[areaid] = {} 

    geometry = [mapping(loc_geom)]
    src = rasterio.open(tif_path)
    out_image, out_transform = mask(src, geometry, crop=True, filled=False)
    co2_sum = float(out_image[0].sum())
    co2_sum_desc = 'Sum of CO2 emission'
    co2_avg = float(out_image[0].filled(0).sum() / out_image[0].mask.sum())
    co2_avg_desc = 'Average CO2 emission'
    result_dict = {'CO2_Emission_Sum':{'val':co2_sum,'desc':co2_sum_desc,'type':'default','weight':None}, 'CO2_Emission_Average':{'val':co2_avg,'desc':co2_avg_desc, 'type':'ratio','weight':loc_weight}}

    data[areaid]["ADM0"] = ccode
    data[areaid]["ADM1"] = adm1_name
    data[areaid]["ADM2"] = adm2_name
    data[areaid][curr_interest] = result_dict

    with open(seg_path, 'w') as file:
        json.dump(data, file, indent=4)

    return result_dict

def get_distance_between_two_locations(
    loc1, 
    loc2
):
    """
    Get distance between the two given locations loc1, loc2
    
    Args:
        loc1: (latitude, longitude) of the given location 1
        loc2: (latitude, longitude) of the given location 2
    
    Return:
        The distance between the two given locations, following WGS84 system.
        Return -1 for error case.
    """
    distance = -1
    lat1, lng1 = loc1
    lat2, lng2 = loc2
    geod = Geod(ellps="WGS84")
    distance_meter = geod.line_length([lng1, lng2], [lat1, lat2])
    distance = distance_meter/1000
    return distance

def get_distance_to_nearest_target(
    loc=(float,float),
    target_name=str
):
    """
    Get distance between the given location and the nearest target,
    and made short description of the result.
    
    Args:
        target_name: name of the target
        loc: (latitude, longitude) of the given location
        
    Return:
        The dictionary, with a form of {<str>:{'val':<float>, 'desc':<str>}}
        key <str> would be a target_name, value <float> for 'val' would be a distance between the given location and the nearest target,
        and value <str> is a string description for this distance.
        The default desciprion would be 'distance to the nearest ' + target_name.
        Return None for error case.
    """
    result_dict = None

    lat, lng = loc
    target_shp_path = getattr(_CONFIG.res_global,target_name)
    col_name = _CONFIG.res_global.col_name
    
    df_target = gpd.read_file(target_shp_path, encoding='UTF-8')
    distance_list = df_target['geometry'].apply(lambda a: get_distance_between_two_locations((lat, lng), (a.y, a.x)))
    distance = min(distance_list)
    distance_idx = distance_list.idxmin()

    description = 'distance to the nearest ' + target_name

    varname = 'distance_'+target_name
    
    result_dict = {varname: {'val': distance, 'desc': description, 'type':'str', 'weight':None}}
    
    return result_dict

def point2adm2poi(
    loc=(float,float),
    zoom_level=None, 
    timeline=None,
    proxy_dir=None
):
    lat, lng = loc
    curr_interest = 'poi'

    if zoom_level is None:
        zoom_level = int(_CONFIG.arcgis.zoom_level)
    if timeline is None:
        timeline = int(_CONFIG.arcgis.timeline)
    if proxy_dir is None:
        proxy_dir = _CONFIG.path.proxy_dir

    ccode, adm1_name, adm2_name, areaid, loc_geom = get_adm2meta(loc)
    res_ccode = 'res_'+ccode

    proxy_dir = Path(proxy_dir)
    seg_path = proxy_dir / f'{ccode}_{curr_interest}.json'

    if seg_path.exists():
        with open(seg_path, 'r') as file:
            data = json.load(file)
           
        if areaid in data.keys() and curr_interest in data[areaid].keys():
            return data[areaid][curr_interest]
    else:
        data = {}

    data[areaid] = {}

    shp_path = getattr(getattr(_CONFIG, res_ccode), curr_interest)
    shp_df = gpd.read_file(shp_path, encoding='UTF-8')

    poi_df = gpd.GeoDataFrame(shp_df[['osm_id', 'name']],geometry=shp_df['geometry'])

    # amenity IS NOT NULL OR man_made IS NOT NULL OR shop IS NOT NULL OR tourism IS NOT NULL
    def osm_category(col1, col2, col3, col4):
        res = ""
        if not pd.isna(col1):
            res = str(col1)
        elif not pd.isna(col2):
            res = str(col2)
        elif not pd.isna(col3):
            res = str(col3)
        else:
            res = str(col4)
        return res
    
    poi_df['category'] = shp_df.apply(lambda a: osm_category(a['amenity'], a['man_made'], a['shop'], a['tourism']), axis=1)
    count_list = poi_df['category'].value_counts()

    # current threshold = 1%
    threshold = len(poi_df) * 0.01
    
    # delete categories <= threshold
    for name, cnt in zip(count_list.index, count_list):
        if cnt < threshold:
            poi_df = poi_df[poi_df['category'] != name]

    area_name = adm2_name+", "+adm1_name+", "+ccode if adm2_name is not None else adm1_name+", "+ccode
    '''
    # In case of defining boundary with y_x blocks
    poi_df.geometry = poi_df.geometry.map(lambda geom: transform(lambda x,y : (y,x), geom))

    img_path_list = get_adm2imgpathlist(loc, zoom_level, timeline)
    y_x_list = _get_y_x(img_path_list)
    polygon_list = []

    for y,x in y_x_list:
        y0_x0_loc = num2deg(x,y,zoom_level)
        y0_x1_loc = num2deg(x+1,y,zoom_level)
        y1_x1_loc = num2deg(x+1,y+1,zoom_level)
        y1_x0_loc = num2deg(x,y+1,zoom_level)
        y_x_polygon = Polygon([y0_x0_loc,y0_x1_loc,y1_x1_loc,y1_x0_loc,y0_x0_loc])
        polygon_list.append(y_x_polygon)
    
    y_x_multipolygon = MultiPolygon(polygon_list)
    loc_poi_df = poi_df[y_x_multipolygon.contains(poi_df['geometry'])]
    '''
    loc_poi_df = poi_df[loc_geom.contains(poi_df['geometry'])]

    cnts = loc_poi_df['category'].value_counts()

    result_dict = {}
    
    for cate, cnt in zip(cnts.index, cnts):
        temp_dict = {}
        temp_dict['val'] = cnt
        temp_dict['desc'] = 'number of '+cate# get category name
        temp_dict['type'] = 'default'
        temp_dict['weight'] = None
        result_dict[cate] = temp_dict

    data[areaid]["ADM0"] = ccode
    data[areaid]["ADM1"] = adm1_name
    data[areaid]["ADM2"] = adm2_name
    data[areaid][curr_interest] = result_dict

    with open(seg_path, 'w') as file:
        json.dump(data, file, indent=4)

    return result_dict
def get_poi_number(
    loc=(float,float),
    target_class=str,
    zoom_level=None, 
    timeline=None,
    proxy_dir=None
):
    lat, lng = loc
    curr_interest = 'poi_health'

    if zoom_level is None:
        zoom_level = int(_CONFIG.arcgis.zoom_level)
    if timeline is None:
        timeline = int(_CONFIG.arcgis.timeline)
    if proxy_dir is None:
        proxy_dir = _CONFIG.path.proxy_dir

    ccode, adm1_name, adm2_name, areaid, loc_geom = get_adm2meta(loc)
    res_ccode = 'res_'+ccode

    proxy_dir = Path(proxy_dir)
    seg_path = proxy_dir / f'{ccode}_{curr_interest}.json'

    if seg_path.exists():
        with open(seg_path, 'r') as file:
            data = json.load(file)
           
        if areaid in data.keys() and curr_interest in data[areaid].keys():
            target_area_info = data[areaid][curr_interest][target_class]
            result_dict = {'number_'+target_class: target_area_info}
            return result_dict
    else:
        data = {}

    data[areaid] = {}

    shp_path = getattr(getattr(_CONFIG, res_ccode), curr_interest)
    shp_df = gpd.read_file(shp_path)

    poi_df = gpd.GeoDataFrame(shp_df[['osm_id', 'name']],geometry=shp_df['geometry'])

    # amenity IS NOT NULL OR man_made IS NOT NULL OR shop IS NOT NULL OR tourism IS NOT NULL
    def osm_category(col1, col2):
        res = ""
        if not pd.isna(col1):
            res = str(col1)
        else:
            res = str(col2)
        return res
    
    poi_df['category'] = shp_df.apply(lambda a: osm_category(a['amenity'], a['healthcare']), axis=1)
#     count_list = poi_df['category'].value_counts()
    
    area_name = adm2_name+", "+adm1_name+", "+ccode if adm2_name is not None else adm1_name+", "+ccode
    '''
    # In case of defining boundary with y_x blocks
    poi_df.geometry = poi_df.geometry.map(lambda geom: transform(lambda x,y : (y,x), geom))

    img_path_list = get_adm2imgpathlist(loc, zoom_level, timeline)
    y_x_list = _get_y_x(img_path_list)
    polygon_list = []

    for y,x in y_x_list:
        y0_x0_loc = num2deg(x,y,zoom_level)
        y0_x1_loc = num2deg(x+1,y,zoom_level)
        y1_x1_loc = num2deg(x+1,y+1,zoom_level)
        y1_x0_loc = num2deg(x,y+1,zoom_level)
        y_x_polygon = Polygon([y0_x0_loc,y0_x1_loc,y1_x1_loc,y1_x0_loc,y0_x0_loc])
        polygon_list.append(y_x_polygon)
    
    y_x_multipolygon = MultiPolygon(polygon_list)
    loc_poi_df = poi_df[y_x_multipolygon.contains(poi_df['geometry'])]
    '''
    loc_poi_df = poi_df[loc_geom.contains(poi_df['geometry'])]

    cnts = loc_poi_df['category'].value_counts()

    result_dict = {}
    
#     category = ['hospital', 'clinic', 'doctors']
    category = ['hospital', 'doctors', 'clinic']
    for cate in category:
        if cate not in cnts.index:
            temp_dict = {}
            temp_dict['val'] = 0
            temp_dict['desc'] = 'number of '+cate# get category name
            temp_dict['type'] = 'default'
            temp_dict['weight'] = None
            result_dict[cate] = temp_dict
        else:
            temp_dict = {}
            temp_dict['val'] = int(cnts.loc[cate])
            temp_dict['desc'] = 'number of '+cate# get category name
            temp_dict['type'] = 'default'
            temp_dict['weight'] = None
            result_dict[cate] = temp_dict
    
    data[areaid]["ADM0"] = ccode
    data[areaid]["ADM1"] = adm1_name
    data[areaid]["ADM2"] = adm2_name
    data[areaid][curr_interest] = result_dict

    with open(seg_path, 'w') as file:
        json.dump(data, file, indent=4)

    return {'number_'+target_class: result_dict[target_class]}
'''
def neighbor_initialize(adm2_name, desc, type_str, ratio):
    desc = val_desc['desc']
    if isinstance(val_desc['val'], str):
        result_dict[var_name] = {'val': f'[{count_idx}] ' + val_desc['val'], 'desc': f'{desc} of the neighboring region(s)' }
    else:
        result_dict[var_name] = {'val': val_desc['val'], 'desc': f'{desc} in the neighboring region(s)' }
'''

def get_aggregate_neighbor_info(
    loc=(float, float),
    func=object,
    proxy_dir=None
):
    """
    Aggregate the function result of neighbors in the given radius of the location.
    
    Args:
        func: The function to get the result.
        loc: The location of the interest
        radius: The radius of the neighborhood.
    
    Return:
        Dictionary that contains the information of neighbors.
        ex)
        {
            (loc_0_x, loc_0_y): 0.0,
            (loc_1_x, loc_1_y): 1.0,
            ...
        }
        Return None for error case.
    """
    lat, lng = loc
    loc_point = get_loc_geometry(loc)

    if proxy_dir is None:
        proxy_dir = _CONFIG.path.proxy_dir

    ccode, adm1_name, adm2_name, areaid, loc_geom = get_adm2meta(loc) 

    proxy_dir = Path(proxy_dir)
    
    neighbor_path = proxy_dir / f'{ccode}_loc.pickle'
    with open(neighbor_path, 'rb') as f:
        loc_data = pickle.load(f)
    
    n_areaid_list = loc_data[areaid]['neighbor']
    repr_loc = loc_data[areaid]['repr_loc']
    func_result_dict = func(repr_loc)
    var_name_list = list(func_result_dict.keys())
    
    result_dict = {}
    
    for var_name in var_name_list:
        result_dict[var_name]={}
    
    neighbor_adm1_list, neighbor_adm2_list, neighbor_areaid_list = [], [], []
    
    data = {}
    n_neighbors = len(n_areaid_list)
    if n_neighbors==0:
        for var_name in var_name_list:
            desc = func_result_dict[var_name]['desc']
            val_type = func_result_dict[var_name]['type']
            val_weight = func_result_dict[var_name]['weight']
            if "address" not in var_name:
                result_dict[var_name] = {'val': 0, 'desc': desc, 'type':val_type, 'weight':None}
            else:
                result_dict[var_name] = {'val': f'None. {adm2_name} is isolated from neighboring administrative regions.', 'desc': f'{desc} of the neighboring region(s)','type':val_type, 'weight':val_weight}
        
        assert len(result_dict)>0
        data[areaid] = result_dict
        
        return result_dict
        
    #count_idx = 1
    for idx in range(len(n_areaid_list)):
        n_repr_loc = loc_data[n_areaid_list[idx]]['repr_loc']
        count_idx = idx +1
        if idx==0:
            result_dict_cache = get_func_repr_dict()
            temp_func_str_cache = get_temp_func_str()
            #print('property, get_aggregate_neighbor_info',temp_func_str_cache)
            if (temp_func_str_cache, n_repr_loc) in result_dict_cache.keys():
                print('cached!', (temp_func_str_cache, n_repr_loc))
                temp_n_areaid_dict = result_dict_cache[(temp_func_str_cache, n_repr_loc)]
            else:
                temp_n_areaid_dict = func(n_repr_loc)
                print(temp_n_areaid_dict)
                set_func_repr_dict((temp_func_str_cache, n_repr_loc), temp_n_areaid_dict)
            for var_name, val_desc in temp_n_areaid_dict.items():
                desc = val_desc['desc']
                print(val_desc)
                val_type= val_desc['type']
                val_weight= val_desc['weight']
                if isinstance(val_desc['val'], str):
                    result_dict[var_name] = {'val': f'[{count_idx}] ' + val_desc['val'], 'desc': f'{desc} of the neighboring region(s)',
                                             'type':val_type, 'weight':val_weight}
                else:
                    result_dict[var_name] = {'val': val_desc['val'], 'desc': f'{desc} in the neighboring region(s)',
                                            'type':val_type, 'weight':val_weight }
        else:
            result_dict_cache = get_func_repr_dict()
            temp_func_str_cache = get_temp_func_str()
            #print('property, get_aggregate_neighbor_info',temp_func_str_cache)
            if (temp_func_str_cache, n_repr_loc) in result_dict_cache.keys():
                print('cached!', (temp_func_str_cache, n_repr_loc))
                temp_n_areaid_dict = result_dict_cache[(temp_func_str_cache, n_repr_loc)]
            else:
                temp_n_areaid_dict = func(n_repr_loc)
                print(temp_n_areaid_dict)
                set_func_repr_dict((temp_func_str_cache, n_repr_loc), temp_n_areaid_dict)
            for var_name, val_desc in temp_n_areaid_dict.items():
                if result_dict.get(var_name) is not None:
                    if isinstance(val_desc['val'], str):
                        result_dict[var_name]['val'] += (f'; [{count_idx}] ' + val_desc['val'])
                    elif val_desc['type']=='ratio':
                        left_weight = result_dict[var_name]['weight']
                        right_weight = val_desc['weight']
                        assert left_weight is not None and right_weight is not None
                        left_ratio = left_weight/(left_weight+right_weight)
                        right_ratio = right_weight/(left_weight+right_weight)
                        new_weight = left_weight+right_weight
                        result_dict[var_name]['val'] = left_ratio * result_dict[var_name]['val'] + right_ratio * val_desc['val']
                        result_dict[var_name]['weight'] = new_weight
                    else:
                        # default
                        result_dict[var_name]['val'] += val_desc['val']
                else:
                    desc = val_desc['desc']
                    val_type= val_desc['type']
                    val_weight= val_desc['weight']
                    if isinstance(val_desc['val'], str):
                        result_dict[var_name] = {'val': f'[{count_idx}] ' + val_desc['val'], 'desc': f'{desc} of the neighboring region(s).',
                                                'type':val_type,'weight':val_weight}
                    else:
                        result_dict[var_name] = {'val': val_desc['val'], 'desc': f'{desc} in the neighboring region(s)',
                                                'type':val_type,'weight':val_weight}
        #count_idx += 1
    
    data[areaid] = result_dict
   
    return result_dict

def aggregate_repr_loc_list(repr_loc_list, func):   
    assert len(repr_loc_list)>0
    #oollect all scores withtin neighbor

    func_result_dict = func(repr_loc_list[0])
    var_name_list = list(func_result_dict.keys())
    result_dict = {}
    
    for var_name in var_name_list:
        result_dict[var_name]={}
    
    for idx in range(len(repr_loc_list)):
        n_repr_loc = repr_loc_list[idx]
        count_idx = idx+1
        if idx==0:
            result_dict_cache = get_func_repr_dict()
            temp_func_str_cache = get_temp_func_str()
            if (temp_func_str_cache, n_repr_loc) in result_dict_cache.keys():
                print('cached!', (temp_func_str_cache, n_repr_loc))
                temp_n_areaid_dict = result_dict_cache[(temp_func_str_cache, n_repr_loc)]
            else:
                temp_n_areaid_dict = func(n_repr_loc)
                #print(temp_n_areaid_dict)
                set_func_repr_dict((temp_func_str_cache, n_repr_loc), temp_n_areaid_dict)
            for var_name, val_desc in temp_n_areaid_dict.items():
                desc = val_desc['desc']
                if isinstance(val_desc['val'], str):
                    result_dict[var_name] = {'val': f'[{count_idx}] ' + val_desc['val'], 'desc': desc }
                else:
                    result_dict[var_name] = {'val': val_desc['val'], 'desc': desc }
        else:
            result_dict_cache = get_func_repr_dict()
            temp_func_str_cache = get_temp_func_str()
            if (temp_func_str_cache, n_repr_loc) in result_dict_cache.keys():
                print('cached!', (temp_func_str_cache, n_repr_loc))
                temp_n_areaid_dict = result_dict_cache[(temp_func_str_cache, n_repr_loc)]
            else:
                temp_n_areaid_dict = func(n_repr_loc)
                #print(temp_n_areaid_dict)
                set_func_repr_dict((temp_func_str_cache, n_repr_loc), temp_n_areaid_dict)
            for var_name, val_desc in temp_n_areaid_dict.items():
                if result_dict.get(var_name) is not None:
                    if isinstance(val_desc['val'], str):
                        result_dict[var_name]['val'] += (f'; [{count_idx}] ' + val_desc['val'])
                    else:
                        result_dict[var_name]['val'] += val_desc['val']
                else:
                    desc = val_desc['desc']
                    if isinstance(val_desc['val'], str):
                        result_dict[var_name] = {'val': f'[{count_idx}] ' + val_desc['val'], 'desc': desc }
                    else:
                        result_dict[var_name] = {'val': val_desc['val'], 'desc': desc }
    
    return result_dict

def get_height(
    loc=(float, float),
) -> float:
    """
    Get height of the given location above sea level.
    
    Args:
        loc: The location of the interest
    
    Return:
        The height of the given location above sea level
        Return -1 for error case.
    """
    height = -1
    
    return height

def get_population(
    loc = (float, float)
) -> float:
    """
    Get population of given location's adm2.
    
    Args:
        loc: The location of the interest
    
    Return:
        The population of given location's adm2
        Return -1 for error case.
    """  
    # if proxy_dir is None:
    proxy_dir = _CONFIG.path.proxy_dir
    proxy_dir = Path(proxy_dir)
    ccode, adm1_name, adm2_name, areaid, loc_geom = get_adm2meta(loc)
    # country = get_country(ccode)
    pop_df_path = proxy_dir / f'{ccode}_population.csv'
    pop_df = pd.read_csv(pop_df_path)
    # print(type(areaid))
    # print(type(pop_df['area_id'].values[0]))
    # print(int(areaid) in pop_df['area_id'].values)
    pop_df['area_id'] = pop_df['area_id'].astype(str)
    if areaid in pop_df['area_id'].values:
        description = "Population of the area"
        result_dict ={"Population": {'val': pop_df.loc[pop_df['area_id'] == areaid, 'population'].values[0], 'desc': description, 'type':'int', 'weight':None}}
        # print(result_dict)
        return result_dict
    # print('hi')
    return -1
