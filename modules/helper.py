import numpy as np
import math
import glob
from arcgis.geoenrichment import *
from arcgis.geometry import Geometry, LengthUnits, AreaUnits, areas_and_lengths
from arcgis.gis import GIS
from pathlib import Path
import requests
import utils
from collections import defaultdict
import pickle
from pyproj import Geod

from tqdm import tqdm
from shapely.geometry import Point
from shapely.geometry import Polygon, MultiPolygon
from shapely.validation import make_valid
from shapely.ops import triangulate
import shapely
import geopandas as gpd
import os, sys

_TEMP_FUNC_STR = ''
_FUNC_REPR_DICT = {}
_CCODE_LOC = {}
_CURR_CCODE = ''
    
def get_temp_func_str():
    return _TEMP_FUNC_STR
    
def set_temp_func_str(temp_func_str):
    _TEMP_FUNC_STR = temp_func_str
    
def get_func_repr_dict():
    return _FUNC_REPR_DICT

def set_func_repr_dict(key, value, overwrite=False):
    if key not in _FUNC_REPR_DICT.keys() or overwrite:
        _FUNC_REPR_DICT[key] = value

def clear_func_repr_dict():
    _FUNC_REPR_DICT = {}

def get_ccode_loc(ccode):
    return _CCODE_LOC
    
def set_ccode_loc(ccode):
    config = utils.read_config()
    proxy_dir = Path(config.path.proxy_dir)
    loc_path = proxy_dir / f'{ccode}_loc.pickle'
    with open(loc_path, 'rb') as f:
        _CCODE_LOC = pickle.load(f)

def get_curr_ccode():
    return _CURR_CCODE

def set_curr_ccode(ccode):
    _CURR_CCODE = ccode

def get_loc_geometry(loc):
    lat, lng = loc
    return Geometry({"x":lng,"y":lat, "spatialReference" : {"wkid" : 4326}})

def get_ring_contained_loc(geom):
    spatial_ref = {"wkid":4326}
    
    result_loc = None
    max_area = 0
    max_geom = None
    # print(geom)
    # print(type(geom))
    # print(result_loc)
        
    if isinstance(geom, MultiPolygon) or isinstance(geom, Polygon):
        # print(geom.centroid)
        lon, lat = geom.centroid.x, geom.centroid.y
        if geom.area > max_area:
            max_area = geom.area
            max_geom = geom
        if get_loc_geometry((lat,lon)).within(geom):
            # print("hi")
            result_loc = (lat,lon)
            # print(result_loc)

    else:
        for i in range(len(geom.rings)):
            curr_geom = Geometry({'rings':[geom.rings[i]], "spatialReference": {"wkid": 4326}})
            lon, lat = curr_geom.centroid
            if curr_geom.area > max_area:
                max_area = curr_geom.area
                max_geom = curr_geom
            if get_loc_geometry((lat,lon)).within(geom):
                result_loc = (lat,lon)
                break
    # print()
    # print(result_loc)
    if result_loc is None:
        #Rare case - triangulation
        print('Corner case - Delaunay, near ', geom.centroid)
        max_valid_geom = make_valid(max_geom.as_shapely) if 'as_shapely' in dir(max_geom) else make_valid(max_geom)
        for triangle in triangulate(max_valid_geom):
            triangle_centroid = get_loc_geometry((triangle.centroid.y,triangle.centroid.x))
            if triangle_centroid.within(geom):
                lon, lat = triangle.centroid.x, triangle.centroid.y
                result_loc = (lat,lon)
                print('Corner case - Delaunay solved')
                print(result_loc)
                break
    
    assert result_loc != None
    
    return result_loc


def get_country(ccode):
    return Country.get(ccode)

def _get_y_x(png_path_list):
    result_list = []
    for png in png_path_list:
        y_x = png.split('/')[-1].split('.')[0]
        y = int(y_x.split('_')[0])
        x = int(y_x.split('_')[1])
        result_list.append((y,x)) 
    return result_list

def _point2areaid(loc):
    
    config = utils.read_config()
    proxy_dir = Path(config.path.proxy_dir)
    
    if _CURR_CCODE == '':
        geom_pkl_list = glob.glob(str(proxy_dir / '*_geom.pickle'))
    else:
        curr_ccode = get_curr_ccode()
        geom_pkl_list = [str(proxy_dir / f'{curr_ccode}_geom.pickle')]
    assert len(geom_pkl_list) > 0
    ccode, result_areaid = "", ""
    
    for geom_pkl in geom_pkl_list:
        ccode = geom_pkl.split('/')[-1][:-len('_geom.pikcle')]
        with open(file=geom_pkl, mode='rb') as f:
            data = pickle.load(f)
        for key, value in data.items():
            if get_loc_geometry(loc).within(value):
                result_areaid = key
                set_curr_ccode(ccode)
                return ccode, result_areaid
    
    return ccode, result_areaid

def _point2address(loc):
    """
    Get address string of the given location
    
    Args:
        loc: The location of the interest
    
    Return:
        The address string of the given location
        Return None for error case.
    """
    
    lat, lng = loc   
    
    rev_geocode_url = 'https://geocode.arcgis.com/arcgis/rest/services/World/GeocodeServer/reverseGeocode?'
    rev_parameter = 'f=pjson&featureTypes=&langCode=en&location={}%2C{}'.format(str(lng),str(lat))
    revgeo_request = rev_geocode_url+rev_parameter
    
    getAddress = requests.get(revgeo_request)
    if getAddress.status_code != 200:
        return None

    revgeo_response =  getAddress.json()

    result_str = revgeo_response['address']['LongLabel']
    print(result_str)
    # return result_str
    if len(result_str.split(', '))<3:
        config = utils.read_config()
        proxy_dir = Path(config.path.proxy_dir)
   
        ccode, areaid = _point2areaid(loc)
        loc_path = proxy_dir / f'{ccode}_loc.pickle'
        
        with open(loc_path,'rb') as f:
            loc_data = pickle.load(f) 
        
        #print(loc_data)
        if 'ADM2' in loc_data[areaid].keys():
            result_str = ', '.join([loc_data[areaid]['ADM2'],loc_data[areaid]['ADM1'],loc_data[areaid]['ADM0']])
        else:
            result_str = ', '.join([loc_data[areaid]['ADM1'],loc_data[areaid]['ADM0']])
    
    return result_str
    
def _point2adm2(loc):
    """
    Get adm2 information of the given location
    
    Args:
        loc: The location of the interest
    
    Return:
        The adm2 information of the given location
        Return None for error case.
    """
    config = utils.read_config()
    proxy_dir = Path(config.path.proxy_dir)
    
    revgeo_singleline = _point2address(loc)
    if revgeo_singleline == None:
        return None
    
    address_component = revgeo_singleline.split(', ')
    address_component.reverse()
    result_list = address_component
    
    ccode, areaid = _point2areaid(loc)
    
    loc_path = proxy_dir / f'{ccode}_loc.pickle'
    with open(file=loc_path, mode='rb') as f:
        loc_data = pickle.load(f)

    if 'ADM2' in loc_data[areaid]:
        if len(result_list)<3:
            #erroneous case - ex) lang_en not supported, postal code, or name of sea
            result_list = ['ccode','adm1','adm2']
        result_list[0] = ccode
        result_list[1] = loc_data[areaid]['ADM1']
        result_list[2] = loc_data[areaid]['ADM2']
    else:
        if len(result_list)<2:
            #erroneous case - ex) lang_en not supported, postal code, or name of sea
            result_list = ['ccode','adm1','adm2']
    
        result_list[0] = ccode
        result_list[1] = loc_data[areaid]['ADM1']
    
    return result_list

def point2adm2meta(loc):
    config = utils.read_config()
    proxy_dir = Path(config.path.proxy_dir)
    
    ccode, areaid = _point2areaid(loc)
    
    loc_path = proxy_dir / f'{ccode}_loc.pickle'
    with open(loc_path,'rb') as f:
        loc_data = pickle.load(f)
    
    adm1_name = loc_data[areaid]['ADM1']
    adm2_name = None
    if 'ADM2' in loc_data[areaid].keys():
        adm2_name = loc_data[areaid]['ADM2']
    
    geom_path = proxy_dir / f'{ccode}_geom.pickle'
    with open(geom_path,'rb') as f:
        geom_data = pickle.load(f)
    loc_geom = geom_data[areaid]
    return ccode, adm1_name, adm2_name, areaid, loc_geom


def _point2adm2area(loc):
    result_area = -1
    ccode, adm1_name, adm2_name, areaid, loc_geom = point2adm2meta(loc)

    if not isinstance(loc_geom, Polygon) and not isinstance(loc_geom, MultiPolygon):
        polygon = make_valid(loc_geom.as_shapely)
    else:
        polygon = loc_geom
    geod = Geod(ellps="WGS84")
    poly_area, poly_perimeter = geod.geometry_area_perimeter(polygon)
    poly_area = abs(poly_area*0.000001)
    result_area = poly_area
    return result_area


def get_repr_locs(ccode):
    country = get_country(ccode)
    adm_level = list(country.levels['level_name'])
    adm_level.reverse()
    adm1_dict = getattr(country.subgeographies, adm_level[1])

    con_loc_dict = {}
    geom_dict = {}
    parent_geom_dict = {}
    parent_id_dict = defaultdict(list)
    
    #with open(file=str(contained_loc_path), mode='wb') as f:
    #    pickle.dump(con_loc_dict, f)
    
    if len(adm_level)==2:
        for adm1_key, adm1_val in tqdm(adm1_dict.items()):
            areaid = adm1_val._areaid
            adm1_name = adm1_key
            
            con_loc_dict[areaid] = {}
            
            con_loc_dict[areaid]['ADM0']=ccode
            con_loc_dict[areaid]['ADM1']=adm1_name
            
            geom = adm1_dict[adm1_key].geometry
            repr_loc = get_ring_contained_loc(geom)
            con_loc_dict[areaid]['repr_loc']=repr_loc
            
            polygon = make_valid(geom.as_shapely)
            geod = Geod(ellps="WGS84")
            poly_area, poly_perimeter = geod.geometry_area_perimeter(polygon)
            poly_area = abs(poly_area*0.000001)
            
            con_loc_dict[areaid]['weight'] = poly_area
            geom_dict[areaid] = geom
            
            for areaid_src, src_val in tqdm(con_loc_dict.items()):
                con_loc_dict[areaid_src]['neighbor']=[]
                for areaid_tar, tar_val in con_loc_dict.items():
                    if (areaid_src != areaid_tar) and (not geom_dict[areaid_src].disjoint(geom_dict[areaid_tar])):
                        con_loc_dict[areaid_src]['neighbor'].append(areaid_tar)

    elif len(adm_level)>=3:
        for adm1_key, adm1_val in tqdm(adm1_dict.items()):
            adm1_id = adm1_val._areaid
            parent_geom = adm1_val.geometry
            parent_geom_dict[adm1_id] = parent_geom
            
            adm2_dict = getattr(adm1_val, adm_level[2])
            for adm2_key, adm2_val in adm2_dict.items():
                adm1_name = adm1_key
                adm2_name = adm2_key
                areaid = adm2_val._areaid
                parent_id_dict[adm1_id].append(areaid)
                
                con_loc_dict[areaid] = {}
                
                con_loc_dict[areaid]['ADM0']=ccode
                con_loc_dict[areaid]['ADM1']=adm1_name
                con_loc_dict[areaid]['ADM2']=adm2_name
                con_loc_dict[areaid]['adm1_id'] = adm1_id
                
                geom = adm2_dict[adm2_key].geometry
                repr_loc =get_ring_contained_loc(geom) 
                con_loc_dict[areaid]['repr_loc']=repr_loc
                
                polygon = make_valid(geom.as_shapely)
                geod = Geod(ellps="WGS84")
                poly_area, poly_perimeter = geod.geometry_area_perimeter(polygon)
                poly_area = abs(poly_area*0.000001)
                
                con_loc_dict[areaid]['weight'] = poly_area
                geom_dict[areaid] = geom
        
        parent_neighbor_dict = defaultdict(list)
        for src_parent, src_geom in tqdm(parent_geom_dict.items()):
            for tar_parent, tar_geom in parent_geom_dict.items():
                if (src_parent != tar_parent) and (not src_geom.disjoint(tar_geom)):
                    parent_neighbor_dict[src_parent].append(tar_parent)
    
        for areaid_src, src_val in tqdm(con_loc_dict.items()):
            con_loc_dict[areaid_src]['neighbor']=[]
            parent_id_src = src_val['adm1_id']
            con_loc_dict[areaid_src]['adm1_neighbor'] = parent_neighbor_dict[parent_id_src]
            con_loc_dict[areaid_src]['adm2_in_adm1_neighbor'] = []
            for nei in con_loc_dict[areaid_src]['adm1_neighbor']:
                con_loc_dict[areaid_src]['adm2_in_adm1_neighbor'].extend(parent_id_dict[nei])
            
            for areaid_tar, tar_val in con_loc_dict.items():
                if (areaid_src != areaid_tar) and (not geom_dict[areaid_src].disjoint(geom_dict[areaid_tar])):
                    con_loc_dict[areaid_src]['neighbor'].append(areaid_tar)
    else:
        print('No subgeography in ArcGIS database.')
    
    return con_loc_dict, geom_dict

def get_repr_locs_SVK(ccode):
    country = get_country(ccode)
    adm_level = list(country.levels['level_name'])
    adm_level.reverse()
    adm1_dict = getattr(country.subgeographies, adm_level[1])

    con_loc_dict = {}
    geom_dict = {}
    parent_geom_dict = {}
    parent_id_dict = defaultdict(list)
    
    for adm1_key, adm1_val in tqdm(adm1_dict.items()):
        adm1_id = adm1_val._areaid
        parent_geom = adm1_val.geometry
        parent_geom_dict[adm1_id] = parent_geom
        
        adm2_dict = getattr(adm1_val, adm_level[2])
        for adm2_key, adm2_val in adm2_dict.items():
            adm1_name = adm1_key
            adm2_name = adm2_key
            areaid = adm2_val._areaid
            parent_id_dict[adm1_id].append(areaid)
            
            con_loc_dict[areaid] = {}
            
            con_loc_dict[areaid]['ADM0']=ccode
            con_loc_dict[areaid]['ADM1']=adm1_name
            con_loc_dict[areaid]['ADM2']=adm2_name
            con_loc_dict[areaid]['adm1_id'] = adm1_id
            
            geom = adm2_dict[adm2_key].geometry
            repr_loc =get_ring_contained_loc(geom) 
            con_loc_dict[areaid]['repr_loc']=repr_loc
            
            polygon = make_valid(geom.as_shapely)
            geod = Geod(ellps="WGS84")
            poly_area, poly_perimeter = geod.geometry_area_perimeter(polygon)
            poly_area = abs(poly_area*0.000001)
            
            con_loc_dict[areaid]['weight'] = poly_area
            geom_dict[areaid] = geom
    
    parent_neighbor_dict = defaultdict(list)
    for src_parent, src_geom in tqdm(parent_geom_dict.items()):
        for tar_parent, tar_geom in parent_geom_dict.items():
            if (src_parent != tar_parent) and (not src_geom.disjoint(tar_geom)):
                parent_neighbor_dict[src_parent].append(tar_parent)

    for areaid_src, src_val in tqdm(con_loc_dict.items()):
        con_loc_dict[areaid_src]['neighbor']=[]
        parent_id_src = src_val['adm1_id']
        con_loc_dict[areaid_src]['adm1_neighbor'] = parent_neighbor_dict[parent_id_src]
        con_loc_dict[areaid_src]['adm2_in_adm1_neighbor'] = []
        for nei in con_loc_dict[areaid_src]['adm1_neighbor']:
            con_loc_dict[areaid_src]['adm2_in_adm1_neighbor'].extend(parent_id_dict[nei])
        
        for areaid_tar, tar_val in con_loc_dict.items():
            if (areaid_src != areaid_tar) and (not geom_dict[areaid_src].disjoint(geom_dict[areaid_tar])):
                con_loc_dict[areaid_src]['neighbor'].append(areaid_tar)

    
    return con_loc_dict, geom_dict

def get_repr_locs_gpkg(ccode, gpkg_addr):
    gpkg_df = gpd.read_file(gpkg_addr)
    adm1_names = set(gpkg_df['NAME_1'])
    adm1_geometries = gpkg_df.dissolve(by='NAME_1')
    adm2_dict = gpkg_df.groupby('NAME_1')['NAME_2'].apply(list).to_dict()
    
    con_loc_dict = {}
    geom_dict = {}
    parent_geom_dict = {}
    parent_id_dict = defaultdict(list)
    
    #with open(file=str(contained_loc_path), mode='wb') as f:
    #    pickle.dump(con_loc_dict, f)
    

    # if len(adm_level)>=3:
    if adm2_dict:
        # adm1_id=1
        for adm1_name in tqdm(adm1_names):
            # adm1_id = gpkg_df[gpkg_df.loc['NAME_1'] == adm1_name, "GID_1"].iloc[0]
            # parent_geom = adm1_val.geometry
            # if not adm1_name in adm1_geometries.index:
            #     continue
            parent_geom = adm1_geometries.loc[adm1_name, "geometry"]
            parent_geom_dict[adm1_name] = parent_geom
            
            for adm2_name in adm2_dict[adm1_name]:
                
                GID_2 = gpkg_df.loc[gpkg_df['NAME_2'] == adm2_name, "GID_2"].iloc[0].split('.')[1:]
                adm1_id = ""
                adm2_id = ""
                if len(GID_2[0]) != 2:
                    adm1_id = '0' + GID_2[0]
                else:
                    adm1_id = GID_2[0]
                # print(GID_2)
                if len(GID_2[1].split('_')[0]) !=2:
                    adm2_id = '00' + GID_2[1].split('_')[0]
                else:
                    adm2_id = '0' + GID_2[1].split('_')[0]
                # print(adm1_id)
                # print(adm2_id)

                areaid = adm1_id + adm2_id
                # print(areaid)
                parent_id_dict[adm1_id].append(areaid)
                
                con_loc_dict[areaid] = {}
                
                con_loc_dict[areaid]['ADM0']=ccode
                con_loc_dict[areaid]['ADM1']=adm1_name
                con_loc_dict[areaid]['ADM2']=adm2_name
                con_loc_dict[areaid]['adm1_id'] = adm1_id
                
                geom = gpkg_df.loc[gpkg_df['NAME_2'] == adm2_name, 'geometry'].iloc[0]
                # print(geom)
                repr_loc =get_ring_contained_loc(geom) 
                con_loc_dict[areaid]['repr_loc']=repr_loc
                if not isinstance(geom, Polygon) and not isinstance(geom, MultiPolygon):
                    polygon = make_valid(geom.as_shapely)
                else:
                    polygon = geom
                geod = Geod(ellps="WGS84")
                poly_area, poly_perimeter = geod.geometry_area_perimeter(polygon)
                poly_area = abs(poly_area*0.000001)
                
                con_loc_dict[areaid]['weight'] = poly_area
                geom_dict[areaid] = geom
        
        parent_neighbor_dict = defaultdict(list)
        for src_parent, src_geom in tqdm(parent_geom_dict.items()):
            for tar_parent, tar_geom in parent_geom_dict.items():
                if (src_parent != tar_parent) and (not src_geom.disjoint(tar_geom)):
                    parent_neighbor_dict[src_parent].append(tar_parent)
    
        for areaid_src, src_val in tqdm(con_loc_dict.items()):
            con_loc_dict[areaid_src]['neighbor']=[]
            parent_id_src = src_val['adm1_id']
            con_loc_dict[areaid_src]['adm1_neighbor'] = parent_neighbor_dict[parent_id_src]
            con_loc_dict[areaid_src]['adm2_in_adm1_neighbor'] = []
            for nei in con_loc_dict[areaid_src]['adm1_neighbor']:
                con_loc_dict[areaid_src]['adm2_in_adm1_neighbor'].extend(parent_id_dict[nei])
            
            for areaid_tar, tar_val in con_loc_dict.items():
                if (areaid_src != areaid_tar) and (not geom_dict[areaid_src].disjoint(geom_dict[areaid_tar])):
                    con_loc_dict[areaid_src]['neighbor'].append(areaid_tar)
    else:
        print('No subgeography in ArcGIS database.')
    
    return con_loc_dict, geom_dict

def get_repr_locs_adm1(con_loc_dict):
    #from {ccode}_loc.pickle --> {ccode}_loc_adm1.pickle
    adm1_areaid_list = []
    loc_adm1_dict = {}
    
    country = get_country(_CURR_CCODE)
    adm_level = list(country.levels['level_name'])
    if len(adm_level)==2:
        return None
    
    for adm2_areaid in con_loc_dict.keys():
        #get all info form ccode_loc (adm2)
        temp_adm1_id = con_loc_dict[adm2_areaid]['adm1_id']
        if loc_adm1_dict.get(temp_adm1_id) is None:
            temp_adm0 = con_loc_dict[adm2_areaid]['ADM0']
            temp_adm1 = con_loc_dict[adm2_areaid]['ADM1']
            temp_adm1_neighbor = con_loc_dict[adm2_areaid]['adm1_neighbor']
            temp_adm2_in_adm1_neighbor = con_loc_dict[adm2_areaid]['adm2_in_adm1_neighbor'] #list
            temp_repr_loc = con_loc_dict[adm2_areaid]['repr_loc']
            temp_weight = con_loc_dict[adm2_areaid]['weight']
            
            loc_adm1_dict[temp_adm1_id] = {}
            
            loc_adm1_dict[temp_adm1_id]['ADM0'] = temp_adm0
            loc_adm1_dict[temp_adm1_id]['ADM1'] = temp_adm1
            loc_adm1_dict[temp_adm1_id]['adm1_neighbor'] = temp_adm1_neighbor
            loc_adm1_dict[temp_adm1_id]['adm2_in_adm1_neighbor'] = temp_adm2_in_adm1_neighbor
            loc_adm1_dict[temp_adm1_id]['adm2_in_adm1_neighbor_repr_loc'] = [con_loc_dict[area_id]['repr_loc'] for area_id in temp_adm2_in_adm1_neighbor]
            loc_adm1_dict[temp_adm1_id]['repr_loc_list'] = [temp_repr_loc]
            loc_adm1_dict[temp_adm1_id]['weight_list'] = [temp_weight]
            loc_adm1_dict[temp_adm1_id]['adm2_id_list'] = [adm2_areaid]
        else:
            temp_repr_loc = con_loc_dict[adm2_areaid]['repr_loc']
            temp_weight = con_loc_dict[adm2_areaid]['weight']
            loc_adm1_dict[temp_adm1_id]['repr_loc_list'].append(temp_repr_loc)
            loc_adm1_dict[temp_adm1_id]['weight_list'].append(temp_weight)
            loc_adm1_dict[temp_adm1_id]['adm2_id_list'].append(adm2_areaid)
    
    return loc_adm1_dict
def get_repr_locs_adm1_gpkg(con_loc_dict):
    #from {ccode}_loc.pickle --> {ccode}_loc_adm1.pickle
    adm1_areaid_list = []
    loc_adm1_dict = {}
    
    # country = get_country(_CURR_CCODE)
    # adm_level = list(country.levels['level_name'])
    # if len(adm_level)==2:
    #     return None
    
    for adm2_areaid in con_loc_dict.keys():
        #get all info form ccode_loc (adm2)
        temp_adm1_id = con_loc_dict[adm2_areaid]['adm1_id']
        if loc_adm1_dict.get(temp_adm1_id) is None:
            temp_adm0 = con_loc_dict[adm2_areaid]['ADM0']
            temp_adm1 = con_loc_dict[adm2_areaid]['ADM1']
            temp_adm1_neighbor = con_loc_dict[adm2_areaid]['adm1_neighbor']
            temp_adm2_in_adm1_neighbor = con_loc_dict[adm2_areaid]['adm2_in_adm1_neighbor'] #list
            temp_repr_loc = con_loc_dict[adm2_areaid]['repr_loc']
            temp_weight = con_loc_dict[adm2_areaid]['weight']
            
            loc_adm1_dict[temp_adm1_id] = {}
            
            loc_adm1_dict[temp_adm1_id]['ADM0'] = temp_adm0
            loc_adm1_dict[temp_adm1_id]['ADM1'] = temp_adm1
            loc_adm1_dict[temp_adm1_id]['adm1_neighbor'] = temp_adm1_neighbor
            loc_adm1_dict[temp_adm1_id]['adm2_in_adm1_neighbor'] = temp_adm2_in_adm1_neighbor
            loc_adm1_dict[temp_adm1_id]['adm2_in_adm1_neighbor_repr_loc'] = [con_loc_dict[area_id]['repr_loc'] for area_id in temp_adm2_in_adm1_neighbor]
            loc_adm1_dict[temp_adm1_id]['repr_loc_list'] = [temp_repr_loc]
            loc_adm1_dict[temp_adm1_id]['weight_list'] = [temp_weight]
            loc_adm1_dict[temp_adm1_id]['adm2_id_list'] = [adm2_areaid]
        else:
            temp_repr_loc = con_loc_dict[adm2_areaid]['repr_loc']
            temp_weight = con_loc_dict[adm2_areaid]['weight']
            loc_adm1_dict[temp_adm1_id]['repr_loc_list'].append(temp_repr_loc)
            loc_adm1_dict[temp_adm1_id]['weight_list'].append(temp_weight)
            loc_adm1_dict[temp_adm1_id]['adm2_id_list'].append(adm2_areaid)
    
    return loc_adm1_dict
        
def _point2adm2imgpathlist(loc, zoom_level=None, timeline=None):
    """
    Load the satelite image of the given location, zoom level, and timeline.
    
    Args:
        loc: The location of the interest
        zoom_level: Zoom level of the image
        timeline: The time of the interest
    
    Return:
        Loaded image path list
        Return empty list for error case.
    """
    
    result_list = []
    ccode, adm1_name, adm2_name, areaid, loc_geom = point2adm2meta(loc)
    if zoom_level is None:
        zoom_level = "*"
    if timeline is None:
        timeline = "*"

    config = utils.read_config()
    img_dir = config.path.img_dir
    
    zoom_level = str(zoom_level)
    timeline = str(timeline)

    adm0_path = img_dir+ccode+"_"+zoom_level+"_"+timeline+"/"
    
    if len(glob.glob(adm0_path))>0:
        adm0_path = glob.glob(adm0_path)[0]

    if adm2_name is None:
        adm2_path = adm0_path+adm1_name+'/'
    else:
        adm2_path = adm0_path+adm1_name+'/'+adm2_name+'/'

    print(adm2_path)
    if os.path.exists(adm2_path):
        result_list = glob.glob(adm2_path+'*_*.png')
    
    return result_list

def deg2num(latitude, longitude, zoom, do_round=True):
    lat_rad = np.radians(latitude)
    n = 2.0 ** zoom
    if do_round:
        f = np.floor
    else:
        f = lambda x: x
    xtile = f((longitude + 180.) / 360. * n)
    ytile = f((1.0 - np.log(np.tan(lat_rad) + (1 / np.cos(lat_rad))) / np.pi) /
              2. * n)
    if do_round:
        if isinstance(xtile, np.ndarray):
            xtile = xtile.astype(np.int32)
        else:
            xtile = int(xtile)
        if isinstance(ytile, np.ndarray):
            ytile = ytile.astype(np.int32)
        else:
            ytile = int(ytile)
    return (xtile, ytile)

def num2deg(xtile, ytile, zoom):
    n = 2.0 ** zoom
    longitude = xtile / n * 360. - 180.
    latitude = np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * ytile / n))))
    return (latitude, longitude)

if __name__=='__main__':
    pass
