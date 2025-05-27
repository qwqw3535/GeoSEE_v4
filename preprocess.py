from modules import get_country, deg2num, num2deg, get_repr_locs, get_repr_locs_adm1,get_repr_locs_euro,get_repr_locs_euro_adm1
from modules import helper
import utils
import pickle
import urllib
from concurrent.futures import as_completed, ThreadPoolExecutor
import math
import numpy as np
import time
from pathlib import Path

import geopandas as gpd
import pandas as pd

from shapely.geometry import Point

def country2list(ccode):
    #get Country by arcgis.geoenrichment.Country.get()
    country = get_country(ccode)
    
    #find subgeography(==adm1, adm2) names for target country
    subgeography_name = list(country.levels['level_name'])
    subgeography_name.reverse()
    
    #dictionary of {<adm1> : <NamedArea>} in adm0 (==counrty)
    adm1_namedarea_dict = getattr(country.subgeographies, subgeography_name[1])
    
    # adm1_namedarea_list
    adm1_namedarea_list = adm1_namedarea_dict.keys()
    
    #Define initial output list
    result_list = []
    for adm1, val in adm1_namedarea_dict.items():
        print(val)
        adm2_namedarea_dict = getattr(val, subgeography_name[2])
        print(list(adm2_namedarea_dict.keys()))
        # print()
        for adm2, val in adm2_namedarea_dict.items():
            adm2_areaid = val._areaid
            adm2_geom = val.geometry
            new_row = {'ADM1':adm1,'ADM2':adm2,'areaid':adm2_areaid,'geometry':adm2_geom}
            result_list.append(new_row)

    return result_list
def country2list_euro(con_loc_dict, geom_dict):

    #Define initial output list
    result_list = []
    for loc_key, loc_val in con_loc_dict.items():
        for geom_key, geom_val in geom_dict.items():
            if loc_key == geom_key:
                new_row = {'ADM1':loc_val['ADM1'],'ADM2':loc_val['ADM2'],'areaid':loc_key,'geometry':geom_val}
        result_list.append(new_row)

    return result_list

def gpkg2list(gpkg_path):
    data = gpd.read_file(gpkg_path)
    result_list = data.apply(lambda row: {'ADM1':row["NAME_1"],'ADM2':row["NAME_2"],'areaid':row["GID_2"],'geometry':row["geometry"]}, axis=1)
    return result_list.to_list()

def gdf2img(ccode, gdf, base_url, base_path, zl, timeline):
    idx_start = gdf.index.start
    idx_stop = gdf.index.stop
    
    for row in gdf.itertuples():
        adm_path = base_path / row.ADM1 / row.ADM2
        adm_path.mkdir(exist_ok=True, parents=True)

        if row.geometry.type == 'MultiPolygon':
            xxyylist = []
            for partpol in list(row.geometry.geoms):
                xxyylist = xxyylist + list(zip(*partpol.exterior.coords.xy))
            xylist = list(map(lambda x:deg2num(x[1],x[0],int(zl),False) ,xxyylist))
        else:
            #xylist = list(map(lambda x:deg2num(x[1],x[0],int(zl),False) , list(zip(*shapely.wkt.loads(row.geometry).exterior.coords.xy))))
            xylist = list(map(lambda x:deg2num(x[1],x[0],int(zl),False) , list(zip(*row.geometry.exterior.coords.xy))))
            
        xylist.sort(key=lambda x:x[0])
        xmin = xylist[0][0]
        xmax = xylist[-1][0]

        xylist.sort(key=lambda x:x[1])
        ymin = xylist[0][1]
        ymax = xylist[-1][1]

        print('Working on index {}, {}, {}, {}th of index {} to {}'.format(row.Index, row.ADM1, row.ADM2, row.Index+1-idx_start, idx_start, idx_stop))
        
        for x in range(math.floor(xmin),math.ceil(xmax)):
            for y in range(math.floor(ymin),math.ceil(ymax)):
                const_path = adm_path / f'{y}_{x}.png'
                urlretrieve_str = base_url+timeline+'/'+zl+'/'+str(y)+'/'+str(x)
                
                templat, templng = num2deg(x+0.5,y+0.5,int(zl))
                if row.geometry.type == 'MultiPolygon':
                    multicon = False
                    multilist = list(row.geometry.geoms)
                    for temppol in multilist:
                        if temppol.contains(Point(templng,templat)):
                            multicon = True
                    if multicon == True:
                        if not const_path.exists():
                            try:
                                urllib.request.urlretrieve(urlretrieve_str, str(const_path))
                            except Exception as e:
                                print('error in {}/{} !'.format(row.ADM1, row.ADM2))
                                print(timeline+'/'+zl+'/'+str(y)+'/'+str(x))
                else:
                    if row.geometry.contains(Point(templng,templat)):
                        if not const_path.exists():
                            try:
                                urllib.request.urlretrieve(urlretrieve_str, str(const_path))
                            except Exception as e:
                                print('error in {}/{} !'.format(row.ADM1, row.ADM2))
                                print(timeline+'/'+zl+'/'+str(y)+'/'+str(x))
                                
        if (row.Index+1-idx_start) % 5 == 0 or row.Index+1-idx_start == idx_stop-idx_start:
            print('{}/{} finished, index {} to {}'.format(row.Index+1-idx_start, idx_stop-idx_start, idx_start, idx_stop))
            print()

if __name__ == "__main__":
    # create_geometry = False
    # create_image = True
    # area_dict = False
    create_geometry = True
    create_image = False
    area_dict = True
    
    config = utils.read_config()
    
    api_key = config.arcgis.api_key
    ccode = config.arcgis.ccode
    zoom_level = config.arcgis.zoom_level
    timeline = config.arcgis.timeline
    threads = int(config.arcgis.threads)
    
    helper._CURR_CCODE = ccode
    img_dir = config.path.img_dir
    preprocessing_dir = Path(config.path.preprocessing_dir)
    
    utils.login_gis_portal(api_key)
    
    gdf_path = preprocessing_dir / f'{ccode}.geojson'
    df_path = preprocessing_dir / f'{ccode}.pkl'
    
    if area_dict:
        force_update = False
        # if len(ccode) >2:
        #     con_loc_dict, geom_dict = get_repr_locs_euro_group(ccode)
        # else:
        con_loc_dict, geom_dict = get_repr_locs_euro(ccode)
        
        proxy_dir = Path(config.path.proxy_dir)
        contained_loc_path = proxy_dir / f'{ccode}_loc.pickle'
        if contained_loc_path.exists() == False or force_update:
            with open(file=str(contained_loc_path), mode='wb') as f:
                pickle.dump(con_loc_dict, f)
        
        geom_path = proxy_dir / f'{ccode}_geom.pickle'
        if geom_path.exists() == False or force_update:
            with open(file=str(geom_path), mode='wb') as f:
                pickle.dump(geom_dict, f)
                
        loc_adm1_dict = get_repr_locs_euro_adm1(con_loc_dict)
        if loc_adm1_dict is not None:
            loc_adm1_path = proxy_dir / f'{ccode}_loc_adm1.pickle'
            if loc_adm1_path.exists() == False:
                with open(file=str(loc_adm1_path), mode='wb') as f:
                    pickle.dump(loc_adm1_dict, f)
    
    # Country to Geometry
    if create_geometry:
        # if len(ccode) >2:
        #     result_list = country2list_euro_group(con_loc_dict, geom_dict)
        # else:
        result_list = country2list_euro(con_loc_dict, geom_dict)
        # result_list = country2list_euro(con_loc_dict, geom_dict)
        print(result_list)
        # time.sleep(1000)
        result_gdf = gpd.GeoDataFrame(result_list,crs='EPSG:4326')
        result_df = pd.DataFrame(result_list)
    
        result_gdf.to_file(gdf_path,driver="GeoJSON")
        result_df.to_pickle(df_path) 
    
    # Geometry to Image
    if create_image:
        base_url = 'https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/WMTS/1.0.0/default028mm/MapServer/tile/'
        base_path = Path(img_dir) / f'{ccode}_{zoom_level}_{timeline}'
        gdf = gpd.read_file(gdf_path)

        split_df = np.array_split(gdf, threads)
            
        with ThreadPoolExecutor(threads) as exe:
            futures = [exe.submit(
                gdf2img,
                ccode,
                cur_gdf,
                base_url,
                base_path,
                zoom_level,
                timeline
            ) for cur_gdf in split_df]