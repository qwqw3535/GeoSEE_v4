import os
import shapely.wkt
import urllib.request
from concurrent.futures import as_completed, ThreadPoolExecutor
import datetime
from arcgis.geoenrichment import *
from arcgis.geometry import Geometry
from arcgis.gis import GIS

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import warnings
warnings.filterwarnings(action='ignore')

import numpy as np
import pandas as pd
import geopandas as gpd

import sys
sys.path.append('../')
try:
    import config
    from helper import *
except ModuleNotFoundError:
    import modules.config
    from modules.helper import *

def geomDataFrameToImage(ccode, geom_df, base_url, base_path, zl,timeline):
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    
    if geom_df is None:
        country = Country.get(ccode)
        geom_df = countryToGeoDataFrame(ccode)
    elif isinstance(geom_df,str):
        geom_df = pd.read_file(geom_df)
    else:
        geom_df = geom_df 
    
    idx_start = geom_df.index.start
    idx_stop = geom_df.index.stop

    for row in geom_df.itertuples():
        if not os.path.exists(base_path+row.ADM1):
            os.makedirs(base_path+row.ADM1)
        if not os.path.exists(base_path+row.ADM1+'/'+row.ADM2):
            os.makedirs(base_path+row.ADM1+'/'+row.ADM2)
            #print(row.ADM1, row.ADM2)
  
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

        print('Working on index {}, {}, {}, {}th of index {} to {}'.format(row.Index+1, row.ADM1, row.ADM2, row.Index+1-idx_start, idx_start, idx_stop))
    
        for x in range(math.floor(xmin),math.ceil(xmax)):
            for y in range(math.floor(ymin),math.ceil(ymax)):
                templat, templng = num2deg(x+0.5,y+0.5,int(zl))
                if row.geometry.type == 'MultiPolygon':
                    multicon = False
                    multilist = list(row.geometry.geoms)
                    for temppol in multilist:
                        if temppol.contains(Point(templng,templat)):
                            multicon = True
                    if multicon == True:
                        const_path = base_path+row.ADM1+'/'+row.ADM2+'/'+str(y)+'_'+str(x)+'.png'
                        if not os.path.exists(const_path):
                            try:
                                urllib.request.urlretrieve(base_url+timeline+'/'+zl+'/'+str(y)+'/'+str(x), const_path)
                            except:
                                print('error in {}/{} !'.format(row.ADM1, row.ADM2))
                                print(timeline+'/'+zl+'/'+str(y)+'/'+str(x))
                else:
                    if row.geometry.contains(Point(templng,templat)):
                        const_path = base_path+row.ADM1+'/'+row.ADM2+'/'+str(y)+'_'+str(x)+'.png'
                        if not os.path.exists(const_path):
                            try:
                                urllib.request.urlretrieve(base_url+timeline+'/'+zl+'/'+str(y)+'/'+str(x), const_path)
                            except:
                                print('error in {}/{} !'.format(row.ADM1, row.ADM2))
                                print(timeline+'/'+zl+'/'+str(y)+'/'+str(x))
        if (row.Index+1-idx_start) % 5 == 0 or row.Index+1-idx_start == idx_stop-idx_start:
            print('{}/{} finished, index {} to {}'.format(row.Index+1-idx_start, idx_stop-idx_start, idx_start, idx_stop))
            print()


if __name__=='__main__':
    api_key = config.api_key
    img_dir = config.img_dir
    ccode = config.ccode
    zoom_level = config.zoom_level
    timeline = config.timeline
    threads = config.threads
    
    zoom_level = str(zoom_level)
    timeline = str(timeline)

    base_url = 'https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/WMTS/1.0.0/default028mm/MapServer/tile/'
    base_path = img_dir + ccode+'_'+zoom_level+'_'+timeline+'/'
    portal = GIS("https://www.arcgis.com", api_key=api_key)

    geom_df = gpd.read_file(ccode+'.geojson')
    split_df = np.array_split(geom_df,threads)
    for i in range(len(split_df)):
        print(i, split_df[i].head())
    #geomDataFrameToImage(ccode, split_df[1], base_url, base_path, zoom_level,timeline)
    
    with ThreadPoolExecutor(threads) as exe:
        futures = [exe.submit(geomDataFrameToImage,ccode,curr_geom_df,base_url,base_path,zoom_level,timeline) for curr_geom_df in split_df]
    '''
    with ProcessPoolExecutor(max_workers=threads) as exe:
        _ = [exe.map(geomDataFrameToImage,ccode,curr_geom_df,base_url,base_path,zoom_level,timeline) for curr_geom_df in split_df]
    '''