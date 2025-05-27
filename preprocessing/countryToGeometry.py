from arcgis.geoenrichment import *
from arcgis.geometry import Geometry
from arcgis.gis import GIS
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


def countryToList(ccode):
    #assign ccode
    ccode = ccode
    
    #Define initial output list
    result_list = []

    #get Country by arcgis.geoenrichment.Country.get()
    country = Country.get(ccode)
    
    #find subgeography(==adm1, adm2) names for target country
    subgeography_name = list(country.levels['level_name'])
    subgeography_name.reverse()
    
    #dictionary of {<adm1> : <NamedArea>} in adm0 (==counrty)
    adm1_namedarea_dict = getattr(country.subgeographies, subgeography_name[1])
    
    # adm1_namedarea_list
    adm1_namedarea_list = adm1_namedarea_dict.keys()
    
    for adm1_namedarea in adm1_namedarea_list:
        adm1 = adm1_namedarea
        print(adm1_namedarea_dict[adm1_namedarea])
        adm2_namedarea_dict = getattr(adm1_namedarea_dict[adm1_namedarea], subgeography_name[2])
        adm2_namedarea_list = adm2_namedarea_dict.keys()
        print(adm2_namedarea_list)
        for adm2_namedarea in adm2_namedarea_list:
            adm2 = adm2_namedarea
            adm2_areaid = adm2_namedarea_dict[adm2_namedarea]._areaid
            adm2_geom = adm2_namedarea_dict[adm2_namedarea].geometry
            new_row = {'ADM1':adm1,'ADM2':adm2,'areaid':adm2_areaid,'geometry':adm2_geom}
            result_list.append(new_row)
            #result_df.append(new_row)
    
    #return result_list
    return result_list

def countryToGeoDataFrame(ccode):
    #to GeoDataFrame
    result_gdf = gpd.GeoDataFrame(countryToList(ccode),crs='EPSG:4326')
    return result_gdf

def countryToDataFrame(ccode):
    #to DataFrame
    result_df = pd.DataFrame(countryToList(ccode))
    return result_df

if __name__=='__main__':
    api_key = config.api_key
    ccode = config.ccode
    portal = GIS("https://www.arcgis.com", api_key=api_key)
    
    result_list = countryToList(ccode)
    result_gdf = gpd.GeoDataFrame(result_list,crs='EPSG:4326')
    result_df = pd.DataFrame(result_list)
    
    result_gdf.to_file(ccode+'.geojson',driver="GeoJSON")
    result_df.to_pickle(ccode+'.pkl')    