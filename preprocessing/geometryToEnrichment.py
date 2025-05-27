from arcgis.geoenrichment import *
from arcgis.geometry import Geometry
from arcgis.gis import GIS
import pandas as pd
import geopandas as gpd

from countryToGeometry import *

import sys
sys.path.append('../')
try:
    import config
    from helper import *
except ModuleNotFoundError:
    import modules.config
    from modules.helper import *

def geomDataFrameToEnrichment(ccode,geom_df,enrich_v):
    #assign ccode
    ccode = ccode
    
    #Define initial output list
    result_list = []

    #get Country by arcgis.geoenrichment.Country.get()
    country = Country.get(ccode)
     
    #find subgeography(==adm1, adm2) names for target country
    subgeography_name = list(country.levels['level_name'])
    subgeography_name.reverse()

    #geom_df := from pkl file or get geometry from ccode
    
    if geom_df is None:
        geom_df = countryToDataFrame(ccode)
    elif isinstance(geom_df,str):
        geom_df = pd.read_pickle(geom_df)
    else:
        geom_df = geom_df
    
    ev_df = country.enrich_variables
    curr_ev_df = ev_df[(ev_df.data_collection.str.lower().str.contains(enrich_v))]
    ev_list = list(curr_ev_df['name'])
    #new_row = {'ADM1':adm1,'ADM2':adm2,'areaid':adm2_areaid,'geometry':adm2_geom}
    
    for row in geom_df.itertuples():
        adm1 = row.ADM1
        adm2 = row.ADM2
        adm2_areaid = row.areaid
        new_row = {'ADM1':adm1,'ADM2':adm2,'areaid':adm2_areaid}
        
        #GeoEnrichment! Credits must be required here.
        adm2_enrich = country.enrich(study_areas= [row.geometry],enrich_variables=curr_ev_df)
        
        for ev in ev_list:
            ev_low = ev.lower()
            if ev_low in adm2_enrich.columns:
                new_row[ev] = adm2_enrich.iloc[0][ev_low]
        result_list.append(new_row)
        print(new_row)
    
    result_df = pd.DataFrame(result_list)
    return result_df
    
if __name__=='__main__':
    api_key = config.api_key
    ccode = config.ccode
    geom_df = ccode+'.pkl'
    enrich_v = config.enrich_v
    portal = GIS("https://www.arcgis.com", api_key=api_key)
    
    result_df = geomDataFrameToEnrichment(ccode,geom_df,'gender')
    
    #result_df.to_pickle(ccode+'.pkl')
    result_df.to_csv(ccode+'_'+enrich_v+'1.csv',index=False)    