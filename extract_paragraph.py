import argparse
import json
import pickle
import os

parser = argparse.ArgumentParser()

parser.add_argument('--gpu', required=True, help='gpu num')
parser.add_argument('--ccode', required=True, help='country code')
parser.add_argument('--var', required=True, help='testing variable')
parser.add_argument('--adm', default='2', help='type "1" only if you want to make paragraph on adm1')
parser.add_argument('--merged', default=False, help='merged data for v4 countries')

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu

import pandas as pd
import numpy as np
from pathlib import Path
from modules import *
from tqdm import tqdm
from utils import query_gpt, parse_and_select, read_config, login_gis_portal
from utils import get_prompt_for_extracting_modules

ask_num = 10
api_key = ''
prompt_version = '2'
target_ccode = args.ccode
target_var = args.var
is_merged = args.merged
adm = args.adm

if target_var == 'overall':
    s_modules_path = f'./selected_modules/{target_var}_{target_ccode}.json'
else:
    s_modules_path = f'./selected_modules/{target_ccode}_{target_var}.json'
    # s_modules_path = f'./selected_modules/{target_ccode}_{target_var}.json'
# if target_var == 'overall':
#     s_modules_path = f'./selected_modules/poi_added/{target_var}_{target_ccode}.json'
# else:
#     s_modules_path = f'./selected_modules/poi_added/{target_ccode}_{target_var}.json'

with open(s_modules_path, 'r') as f:
    data = json.load(f)
    
s_modules = data['s_modules']
print(s_modules)
if is_merged:
    original_ccode =target_ccode
    target_ccode = 'PL_SK_CZ_HU'
else:
    original_ccode =target_ccode



config = read_config()
arcgis_api_key = config.arcgis.api_key
timeline = config.arcgis.timeline
login_gis_portal(arcgis_api_key)

def get_temp_func_str():
    return helper._TEMP_FUNC_STR
    
def set_temp_func_str(temp_func_str):
    helper._TEMP_FUNC_STR = temp_func_str
    
def get_func_repr_dict():
    return helper._FUNC_REPR_DICT

def set_func_repr_dict(key, value, overwrite=False):
    if key not in helper._FUNC_REPR_DICT.keys() or overwrite:
        helper._FUNC_REPR_DICT[key] = value

def clear_func_repr_dict():
    helper._FUNC_REPR_DICT = {}

def set_ccode_loc(ccode):
    config = read_config()
    proxy_dir = Path(config.path.proxy_dir)
    loc_path = proxy_dir / f'{ccode}_loc.pickle'
    with open(loc_path, 'rb') as f:
        helper._CCODE_LOC = pickle.load(f)
        
def set_curr_ccode(ccode):
    helper._CURR_CCODE = ccode
        
proxy_dir = Path(config.path.proxy_dir)
set_ccode_loc(target_ccode)
set_curr_ccode(target_ccode)
set_temp_func_str('')
clear_func_repr_dict()

desc_results = []
adm_offset = '' if adm=='2' else '_adm1'

if adm=='2':
    ccode_loc_path = proxy_dir / f'{target_ccode}_loc{adm_offset}.pickle'
    with open(ccode_loc_path,'rb') as f:
        area_dict = pickle.load(f)
    for area_id, loc in tqdm(area_dict.items()):
        print(area_id, s_modules)
        Loc = loc['repr_loc']
        result_dict = {}
        for module in s_modules:
            print(area_id, module)
            module_func = None
            if 'get_aggregate_neighbor_info' in module:
                module_func = module.replace('get_aggregate_neighbor_info(Loc, ','').replace('))',')')
            else:
                module_func = 'lambda x:'+module.replace('Loc','loc=x')
            set_temp_func_str(module_func)
            # print(f'This is module_func: {module_func}')
            ## args 안의 loc 등을 모르는데 어떻게 실행이 되는??
            ## A. 위의 for문의 loc을 가져다 씀 
            exec("results = " + module)
            # print(f'This is result: {results}')
            result_dict[module] = results
                
        paragraph = ""
        for fname, desc in result_dict.items():
            for key, info in desc.items():
                paragraph += f"{info['desc']} is {info['val']}"
                paragraph += "\n"
        paragraph = paragraph.strip()
        
        if target_ccode == 'KOR':
            paragraph = paragraph.replace('Gyeongsangbuk-do, Daegu', 'Gyeongsangbuk-do')
        
        desc_aggregate=[]
        for line in paragraph.split('\n'):
            line_split = line.split('is ')
            curr_float_candidate = line_split[-1]
            curr_replace = None
            try:
                curr_float = float(curr_float_candidate)
                curr_float = round(curr_float,3)
                curr_replace = str(curr_float)
            except:
                curr_replace = line_split[-1]
            post_line = line_split[:-1]+[curr_replace]
            desc_aggregate.append('is '.join(post_line))
        paragraph = '\n'.join(desc_aggregate)
        desc_results.append([area_id, paragraph])
        
        df = pd.DataFrame(desc_results)
        df.columns = ['area_id', 'desc']
        df.to_csv(f'./extracted_paragraphs/landuse/{target_var}/{target_ccode}_{target_var}_{original_ccode}_adm2_{timeline}_paragraph_output.csv', index=False, header=True)
else:
    ccode_loc_path = proxy_dir / f'{target_ccode}_loc{adm_offset}.pickle'
    with open(ccode_loc_path,'rb') as f:
        area_dict = pickle.load(f)
    for area_id, loc in tqdm(area_dict.items()):
        result_dict = {}
        repr_loc_list = []
        for module in s_modules:
            print(area_id,module)
            if 'get_aggregate_neighbor_info' in module:
                repr_loc_list = loc['adm2_in_adm1_neighbor_repr_loc']
                module_func = module.replace('get_aggregate_neighbor_info(Loc, ','').replace('))',')')
                if len(repr_loc_list)==0:
                    temp_loc = loc['repr_loc_list'][0]
                    exec(f'temp_result_dict = ({module_func})(temp_loc)')
                    var_name_list = list(temp_result_dict.keys())
                    for var_name in var_name_list:
                        desc = temp_result_dict[var_name]['desc']
                        adm1_name = loc['ADM1']
                        temp_result_dict[var_name] = {'val': f'of the neighboring region(s) is not defined. {adm1_name} is isolated from neighboring administrative regions.', 'desc':desc}
                    results = temp_result_dict
                else:
                    set_temp_func_str(module_func)
                    exec(f'results = aggregate_repr_loc_list(repr_loc_list, {module_func})')
                    for key in results.keys():
                        results[key]['desc'] += ' of the neighboring region(s)' 
            elif 'get_address' in module:
                val_address = area_dict[area_id]["ADM1"] + ', '+ area_dict[area_id]["ADM0"]
                loc_info = {'val': val_address, 'desc': 'full address of the given location' }
                results = {'address': loc_info}
            else:
                repr_loc_list = loc['repr_loc_list']
                module_func = 'lambda x:'+module.replace('Loc','loc=x')
                set_temp_func_str(module_func)
                exec(f'results = aggregate_repr_loc_list(repr_loc_list, {module_func})')
            result_dict[module] = results
        #print(result_dict)
        paragraph = ""
        for fname, desc in result_dict.items():
            for key, info in desc.items():
                paragraph += f"{info['desc']} is {info['val']}"
                paragraph += "\n"
        paragraph = paragraph.strip()
        print(paragraph)
        
        #Post-hoc
        if target_ccode == 'KOR':
            paragraph = paragraph.replace('Gyeongsangbuk-do, Daegu', 'Gyeongsangbuk-do')

        desc_aggregate=[]
        for line in paragraph.split('\n'):
            line_split = line.split('is ')
            curr_float_candidate = line_split[-1]
            curr_replace = None
            try:
                curr_float = float(curr_float_candidate)
                curr_float = round(curr_float,3)
                curr_replace = str(curr_float)
            except:
                curr_replace = line_split[-1]
            post_line = line_split[:-1]+[curr_replace]
            desc_aggregate.append('is '.join(post_line))
        paragraph = '\n'.join(desc_aggregate)
        desc_results.append([area_id, paragraph])
        
        df = pd.DataFrame(desc_results)
        df.columns = ['area_id', 'desc']
        df.to_csv(f'./extracted_paragraphs/landuse/{target_var}/{target_ccode}_{target_var}{adm_offset}_{timeline}_paragraph_output.csv', index=False, header=True)
