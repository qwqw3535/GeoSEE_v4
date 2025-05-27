import re
import json
import time
import openai
import numpy as np
import pandas as pd
from tqdm import tqdm

import configparser
from argparse import Namespace
from sklearn.preprocessing import StandardScaler

import openai
from arcgis.gis import GIS

def read_config(
    config_path='./config.ini'
):
    config = configparser.ConfigParser()
    config.read(config_path)
    
    if len(config.sections()) == 0:
        return None
    
    ns_config = Namespace()
    for section in config.sections():
        setattr(ns_config, section, Namespace())
        for attr, val in config[section].items():
            setattr(getattr(ns_config, section), attr, val)
    
    return ns_config

def login_gis_portal(
    api_key,
    url="https://www.arcgis.com"
):
    portal = GIS(url, api_key=api_key)
    
    return portal

def query_gpt(text_list, api_key, model, max_completion_tokens=30, temperature=0, max_try_num=10, tqdm_disable=False):
    openai.api_key = api_key
    if model=='deepseek-chat' or model=='deepseek-reasoner':
        openai.base_url = "https://api.deepseek.com"
    result_list = []
    for prompt in tqdm(text_list, disable=tqdm_disable):
        curr_try_num = 0
        while curr_try_num < max_try_num:
            try:
                response = openai.chat.completions.create(
                    model=model,
                    messages=[{"role":"user", "content":prompt}],
                    #temperature = temperature,
                    max_completion_tokens = max_completion_tokens,
                    top_p = 1
                )
                message = response.choices[0].message
                result = message.content
                reasoning_result = result
                #reasoning_result = response.choices[0].message.reasoning_content
                if hasattr(message, "reasoning_content"):
                    reasoning_result = message.reasoning_content
                result_list.append((result, reasoning_result))
                break
            except openai.BadRequestError as e:
                print(e)
                curr_try_num += 1
                if curr_try_num >= max_try_num:
                    result_list.append(("BadRequest","BadRequest"))
                # return result_list
            except Exception as e:
                print(e)
                print(curr_try_num)
                curr_try_num += 1
                if curr_try_num >= max_try_num:
                    result_list.append(("Exception on response","Exception on response"))
                time.sleep(10)
    return result_list

def query_gpt_logprobs(text_list, api_key, model, max_completion_tokens=30, temperature=0, max_try_num=10, tqdm_disable=False):
    openai.api_key = api_key
    if model=='deepseek-chat' or model=='deepseek-reasoner':
        openai.base_url = "https://api.deepseek.com"
    result_list = []
    for prompt in tqdm(text_list, disable=tqdm_disable):
        curr_try_num = 0
        while curr_try_num < max_try_num:
            try:
                response = openai.chat.completions.create(
                    model=model,
                    messages=[{"role":"user", "content":prompt}],
                    #temperature = temperature,
                    max_completion_tokens = max_completion_tokens,
                    top_p = 1,
                    logprobs = True,
                    top_logprobs = 5
                )
                message = response.choices[0].message
                result = message.content
                reasoning_result = result
                print(response)
                print(message)
                assert(1==0)
                #reasoning_result = response.choices[0].message.reasoning_content
                if hasattr(message, "reasoning_content"):
                    reasoning_result = message.reasoning_content
                result_list.append((result, reasoning_result))
                break
            except openai.BadRequestError as e:
                print(e)
                curr_try_num += 1
                if curr_try_num >= max_try_num:
                    result_list.append(("BadRequest","BadRequest"))
                # return result_list
            except Exception as e:
                print(e)
                print(curr_try_num)
                curr_try_num += 1
                if curr_try_num >= max_try_num:
                    result_list.append(("Exception on response","Exception on response"))
                time.sleep(10)
    return result_list


def fill_in_templates(fill_in_dict, template_str):
    for key, value in fill_in_dict.items():
        if key in template_str:
            template_str = template_str.replace(key, value)
    return template_str   


def get_prompt_for_extracting_modules(question, desc='function_desc.txt', version='1'):
    with open(f"./templates/{desc}", "r") as f:
        function_desc = f.read()
        
    if version == '1':
        postfix = ''
    elif version == '2':
        postfix = '_v2'
    else:
        assert(0)
        
    with open(f"./templates/get_modules_template{postfix}.txt", "r") as f:
        template = f.read()

    fill_in_dict = {
        "<MODULE_DESC>": function_desc, 
        "<QUESTION>": question
    }
    template = fill_in_templates(fill_in_dict, template)
    return template


def get_prompt_for_inference_unsupervised(df, index, target_question, in_context_num=5):        
    template = "<EXAMPLES>\n\n<TARGET>"

    target_df = df.iloc[index]
    candidate_df = df.drop(index)
        
    # choose in-context samples
    scored_df = candidate_df[candidate_df['score'] != -1]
    if len(scored_df) < 1:
        example_desc = ""
    else:
        if len(scored_df) < 5:
            example_df = scored_df
        else:
            quantile_list = [0, 0.25, 0.5, 0.75, 1]
            selected_df_list = []
            selected_index_list = []
            for quant_val in quantile_list:
                selected_df = scored_df[scored_df['score'] == scored_df['score'].quantile(quant_val, interpolation='lower')]
                selected_df = selected_df.sample(1)
                selected_df_list.append(selected_df)
                selected_index_list.append(selected_df.index[0])
                
            remaining_df = scored_df.drop(selected_index_list, axis=0).reset_index(drop=True)
            if len(remaining_df) > 5:
                extracted_vals = []
                for desc in remaining_df['desc']:
                    extracted_vals.append(np.array([float(val) for val in re.findall(r"[-+]?(?:\d*\.*\d+)", '\n'.join(desc.split('\n')[1:]))]))
                extracted_vals = np.stack(extracted_vals)
                target_val = np.array([float(val) for val in re.findall(r"[-+]?(?:\d*\.*\d+)", '\n'.join(target_df.desc.split('\n')[1:]))])

                scaler = StandardScaler()
                extracted_vals_normalized = scaler.fit_transform(extracted_vals)
                target_val_normalized = scaler.transform(target_val.reshape(1, -1))
                sample_dist = np.linalg.norm(extracted_vals_normalized - target_val_normalized, axis=1)
                selected_neighbors = sample_dist.argsort()[:5]
                for neighbor in selected_neighbors:
                    selected_df_list.append(pd.DataFrame(remaining_df.iloc[neighbor]).T)
                
            example_df = pd.concat(selected_df_list, axis=0, ignore_index=True)
        example_desc = '\n\n'.join(['. '.join(row.desc.split('\n')) + f'.\n{target_question}\nAnswer: ' + str(round(row.score, 3)) for _, row in example_df.iterrows()])
        
    # Serialize the dataframe
    target_desc = '. '.join(target_df['desc'].split('\n'))
    target_desc += f'.\n{target_question}\nAnswer: '
        
    fill_in_dict = {
        "<EXAMPLES>": example_desc,
        "<TARGET>": target_desc
    }
    template = fill_in_templates(fill_in_dict, template)
    return template



def get_prompt_for_inference_in_context(df, df_record, index, X_train, y_train, X_test, target_question, target_ccode=False, addr_only = False, classification = False):        
    template = "<EXAMPLES>\n\n<TARGET>"
                
    # choose in-context samples
    X_train_all = X_train.copy()
    X_train_all['score'] = y_train.to_numpy().ravel()
    
    example_df = X_train_all
    candidate_df = df_record.copy()
    if target_ccode:
        X_test_target_country = X_test.loc[X_test['area_id'].apply(lambda x: target_ccode in x)]
        test_desc = df[df['area_id'] == X_test_target_country['area_id'].iloc[index]].iloc[0]['desc']
    else:
        print(df)
        print(X_test['area_id'].iloc[index])
        test_desc = df[df['area_id'] == X_test['area_id'].iloc[index]].iloc[0]['desc']
        
    # choose in-context samples
    scored_df = candidate_df[candidate_df['score'] != -1]        
    if len(scored_df) < 3 and len(scored_df) > 0:
        add_example_df = scored_df
        example_df = pd.concat([example_df, add_example_df], axis=0)
    elif len(scored_df) >= 3:
        quantile_list = [0.0, 0.5, 1.0]        
        selected_df_list = []
        selected_index_list = []
        for quant_val in quantile_list:
            selected_df = scored_df[scored_df['score'] == scored_df['score'].quantile(quant_val, interpolation='lower')]
            selected_df = selected_df.sample(1)
            selected_df_list.append(selected_df)
            selected_index_list.append(selected_df.index[0])
        remaining_df = scored_df.drop(selected_index_list, axis=0).reset_index(drop=True)
        # print(remaining_df)
        if len(remaining_df) > 5 and not addr_only:
            extracted_vals = []  
            # print(df)          
            for remaining_area_id in remaining_df['area_id']:
                # print(df['area_id'])   
                # print(remaining_area_id)   
                # print(df[df['area_id'] == remaining_area_id])
                desc = df[df['area_id'] == remaining_area_id].iloc[0]['desc']
                extracted_vals.append(np.array([float(val) for val in re.findall(r"[-+]?(?:\d*\.*\d+)", '\n'.join(desc.split('\n')[1:]))]))
            extracted_vals = np.stack(extracted_vals)
            # print(extracted_vals)
            target_val = np.array([float(val) for val in re.findall(r"[-+]?(?:\d*\.*\d+)", '\n'.join(test_desc.split('\n')[1:]))])

           
            scaler = StandardScaler()
            extracted_vals_normalized = scaler.fit_transform(extracted_vals)
            target_val_normalized = scaler.transform(target_val.reshape(1, -1))
            sample_dist = np.linalg.norm(extracted_vals_normalized - target_val_normalized, axis=1)
            selected_neighbors = sample_dist.argsort()[:2]
            for neighbor in selected_neighbors:
                selected_df_list.append(pd.DataFrame(remaining_df.iloc[neighbor]).T)
            
        add_example_df = pd.concat(selected_df_list, axis=0)
        # example_df = pd.concat([example_df, add_example_df], axis=0)
        # print(selected_df_list)
        example_df = pd.concat([example_df, add_example_df], axis=0)
        # print(example_df)
    example_desc_list = []
    for example_area_id in example_df['area_id']:
        # print(example_area_id)
        # print(df['area_id'])
        # print(df[df['area_id'] == example_area_id])
        example_desc = df[df['area_id'] == example_area_id].iloc[0]['desc']
        example_desc_list.append(example_desc)
    
    example_df['desc'] = example_desc_list    
    # example_desc = '\n\n'.join(['. '.join(df[df['area_id'] == row['area_id']].iloc[0].desc.split('\n')) + f'.\n{target_question}\nAnswer: ' + str(round(row.score, 3)) for _, row in example_df.iterrows()])
    if classification:
        example_desc = '\n'.join(['. '.join(df[df['area_id'] == row['area_id']].iloc[0].desc.split('\n')) + f'.\nAnswer: ' + row.score for _, row in example_df.iterrows()])
    else:
        example_desc = '\n'.join(['. '.join(df[df['area_id'] == row['area_id']].iloc[0].desc.split('\n')) + f'.\nAnswer: ' + str(round(row.score, 3)) for _, row in example_df.iterrows()])
        
        
    # Serialize the dataframe
    target_desc = '. '.join(test_desc.split('\n'))
    # target_desc += f'.\n{target_question}'
    # if target_ccode =='PRK':
    #     target_desc += f'\nImportant considerations: North Korea\'s electricity supply is controlled by authorities and may focus on idolization projects, not accurately reflecting actual economic activity.
    # Nightlight intensity data does not capture daytime economic activities.'
    target_desc += f'\nAnswer: '
    
    example_desc = target_question +'\n\n' +example_desc
    fill_in_dict = {
        "<EXAMPLES>": example_desc,
        "<TARGET>": target_desc
    }
    template = fill_in_templates(fill_in_dict, template)
    return template



def get_prompt_for_inference_in_context_class(df, df_record, index, X_train, y_train, X_test, target_question, target_ccode=False, addr_only = False, classification = False):        
    template = "<EXAMPLES>\n\n<TARGET>"
                
    # choose in-context samples
    X_train_all = X_train.copy()
    print(X_train_all)
    print(y_train)
    print(y_train.to_numpy().ravel())
    X_train_all['score'] = y_train.to_numpy().ravel()
    
    example_df = X_train_all
    candidate_df = df_record.copy()
    if target_ccode:
        X_test_target_country = X_test.loc[X_test['area_id'].apply(lambda x: target_ccode in x)]
        test_desc = df[df['area_id'] == X_test_target_country['area_id'].iloc[index]].iloc[0]['desc']
    else:
        test_desc = df[df['area_id'] == X_test['area_id'].iloc[index]].iloc[0]['desc']
        
    # choose in-context samples
    '''
    scored_df = candidate_df[candidate_df['score'] != -1]        
    if len(scored_df) < 3 and len(scored_df) > 0:
        add_example_df = scored_df
        example_df = pd.concat([example_df, add_example_df], axis=0)
    elif len(scored_df) >= 3:
        add_example_df = scored_df.iloc[-3:]
        example_df = pd.concat([example_df, add_example_df], axis=0)
        # print(example_df)
    '''
    example_desc_list = []
    for example_area_id in example_df['area_id']:
        # print(example_area_id)
        # print(df['area_id'])
        # print(df[df['area_id'] == example_area_id])
        example_desc = df[df['area_id'] == example_area_id].iloc[0]['desc']
        example_desc_list.append(example_desc)
    
    example_df['desc'] = example_desc_list    
    # example_desc = '\n\n'.join(['. '.join(df[df['area_id'] == row['area_id']].iloc[0].desc.split('\n')) + f'.\n{target_question}\nAnswer: ' + str(round(row.score, 3)) for _, row in example_df.iterrows()])
    if classification:
        example_desc = '\n'.join(['. '.join(df[df['area_id'] == row['area_id']].iloc[0].desc.split('\n')) + f'.\nAnswer: ' + row.score for _, row in example_df.iterrows()])
    else:
        example_desc = '\n'.join(['. '.join(df[df['area_id'] == row['area_id']].iloc[0].desc.split('\n')) + f'.\nAnswer: ' + str(round(row.score, 3)) for _, row in example_df.iterrows()])
        
    # target_question : 250508 // only for 'PL_SK_CZ_HU_GRDP_adm2_9549_o1mini_landuse_wo_neighbor_paragraph_output'
    ccode_metadata = ""
    test_area_id = df[df['area_id'] == X_test['area_id'].iloc[index]].iloc[0]['area_id']
    print(test_area_id)
    if 'PL' in test_area_id:
        ccode_metadata += "As supporting information for solving the following issue, Poland has 79 'Districts', which represent the second highest postal level of geography in the country. Poland's geographical hierarchy is structured as follows: Country > Provinces > Districts. Please respond to the following question regarding the prediction of indicators for 'Districts'."
    elif 'CZ' in test_area_id:
        ccode_metadata += "As supporting information for solving the following issue, Czech repulic has 14 'Regions', which represent the highest administrative level of geography in the country. Czech republic's geographical hierarchy is structured as follows: Country > Regions > Districts. Please respond to the following question regarding the prediction of indicators for 'Regions'."
    elif 'SK' in test_area_id:
        ccode_metadata += "As supporting information for solving the following issue, Slovakia has 8 'Regions', named Kraje, which represent the highest administrative level of geography in the country. Slovakia's geographical hierarchy is structured as follows: Country > Regions > Districts. Please respond to the following question regarding the prediction of indicators for 'Regions'."
    elif 'HU' in test_area_id:
        ccode_metadata += "As supporting information for solving the following issue, Hungary has 20 'Counties', named Megyék, which represent the second administrative level of geography in the country. Hungary's geographical hierarchy is structured as follows: Country > Regions > Counties. Please respond to the following question regarding the prediction of indicators for 'Counties'."
    else:
        ccode_metadata += ""
    
    # Serialize the dataframe
    target_desc = '. '.join(test_desc.split('\n'))
    # target_desc += f'.\n{target_question}'
    # if target_ccode =='PRK':
    #     target_desc += f'\nImportant considerations: North Korea\'s electricity supply is controlled by authorities and may focus on idolization projects, not accurately reflecting actual economic activity.
    # Nightlight intensity data does not capture daytime economic activities.'
    target_desc += f'\nAnswer: '
    
    example_desc = ccode_metadata +'\n\n' +target_question +'\n\n' +example_desc
    fill_in_dict = {
        "<EXAMPLES>": example_desc,
        "<TARGET>": target_desc
    }
    template = fill_in_templates(fill_in_dict, template)
    return template

def get_prompt_for_inference_in_context_logprobs_class(df, df_record, index, X_train, y_train, X_test, target_question, target_ccode=False, addr_only = False, classification = False):        
    template = "<EXAMPLES>\n\n<TARGET>"
                
    # choose in-context samples
    X_train_all = X_train.copy()
    print(X_train_all)
    print(y_train)
    print(y_train.to_numpy().ravel())
    X_train_all['score'] = y_train.to_numpy().ravel()
    
    example_df = X_train_all
    candidate_df = df_record.copy()
    if target_ccode:
        X_test_target_country = X_test.loc[X_test['area_id'].apply(lambda x: target_ccode in x)]
        test_desc = df[df['area_id'] == X_test_target_country['area_id'].iloc[index]].iloc[0]['desc']
    else:
        test_desc = df[df['area_id'] == X_test['area_id'].iloc[index]].iloc[0]['desc']    
    example_desc_list = []
    for example_area_id in example_df['area_id']:
        # print(example_area_id)
        # print(df['area_id'])
        # print(df[df['area_id'] == example_area_id])
        example_desc = df[df['area_id'] == example_area_id].iloc[0]['desc']
        example_desc_list.append(example_desc)
    
    example_df['desc'] = example_desc_list    
    # example_desc = '\n\n'.join(['. '.join(df[df['area_id'] == row['area_id']].iloc[0].desc.split('\n')) + f'.\n{target_question}\nAnswer: ' + str(round(row.score, 3)) for _, row in example_df.iterrows()])
    if classification:
        example_desc = '\n'.join(['. '.join(df[df['area_id'] == row['area_id']].iloc[0].desc.split('\n')) + f'.\nAnswer: ' + row.score for _, row in example_df.iterrows()])
    else:
        example_desc = '\n'.join(['. '.join(df[df['area_id'] == row['area_id']].iloc[0].desc.split('\n')) + f'.\nAnswer: ' + str(round(row.score, 3)) for _, row in example_df.iterrows()])
        
        
    # Serialize the dataframe
    target_desc = '. '.join(test_desc.split('\n'))
    # target_desc += f'.\n{target_question}'
    # if target_ccode =='PRK':
    #     target_desc += f'\nImportant considerations: North Korea\'s electricity supply is controlled by authorities and may focus on idolization projects, not accurately reflecting actual economic activity.
    # Nightlight intensity data does not capture daytime economic activities.'
    target_desc += f'\nAnswer: '
    
    example_desc = target_question +'\n\n' +example_desc
    fill_in_dict = {
        "<EXAMPLES>": example_desc,
        "<TARGET>": target_desc
    }
    template = fill_in_templates(fill_in_dict, template)
    return template



def parse_and_select(answers, version='1'):
    ans_dict = {}
    for answer in answers:
        answer = '1. ' + '1. '.join(answer.split('1. ')[1:]).strip()
        answer = answer.replace('**', '')
        if version == '1':
            parsed_ans = []
            for ans in answer.split('\n'):
                if '):' in ans:
                    parsed_ans.append(ans[3:].split('):')[0].strip() + ')')
                else:
                    parsed_ans.append(ans[3:].strip())                
        elif version == '2':
            parsed_ans = [ans[3:].strip() for ans in answer.split('The total list of selected modules')[1].split('\n')[1:]]
        else:
            assert(0)
                        
        for parsed in parsed_ans:
            if len(parsed) <= 3:
                continue
                
            if 'Loc' not in parsed:
                continue
                
            # Handle exceptional cases here
            if 'get_aggregate_neighbor_info' in parsed:
                parsed = parsed.replace('Func=', '')
            elif 'count_area' in parsed:
                parsed = parsed.replace('Class=', '')
                                
            if parsed not in [*ans_dict]:
                ans_dict[parsed] = 1
            else:
                ans_dict[parsed] += 1
                
    selected_modules = np.array([*ans_dict.keys()])[np.array([*ans_dict.values()]) >= len(answers) // 3]
    return selected_modules

def save_dataset(file_path, info_path, description, answer):
    message_lst = []
    for i in range(len(description)):
        message_lst.append({
            "messages": [
                {"role": "user", "content": description[i]}, 
                {"role": "assistant", "content": str(round(answer[i], 3))}
            ]
        })
        
    with open(file_path, 'w', encoding='utf-8') as f:
        for entry in message_lst:
            json.dump(entry, f, ensure_ascii=False)
            f.write('\n')
            
    fileinfo = openai.File.create(
        file=open(file_path, 'rb'),
        purpose='fine-tune'
    )

    with open(info_path, "w") as json_file:
        json.dump(fileinfo, json_file)

    return fileinfo['id']

def random_selection_nightlight(nl_json_path, n_shot, seed, target_ccode =False):
    #return : dictionary with {'selected_area_id':1, 'not_selected_area_id'}
    np.random.seed(seed)
    with open(nl_json_path,'r') as f:
        data = json.load(f)
    print(data)
    if target_ccode:
        area_id_list = [i for i in data.keys() if target_ccode in i]
    else:
        area_id_list = list(data.keys())
    area_id_list.sort(key=lambda area_id: data[area_id]['nightlight']['Nightlight_Average']['val'])
    
    n = int((len(area_id_list) / n_shot))
    n_partition = [area_id_list[i:i+n] for i in range(0, len(area_id_list), n)]
    if len(n_partition) > n_shot:
        n_end = n_partition.pop()
        n_partition[-1]+=n_end
    n_random_selection = [np.random.choice(x) for x in n_partition]
    return n_random_selection