import os
import pandas as pd
import numpy as np
import json
import re
from modules import *
from tqdm import tqdm
from utils import query_gpt, parse_and_select, read_config, login_gis_portal, random_selection_nightlight
from utils import get_prompt_for_extracting_modules, get_prompt_for_inference_in_context_class
from collections import Counter
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor

#functions

def normalize_points(points):
    """0~1 스케일로 정규화"""
    points = np.array(points)
    return (points - points.min()) / (points.max() - points.min())

def simulate_k_partition_with_ids(values, ids, k=5, n_trials=10000, seed=42):
    assert len(values) == len(ids) == 2 * k, "length mismatch!"
    
    # 1. 정규화
    normed = (np.array(values) - min(values)) / (max(values) - min(values))

    # 2. 시뮬레이션
    np.random.seed(seed)
    counter = Counter()
    total_trials = 0
    attempts = 0

    with tqdm(total=n_trials, desc="시뮬레이션 진행 중") as pbar:
        while total_trials < n_trials:
            attempts += 1
            cuts = np.sort(np.random.rand(k - 1))
            cuts = np.concatenate([[0], cuts, [1]])

            group_counts = np.histogram(normed, bins=cuts)[0]

            if np.any(group_counts == 0):
                continue

            counter[tuple(group_counts)] += 1
            total_trials += 1
            pbar.update(1)

    print(f"\n✅ 시뮬레이션 완료: {total_trials} 유효 / {attempts} 시도 (성공률 {total_trials/attempts:.2%})")
    return counter

def map_ids_to_partition(items, partition):
    grouped = []
    idx = 0
    for size in partition:
        grouped.append(items[idx:idx+size])
        idx += size
    return grouped

# n_samples : given shots, make a split sample for each class_iter
def sample_partitions_with_details_df_list(counter, values, ids, score_names, n_samples=10, seed=42):
    random.seed(seed)
    dist_list = list(counter.keys())
    weights = [counter[d] for d in dist_list]
    total = sum(weights)
    probs = [w / total for w in weights]

    samples = random.choices(dist_list, weights=probs, k=n_samples)
    df_list = []

    # area_id → ground_val 매핑
    ground_map = dict(zip(ids, values))

    for dist in samples:
        grouped_ids = map_ids_to_partition(ids, dist)
        grouped_vals = map_ids_to_partition(values, dist)
        rows = []

        for score_label, id_group, val_group in zip(score_names, grouped_ids, grouped_vals):
            avg_val = sum(val_group) / len(val_group)

            for area_id in id_group:
                rows.append({
                    'area_id': area_id,
                    'score': score_label,
                    'sample_val': round(avg_val, 4),
                    'ground_val': ground_map[area_id]
                })

        df_list.append(pd.DataFrame(rows))

    return df_list

#parameters
parser = argparse.ArgumentParser()

parser.add_argument('--ablation', required=False, help='ablation',default = '')
parser.add_argument('--target_var', required=True, help='testing variable')
parser.add_argument('--gpt_version', required=True, help='gpt version')
parser.add_argument('--exp_name', default='', help='experiment name')
parser.add_argument('--dir_name', default='', help='directory name')

args = parser.parse_args()
if args.ablation:
    ablation = '_' + args.ablation
else:
    ablation = ''

target_var = args.target_var
target_var_desc = ''
gpt_version = args.gpt_version
file_path_var = target_var
col_var = target_var
adm_offset = ''
dir_name = 'v4_classification'
exp_name = '_landuse'
target_ccode = 'PL_SK_CZ_HU'
country_name = 'the Czech Republic, Hungary, Poland, and Slovakia'
# adm_offset = '_adm1'

# parameters remain unchanged
# simulation parameter
n_simulation_trials = 10000
iter_init_offset = 42

# set overall progress iteration
iter_num = 3

#class & iteration
classes = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
num_classes = len(classes)
shot = 2 * num_classes
class_iter_num = 10

#if just for print out shots
just_shot = False

#fewshot target question
fewshot_target_question = f"Infer {target_var_desc} from the given location's description by leveraging multiple data aspects from the provided dataset and incorporating any additional insights you have about the region. You are provided with qualitative ground truths for five classes—Very Low, Low, Medium, High, and Very High—that serve as benchmarks for other regions of {country_name} and neighboring countries. Based on these qualitative benchmarks, please deliver your final answer as one of these five classes."

ask_num = 10
api_key = ''
if gpt_version=='deepseek-chat' or gpt_version=='deepseek-reasoner':
    api_key=''

prompt_version = '2'

timeline ='9549'
# target_var = 'GRDP'
# target_var_desc = 'a regional GDP (in milion dollars)'
if target_var =='GRDP':
    target_var_desc = 'a regional GDP (in milion euros)'
elif target_var =='population':
    target_var_desc = 'the population'
elif target_var == 'work_accidents':
    target_var_desc = 'the number of accidents at work'
elif target_var == 'male_life_expectancy_deviation':
    target_var_desc = 'the deviation of male life expectancy at birth from the average(i.e., observed value - average)'
elif target_var == 'female_life_expectancy_deviation':
    target_var_desc = 'the deviation of female life expectancy at birth from the average(i.e., observed value - average)'
elif target_var == 'life_expectancy_male':
    target_var_desc = 'the male life expectancy at birth'
elif target_var == 'life_expectancy_female':
    target_var_desc = 'the female life expectancy at birth'
elif target_var == 'life_expectancy':
    target_var_desc = 'the life expectancy at birth'
elif target_var == 'LPR':
    target_var_desc = 'the labour force participation rate'
elif target_var == 'ELP':
    target_var_desc = 'the population of elderly people (aged 65 and older)' 
elif target_var == 'HER':
    target_var_desc = 'the proportion of highly-educated individuals (at least university graduates)'
else:
    print('no target var desc')
    assert(1==0)

result_df_overall = pd.DataFrame(columns=['spearman_corr_mean','spearman_corr_std','pearson_corr_mean', 'pearson_corr_std', 'r2_mean','r2_std', 'test_rmse_mean','test_rmse_std'], index=['HU','SK','PL','CZ'])

fewshot_target_question = f"Infer {target_var_desc} from the given location's description by leveraging multiple data aspects from the provided dataset and incorporating any additional insights you have about the region. You are provided with qualitative ground truths for five classes—Very Low, Low, Medium, High, and Very High—that serve as benchmarks for other regions of {country_name} and neighboring countries. Based on these qualitative benchmarks, please deliver your final answer as one of these five classes."
gpt_name = gpt_version.replace('-', '')
df = pd.read_csv(f"./extracted_paragraphs/landuse/{target_var}/{target_ccode}_{target_var}{ablation}_adm2_{timeline}_o1mini{exp_name}_wo_neighbor_paragraph_output.csv")

random_seed = range(iter_init_offset,iter_init_offset+iter_num)
ground_truth = pd.read_csv(f'./data/label/{target_ccode}_{file_path_var}{adm_offset}.csv', encoding='utf-8')[['area_id', 'area_name',col_var]]
COT_reasonings = pd.DataFrame(columns=["desc"])

# manage directories
# reads
# f'./data/proxy/{target_ccode}_nightlight{adm_offset}.json',shot,random_seed[idx]
# f'./extracted_paragraphs/landuse/{target_var}/{target_ccode}_{target_var}{ablation}_adm2_{timeline}_{gpt_name}{exp_name}_wo_neighbor_paragraph_output.csv'
# writes
# f"./COT_reasoning/{target_var}/{dir_name}/{target_var}/{target_ccode}_{target_var}_{save_name}{ablation}_{gpt_version}_iter_num_{idx}_class_sample_{i}.csv"
# f'./results/{target_var}/{dir_name}/{target_ccode}_{target_var}_{save_name}{ablation}_{gpt_version}_{len(X_train)}shots_iter_num_{idx}_class_sample_{class_iter}.csv'

# make directory
os.makedirs(f'./COT_reasoning/{target_var}/{dir_name}/',exist_ok=True)
os.makedirs(f'./results/{target_var}/{dir_name}/',exist_ok=True)

#main code
shot_simulation_counter = None
#임시로 iter는 1번
print(fewshot_target_question)

# 1. 개별 API 호출을 병렬화할 함수 (innermost loop)
def process_single_inference(i, X_test, df, df_record, X_train, y_train, target_question, gpt_version, api_key):
    # 추론 프롬프트 생성
    inference_prompt = get_prompt_for_inference_in_context_class(
        df, df_record, i, X_train, y_train, X_test,
        target_question, target_ccode=False, addr_only=False, classification=True
    )
    # API 호출 (query_gpt는 API 호출 함수)
    model_output, reasoning_result = query_gpt(
        [inference_prompt], api_key, temperature=0, max_completion_tokens=30000,
        tqdm_disable=True, model=gpt_version
    )[0]
    model_output = model_output.replace(',', '')
    reasoning_result = reasoning_result.replace(',', '')
    
    try:
        candidates = re.findall(r'\b(Very Low|Low|Medium|High|Very High)\b', model_output)[-1]
        answer = candidates
    except Exception as e:
        answer = ''
    
    return i, reasoning_result, answer

# 2. 하나의 class_iter에 대해 처리하는 함수 (내부의 for i loop 병렬화)
def process_class_iter(idx, class_iter, shot_df_list, ground_truth, col_var, target_question,
                       gpt_version, api_key, target_var, target_ccode, save_name, ablation, shot,
                       dir_name, just_shot, base_df):
    # train/test 분할
    train_area_id_list = list(shot_df_list[class_iter]['area_id'])
    X_train = shot_df_list[class_iter]['area_id']
    y_train = shot_df_list[class_iter]['score']
    X_test = ground_truth[ground_truth['area_id'].isin(train_area_id_list)==False]['area_id']
    y_test = ground_truth[ground_truth['area_id'].isin(train_area_id_list)==False][col_var]
    
    # shots 폴더에 해당 class_iter 결과 저장 (먼저 snapshot)
    shot_csv_path = f'./shots/{target_var}/{dir_name}/{target_ccode}_{target_var}_{save_name}{ablation}_{gpt_version}_{shot}shots_iter_num_{idx}_class_sample_{class_iter}.csv'
    os.makedirs(os.path.dirname(shot_csv_path), exist_ok=True)
    shot_df_list[class_iter].to_csv(shot_csv_path, index=False, header=True)
    
    if just_shot:
        return
    
    # 재설정 (인덱스 초기화)
    X_train = X_train.reset_index().drop('index', axis=1)
    X_test = X_test.reset_index().drop('index', axis=1)
    y_train = y_train.reset_index().drop('index', axis=1)
    y_test = y_test.reset_index().drop('index', axis=1)
    
    # 결과 기록용 DataFrame: 각 테스트 area_id에 대해 score와 ground truth 기록
    df_record = pd.DataFrame(X_test.copy())
    df_record['score'] = [-1 for _ in range(len(df_record))]
    df_record['ground_truth'] = y_test.to_numpy()
    
    # innermost loop 병렬화: 각 테스트 샘플에 대해 API 호출
    # base_df는 추론 프롬프트 생성에 필요한 데이터 프레임(예: 전체 df)
    COT_reasonings = pd.DataFrame(columns=["desc"])

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(
                process_single_inference, i, X_test, base_df, df_record, X_train, y_train,
                target_question, gpt_version, api_key
            )
            for i in range(len(X_test))
        ]
        
        results = []
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc=f'Iter {idx} - Class sample {class_iter}'):
            results.append(future.result())
        
        # 정렬: 결과를 원래 인덱스 순서대로 정렬
        results.sort(key=lambda x: x[0])
        
        for i, reasoning_result, answer in results:
            COT_reasonings.loc[i] = reasoning_result
            df_record.loc[i, 'score'] = answer
            
    
    # 결과 저장 (results 폴더)
    COT_reasonings_csv_path = f'./COT_reasoning/{target_var}/{dir_name}/{target_ccode}_{target_var}_{save_name}{ablation}_{gpt_version}_iter_num_{idx}_class_sample_{class_iter}.csv'
    result_csv_path = f'./results/{target_var}/{dir_name}/{target_ccode}_{target_var}_{save_name}{ablation}_{gpt_version}_{shot}shots_iter_num_{idx}_class_sample_{class_iter}.csv'
    os.makedirs(os.path.dirname(COT_reasonings_csv_path), exist_ok=True)
    COT_reasonings.to_csv(COT_reasonings_csv_path, index=True, header=True)
    os.makedirs(os.path.dirname(result_csv_path), exist_ok=True)
    df_record.to_csv(result_csv_path, index=False, header=True)

def process_iteration(idx, ground_truth, col_var, target_question,
                      gpt_version, api_key, target_var, target_ccode,
                      save_name, ablation, shot, dir_name, just_shot,
                      n_samples, class_iter_num, base_df):
    """
    idx에 해당하는 반복 작업: random_ids 추출, shot_df_list 생성, 그리고 각 class_iter 처리
    """
    shot_df_list = []
    for class_iter in range(class_iter_num):
        shot_csv_path = f'./shots/{target_var}/{dir_name}/{target_ccode}_{target_var}_{save_name}{ablation}_o1-mini_{shot}shots_iter_num_{idx}_class_sample_{class_iter}.csv'
        try:
            df = pd.read_csv(shot_csv_path)
            shot_df_list.append(df)
            print(f"Successfully loaded: {shot_csv_path}")
        except Exception as e:
            print(f"Failed to load {shot_csv_path}. Error: {e}")
    
    pre_exist_flag = (len(shot_df_list)==class_iter_num)
    
    if pre_exist_flag:
        random_ids = list(shot_df_list[0]['area_id'])
    else:
        # 예시: random_ids 추출 및 shot_df_list 생성
        random_ids = random_selection_nightlight(f'./data/proxy/{target_ccode}_nightlight{adm_offset}.json', shot, random_seed[idx])
    
    ground_truth_shot = ground_truth[ground_truth['area_id'].isin(random_ids)]
    ground_truth_shot_sorted = ground_truth_shot.sort_values(by=target_var)
    
    X_train_all = ground_truth_shot_sorted['area_id']
    y_train_all = ground_truth_shot_sorted[col_var]
    
    shot_ids = list(X_train_all)
    shot_values = list(y_train_all)
    shot_classes = classes  # 미리 정의된 클래스 리스트
    
    k = num_classes         # 미리 정의된 클래스 수
    n_trials = n_simulation_trials  # 시뮬레이션 횟수
    
    # 각 idx마다 shot_simulation_counter 생성 (seed 적용)
    if not pre_exist_flag:
        shot_sim_counter = simulate_k_partition_with_ids(shot_values, shot_ids, k=k, n_trials=n_trials, seed=random_seed[idx])
        shot_df_list = sample_partitions_with_details_df_list(shot_sim_counter, shot_values, shot_ids, shot_classes, n_samples=n_samples, seed=random_seed[idx])
    
    # 각 class_iter (샘플 분할) 처리: 내부에서 ThreadPoolExecutor로 병렬화됨
    for class_iter in range(class_iter_num):
        process_class_iter(idx, class_iter, shot_df_list, ground_truth, col_var, target_question,
                           gpt_version, api_key, target_var, target_ccode, save_name, ablation,
                           shot, dir_name, just_shot, base_df)
    # 필요에 따라 각 idx마다 반환값을 만들거나 로그를 남길 수 있음.
    return f"Iteration {idx} 완료"

# ProcessPoolExecutor를 사용하여 idx 반복(outer loop) 병렬 실행
def run_all_iterations(iter_num, ground_truth, col_var, target_question,
                       gpt_version, api_key, target_var, target_ccode,
                       save_name, ablation, shot, dir_name, just_shot,
                       n_samples, class_iter_num, base_df):
    results = []
    with ProcessPoolExecutor(max_workers=min(iter_num, 4)) as executor:
        futures = [executor.submit(
                        process_iteration,
                        idx, ground_truth, col_var, target_question,
                        gpt_version, api_key, target_var, target_ccode,
                        save_name, ablation, shot, dir_name, just_shot,
                        n_samples, class_iter_num, base_df)
                   for idx in range(iter_num)]
        for future in as_completed(futures):
            try:
                res = future.result()
                results.append(res)
                print(res)
            except Exception as e:
                print(f"Error in iteration: {e}")
    return results

# 예시 사용
# (아래 변수들은 모두 미리 정의되어 있어야 함: iter_num, ground_truth, col_var, fewshot_target_question, gpt_version, api_key, 
# target_var, target_ccode, random_seed, shot, num_classes, n_simulation_trials, class_iter_num, just_shot, base_df, adm_offset, classes)
run_all_iterations(
    iter_num=iter_num,
    ground_truth=ground_truth,
    col_var=col_var,
    target_question=fewshot_target_question,
    gpt_version=gpt_version,
    api_key=api_key,
    target_var=target_var,
    target_ccode=target_ccode,
    save_name='classification',
    ablation=ablation,
    shot=shot,
    dir_name='v4_classification',
    just_shot=just_shot,
    n_samples=class_iter_num,
    class_iter_num=class_iter_num,
    base_df=df  # 예: 전체 데이터프레임
)