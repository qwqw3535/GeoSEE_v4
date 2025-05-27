import csv
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset
import pandas as pd
import warnings
from utils.siScore_utils import *
from utils.parameters import *
from skimage import io, transform
from skimage.color import rgba2rgb
import os

warnings.filterwarnings("ignore")

args = extract_score_parser()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

input_dim = len(pd.read_csv(args.test).columns) - 2
model = nn.Sequential(
    nn.Linear(input_dim, 128),
    nn.ReLU(),
    nn.Linear(128, 1)
)
calibration_model = nn.Sequential(
    nn.Linear(1, 16),
    nn.ReLU(),
    nn.Linear(16, 1)
)

if args.exp_name == '':
    exp_name = args.llm
else:
    exp_name = args.exp_name



if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
    calibration_model = nn.DataParallel(calibration_model)
if args.shot == None:
    csv_name = args.dir_name.split('/')[-1].replace(args.llm, 'o1-mini').replace('addr_', '').replace('random_', '')
    indicator = csv_name.split('PL_SK_CZ_HU_')[-1].split('_classification')[0]
    shot_df = pd.read_csv(f'shot/{indicator}/v4_classification/{csv_name}')
else:
    shot_df = pd.read_csv(args.shot)
    indicator = args.model.split('/')[-1].split('_iter')[0]
model_path = args.model
# calibration_model_path = model_path.replace(indicator, indicator+'_calibration')

model.load_state_dict(torch.load(model_path)['model'], strict=True)
model.to(device)

# calibration_model.load_state_dict(torch.load(calibration_model_path)['model'], strict=True)
# calibration_model.to(device)
print("Load Finished")
min_val = shot_df['ground_val'].min()
max_val = shot_df['ground_val'].max()

class VectorDataset(Dataset):
    def __init__(self, csv_name):
        df = pd.read_csv(csv_name)
        self.names = df['area_id'].values
        self.vectors = df.drop(columns=["area_id", "cluster"]).values
        self.vectors = torch.tensor(self.vectors, dtype=torch.float32)
    def __len__(self):
        return len(self.vectors)

    def __getitem__(self, idx):
        return self.vectors[idx], self.names[idx]

# To enforce the batch normalization during the evaluation
model.eval()    
    
test_dataset = VectorDataset(args.test)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=4, drop_last=False)
print(len(test_dataset))

# df2 = pd.read_csv('./data/{}'.format(args.test))
df2 = pd.read_csv(args.test)
df2['score'] = -1
df2['original_score'] = -1
result=[]
with torch.no_grad():
    for batch_idx, (data1, name) in enumerate(test_loader):
        data1 = data1.to(device)
        scores = model(data1).squeeze()
        if scores.dim() == 0:
            scores = scores.unsqueeze(0)
            
        count = 0
        for each_name in name:
            if scores.dim() == 0:
                score = scores.item()
            else:
                score = scores[count].cpu().data.numpy()
            df2.loc[df2['area_id'] == each_name, 'score'] = score
            df2.loc[df2['area_id'] == each_name, 'original_score'] = score * (max_val - min_val) + min_val
            count += 1

model_name = (args.model.split('/')[-1])[:-len(".ckpt")]
os.makedirs(f"./result/{exp_name}/{indicator}/", exist_ok=True)
df2.to_csv(f"./result/{exp_name}/{indicator}/siScore_{model_name}.csv", index=False)
