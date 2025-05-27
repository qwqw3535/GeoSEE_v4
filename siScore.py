import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import torch.backends.cudnn as cudnn
import pandas as pd
from torch.utils.data import DataLoader
from utils.graph import *
from utils.siScore_utils import *
from utils.parameters import *
from itertools import permutations
import os
import math
# os.environ["CUDA_VISIBLE_DEVICES"]= "2,3"
shot_median = []
classes = {
    0: 'Very Low',
    1: 'Low',
    2: 'Medium',
    4: 'Very High',
    3: 'High',
}
def make_data_loader(cluster_dir, batch_sz):
    cluster_dataset = VectorDataset(args.dir_name, cluster_dir)
    cluster_loader = torch.utils.data.DataLoader(cluster_dataset, batch_size=batch_sz, shuffle=True, num_workers=4, drop_last=True)
    return cluster_loader
    
    
def generate_loader_dict(total_list, unified_cluster_list, batch_sz):
    loader_dict = {}
    for cluster_id in total_list:
        cluster_loader = make_data_loader([cluster_id], batch_sz)
        loader_dict[cluster_id] = cluster_loader        
    
    for cluster_tuple in unified_cluster_list:
        cluster_loader = make_data_loader(cluster_tuple, batch_sz)
        for cluster_num in cluster_tuple:
            loader_dict[cluster_num] = cluster_loader
    return loader_dict


def deactivate_batchnorm(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_parameters()
            m.eval()
            with torch.no_grad():
                m.weight.fill_(1.0)
                m.bias.zero_()
                

def train(args, epoch, model, optimizer, loader_list, cluster_path_list, device):
    global shot_median
    model.train()
    # Deactivate the batch normalization before training
    if torch.cuda.device_count() > 1:
        deactivate_batchnorm(model.module)
    else:
        deactivate_batchnorm(model)
    
    train_loss = AverageMeter()
    reg_loss = AverageMeter()
    
    # For each cluster route
    path_idx = 0
    avg_loss = 0
    shot_loss = AverageMeter()
    count = 0
    for cluster_path in cluster_path_list:
        path_idx += 1
#         dataloaders = []
        dataloaders = [(cid, loader_list[cid]) for cid in cluster_path]
#         for cluster_id in cluster_path:
#             dataloaders.append(loader_list[cluster_id])
        for batch_idx, data in enumerate(zip(*[loader for _, loader in dataloaders])):
            cluster_num = len(data)
            data_zip = torch.cat(data, 0).to(device)
            classes = []
            
            for i, features in enumerate(data):
                cluster_id = dataloaders[i][0]
                class_tensor = torch.full((features.shape[0],), cluster_id, dtype=torch.long)
                classes.append(class_tensor)
            
            classes = torch.cat(classes, 0)
            # Generating Score
            scores = model(data_zip).squeeze()
#             scores = torch.clamp(scores, min=0, max=1)
            score_list = torch.split(scores, args.batch_sz, dim = 0)
            
            # Standard deviation as a loss
            loss_var = torch.zeros(1).to(device)
            for score in score_list:
                loss_var += score.var()
            loss_var /= len(score_list)
            
            # gt랑 비교
            loss_shot = torch.zeros(1).to(device)
            for score, cls in zip(scores, classes):
#                 weight = 1.0 / ((math.log(shot_median[cls.item()]) + 1e-6))
#                 loss_shot += weight * (score - shot_median[cls.item()])**2
                loss_shot += (score - shot_median[cls.item()])**2

            # Differentiable Ranking with sigmoid function
            rank_matrix = torch.zeros((args.batch_sz, cluster_num, cluster_num)).to(device)
            for itertuple in list(permutations(range(cluster_num), 2)):
                score1 = score_list[itertuple[0]]
                score2 = score_list[itertuple[1]]
                diff = args.lamb * (score2 - score1)
                results = torch.sigmoid(diff)
                rank_matrix[:, itertuple[0], itertuple[1]] = results
                rank_matrix[:, itertuple[1], itertuple[0]] = 1 - results

            rank_predicts = rank_matrix.sum(1)
            temp = torch.Tensor(range(cluster_num))
            target_rank = temp.unsqueeze(0).repeat(args.batch_sz, 1).to(device)

            # Equivalent to spearman rank correlation loss
            loss_train = ((rank_predicts - target_rank)**2).mean()
            
            loss = loss_train + loss_var * args.alpha + loss_shot
            if args.batch_sz == 1:
                loss = loss_train + loss_shot
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss.update(loss_train.item(), args.batch_sz)
            reg_loss.update(loss_var.item(), args.batch_sz)
            shot_loss.update(loss_shot.item(), args.batch_sz)
            avg_loss += loss.item()
            count += 1

            # Print status
            if batch_idx % 10 == 0:
                print('Epoch: [{epoch}][{path_idx}][{elps_iters}] '
                      'Train loss: {train_loss.val:.4f} ({train_loss.avg:.4f}) '
                      'Reg loss: {reg_loss.val:.4f} ({reg_loss.avg:.4f}) '
                      'shot_loss: {shot_loss.val:.4f} ({shot_loss.avg:.4f})'.format(
                          epoch=epoch, path_idx=path_idx, elps_iters=batch_idx, train_loss=train_loss, reg_loss=reg_loss, shot_loss=shot_loss))
                
    return avg_loss / count
   

def main(args):
    global shot_median
    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # Input example
    cluster_number = args.cluster_num
    
    # Graph generation mode
    if args.graph_config:
        graph_config = args.graph_config  
    else:
        raise ValueError
    
    if args.exp_name == '':
        exp_name = args.llm
    else:
        exp_name = args.exp_name
        
    # Dataloader definition   
    start, end, partial_order, cluster_unify = graph_process(graph_config)
    loader_list = generate_loader_dict(range(cluster_number), cluster_unify, args.batch_sz)
    cluster_graph = generate_graph(partial_order, cluster_number)
    cluster_path_list = cluster_graph.printPaths(start, end)
    print("Cluster_path: ", cluster_path_list)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    
    input_dim = len(pd.read_csv(args.dir_name).columns) - 2
    model = nn.Sequential(
        nn.Linear(input_dim, 128),
        nn.ReLU(),
        nn.Linear(128, 1)
    )
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model) 
        cudnn.benchmark = True

    # model.load_state_dict(torch.load(args.pretrained_path), strict = True)
    # model.module.fc = nn.Sequential(nn.Linear(512, 1))
    model.to(device)
    
    if args.shot == None:
        csv_name = args.dir_name.split('/')[-1].replace(args.llm, 'o1-mini').replace('addr_', '').replace('random_', '')
        indicator = csv_name.split('PL_SK_CZ_HU_')[-1].split('_classification')[0]
        shot_df = pd.read_csv(f'shot/{indicator}/v4_classification/{csv_name}')
    else:
        shot_df = pd.read_csv(args.shot)
        
    min_val = shot_df['ground_val'].min()
    max_val = shot_df['ground_val'].max()
    shot_df['ground_val_norm'] = (shot_df['ground_val'] - min_val) / (max_val - min_val)

    for i in range(5):
        temp = shot_df[shot_df['score']==classes[i]]
        shot_median.append(temp['ground_val_norm'].median())
    
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
    # print("Pretrained net load finished")
    
    
    best_loss = float('inf')
    if args.load == False:    
        for epoch in range(args.epochs):          
            loss = train(args, epoch, model, optimizer, loader_list, cluster_path_list, device)

            if epoch % 10 == 0 and epoch != 0:                
                if best_loss > loss:
                    print("state saving...")
                    state = {
                        'model': model.state_dict(),
                        'loss': loss
                    }
                    if not os.path.isdir(f'checkpoint/{exp_name}'):
                        os.mkdir(f'checkpoint/{exp_name}')
                    torch.save(state, './checkpoint/{}/{}'.format(exp_name, args.name))
                    best_loss = loss
                    print("best loss: %.4f\n" % (best_loss))

if __name__ == "__main__":
    args = siScore_parser()
    main(args)