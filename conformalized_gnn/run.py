import argparse
import os.path as osp
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import pandas as pd
import copy
import os
import wandb
import pickle
import sys 
sys.path.append('..')
from utills.function import generate_graph_data, generate_noisy_graph_data, load_county_graph_data, load_twitch_graph_data, \
            load_wiki_graph_data, load_trans_graph_data, create_ba_graph_pyg, create_er_graph_pyg, create_grid_graph_pyg, create_tree_graph_pyg
            
from torch_geometric.logging import log
from torch_geometric.data import Data
from scipy.stats import pearsonr
import torch_geometric.transforms as T

from conformalized_gnn.model import GNN, ConfGNN, ConfMLP
from conformalized_gnn.calibrator import TS, VS, ETS, CaGCN, GATS
from conformalized_gnn.conformal import run_conformal_regression

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="basic", help='dataset_name')
parser.add_argument('--nodes', type=float, default=1000, help='num_nodes')
parser.add_argument('--noise', type=float, default=0.3, help='noise_level')

parser.add_argument('--hidden_channels', type=int, default=64)
parser.add_argument('--model', type=str, default='GCN', choices = ['GAT', 'GCN', 'GraphSAGE', 'SGC'])
parser.add_argument('--heads', type=int, default=1)
parser.add_argument('--aggr', type=str, default='sum')
parser.add_argument('--alpha', type=float, default=0.1)

parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--epochs', type=int, default=500)

parser.add_argument('--wandb', action='store_true', help='Track experiment')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--conformal_score', type=str, default='cqr', choices = ['aps', 'cqr'])

parser.add_argument('--conf_correct_model', type=str, default='gnn', choices = ['gnn', 'mlp', 'Calibrate', 'mcdropout', 'mcdropout_std', 'QR'])
parser.add_argument('--quantile', action='store_true', default = False)
parser.add_argument('--bnn', action='store_true', default = False)

parser.add_argument('--target_size', type=int, default=0)
parser.add_argument('--confnn_hidden_dim', type=int, default=64)
parser.add_argument('--confgnn_num_layers', type=int, default=1)
parser.add_argument('--confgnn_base_model', type=str, default='GCN', choices = ['GAT', 'GCN', 'GraphSAGE', 'SGC'])
parser.add_argument('--confgnn_lr', type=float, default=1e-3)

parser.add_argument('--tau', type=float, default=0.1)
parser.add_argument('--size_loss_weight', type=float, default=1)
parser.add_argument('--reg_loss_weight', type=float, default=1)

parser.add_argument('--not_save_res', action='store_true', default = False)
parser.add_argument('--num_runs', type=int, default=10)
parser.add_argument('--retrain', action='store_true', default = False)
parser.add_argument('--data_seed', type=int, default=0)
parser.add_argument('--cond_cov_loss', action='store_true', default = False)

parser.add_argument('--calib_fraction', type=float, default=0.5)


args = parser.parse_args()

global task
task = 'regression'
metric = 'eff_valid_cqr'
if args.conformal_score != 'cqr':
    raise ValueError('For regression task, the training conformal score should be cqr!')

if args.bnn:
    args.quantile = False
else:    
    args.quantile = True 
    
device = torch.device(args.device)

name = args.dataset + '_' + args.model
if args.alpha != 0.1:
    name += '_alpha_' + str(args.alpha)    
if args.bnn:
    name += '_bnn'

if args.wandb:
    wandb.init(project='ConformalGNN_' + args.dataset + '_' + args.model, name=name, config = args)
 
if args.dataset == 'basic':
    graph_data = generate_graph_data(num_nodes=args.nodes)
elif args.dataset in ('gaussian', 'uniform', 'outlier', 'edge'):
    graph_data = generate_noisy_graph_data(num_nodes=args.nodes, noise_type=args.dataset, noise_level=args.noise)
elif args.dataset in ('education', 'election', 'income', 'unemployment'):
    graph_data = load_county_graph_data(args.dataset, 2012)
elif args.dataset in ('DE', 'ENGB', 'ES', 'FR', 'PTBR', 'RU'):
    graph_data = load_twitch_graph_data(args.dataset)
elif args.dataset in ('chameleon', 'crocodile', 'squirrel'):
    graph_data = load_wiki_graph_data(args.dataset)
elif args.dataset in ('Anaheim', 'ChicagoSketch'):
    graph_data = load_trans_graph_data(args.dataset)
elif args.dataset == 'BA':
    graph_data = create_ba_graph_pyg(n=args.nodes)
elif args.dataset == 'ER':
    graph_data = create_er_graph_pyg(n=args.nodes)
elif args.dataset == 'grid':
    graph_data = create_grid_graph_pyg()
elif args.dataset == 'tree':
    graph_data = create_tree_graph_pyg()

def gaussian_nll_loss(mean, log_var, y_true):
    # Compute the negative log likelihood for a Gaussian distribution
    precision = torch.exp(-log_var)
    mse_loss = F.mse_loss(mean, y_true, reduction='none')
    nll_loss = 0.5 * (mse_loss * precision + log_var + torch.log(torch.tensor(2 * np.pi)))
    return torch.mean(nll_loss)

def train(epoch, model, data, optimizer, alpha):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index, data.edge_attr)

    if args.quantile:
        mid = out[:, 0][data.train_mask].reshape(-1,1)
        label = data.y[data.train_mask].reshape(-1,1)
        mse_loss = F.mse_loss(mid, label)
        low_bound = alpha/2
        upp_bound = 1 - alpha/2
        lower = out[:, 1][data.train_mask].reshape(-1,1)
        upper = out[:, 2][data.train_mask].reshape(-1,1)
        low_loss = torch.mean(torch.max((low_bound - 1) * (label - lower), low_bound * (label - lower)))
        upp_loss = torch.mean(torch.max((upp_bound - 1) * (label - upper), upp_bound * (label - upper)))
        loss = mse_loss + low_loss + upp_loss
    elif args.bnn:
        mu = out[:, 0][data.train_mask].reshape(-1,1)
        logvar = out[:, 1][data.train_mask].reshape(-1,1)
        loss = gaussian_nll_loss(mu, logvar, data.y[data.train_mask])
    else:
        loss = F.mse_loss(out[data.train_mask], data.y[data.train_mask])
        
    loss.backward()
    optimizer.step()
    
    if args.quantile:
        return (float(loss), mse_loss, low_loss, upp_loss)
    elif args.bnn:
        return float(loss)
    else:
        return float(loss)

@torch.no_grad()
def test(model, data, alpha, tau, target_size, size_loss = False):
    model.eval()
    
    if size_loss:
        pred_raw, ori_pred_raw = model(data.x, data.edge_index)
    else:
        pred_raw = model(data.x, data.edge_index)
        
    if args.quantile:
        pred = pred_raw[:, 0]
    elif args.bnn:
        pred = pred_raw[:, 0]
    else:
        pred = pred_raw
        
    accs = []
    for mask in [data.train_mask, data.valid_mask, data.calib_test_mask]:
        accs.append(pearsonr(pred[mask].detach().cpu().numpy().reshape(-1), 
                             data.y[mask].detach().cpu().numpy().reshape(-1))[0])
    if size_loss:
        if task == 'regression':
            if args.quantile:
                query_idx = np.where(data.valid_mask)[0]
                np.random.seed(0)
                np.random.shuffle(query_idx)

                train_train_idx = query_idx[:int(len(query_idx)/2)]
                train_calib_idx = query_idx[int(len(query_idx)/2):]
                
                n_temp = len(train_calib_idx)
                ### use only train_train nodes
                mid = pred_raw[:, 0][train_calib_idx].reshape(-1,1)
                label = data.y[train_calib_idx].reshape(-1,1)
                mse_loss = F.mse_loss(mid, label)
                low_bound = alpha/2
                upp_bound = 1 - alpha/2
                lower = pred_raw[:, 1][train_calib_idx].reshape(-1,1)
                upper = pred_raw[:, 2][train_calib_idx].reshape(-1,1)
                low_loss = torch.mean(torch.max((low_bound - 1) * (label - lower), low_bound * (label - lower)))
                upp_loss = torch.mean(torch.max((upp_bound - 1) * (label - upper), upp_bound * (label - upper)))
                
                ## CQR loss
                size_loss = 0
                lower_calib = pred_raw[:, 1][train_train_idx].reshape(-1,1)
                upper_calib = pred_raw[:, 2][train_train_idx].reshape(-1,1)
                label_calib = data.y[train_train_idx].reshape(-1,1)

                cal_scores = torch.maximum(label_calib-upper_calib, lower_calib-label_calib)
                # Get the score quantile
                qhat = torch.quantile(cal_scores, np.ceil((n_temp+1)*(1-alpha))/n_temp, interpolation='higher')
                size_loss = torch.mean(upper_calib + qhat - (lower_calib - qhat))
                pred_loss = mse_loss + low_loss + upp_loss
        elif args.bnn:
            raise ValueError('Not implemented....')
        else:
            out_softmax = F.softmax(pred_raw, dim = 1)
            query_idx = np.where(data.valid_mask)[0]
            np.random.seed(0)
            np.random.shuffle(query_idx)

            train_train_idx = query_idx[:int(len(query_idx)/2)]
            train_calib_idx = query_idx[int(len(query_idx)/2):]

            n_temp = len(train_calib_idx)
            q_level = np.ceil((n_temp+1)*(1-alpha))/n_temp

            tps_conformal_score = out_softmax[train_calib_idx][torch.arange(len(train_calib_idx)), data.y[train_calib_idx]]
            qhat = torch.quantile(tps_conformal_score, 1 - q_level, interpolation='higher')
            c = torch.sigmoid((out_softmax[train_train_idx] - qhat)/tau)
            size_loss = torch.mean(torch.relu(torch.sum(c, axis = 1) - target_size))
            
        return accs, pred_raw, size_loss.item()
    else:
        return accs, pred_raw  
    
def main(args):
    data = Data(x=graph_data.x, edge_index=graph_data.edge_index, y=graph_data.y)
    x = data.x
    y = data.y
    
    idx = np.array(range(len(y)))  
    np.random.seed(args.data_seed)
    np.random.shuffle(idx)
    split_res = np.split(idx, [int(0.5 * len(idx)), int(0.6 * len(idx)), len(idx)])
    train_idx, valid, calib_test = split_res[0], split_res[1], split_res[2]

    data.train_mask = np.array([False] * len(y)) 
    data.train_mask[train_idx] = True

    data.valid_mask = np.array([False] * len(y)) 
    data.valid_mask[valid] = True

    data.calib_test_mask = np.array([False] * len(y)) 
    data.calib_test_mask[calib_test] = True
    
    n_trials = 100
    n = min(1000, int(calib_test.shape[0]/2))
    alpha = args.alpha
    tau = args.tau
    target_size = args.target_size
    num_conf_layers = args.confgnn_num_layers
    base_model = args.confgnn_base_model
    optimal_examine_res = {}
    tau2res = {} 
    
    for run in tqdm(range(args.num_runs)):
        result_this_run = {}
        
        if args.quantile:
            if args.alpha == 0.1:
                model_checkpoint = './model/' + args.model + '_' + args.dataset + '_' + str(run+1) + '_quantile_0410.pt'
            else:
                model_checkpoint = './model/' + args.model + '_' + args.dataset + '_' + str(run+1) + '_quantile_' + str(args.alpha) + '_0410.pt'
        elif args.bnn:
            model_checkpoint = './model/' + args.model + '_' + args.dataset + '_' + str(run+1) + '_bnn_' + str(args.alpha) + '_0410.pt'
        else:
            model_checkpoint = './model/' + args.model + '_' + args.dataset + '_' + str(run+1) + '_0410.pt'
            
        if args.quantile:
            output_dim = 3
        elif args.bnn:
            output_dim = 2
        else:
            output_dim = 1
        num_features = x.shape[1]
        
        if (os.path.exists(model_checkpoint)) and (not args.retrain):
            print('loading saved base model...')
            model = torch.load(model_checkpoint, map_location = device)
            model, data = model.to(device), data.to(device)
            model.eval()
            pred = model(data.x, data.edge_index)
            best_model = model
            best_pred = pred
        else:
            print('training base model from scratch...')
            model = GNN(num_features, args.hidden_channels, output_dim, args.model, args.heads, args.aggr)    

            model, data = model.to(device), data.to(device)
            optimizer = torch.optim.Adam([
                dict(params=model.conv1.parameters(), weight_decay=5e-4),
                dict(params=model.conv2.parameters(), weight_decay=0)
            ], lr=args.lr)  # Only perform weight-decay on first convolution.

            best_val_acc = final_test_acc = 0
            for epoch in range(1, args.epochs + 1):
                loss = train(epoch, model, data, optimizer, alpha)
                if args.quantile:
                    mse = loss[1]
                    lower = loss[2]
                    upper = loss[3]
                    loss = loss[0]
                
                (train_acc, val_acc, tmp_test_calib_acc), pred = test(model, data, alpha, tau, target_size)
                if val_acc > best_val_acc:
                    #torch.save(best_model, model_checkpoint)
                    best_model = copy.deepcopy(model)
                    best_val_acc = val_acc
                    test_acc = tmp_test_calib_acc
                    best_pred = pred
                
            # torch.save(best_model, model_checkpoint)
            pred = best_pred

        (train_acc, val_acc, test_acc), _ = test(best_model, data, alpha, tau, target_size, size_loss = False)
            
        result_this_run['gnn'] = {}
        
        if args.bnn:
            pred = pred.detach().cpu().numpy()
            pred_all = pred[:, 0].reshape(-1)
            pred_logvar = pred[:, 1].reshape(-1)
            pred_std = np.sqrt(np.exp(pred_logvar))
            mu = pred_all
            pred = np.vstack([mu, mu-1.96 * pred_std,mu+1.96 * pred_std]).T        
    
        result_this_run['gnn']['CQR'] = run_conformal_regression(pred, data, n, alpha, calib_eval = False)
        
        condcov_epochs = []
        result_this_run['conf_gnn'] = {}
        if args.bnn:
            result_this_run['conf_gnn']['Raw'] = run_conformal_regression(pred, data, n, alpha, score = 'qr', calib_eval = False)        
        else:    
            model_to_correct = copy.deepcopy(model)
            if args.conf_correct_model == 'gnn':
                confmodel = ConfGNN(model_to_correct, data, args, num_conf_layers, base_model, output_dim, task).to(args.device)
            elif args.conf_correct_model == 'mlp':
                confmodel = ConfMLP(model_to_correct, data, output_dim, task).to(args.device)
            
            optimizer = torch.optim.Adam(confmodel.parameters(), weight_decay=5e-4, lr=args.confgnn_lr)  # Only perform weight-decay on first convolution.
            pred_loss_hist, size_loss_hist, cons_loss_hist, val_size_loss_hist = [], [], [], []
            best_size_loss = 10000
            best_val_acc = 0
            
            print('Starting topology-aware conformal correction...')
            for epoch in range(1, args.epochs + 1):  
                if (not args.conftr_holdout) and (not args.conftr_calib_holdout) and (not args.conftr_valid_holdout):
                    train_idx = np.where(data.train_mask)[0]
                    np.random.seed(epoch)
                    np.random.shuffle(train_idx)
                    train_train_idx = train_idx[:int(len(train_idx)/2)]
                    train_calib_idx = train_idx[int(len(train_idx)/2):]
                    train_test_idx = train_train_idx
                confmodel.train()
                optimizer.zero_grad()
                out, ori_out = confmodel(data.x, data.edge_index)
                
                if task == 'regression':
                    if args.quantile:
                        ### use only train_train nodes
                        mid = out[:, 0][train_train_idx].reshape(-1,1)
                        label = data.y[train_train_idx].reshape(-1,1)
                        mse_loss = F.mse_loss(mid, label)
                        low_bound = alpha/2
                        upp_bound = 1 - alpha/2
                        lower = out[:, 1][train_train_idx].reshape(-1,1)
                        upper = out[:, 2][train_train_idx].reshape(-1,1)
                        low_loss = torch.mean(torch.max((low_bound - 1) * (label - lower), low_bound * (label - lower)))
                        upp_loss = torch.mean(torch.max((upp_bound - 1) * (label - upper), upp_bound * (label - upper)))
                        pred_loss = mse_loss + low_loss + upp_loss

                        n_temp = len(train_calib_idx)
                        ## CQR loss
                        lower_calib = out[:, 1][train_calib_idx].reshape(-1,1)
                        upper_calib = out[:, 2][train_calib_idx].reshape(-1,1)
                        label_calib = data.y[train_calib_idx].reshape(-1,1)

                        cal_scores = torch.maximum(label_calib-upper_calib, lower_calib-label_calib)
                        # Get the score quantile
                        qhat = torch.quantile(cal_scores, np.ceil((n_temp+1)*(1-alpha))/n_temp, interpolation='higher')

                        lower_test = out[:, 1][train_test_idx].reshape(-1,1)
                        upper_test = out[:, 2][train_test_idx].reshape(-1,1)
                        
                        lower_deviate_loss = F.mse_loss(out[:, 1].reshape(-1,1), ori_out[:, 1].reshape(-1,1))
                        upper_deviate_loss = F.mse_loss(out[:, 2].reshape(-1,1), ori_out[:, 2].reshape(-1,1))
                                                
                        size_loss = torch.mean(upper_test + qhat - (lower_test - qhat))
                        
                        loss = pred_loss
                        
                        loss.backward()
                        optimizer.step()
                        
                        loss = float(loss)
                        pred_loss_hist.append(pred_loss.item())
                        size_loss_hist.append(size_loss.item())
                        
                        (train_acc, val_acc, tmp_test_calib_acc), pred, size_loss = test(confmodel, data, alpha, tau, target_size, size_loss = True)
                
                        eff_valid = run_conformal_regression(pred, data, n, alpha, validation_set = True)[1]
                        
                        val_size_loss_hist.append(size_loss)
                        if val_acc > best_val_acc:
                            best_val_acc = val_acc
                            test_acc = tmp_test_calib_acc
                            best_pred = pred  
                        
            result_this_run['conf_gnn'] = {}
            result_this_run['conf_gnn']['CQR'] = run_conformal_regression(best_pred, data, n, alpha, calib_eval = args.conftr_calib_holdout, calib_fraction = args.calib_fraction)
            result_this_run['conf_gnn']['eff_valid'] = run_conformal_regression(best_pred, data, n, alpha, validation_set = True)[1]

    tau2res[run] = result_this_run
    print('Finished training this run!')
    
    if not os.path.exists('./pred'):
        os.mkdir('./pred')
    if not args.not_save_res:
        print('Saving results to', './pred/' + name +'.pkl')
        with open('./pred/' + name +'.pkl', 'wb') as f:
            pickle.dump(tau2res, f)
            
main(args)
