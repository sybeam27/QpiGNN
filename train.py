import sys
sys.path('./utills')
import os
import re
import random
import argparse
import pickle
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx
import torch.nn.functional as F
import numpy as np
import pandas as pd
import seaborn as sns
import json
from datetime import datetime
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix, coo_matrix
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PyPDF2 import PdfMerger
from torch.distributions.normal import Normal
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import from_networkx, to_networkx
from torch_geometric.nn import SAGEConv, GATConv, GCNConv, GraphSAGE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch_geometric.transforms import RandomNodeSplit
from utills.function import set_seed, generate_graph_data, generate_noisy_graph_data, load_county_graph_data, load_twitch_graph_data, \
            load_wiki_graph_data, load_trans_graph_data, create_ba_graph_pyg, create_er_graph_pyg, create_grid_graph_pyg, create_tree_graph_pyg, \
            normalize, split_graph_data, split_cp_graph_data, evaluate_model_performance, sort_by_y, coverage_width
from utills.model import GQNN_R, GQNN_N, GNN_CP, BayesianGNN, MCDropoutGNN, GQNN, QRLoss, RQRLoss, GQNNLoss, GQNNLoss2

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ["TORCH_USE_CUDA_DSA"] = "1"

set_seed(1127)  

parser = argparse.ArgumentParser(description='Train GQNN Model')
parser.add_argument('--gpu', type=str, default="0", help='gpu_number')
parser.add_argument('--date', type=float, default=0000, help='index_dates')
parser.add_argument('--pdf', type=bool, default=True, help='pdf_save')

parser.add_argument('--dataset', type=str, default="basic", help='dataset_name')
parser.add_argument('--nodes', type=float, default=1000, help='num_nodes')
parser.add_argument('--noise', type=float, default=0.3, help='noise_level')
parser.add_argument('--model', type=str, default="GQNN", help='model_name')
parser.add_argument('--device', type=str, default='cuda:0')

parser.add_argument('--hidden_dim', type=float, default=64, help='hidden_dim')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning_rate')
parser.add_argument('--weight', type=float, default=1e-3, help='weight_decay')
parser.add_argument('--epochs', type=float, default=500, help='num_epochs')
parser.add_argument('--runs', type=int, default=10, help='num_runs')

# parser.add_argument('--tau_low', type=float, default=0.05, help='tau_low')
# parser.add_argument('--tau_upper', type=float, default=0.95, help='tau_upper')
parser.add_argument('--target_coverage', type=float, default=0.9, help='target_coverage')

parser.add_argument('--lambda_factor', type=float, default=1, help='lambda_factor')
parser.add_argument('--num_samples', type=float, default=100, help='num_samples')
parser.add_argument('--dropout', type=float, default=0.2, help='dropout')

args = parser.parse_args()

device = torch.device(args.device)

if args.dataset != '':
    if args.dataset_name == 'basic':
        graph_data = generate_graph_data(num_nodes=args.num_nodes)
    elif args.dataset_name in ('gaussian', 'uniform', 'outlier', 'edge'):
        graph_data = generate_noisy_graph_data(num_nodes=args.num_nodes, noise_type=args.dataset_name, noise_level=args.noise_level)
    elif args.dataset_name in ('education', 'election', 'income', 'unemployment'):
        graph_data = load_county_graph_data(args.dataset_name, 2012)
    elif args.dataset_name in ('DE', 'ENGB', 'ES', 'FR', 'PTBR', 'RU'):
        graph_data = load_twitch_graph_data(args.dataset_name)
    elif args.dataset_name in ('chameleon', 'crocodile', 'squirrel'):
        graph_data = load_wiki_graph_data(args.dataset_name)
    elif args.dataset_name in ('Anaheim', 'ChicagoSketch'):
        graph_data = load_trans_graph_data(args.dataset_name)
    elif args.dataset_name == 'BA':
        graph_data = create_ba_graph_pyg(n=args.num_nodes)
    elif args.dataset_name == 'ER':
        graph_data = create_er_graph_pyg(n=args.num_nodes)
    elif args.dataset_name == 'grid':
        graph_data = create_grid_graph_pyg()
    elif args.dataset_name == 'tree':
        graph_data = create_tree_graph_pyg()
    
# split data & normalize
train_data, test_data = split_graph_data(graph_data, test_ratio=0.2)
train_min, train_max, y_min, y_max = train_data.x.min(), train_data.x.max(), train_data.y.min(), train_data.y.max()
train_data.x, test_data.x, train_data.y, test_data.y= normalize(train_data.x, train_min, train_max), normalize(test_data.x, train_min, train_max), normalize(train_data.y, y_min, y_max), normalize(test_data.y, y_min, y_max)

print(f"Train data: {train_data.x.shape[0]} nodes, {train_data.edge_index.shape[1]} edges")
print(f"Train edge_index 최대값: {train_data.edge_index.max().item()}")
print(f"Test data: {test_data.x.shape[0]} nodes, {test_data.edge_index.shape[1]} edges")
print(f"Test edge_index 최대값: {test_data.edge_index.max().item()}")

if args.model == 'CP':
    cp_train_data, calibration_data = split_cp_graph_data(train_data, cali_ratio=0.2)

    print(f"Train data: {cp_train_data.x.shape[0]} nodes, {cp_train_data.edge_index.shape[1]} edges")
    print(f"Train edge_index 최대값: {cp_train_data.edge_index.max().item()}")
    print(f"Calibration data: {calibration_data.x.shape[0]} nodes, {calibration_data.edge_index.shape[1]} edges")
    print(f"Calibration edge_index 최대값: {calibration_data.edge_index.max().item()}")
    print(f"Test data: {test_data.x.shape[0]} nodes, {test_data.edge_index.shape[1]} edges")
    print(f"Test edge_index 최대값: {test_data.edge_index.max().item()}")

# result folder & file
root_dir = f"./result"
csv_dir = os.path.join(root_dir, 'eval')
os.makedirs(csv_dir, exist_ok=True)
pdf_dir = os.path.join(root_dir, 'img')
os.makedirs(pdf_dir, exist_ok=True)

file_name = args.dataset + '_' + args.model
if args.model == 'GQNN':
    file_name += f'_lf({args.lambda_factor})'

# Training..
print('-' * 40, f'{args.model}: {args.dataset_name} training is starting... ', '-' * 40)

in_dim = train_data.x.shape[1]
train_data = train_data.to(device)
pastel_colors = sns.color_palette('Dark2')

results = {}

for run in tqdm(range(args.runs)):
    result_this_run = {}
    
    if args.model == 'SQR':
        tau_low = (1 - args.target_coverage)/2
        tau_upper = 1 - tau_low
        color = pastel_colors[0]
        model = GQNN_R(in_dim=in_dim, hidden_dim=args.hidden_dim).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight)
        criterion = QRLoss()
        
        for epoch in tqdm(range(args.epochs)):
            model.train()
            optimizer.zero_grad()
            
            taus = torch.rand(train_data.x.size(0), 1, dtype=torch.float32, device=device)
            preds = model(train_data.x, train_data.edge_index, taus)
            loss = criterion(preds, train_data.y, taus)
                
            loss.backward()
            optimizer.step()
                        
        print('-' * 40, f'{args.model}: {args.dataset_name} Train Evaluation... ', '-' * 40)
        model.eval()
        tau_lows = torch.full((train_data.x.size(0), 1), tau_low, dtype=torch.float32, device=device)
        tau_uppers = torch.full((train_data.x.size(0), 1), tau_upper, dtype=torch.float32, device=device)

        with torch.no_grad():
            train_low_preds = model(train_data.x, train_data.edge_index, tau_lows).cpu().numpy()
            train_upper_preds = model(train_data.x, train_data.edge_index, tau_uppers).cpu().numpy()
            train_targets = train_data.y.cpu().numpy()
        train_eval = evaluate_model_performance(train_low_preds, train_upper_preds, train_targets, target=args.target_coverage)
        result_this_run['train_metrics'] = train_eval
        
        print('-' * 40, f'{args.model}: {args.dataset_name} Test Evaluation... ', '-' * 40)
        test_data = test_data.to(device)
        tau_lows = torch.full((test_data.x.size(0), 1), tau_low, dtype=torch.float32, device=device)
        tau_uppers = torch.full((test_data.x.size(0), 1), tau_upper, dtype=torch.float32, device=device)

        with torch.no_grad():
            test_low_preds = model(test_data.x, test_data.edge_index, tau_lows).cpu().numpy()
            test_upper_preds = model(test_data.x, test_data.edge_index, tau_uppers).cpu().numpy()
            test_targets = test_data.y.cpu().numpy()
        test_eval = evaluate_model_performance(test_low_preds, test_upper_preds, test_targets, target=args.target_coverage)
        result_this_run['test_metrics'] = test_eval
        
    elif args.model == 'RQR':
        color = pastel_colors[1]
        model = GQNN_N(in_dim=in_dim, hidden_dim=args.hidden_dim).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight)
        criterion = RQRLoss(target=args.target_coverage, lambda_factor=args.lambda_factor)  # lambda_factor 고정함
        
        for epoch in tqdm(range(args.epochs)):
            model.train()
            optimizer.zero_grad()
            
            preds = model(train_data.x, train_data.edge_index)
            loss = criterion(preds, train_data.y)
                
            loss.backward()
            optimizer.step()

        print('-' * 40, f'{args.model}: {args.dataset_name} Train Evaluation... ', '-' * 40)
        model.eval()

        with torch.no_grad():
            train_preds = model(train_data.x, train_data.edge_index)
            train_low_preds = train_preds[:, 0].cpu().numpy()
            train_upper_preds = train_preds[:, 1].cpu().numpy()
            train_targets = train_data.y.cpu().numpy()
        train_eval = evaluate_model_performance(train_low_preds, train_upper_preds, train_targets, target=args.target_coverage)
        result_this_run['train_metrics'] = train_eval
        
        print('-' * 40, f'{args.model}: {args.dataset_name} Test Evaluation... ', '-' * 40)
        test_data = test_data.to(device)

        with torch.no_grad():
            test_preds = model(test_data.x, test_data.edge_index)
            test_low_preds = test_preds[:, 0].cpu().numpy()
            test_upper_preds = test_preds[:, 1].cpu().numpy()
            test_targets = test_data.y.cpu().numpy()
        test_eval = evaluate_model_performance(test_low_preds, test_upper_preds, test_targets, target=args.target_coverage)
        result_this_run['test_metrics'] = test_eval
        
    elif args.model == 'CP':
        color = pastel_colors[2]
        cp_train_data = cp_train_data.to(device)
        
        model = GNN_CP(in_dim=in_dim, hidden_dim=args.hidden_dim).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight)
        
        epochs = []
        losses = []
        for epoch in tqdm(range(args.epochs)):
            model.train()
            optimizer.zero_grad()
            
            preds = model(cp_train_data.x, cp_train_data.edge_index)
            loss = F.mse_loss(preds, cp_train_data.y)
            
            loss.backward()
            optimizer.step()
            
            epochs.append(epoch+1)
            losses.append(loss.item())
        
        print('-' * 40, f'{args.model}: {args.dataset_name} Train Evaluation... ', '-' * 40)
        model.eval()
        calibration_data = calibration_data.to(device)
        test_data = test_data.to(device)

        with torch.no_grad():
            preds_cal = model(calibration_data.x, calibration_data.edge_index)
            preds_train = model(train_data.x, train_data.edge_index).cpu().numpy()
            preds_test = model(test_data.x, test_data.edge_index).cpu().numpy()

        conformal_scores = torch.abs(calibration_data.y- preds_cal).cpu().numpy()
        q_hat = np.quantile(conformal_scores, args.target_coverage)

        train_low_preds = preds_train - q_hat
        train_upper_preds = preds_train + q_hat
        train_targets = train_data.y.cpu().numpy()
        train_eval = evaluate_model_performance(train_low_preds, train_upper_preds, train_targets, target=args.target_coverage)
        result_this_run['train_metrics'] = train_eval
        
        print('-' * 40, f'{args.model}: {args.dataset_name} Test Evaluation... ', '-' * 40)
        test_low_preds = preds_test - q_hat
        test_upper_preds = preds_test + q_hat
        test_targets = test_data.y.cpu().numpy()
        test_eval = evaluate_model_performance(test_low_preds, test_upper_preds, test_targets, target=args.target_coverage)
        result_this_run['test_metrics'] = test_eval
        
    elif args.model == 'BNN':
        color = pastel_colors[3]
        model = BayesianGNN(in_dim=in_dim, hidden_dim=args.hidden_dim).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight)
        
        epochs = []
        losses = []
        for epoch in tqdm(range(args.epochs)):
            model.train()
            optimizer.zero_grad()
            
            preds = model(train_data.x, train_data.edge_index)
            loss = F.mse_loss(preds, train_data.y.squeeze())
                
            loss.backward()
            optimizer.step()
                    
            epochs.append(epoch+1)
            losses.append(loss.item())
                    
        print('-' * 40, f'{args.model}: {args.dataset_name} Train Evaluation... ', '-' * 40)
        model.eval()

        preds_list = []
        with torch.no_grad():
            for _ in range(args.num_samples):
                preds = model(train_data.x, train_data.edge_index)  
                preds_list.append(preds.cpu().numpy())

        preds_array = np.array(preds_list)  # (num_samples, num_nodes)
        mean_preds = preds_array.mean(axis=0)  # 평균 예측값
        std_preds = preds_array.std(axis=0)    # 표준편차

        if args.target_coverage == 0.9:
            t = 1.645
        elif args.target_coverage == 0.95:
            t = 1.96
        # 80%: 1.28 / 90%: 1.645 / 95%: 1.96 / 99%: 2.576

        train_low_preds = mean_preds - t * std_preds  
        train_upper_preds = mean_preds + t * std_preds 
        train_targets = train_data.y.cpu().numpy()
        train_eval = evaluate_model_performance(train_low_preds, train_upper_preds, train_targets, target=args.target_coverage)
        result_this_run['train_metrics'] = train_eval
        
        print('-' * 40, f'{args.model}: {args.dataset_name} Test Evaluation... ', '-' * 40)
        test_data = test_data.to(device)

        preds_list = []
        with torch.no_grad():
            for _ in range(args.num_samples):
                preds = model(test_data.x, test_data.edge_index)  # Bayesian Sampling
                preds_list.append(preds.cpu().numpy())

        preds_array = np.array(preds_list)  
        mean_preds = preds_array.mean(axis=0)  
        std_preds = preds_array.std(axis=0)    

        test_low_preds = mean_preds - t * std_preds 
        test_upper_preds = mean_preds + t * std_preds 
        test_targets = test_data.y.cpu().numpy()
        test_eval = evaluate_model_performance(test_low_preds, test_upper_preds, test_targets, target=args.target_coverage)
        result_this_run['test_metrics'] = test_eval
        
    elif args.model == 'MC':
        color = pastel_colors[4]
        model = MCDropoutGNN(in_dim=in_dim, hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight)

        epochs = []
        losses = []
        for epoch in tqdm(range(args.epochs)):
            model.train()
            optimizer.zero_grad()
            
            preds = model(train_data.x, train_data.edge_index, training=True)
            loss = F.mse_loss(preds, train_data.y.squeeze())
                
            loss.backward()
            optimizer.step()
                    
            epochs.append(epoch+1)
            losses.append(loss.item())
                        
        print('-' * 40, f'{args.model}: {args.dataset_name} Train Evaluation... ', '-' * 40)
        model.eval()
        
        preds_list = []
        with torch.no_grad():
            for _ in range(args.num_samples):
                preds = model(train_data.x, train_data.edge_index, training=True)  # Dropout 유지
                preds_list.append(preds.cpu().numpy())

        preds_array = np.array(preds_list)  # (num_samples, num_nodes)
        mean_preds = preds_array.mean(axis=0)  # 평균 예측값
        std_preds = preds_array.std(axis=0)    # 표준편차
        
        if args.target_coverage == 0.9:
            t = 1.645
        elif args.target_coverage == 0.95:
            t = 1.96
            
        train_low_preds = mean_preds - t * std_preds 
        train_upper_preds = mean_preds + t * std_preds 
        train_targets = train_data.y.cpu().numpy()
        train_eval = evaluate_model_performance(train_low_preds, train_upper_preds, train_targets, target=args.target_coverage)
        result_this_run['train_metrics'] = train_eval
        
        print('-' * 40, f'{args.model}: {args.dataset_name} Test Evaluation... ', '-' * 40)
        test_data = test_data.to(device)
        
        preds_list = []
        with torch.no_grad():
            for _ in range(args.num_samples):
                preds = model(test_data.x, test_data.edge_index, training=True)  # Dropout 유지
                preds_list.append(preds.cpu().numpy())

        preds_array = np.array(preds_list)  # (num_samples, num_nodes)
        mean_preds = preds_array.mean(axis=0)  # 평균 예측값
        std_preds = preds_array.std(axis=0)    # 표준편차

        test_low_preds = mean_preds - t * std_preds  # 95% 신뢰구간 하한
        test_upper_preds = mean_preds + t * std_preds  # 95% 신뢰구간 상한
        test_targets = test_data.y.cpu().numpy()
        test_eval = evaluate_model_performance(test_low_preds, test_upper_preds, test_targets, target=args.target_coverage)
        result_this_run['test_metrics'] = test_eval
        
    elif args.model == 'GQNN':
        color = pastel_colors[6]
        model = GQNN(in_dim=in_dim, hidden_dim=args.hidden_dim).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight)
        criterion = GQNNLoss(target_coverage=args.target_coverage, lambda_factor=args.lambda_factor)
        
        for epoch in tqdm(range(args.epochs)):
            model.train()
            optimizer.zero_grad()
            
            preds_low, preds_upper = model(train_data.x, train_data.edge_index)
            loss = criterion(preds_low, preds_upper, train_data.y)
                
            loss.backward()
            optimizer.step()
            
            cvg, wdt = coverage_width(train_data.y, preds_low, preds_upper)
            
        print('-' * 40, f'{args.model}: {args.dataset_name} Train Evaluation... ', '-' * 40)
        model.eval()

        with torch.no_grad():
            preds_low, preds_upper = model(train_data.x, train_data.edge_index)    
            train_low_preds = preds_low.cpu().numpy()
            train_upper_preds = preds_upper.cpu().numpy()
            train_targets = train_data.y.cpu().numpy()
        train_eval = evaluate_model_performance(train_low_preds, train_upper_preds, train_targets, target=args.target_coverage)
        result_this_run['train_metrics'] = train_eval
        
        print('-' * 40, f'{args.model}: {args.dataset_name} Test Evaluation... ', '-' * 40)
        test_data = test_data.to(device)

        with torch.no_grad():
            preds_low, preds_upper = model(test_data.x, test_data.edge_index)    
            test_low_preds = preds_low.cpu().numpy()
            test_upper_preds = preds_upper.cpu().numpy()
            test_targets = test_data.y.cpu().numpy()
        test_eval = evaluate_model_performance(test_low_preds, test_upper_preds, test_targets, target=args.target_coverage)
        result_this_run['test_metrics'] = test_eval

    elif args.model == 'GQNN_2':
        color = pastel_colors[7]
        model = GQNN(in_dim=in_dim, hidden_dim=args.hidden_dim).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight)
        criterion = GQNNLoss2(target_coverage=args.target_coverage, lambda_width=args.lambda_factor) # 고정
        
        for epoch in tqdm(range(args.epochs)):
            model.train()
            optimizer.zero_grad()
            
            preds_low, preds_upper = model(train_data.x, train_data.edge_index)
            loss = criterion(preds_low, preds_upper, train_data.y)
                
            loss.backward()
            optimizer.step()
            
            cvg, wdt = coverage_width(train_data.y, preds_low, preds_upper)
            
        print('-' * 40, f'{args.model}: {args.dataset_name} Train Evaluation... ', '-' * 40)
        model.eval()

        with torch.no_grad():
            preds_low, preds_upper = model(train_data.x, train_data.edge_index)    
            train_low_preds = preds_low.cpu().numpy()
            train_upper_preds = preds_upper.cpu().numpy()
            train_targets = train_data.y.cpu().numpy()
        train_eval = evaluate_model_performance(train_low_preds, train_upper_preds, train_targets, target=args.target_coverage)
        result_this_run['train_metrics'] = train_eval
        
        print('-' * 40, f'{args.model}: {args.dataset_name} Test Evaluation... ', '-' * 40)
        test_data = test_data.to(device)

        with torch.no_grad():
            preds_low, preds_upper = model(test_data.x, test_data.edge_index)    
            test_low_preds = preds_low.cpu().numpy()
            test_upper_preds = preds_upper.cpu().numpy()
            test_targets = test_data.y.cpu().numpy()
        test_eval = evaluate_model_performance(test_low_preds, test_upper_preds, test_targets, target=args.target_coverage)
        result_this_run['test_metrics'] = test_eval
        
    if args.pdf:
        existing_pdf = os.path.join(pdf_dir, f"{args.model}_eval_plots.pdf")
        new_pdf = os.path.join(pdf_dir, "new_train_plot.pdf")
        merged_pdf = os.path.join(pdf_dir, "merged_train_plots.pdf")

        x_st, y_st, (low_r_st, upper_r_st) = sort_by_y(train_data.x, train_data.y, train_low_preds, train_upper_preds)
        
        with PdfPages(new_pdf) as pdf:
            fig, axes = plt.subplots(1, 2, figsize=(16, 4))  
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M") 
            plt.suptitle(f"Model: {args.model}, Dataset: {args.dataset_name}, Time: {timestamp} (Train)", fontsize=12, fontweight='bold')

            axes[0].scatter(range(len(x_st)), y_st, alpha=0.3, color='blue', label="True Values", s=15)
            axes[0].fill_between(range(len(x_st)), low_r_st, upper_r_st, color=color, alpha=0.5)
            axes[0].set_xlabel("Sotred Node Index") 
            axes[0].set_ylabel("Values")

            axes[1].scatter(range(len(x_st)), train_targets, alpha=0.3, color='blue', label="True Values", s=15)
            axes[1].fill_between(range(len(x_st)), train_low_preds.squeeze(), train_upper_preds.squeeze(), color=color, alpha=0.5)
            axes[1].set_xlabel("Node Index")  
            axes[1].set_ylabel("Values") 
            
            pdf.savefig(fig)
            plt.close(fig)
            
        if os.path.exists(existing_pdf):
            merger = PdfMerger()
            merger.append(existing_pdf)
            merger.append(new_pdf)
            merger.write(merged_pdf)
            merger.close()

            os.replace(merged_pdf, existing_pdf)
            os.remove(new_pdf)
        else:
            os.replace(new_pdf, existing_pdf)
            
        x_st, y_st, (low_r_st, upper_r_st) = sort_by_y(test_data.x, test_data.y, test_low_preds, test_upper_preds)
        
        with PdfPages(new_pdf) as pdf:
            fig, axes = plt.subplots(1, 2, figsize=(16, 4))  
            plt.suptitle(f"Model: {args.model}, Dataset: {args.dataset_name}, Time: {timestamp} (Test)", fontsize=12, fontweight='bold')

            axes[0].scatter(range(len(x_st)), y_st, alpha=0.3, color='blue', label="True Values", s=15)
            axes[0].fill_between(range(len(x_st)), low_r_st, upper_r_st, color=color, alpha=0.5)
            axes[0].set_xlabel("Sotred Node Index") 
            axes[0].set_ylabel("Values")

            axes[1].scatter(range(len(x_st)), test_targets, alpha=0.3, color='blue', label="True Values", s=15)
            axes[1].fill_between(range(len(x_st)), test_low_preds.squeeze(), test_upper_preds.squeeze(), color=color, alpha=0.5)
            axes[1].set_xlabel("Node Index")  
            axes[1].set_ylabel("Values") 
            
            pdf.savefig(fig)
            plt.close(fig)
            
        if os.path.exists(existing_pdf):
            merger = PdfMerger()
            merger.append(existing_pdf)
            merger.append(new_pdf)
            merger.write(merged_pdf)
            merger.close()

            os.replace(merged_pdf, existing_pdf)
            os.remove(new_pdf)
        else:
            os.replace(new_pdf, existing_pdf)
    
    results[run] = result_this_run
    print(f'Finished training {run} run!')


print('Saving results to', './result/eval' + file_name +'.pkl')
with open('./pred/' + file_name +'.pkl', 'wb') as f:
    pickle.dump(results, f)