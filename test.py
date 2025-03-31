import sys
sys.path.append('./utils')

import os
import re
import random
import argparse
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
            normalize, split_graph_data, evaluate_model_performance, sort_by_y, coverage_width
from utills.model import GQNN_R, GQNN_N, GNN_CP, BayesianGNN, MCDropoutGNN, GQNN, QRLoss, RQRLoss, GQNNLoss, GQNNLoss2

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ["TORCH_USE_CUDA_DSA"] = "1"

set_seed(1127)  

parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('--gpu', type=str, default="0", help='gpu_number')
parser.add_argument('--pdf_save', type=bool, default=False, help='pdf_save')
parser.add_argument('--dataset', type=str, default="basic", help='dataset_name')
parser.add_argument('--nodes', type=float, default=1000, help='num_nodes')
parser.add_argument('--noise', type=float, default=0.3, help='noise_level')
parser.add_argument('--model', type=str, default="GQNN", help='model_name')
parser.add_argument('--hidden_dim', type=float, default=64, help='hidden_dim')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning_rate')
parser.add_argument('--weight', type=float, default=1e-3, help='weight_decay')
parser.add_argument('--tau_low', type=float, default=0.05, help='tau_low')
parser.add_argument('--tau_upper', type=float, default=0.95, help='tau_upper')
parser.add_argument('--epochs', type=float, default=500, help='num_epochs')
parser.add_argument('--target_coverage', type=float, default=0.9, help='target_coverage')
parser.add_argument('--lambda_factor', type=float, default=1, help='lambda_factor')
parser.add_argument('--num_samples', type=float, default=100, help='num_samples')
parser.add_argument('--dropout', type=float, default=0.2, help='dropout')

args = parser.parse_args()
gpu_number = args.gpu
pdf_save  = args.pdf_save
dataset_name = args.dataset
num_nodes = args.nodes
noise_level = args.noise
model_name = args.model
hidden_dim = args.hidden_dim
learning_rate = args.learning_rate
weight = args.weight
tau_low = args.tau_low
tau_upper = args.tau_upper
num_epochs = args.epochs
target = args.target_coverage
lambda_factor = args.lambda_factor
num_samples = args.num_samples
dropout= args.dropout

# dataset
if dataset_name == 'basic':
    graph_data = generate_graph_data(num_nodes=num_nodes)
elif dataset_name in ('gaussian', 'uniform', 'outlier', 'edge'):
    graph_data = generate_noisy_graph_data(num_nodes=num_nodes, noise_type=dataset_name, noise_level=noise_level)
elif dataset_name in ('education', 'election', 'income', 'unemployment'):
    graph_data = load_county_graph_data(dataset_name, 2012)
elif dataset_name in ('DE', 'ENGB', 'ES', 'FR', 'PTBR', 'RU'):
    graph_data = load_twitch_graph_data(dataset_name)
elif dataset_name in ('chameleon', 'crocodile', 'squirrel'):
    graph_data = load_wiki_graph_data(dataset_name)
elif dataset_name in ('Anaheim', 'ChicagoSketch'):
    graph_data = load_trans_graph_data(dataset_name)
elif dataset_name == 'BA':
    graph_data = create_ba_graph_pyg(n=num_nodes)
elif dataset_name == 'ER':
    graph_data = create_er_graph_pyg(n=num_nodes)
elif dataset_name == 'grid':
    graph_data = create_grid_graph_pyg()
elif dataset_name == 'tree':
    graph_data = create_tree_graph_pyg()
    
# split data & normalize
train_data, test_data = split_graph_data(graph_data)

train_min = train_data.x.min()
train_max = train_data.x.max()
y_min = train_data.y.min()
y_max = train_data.y.max()

train_data.x = normalize(train_data.x, train_min, train_max)
test_data.x = normalize(test_data.x, train_min, train_max)
train_data.y = normalize(train_data.y, y_min, y_max)
test_data.y = normalize(test_data.y, y_min, y_max)

print(f"Train data: {train_data.x.shape[0]} nodes, {train_data.edge_index.shape[1]} edges")
print(f"Train edge_index 최대값: {train_data.edge_index.max().item()}")
print(f"Test data: {test_data.x.shape[0]} nodes, {test_data.edge_index.shape[1]} edges")
print(f"Test edge_index 최대값: {test_data.edge_index.max().item()}")

if model_name == 'CP-GNN':
    cp_train_data, calibration_data = split_graph_data(train_data)
    num_train = train_data.x.shape[0]
    num_calibration = int(num_train * 0.2)  # Calibration 비율 20%

    # 랜덤하게 인덱스 생성
    indices = torch.randperm(num_train)
    calibration_indices = indices[:num_calibration]
    cp_train_indices = indices[num_calibration:]

    # 노드 마스크 생성
    cp_train_mask = torch.zeros(num_train, dtype=torch.bool)
    calibration_mask = torch.zeros(num_train, dtype=torch.bool)
    cp_train_mask[cp_train_indices] = True
    calibration_mask[calibration_indices] = True

    # cp_train용 edge_index 필터링 및 재매핑
    cp_train_edge_mask = cp_train_mask[train_data.edge_index[0]] & cp_train_mask[train_data.edge_index[1]]
    cp_train_edge_index = train_data.edge_index[:, cp_train_edge_mask]
    cp_train_node_map = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(cp_train_indices)}
    cp_train_edge_index = torch.tensor(
        [[cp_train_node_map[idx.item()] for idx in cp_train_edge_index[0]],
        [cp_train_node_map[idx.item()] for idx in cp_train_edge_index[1]]],
        dtype=torch.long
    )

    # calibration용 edge_index 필터링 및 재매핑
    calibration_edge_mask = calibration_mask[train_data.edge_index[0]] & calibration_mask[train_data.edge_index[1]]
    calibration_edge_index = train_data.edge_index[:, calibration_edge_mask]
    calibration_node_map = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(calibration_indices)}
    calibration_edge_index = torch.tensor(
        [[calibration_node_map[idx.item()] for idx in calibration_edge_index[0]],
        [calibration_node_map[idx.item()] for idx in calibration_edge_index[1]]],
        dtype=torch.long
    )

    # cp_train_data 생성
    cp_train_data = Data(
        x=train_data.x[cp_train_indices],
        y=train_data.y[cp_train_indices],
        edge_index=cp_train_edge_index
    )

    # calibration_data 생성
    calibration_data = Data(
        x=train_data.x[calibration_indices],
        y=train_data.y[calibration_indices],
        edge_index=calibration_edge_index
    )
    
    print(f"Train data: {cp_train_data.x.shape[0]} nodes, {cp_train_data.edge_index.shape[1]} edges")
    print(f"Train edge_index 최대값: {cp_train_data.edge_index.max().item()}")
    print(f"Calibration data: {calibration_data.x.shape[0]} nodes, {calibration_data.edge_index.shape[1]} edges")
    print(f"Calibration edge_index 최대값: {calibration_data.edge_index.max().item()}")
    print(f"Test data: {test_data.x.shape[0]} nodes, {test_data.edge_index.shape[1]} edges")
    print(f"Test edge_index 최대값: {test_data.edge_index.max().item()}")

# result folder & file
root_dir = f"./result_{model_name}"
csv_dir = os.path.join(root_dir, 'eval')
pdf_dir = os.path.join(root_dir, 'img')
os.makedirs(csv_dir, exist_ok=True)
os.makedirs(pdf_dir, exist_ok=True)
train_csv_file = os.path.join(csv_dir, f'{dataset_name}_train_eval.csv')
test_csv_file = os.path.join(csv_dir, f'{dataset_name}_test_eval.csv')

# Training..
print('-' * 40, f'{model_name}: {dataset_name} training is starting... ', '-' * 40)

in_dim = train_data.x.shape[1]
device = torch.device(f"cuda:{gpu_number}" if torch.cuda.is_available() else 'cpu')
train_data = train_data.to(device)
pastel_colors = sns.color_palette('Dark2')
# SQR-GNN
if model_name == 'SQR-GNN':
    color = pastel_colors[0]
    model = GQNN_R(in_dim=in_dim, hidden_dim=hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight)
    criterion = QRLoss()
    
    epochs = []
    losses = []
    for epoch in tqdm(range(num_epochs)):
        model.train()
        optimizer.zero_grad()
        
        taus = torch.rand(train_data.x.size(0), 1, dtype=torch.float32, device=device)
        preds = model(train_data.x, train_data.edge_index, taus)
        loss = criterion(preds, train_data.y, taus)
            
        loss.backward()
        optimizer.step()
                
    #     epochs.append(epoch+1)
    #     losses.append(loss.item())
        
    # print('-' * 40, f'{model_name}: {dataset_name} training is ended... ', '-' * 40)
    # plt.figure(figsize=(5, 2))
    # plt.plot(epochs, losses, color=color)
    # plt.title(f'{model_name} Loss', size=10)
    # plt.show()
    
    print('-' * 40, f'{model_name}: {dataset_name} Train Evaluation... ', '-' * 40)
    model.eval()
    tau_lows = torch.full((train_data.x.size(0), 1), tau_low, dtype=torch.float32, device=device)
    tau_uppers = torch.full((train_data.x.size(0), 1), tau_upper, dtype=torch.float32, device=device)

    with torch.no_grad():
        train_low_preds = model(train_data.x, train_data.edge_index, tau_lows).cpu().numpy()
        train_upper_preds = model(train_data.x, train_data.edge_index, tau_uppers).cpu().numpy()
        train_targets = train_data.y.cpu().numpy()
    train_eval = evaluate_model_performance(train_low_preds, train_upper_preds, train_targets, target=target)
    train_eval = {k: round(v, 4) for k, v in train_eval.items()}
    train_new_row = pd.DataFrame([train_eval])

    if os.path.exists(train_csv_file):
        df_tr = pd.read_csv(train_csv_file)
        df_tr = pd.concat([df_tr, train_new_row], ignore_index=True)
    else:
        df_tr = train_new_row
    df_tr.to_csv(train_csv_file, index=False, float_format='%.4f')
    
    print('-' * 40, f'{model_name}: {dataset_name}s Test Evaluation... ', '-' * 40)
    test_data = test_data.to(device)
    tau_lows = torch.full((test_data.x.size(0), 1), tau_low, dtype=torch.float32, device=device)
    tau_uppers = torch.full((test_data.x.size(0), 1), tau_upper, dtype=torch.float32, device=device)

    with torch.no_grad():
        test_low_preds = model(test_data.x, test_data.edge_index, tau_lows).cpu().numpy()
        test_upper_preds = model(test_data.x, test_data.edge_index, tau_uppers).cpu().numpy()
        test_targets = test_data.y.cpu().numpy()
    test_eval = evaluate_model_performance(test_low_preds, test_upper_preds, test_targets, target=target)
    test_eval = {k: round(v, 4) for k, v in test_eval.items()}
    test_new_row = pd.DataFrame([test_eval])

    if os.path.exists(test_csv_file):
        df_ts = pd.read_csv(test_csv_file)
        df_ts = pd.concat([df_ts, test_new_row], ignore_index=True)
    else:
        df_ts = test_new_row
    df_ts.to_csv(test_csv_file, index=False, float_format='%.4f')
    
elif model_name == 'RQR-GNN':
    color = pastel_colors[1]
    model = GQNN_N(in_dim=in_dim, hidden_dim=hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight)
    criterion = RQRLoss(target=target, lambda_factor=lambda_factor)
    
    epochs, losses, coverages, widths = [], [], [], []
    for epoch in tqdm(range(num_epochs)):
        model.train()
        optimizer.zero_grad()
        
        preds = model(train_data.x, train_data.edge_index)
        loss = criterion(preds, train_data.y)
            
        loss.backward()
        optimizer.step()
        
        epochs.append(epoch+1)
        losses.append(loss.item())
        
    #     cvg, wdt = coverage_width(train_data.y, preds[:, 0], preds[:, 1])
        
    #     coverages.append(cvg.item())
    #     widths.append(wdt.item())
        
    # print('-' * 40, f'{model_name}: {dataset_name} training is ended... ', '-' * 40)
    # color = sns.color_palette("colorblind")    
    # plt.figure(figsize=(20, 3))
    # plt.subplot(131)
    # plt.plot(epochs, losses, color=color[0])
    # plt.title(f'{model_name} Loss', size=10)

    # plt.subplot(132)
    # plt.plot(epochs, coverages, color=color[1])
    # plt.hlines(y=0.9, xmin=min(epochs), xmax=max(epochs), colors='red', linestyles='--', label='target coverage: 0.9')
    # plt.title(f'{model_name} Coverage', size=10)
    # plt.legend()

    # plt.subplot(133)
    # plt.plot(epochs, widths, color=color[2])
    # plt.hlines(y=0.3, label='0.3', xmin=min(epochs), xmax=max(epochs), colors='green', linestyles='--')
    # plt.hlines(y=0.1, label='0.1', xmin=min(epochs), xmax=max(epochs), colors='purple', linestyles='-.')
    # plt.title(f'{model_name} Width', size=10)
    # plt.legend()
    # plt.show()     

    print('-' * 40, f'{model_name}: {dataset_name} Train Evaluation... ', '-' * 40)
    model.eval()

    with torch.no_grad():
        train_preds = model(train_data.x, train_data.edge_index)
        train_low_preds = train_preds[:, 0].cpu().numpy()
        train_upper_preds = train_preds[:, 1].cpu().numpy()
        train_targets = train_data.y.cpu().numpy()
    train_eval = evaluate_model_performance(train_low_preds, train_upper_preds, train_targets, target=target)
    train_eval = {k: round(v, 4) for k, v in train_eval.items()}
    train_new_row = pd.DataFrame([train_eval])

    if os.path.exists(train_csv_file):
        df_tr = pd.read_csv(train_csv_file)
        df_tr = pd.concat([df_tr, train_new_row], ignore_index=True)
    else:
        df_tr = train_new_row
    df_tr.to_csv(train_csv_file, index=False, float_format='%.4f')
    
    print('-' * 40, f'{model_name}: {dataset_name}s Test Evaluation... ', '-' * 40)
    test_data = test_data.to(device)

    with torch.no_grad():
        test_preds = model(test_data.x, test_data.edge_index)
        test_low_preds = test_preds[:, 0].cpu().numpy()
        test_upper_preds = test_preds[:, 1].cpu().numpy()
        test_targets = test_data.y.cpu().numpy()
    test_eval = evaluate_model_performance(test_low_preds, test_upper_preds, test_targets, target=target)
    test_eval = {k: round(v, 4) for k, v in test_eval.items()}
    test_new_row = pd.DataFrame([test_eval])

    if os.path.exists(test_csv_file):
        df_ts = pd.read_csv(test_csv_file)
        df_ts = pd.concat([df_ts, test_new_row], ignore_index=True)
    else:
        df_ts = test_new_row
    df_ts.to_csv(test_csv_file, index=False, float_format='%.4f')
    
elif model_name == 'CP-GNN':
    color = pastel_colors[2]
    cp_train_data = cp_train_data.to(device)
    
    model = GNN_CP(in_dim=in_dim, hidden_dim=hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight)
    
    epochs = []
    losses = []
    for epoch in tqdm(range(num_epochs)):
        model.train()
        optimizer.zero_grad()
        
        preds = model(cp_train_data.x, cp_train_data.edge_index)
        loss = F.mse_loss(preds, cp_train_data.y)
        
        loss.backward()
        optimizer.step()
        
        epochs.append(epoch+1)
        losses.append(loss.item())
    
    print('-' * 40, f'{model_name}: {dataset_name} Train Evaluation... ', '-' * 40)
    model.eval()
    calibration_data = calibration_data.to(device)
    test_data = test_data.to(device)

    with torch.no_grad():
        preds_cal = model(calibration_data.x, calibration_data.edge_index)
        preds_train = model(train_data.x, train_data.edge_index).cpu().numpy()
        preds_test = model(test_data.x, test_data.edge_index).cpu().numpy()

    conformal_scores = torch.abs(calibration_data.y- preds_cal).cpu().numpy()
    q_hat = np.quantile(conformal_scores, target)

    train_low_preds = preds_train - q_hat
    train_upper_preds = preds_train + q_hat
    train_targets = train_data.y.cpu().numpy()
    train_eval = evaluate_model_performance(train_low_preds, train_upper_preds, train_targets, target=target)
    train_new_row = pd.DataFrame([train_eval])

    if os.path.exists(train_csv_file):
        df_tr = pd.read_csv(train_csv_file)
        df_tr = pd.concat([df_tr, train_new_row], ignore_index=True)
    else:
        df_tr = train_new_row
    df_tr.to_csv(train_csv_file, index=False, float_format='%.4f')
    
    print('-' * 40, f'{model_name}: {dataset_name}s Test Evaluation... ', '-' * 40)
    test_low_preds = preds_test - q_hat
    test_upper_preds = preds_test + q_hat
    test_targets = test_data.y.cpu().numpy()
    test_eval = evaluate_model_performance(test_low_preds, test_upper_preds, test_targets, target=target)
    test_new_row = pd.DataFrame([test_eval])

    if os.path.exists(test_csv_file):
        df_ts = pd.read_csv(test_csv_file)
        df_ts = pd.concat([df_ts, test_new_row], ignore_index=True)
    else:
        df_ts = test_new_row
    df_ts.to_csv(test_csv_file, index=False, float_format='%.4f')
    
elif model_name == 'BNN':
    color = pastel_colors[3]
    model = BayesianGNN(in_dim=in_dim, hidden_dim=hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight)
    
    epochs = []
    losses = []
    for epoch in tqdm(range(num_epochs)):
        model.train()
        optimizer.zero_grad()
        
        preds = model(train_data.x, train_data.edge_index)
        loss = F.mse_loss(preds, train_data.y.squeeze())
            
        loss.backward()
        optimizer.step()
                
        epochs.append(epoch+1)
        losses.append(loss.item())
                
    print('-' * 40, f'{model_name}: {dataset_name} Train Evaluation... ', '-' * 40)
    model.eval()

    preds_list = []
    with torch.no_grad():
        for _ in range(num_samples):
            preds = model(train_data.x, train_data.edge_index)  
            preds_list.append(preds.cpu().numpy())

    preds_array = np.array(preds_list)  # (num_samples, num_nodes)
    mean_preds = preds_array.mean(axis=0)  # 평균 예측값
    std_preds = preds_array.std(axis=0)    # 표준편차

    # 80%: 1.28 / 90%: 1.645 / 95%: 1.96 / 99%: 2.576
    train_low_preds = mean_preds - 1.645 * std_preds  
    train_upper_preds = mean_preds + 1.645 * std_preds 
    train_targets = train_data.y.cpu().numpy()
    train_eval = evaluate_model_performance(train_low_preds, train_upper_preds, train_targets, target=target)
    train_eval = {k: round(v, 4) for k, v in train_eval.items()}
    train_new_row = pd.DataFrame([train_eval])

    if os.path.exists(train_csv_file):
        df_tr = pd.read_csv(train_csv_file)
        df_tr = pd.concat([df_tr, train_new_row], ignore_index=True)
    else:
        df_tr = train_new_row
    df_tr.to_csv(train_csv_file, index=False, float_format='%.4f')
    
    print('-' * 40, f'{model_name}: {dataset_name}s Test Evaluation... ', '-' * 40)
    test_data = test_data.to(device)

    preds_list = []
    with torch.no_grad():
        for _ in range(num_samples):
            preds = model(test_data.x, test_data.edge_index)  # Bayesian Sampling
            preds_list.append(preds.cpu().numpy())

    preds_array = np.array(preds_list)  # (num_samples, num_nodes)
    mean_preds = preds_array.mean(axis=0)  # 평균 예측값
    std_preds = preds_array.std(axis=0)    # 표준편차

    test_low_preds = mean_preds - 1.645 * std_preds  # 95% 신뢰구간 하한
    test_upper_preds = mean_preds + 645.96 * std_preds  # 95% 신뢰구간 상한
    test_targets = test_data.y.cpu().numpy()
    test_eval = evaluate_model_performance(test_low_preds, test_upper_preds, test_targets, target=target)
    test_eval = {k: round(v, 4) for k, v in test_eval.items()}
    test_new_row = pd.DataFrame([test_eval])

    if os.path.exists(test_csv_file):
        df_ts = pd.read_csv(test_csv_file)
        df_ts = pd.concat([df_ts, test_new_row], ignore_index=True)
    else:
        df_ts = test_new_row
    df_ts.to_csv(test_csv_file, index=False, float_format='%.4f')
    
elif model_name == 'MC':
    color = pastel_colors[4]
    model = MCDropoutGNN(in_dim=in_dim, hidden_dim=hidden_dim, dropout=dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight)

    epochs = []
    losses = []
    for epoch in tqdm(range(num_epochs)):
        model.train()
        optimizer.zero_grad()
        
        preds = model(train_data.x, train_data.edge_index, training=True)
        loss = F.mse_loss(preds, train_data.y.squeeze())
            
        loss.backward()
        optimizer.step()
                
        epochs.append(epoch+1)
        losses.append(loss.item())
                    
    print('-' * 40, f'{model_name}: {dataset_name} Train Evaluation... ', '-' * 40)
    model.eval()
    
    preds_list = []
    with torch.no_grad():
        for _ in range(num_samples):
            preds = model(train_data.x, train_data.edge_index, training=True)  # Dropout 유지
            preds_list.append(preds.cpu().numpy())

    preds_array = np.array(preds_list)  # (num_samples, num_nodes)
    mean_preds = preds_array.mean(axis=0)  # 평균 예측값
    std_preds = preds_array.std(axis=0)    # 표준편차

    train_low_preds = mean_preds - 1.645 * std_preds 
    train_upper_preds = mean_preds + 1.645 * std_preds 
    train_targets = train_data.y.cpu().numpy()
    train_eval = evaluate_model_performance(train_low_preds, train_upper_preds, train_targets, target=target)
    train_eval = {k: round(v, 4) for k, v in train_eval.items()}
    train_new_row = pd.DataFrame([train_eval])

    if os.path.exists(train_csv_file):
        df_tr = pd.read_csv(train_csv_file)
        df_tr = pd.concat([df_tr, train_new_row], ignore_index=True)
    else:
        df_tr = train_new_row
    df_tr.to_csv(train_csv_file, index=False, float_format='%.4f')
    
    print('-' * 40, f'{model_name}: {dataset_name}s Test Evaluation... ', '-' * 40)
    test_data = test_data.to(device)
    
    preds_list = []
    with torch.no_grad():
        for _ in range(num_samples):
            preds = model(test_data.x, test_data.edge_index, training=True)  # Dropout 유지
            preds_list.append(preds.cpu().numpy())

    preds_array = np.array(preds_list)  # (num_samples, num_nodes)
    mean_preds = preds_array.mean(axis=0)  # 평균 예측값
    std_preds = preds_array.std(axis=0)    # 표준편차

    test_low_preds = mean_preds - 1.645 * std_preds  # 95% 신뢰구간 하한
    test_upper_preds = mean_preds + 1.645 * std_preds  # 95% 신뢰구간 상한
    test_targets = test_data.y.cpu().numpy()
    test_eval = evaluate_model_performance(test_low_preds, test_upper_preds, test_targets, target=target)
    test_eval = {k: round(v, 4) for k, v in test_eval.items()}
    test_new_row = pd.DataFrame([test_eval])

    if os.path.exists(test_csv_file):
        df_ts = pd.read_csv(test_csv_file)
        df_ts = pd.concat([df_ts, test_new_row], ignore_index=True)
    else:
        df_ts = test_new_row
    df_ts.to_csv(test_csv_file, index=False, float_format='%.4f')
    
elif model_name == 'GQNN':
    color = pastel_colors[6]
    model = GQNN(in_dim=in_dim, hidden_dim=hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight)
    criterion = GQNNLoss(target_coverage=target, lambda_factor=lambda_factor)
    
    epochs, losses, coverages, widths = [], [], [], []
    for epoch in tqdm(range(num_epochs)):
        model.train()
        optimizer.zero_grad()
        
        preds_low, preds_upper = model(train_data.x, train_data.edge_index)
        loss = criterion(preds_low, preds_upper, train_data.y)
            
        loss.backward()
        optimizer.step()
        
        epochs.append(epoch+1)
        losses.append(loss.item())

        cvg, wdt = coverage_width(train_data.y, preds_low, preds_upper)
        
        coverages.append(cvg.item())
        widths.append(wdt.item())
        
    print('-' * 40, f'{model_name}: {dataset_name} Train Evaluation... ', '-' * 40)
    model.eval()

    with torch.no_grad():
        preds_low, preds_upper = model(train_data.x, train_data.edge_index)    
        train_low_preds = preds_low.cpu().numpy()
        train_upper_preds = preds_upper.cpu().numpy()
        train_targets = train_data.y.cpu().numpy()
    train_eval = evaluate_model_performance(train_low_preds, train_upper_preds, train_targets, target=target)
    train_eval = {k: round(v, 4) for k, v in train_eval.items()}
    train_eval['lambda_factor'] = lambda_factor
    train_new_row = pd.DataFrame([train_eval])

    if os.path.exists(train_csv_file):
        df_tr = pd.read_csv(train_csv_file)
        df_tr = pd.concat([df_tr, train_new_row], ignore_index=True)
    else:
        df_tr = train_new_row
    df_tr.to_csv(train_csv_file, index=False, float_format='%.4f')
    
    print('-' * 40, f'{model_name}: {dataset_name}s Test Evaluation... ', '-' * 40)
    test_data = test_data.to(device)

    with torch.no_grad():
        preds_low, preds_upper = model(test_data.x, test_data.edge_index)    
        test_low_preds = preds_low.cpu().numpy()
        test_upper_preds = preds_upper.cpu().numpy()
        test_targets = test_data.y.cpu().numpy()
    test_eval = evaluate_model_performance(test_low_preds, test_upper_preds, test_targets, target=target)
    test_eval = {k: round(v, 4) for k, v in test_eval.items()}
    test_eval['lambda_factor'] = lambda_factor
    test_new_row = pd.DataFrame([test_eval])

    if os.path.exists(test_csv_file):
        df_ts = pd.read_csv(test_csv_file)
        df_ts = pd.concat([df_ts, test_new_row], ignore_index=True)
    else:
        df_ts = test_new_row
    df_ts.to_csv(test_csv_file, index=False, float_format='%.4f')

elif model_name == 'GQNN_2':
    color = pastel_colors[7]
    model = GQNN(in_dim=in_dim, hidden_dim=hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight)
    criterion = GQNNLoss2(target_coverage=target, lambda_width=lambda_factor)
    
    epochs, losses, coverages, widths = [], [], [], []
    for epoch in tqdm(range(num_epochs)):
        model.train()
        optimizer.zero_grad()
        
        preds_low, preds_upper = model(train_data.x, train_data.edge_index)
        loss = criterion(preds_low, preds_upper, train_data.y)
            
        loss.backward()
        optimizer.step()
        
        epochs.append(epoch+1)
        losses.append(loss.item())

        cvg, wdt = coverage_width(train_data.y, preds_low, preds_upper)
        
        coverages.append(cvg.item())
        widths.append(wdt.item())
        
    print('-' * 40, f'{model_name}: {dataset_name} Train Evaluation... ', '-' * 40)
    model.eval()

    with torch.no_grad():
        preds_low, preds_upper = model(train_data.x, train_data.edge_index)    
        train_low_preds = preds_low.cpu().numpy()
        train_upper_preds = preds_upper.cpu().numpy()
        train_targets = train_data.y.cpu().numpy()
    train_eval = evaluate_model_performance(train_low_preds, train_upper_preds, train_targets, target=target)
    train_eval = {k: round(v, 4) for k, v in train_eval.items()}
    train_eval['lambda_factor'] = lambda_factor
    train_new_row = pd.DataFrame([train_eval])

    if os.path.exists(train_csv_file):
        df_tr = pd.read_csv(train_csv_file)
        df_tr = pd.concat([df_tr, train_new_row], ignore_index=True)
    else:
        df_tr = train_new_row
    df_tr.to_csv(train_csv_file, index=False, float_format='%.4f')
    
    print('-' * 40, f'{model_name}: {dataset_name}s Test Evaluation... ', '-' * 40)
    test_data = test_data.to(device)

    with torch.no_grad():
        preds_low, preds_upper = model(test_data.x, test_data.edge_index)    
        test_low_preds = preds_low.cpu().numpy()
        test_upper_preds = preds_upper.cpu().numpy()
        test_targets = test_data.y.cpu().numpy()
    test_eval = evaluate_model_performance(test_low_preds, test_upper_preds, test_targets, target=target)
    test_eval = {k: round(v, 4) for k, v in test_eval.items()}
    test_eval['lambda_factor'] = lambda_factor
    test_new_row = pd.DataFrame([test_eval])

    if os.path.exists(test_csv_file):
        df_ts = pd.read_csv(test_csv_file)
        df_ts = pd.concat([df_ts, test_new_row], ignore_index=True)
    else:
        df_ts = test_new_row
    df_ts.to_csv(test_csv_file, index=False, float_format='%.4f')
       
if pdf_save:
    existing_train_pdf = os.path.join(pdf_dir, "train_plots.pdf")
    new_train_pdf = os.path.join(pdf_dir, "new_train_plot.pdf")
    merged_train_pdf = os.path.join(pdf_dir, "merged_train_plots.pdf")

    x_st, y_st, (low_r_st, upper_r_st) = sort_by_y(train_data.x, train_data.y, train_low_preds, train_upper_preds)
    
    with PdfPages(new_train_pdf) as pdf:
        fig, axes = plt.subplots(1, 2, figsize=(16, 3))  
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S") 
        plt.suptitle(f"Model: {model_name}, Dataset: {dataset_name} (Train), Time: {timestamp}", fontsize=14, fontweight='bold')

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
        
    if os.path.exists(existing_train_pdf):
        merger = PdfMerger()
        merger.append(existing_train_pdf)
        merger.append(new_train_pdf)
        merger.write(merged_train_pdf)
        merger.close()

        os.replace(merged_train_pdf, existing_train_pdf)
        os.remove(new_train_pdf)
    else:
        os.replace(new_train_pdf, existing_train_pdf)
        
    existing_test_pdf = os.path.join(pdf_dir, "test_plots.pdf")
    new_test_pdf = os.path.join(pdf_dir, "new_test_plot.pdf")
    merged_test_pdf = os.path.join(pdf_dir, "merged_test_plots.pdf")

    x_st, y_st, (low_r_st, upper_r_st) = sort_by_y(test_data.x, test_data.y, test_low_preds, test_upper_preds)
    
    with PdfPages(new_test_pdf) as pdf:
        fig, axes = plt.subplots(1, 2, figsize=(16, 3))  
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S") 
        plt.suptitle(f"Model: {model_name}, Dataset: {dataset_name} (Test), Time: {timestamp}", fontsize=14, fontweight='bold')

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
        
    if os.path.exists(existing_test_pdf):
        merger = PdfMerger()
        merger.append(existing_test_pdf)
        merger.append(new_test_pdf)
        merger.write(merged_test_pdf)
        merger.close()

        os.replace(merged_test_pdf, existing_test_pdf)
        os.remove(new_test_pdf)
    else:
        os.replace(new_test_pdf, existing_test_pdf)
