import sys
import os
import re
import random
import argparse
import pickle
from tqdm import tqdm
import time
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
            normalize, split_graph_data, split_cp_graph_data, evaluate_model_performance, sort_by_y, coverage_width, \
                get_gpu_memory, get_cpu_memory, count_parameters

set_seed(1127)  

parser = argparse.ArgumentParser(description='Train GQNN Model')
parser.add_argument('--dataset', type=str, default="basic", help='dataset_name')
parser.add_argument('--nodes', type=float, default=1000, help='num_nodes')
parser.add_argument('--noise', type=float, default=0.3, help='noise_level')
parser.add_argument('--target_coverage', type=float, default=0.9, help='target_coverage')

parser.add_argument('--hidden_dim', type=float, default=64, help='hidden_dim')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning_rate')
parser.add_argument('--weight', type=float, default=1e-3, help='weight_decay')

parser.add_argument('--epochs', type=float, default=500, help='num_epochs')
parser.add_argument('--runs', type=int, default=10, help='num_runs')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--pdf', type=bool, default=False, help='pdf_save')
parser.add_argument('--lambda_factor', type=float, default=1, help='lambda_factor')


args = parser.parse_args()

device = torch.device(args.device)

if args.dataset != '':
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
    
# split data & normalize
train_data, test_data = split_graph_data(graph_data, test_ratio=0.2)
train_min, train_max, y_min, y_max = train_data.x.min(), train_data.x.max(), train_data.y.min(), train_data.y.max()
train_data.x, test_data.x, train_data.y, test_data.y= normalize(train_data.x, train_min, train_max), normalize(test_data.x, train_min, train_max), normalize(train_data.y, y_min, y_max), normalize(test_data.y, y_min, y_max)

print(f"Train data: {train_data.x.shape[0]} nodes, {train_data.edge_index.shape[1]} edges")
print(f"Train edge_index 최대값: {train_data.edge_index.max().item()}")
print(f"Test data: {test_data.x.shape[0]} nodes, {test_data.edge_index.shape[1]} edges")
print(f"Test edge_index 최대값: {test_data.edge_index.max().item()}")

# result folder & file
root_dir = f"./pred/{args.model}/"
os.makedirs(root_dir, exist_ok=True)

if args.pdf:
    pdf_dir = os.path.join(root_dir, 'img')
    os.makedirs(pdf_dir, exist_ok=True)

file_name = args.dataset + '_' + args.model
if args.model == 'GQNN':
    file_name += f'_lf({args.lambda_factor})'
    
class GQNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, dual_output=True):
        super().__init__()
        self.dual_output = dual_output
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        
        if self.dual_output:
            self.fc_pred = nn.Linear(hidden_dim, 1)
            self.fc_diff = nn.Linear(hidden_dim, 1)
        else:
            self.fc_pred_low = nn.Linear(hidden_dim, 1)
            self.fc_pred_upper = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))

        if self.dual_output:
            preds = self.fc_pred(x)
            diffs = torch.sigmoid(self.fc_diff(x))
            preds_low = preds - diffs
            preds_upper = preds + diffs
        else:
            preds_low = self.fc_pred_low(x)
            preds_upper = self.fc_pred_upper(x)

        return preds_low, preds_upper

class GQNNLoss(nn.Module):
    def __init__(self, target_coverage=0.9, lambda_factor=0.1,
                 use_sample_loss=True, use_coverage_loss=True, use_width_loss=True):
        super().__init__()
        self.target_coverage = target_coverage
        self.lf = lambda_factor
        self.use_sample_loss = use_sample_loss
        self.use_coverage_loss = use_coverage_loss
        self.use_width_loss = use_width_loss

    def forward(self, preds_low, preds_upper, target):
        diffs = (preds_upper - preds_low) / 2
        
        # (1) 샘플 손실: 구간 밖 거리
        below_loss = torch.relu(preds_low - target)
        above_loss = torch.relu(target - preds_upper)
        sample_loss = below_loss + above_loss
        mean_sample_loss = sample_loss.mean() if self.use_sample_loss else 0.0

        # (2) 커버리지 손실
        covered = (preds_low <= target) & (target <= preds_upper)
        current_coverage = covered.float().mean()
        coverage_penalty = (self.target_coverage - current_coverage) ** 2 if self.use_coverage_loss else 0.0

        # (3) 폭 패널티
        width_loss = self.lf * 2 * diffs.mean() if self.use_width_loss else 0.0

        total_loss = mean_sample_loss + coverage_penalty + width_loss
        return total_loss

def train_and_evaluate(model, criterion, optimizer, train_data, test_data, args, run, device, results, result_this_run, color):
    # -- 리소스 초기화 --
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()

    # -- 학습 루프 --
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()

        preds_low, preds_upper = model(train_data.x, train_data.edge_index)
        loss = criterion(preds_low, preds_upper, train_data.y)

        loss.backward()
        optimizer.step()

        # 커버리지 및 폭 추적
        # cvg, wdt = coverage_width(train_data.y, preds_low, preds_upper)

    # -- 학습 리소스 기록 --
    training_time = time.time() - start_time
    gpu_mem = get_gpu_memory()
    cpu_mem = get_cpu_memory()
    param_count = count_parameters(model)

    result_this_run['training_time_sec'] = round(training_time, 2)
    result_this_run['gpu_mem_MB'] = round(gpu_mem, 2)
    result_this_run['cpu_mem_MB'] = round(cpu_mem, 2)
    result_this_run['param_count'] = param_count

    print(f"[{args.model}] Training Time: {training_time:.2f}s | GPU Peak: {gpu_mem:.1f}MB | CPU: {cpu_mem:.1f}MB | Params: {param_count:,}")

    # -- 평가: Train --
    print(f"{args.model}: {args.dataset} Train Evaluation...")
    model.eval()
    with torch.no_grad():
        preds_low, preds_upper = model(train_data.x, train_data.edge_index)
        train_low_preds = preds_low.cpu().numpy()
        train_upper_preds = preds_upper.cpu().numpy()
        train_targets = train_data.y.cpu().numpy()

    train_eval = evaluate_model_performance(train_low_preds, train_upper_preds, train_targets, target=args.target_coverage)
    result_this_run['train_metrics'] = train_eval

    # -- 평가: Test --
    print(f"{args.model}: {args.dataset} Test Evaluation...")
    test_data = test_data.to(device)
    with torch.no_grad():
        preds_low, preds_upper = model(test_data.x, test_data.edge_index)
        test_low_preds = preds_low.cpu().numpy()
        test_upper_preds = preds_upper.cpu().numpy()
        test_targets = test_data.y.cpu().numpy()

    test_eval = evaluate_model_performance(test_low_preds, test_upper_preds, test_targets, target=args.target_coverage)
    result_this_run['test_metrics'] = test_eval

    # -- PDF 저장 & 병합 --
    if args.pdf:
        save_prediction_plots(
            phase='Train',
            x=train_data.x,
            y=train_data.y,
            low=train_low_preds,
            upper=train_upper_preds,
            true_y=train_targets,
            args=args,
            color=color
        )

        save_prediction_plots(
            phase='Test',
            x=test_data.x,
            y=test_data.y,
            low=test_low_preds,
            upper=test_upper_preds,
            true_y=test_targets,
            args=args,
            color=color
        )

    # -- 결과 저장 --
    results[run] = result_this_run
    print(f'Finished training run {run}')

def save_prediction_plots(phase, x, y, low, upper, true_y, args, color):
    pdf_dir = args.pdf_dir
    os.makedirs(pdf_dir, exist_ok=True)

    existing_pdf = os.path.join(pdf_dir, f"{args.model}_eval_plots.pdf")
    new_pdf = os.path.join(pdf_dir, f"new_{phase.lower()}_plot.pdf")
    merged_pdf = os.path.join(pdf_dir, "merged_temp.pdf")

    x_st, y_st, (low_r, upper_r) = sort_by_y(x, y, low, upper)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    with PdfPages(new_pdf) as pdf:
        fig, axes = plt.subplots(1, 2, figsize=(16, 4))
        plt.suptitle(f"Model: {args.model}, Dataset: {args.dataset}, Time: {timestamp} ({phase})", fontsize=12, fontweight='bold')

        # Plot 1
        axes[0].scatter(range(len(x_st)), y_st, alpha=0.3, color='blue', label="True Values", s=15)
        axes[0].fill_between(range(len(x_st)), low_r, upper_r, color=color, alpha=0.5)
        axes[0].set_xlabel("Sorted Node Index")
        axes[0].set_ylabel("Values")

        # Plot 2
        axes[1].scatter(range(len(x_st)), true_y, alpha=0.3, color='blue', label="True Values", s=15)
        axes[1].fill_between(range(len(x_st)), low.squeeze(), upper.squeeze(), color=color, alpha=0.5)
        axes[1].set_xlabel("Node Index")
        axes[1].set_ylabel("Values")

        pdf.savefig(fig)
        plt.close(fig)

    # Merge PDFs
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


def save_final_results(results, path):
    print('Saving results to', path)
    with open(path, 'wb') as f:
        pickle.dump(results, f)
