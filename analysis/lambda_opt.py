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

sys.path('../utills/')
from utills.function import set_seed, generate_graph_data, generate_noisy_graph_data, load_county_graph_data, load_twitch_graph_data, \
            load_wiki_graph_data, load_trans_graph_data, create_ba_graph_pyg, create_er_graph_pyg, create_grid_graph_pyg, create_tree_graph_pyg, \
            normalize, split_graph_data, split_cp_graph_data, evaluate_model_performance, sort_by_y, coverage_width, \
                get_gpu_memory, get_cpu_memory, count_parameters
from utills.model import GQNN_R, GQNN_N, BayesianGNN, MCDropoutGNN, GQNN, QRLoss, RQRLoss, GQNNLoss, GQNNLoss2

import matplotlib.pyplot as plt
import os
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from skopt.plots import plot_convergence

# 실험용 설정
target_coverage = 0.90
lambda_list = [0.005, 0.01, 0.03, 0.05, 0.07, 0.1]
dataset_list = ["Basic", "Edge", "ER", "BA", "Tree"]
runs = 5
epochs = 500
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def select_optimal_lambda(dataset_name, lambda_list, target_coverage):
    best_lambda = None
    best_mpiw = float("inf")
    log = []

    data = load_graph_dataset(dataset_name)

    for lam in lambda_list:
        picps, mpiws = [], []

        for _ in range(runs):
            picp, mpiw = train_and_evaluate(data, lam, target_coverage, epochs)
            picps.append(picp)
            mpiws.append(mpiw)

        mean_picp = np.mean(picps)
        mean_mpiw = np.mean(mpiws)

        log.append({'lambda': lam, 'PICP': mean_picp, 'MPIW': mean_mpiw})

        if mean_picp >= target_coverage and mean_mpiw < best_mpiw:
            best_lambda = lam
            best_mpiw = mean_mpiw

        print(f"[{dataset_name}] λ={lam:.3f} → PICP={mean_picp:.3f}, MPIW={mean_mpiw:.3f}")

    return best_lambda, best_mpiw, log

def load_graph_dataset(name, num_nodes=1000, noise_level=0.3):
    if name == 'basic':
        return generate_graph_data(num_nodes)
    elif name in ('gaussian', 'uniform', 'outlier', 'edge'):
        return generate_noisy_graph_data(num_nodes, noise_type=name, noise_level=noise_level)
    elif name in ('education', 'election', 'income', 'unemployment'):
        return load_county_graph_data(name, 2012)
    elif name in ('DE', 'ENGB', 'ES', 'FR', 'PTBR', 'RU'):
        return load_twitch_graph_data(name)
    elif name in ('chameleon', 'crocodile', 'squirrel'):
        return load_wiki_graph_data(name)
    elif name in ('Anaheim', 'ChicagoSketch'):
        return load_trans_graph_data(name)
    elif name == 'BA':
        return create_ba_graph_pyg(n=num_nodes)
    elif name == 'ER':
        return create_er_graph_pyg(n=num_nodes)
    elif name == 'grid':
        return create_grid_graph_pyg()
    elif name == 'tree':
        return create_tree_graph_pyg()
    else:
        raise ValueError(f"Unknown dataset: {name}")

def train_and_evaluate(data, lam, target_coverage, epochs=500):
    train_data, test_data = split_graph_data(data, test_ratio=0.2)

    # Normalize
    train_min, train_max = train_data.x.min(), train_data.x.max()
    y_min, y_max = train_data.y.min(), train_data.y.max()
    train_data.x = normalize(train_data.x, train_min, train_max)
    test_data.x = normalize(test_data.x, train_min, train_max)
    train_data.y = normalize(train_data.y, y_min, y_max)
    test_data.y = normalize(test_data.y, y_min, y_max)

    model = GQNN(in_dim=train_data.x.shape[1], hidden_dim=64).to(device)
    criterion = GQNNLoss(target_coverage=target_coverage, lambda_factor=lam)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        pl, pu = model(train_data.x.to(device), train_data.edge_index.to(device))
        loss = criterion(pl, pu, train_data.y.to(device))
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        pl, pu = model(test_data.x.to(device), test_data.edge_index.to(device))
        metrics = evaluate_model_performance(
            pl.cpu().numpy(), pu.cpu().numpy(), test_data.y.cpu().numpy(), target=target_coverage
        )

    return metrics["PICP"], metrics["MPIW"]

def plot_lambda_sweep_subplots(logs_dict, result_dict, target_coverage):

    num_datasets = len(logs_dict)
    num_cols = 3
    num_rows = int(np.ceil(num_datasets / num_cols))

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(6 * num_cols, 4 * num_rows))
    axes = axes.flatten()

    all_handles = []  # 전체 legend용

    for idx, (dataset, log) in enumerate(logs_dict.items()):
        lambdas = [entry['lambda'] for entry in log]
        picps = [entry['PICP'] for entry in log]
        mpiws = [entry['MPIW'] for entry in log]
        
        sorted_idx = np.argsort(lambdas)
        lambdas = [lambdas[i] for i in sorted_idx]
        picps   = [picps[i]   for i in sorted_idx]
        mpiws   = [mpiws[i]   for i in sorted_idx]

        best_lambda, best_mpiw = result_dict[dataset]
        ax1 = axes[idx]
        ax2 = ax1.twinx()

        # plot 및 legend 추출
        line1, = ax1.plot(lambdas, picps, 'o-', color='tab:blue', label='PICP')
        line2, = ax2.plot(lambdas, mpiws, 's--', color='tab:red', label='MPIW')

        ax1.axhline(target_coverage, color='tab:blue', linestyle='--', alpha=0.3)

        ax1.set_title(dataset)
        ax1.set_xlabel("Lambda")
        ax1.set_ylabel("PICP", color='tab:blue')
        ax2.set_ylabel("MPIW", color='tab:red')

        if best_lambda is not None and best_lambda in lambdas:
            best_idx = lambdas.index(best_lambda)
            best_x = best_lambda
            best_picp = picps[best_idx]
            best_mpiw = mpiws[best_idx]

            # 마커 표시
            ax1.plot(best_x, best_picp, 'o', color='blue', markerfacecolor='none', markersize=10)
            ax2.plot(best_x, best_mpiw, 's', color='red', markerfacecolor='none', markersize=10)

            # # 텍스트 라벨 추가 (PICP 쪽)
            # ax1.text(
            #     best_x, best_picp + 1.0,  # y위치 살짝 위로
            #     f"λ={best_x:.2f}",
            #     fontsize=10,
            #     color='blue',
            #     ha='center'
            # )

        # 첫 번째 subplot에서만 handles 저장
        if idx == 0:
            all_handles.extend([line1, line2,
                plt.Line2D([], [], linestyle='--', color='tab:blue', alpha=0.3, label='Target Coverage')])

    # Unused axes 제거
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    # 공통 legend 추가
    fig.legend(
        handles=all_handles,
        loc='lower center',
        ncol=3,
        frameon=False,
        fontsize='large',
        bbox_to_anchor=(0.5, -0.02)
    )

    fig.tight_layout(rect=[0, 0.03, 1, 1])  # bottom 여백 확보
    os.makedirs("lambda/figs", exist_ok=True)
    fig_path = "lambda/figs/lambda_sweep_all_subplots.png"
    fig.savefig(fig_path, bbox_inches='tight')
    print(f"Saved subplot figure with legend: {fig_path}")
    plt.close()

def get_lambda_list(logscale=False, num_points=10):
    if logscale:
        return np.round(np.logspace(np.log10(0.001), np.log10(0.1), num=num_points), 5).tolist()
    else:
        return np.round(np.linspace(0.005, 0.1, num=num_points), 5).tolist()
    
# ---------------- Bayesian Optimization ---------------- #
space = [Real(0.001, 0.1, name='lam')]

def run_lambda_optimization(data, target_coverage, epochs=300, runs=3):
    eval_log = []  # ← 여기에 로그 저장

    @use_named_args(space)
    def objective(lam):
        picps, mpiws = [], []
        for _ in range(runs):
            picp, mpiw = train_and_evaluate(data, lam, target_coverage, epochs)
            picps.append(picp)
            mpiws.append(mpiw)

        mean_picp = np.mean(picps)
        mean_mpiw = np.mean(mpiws)

        eval_log.append({'lambda': lam, 'PICP': mean_picp, 'MPIW': mean_mpiw})  # ← 로그 추가

        if mean_picp < target_coverage:
            return mean_mpiw + (target_coverage - mean_picp) * 100
        else:
            return mean_mpiw

    result = gp_minimize(
        func=objective,
        dimensions=space,
        acq_func="EI",
        n_calls=20,
        n_random_starts=5,
        random_state=42
    )
    return result, eval_log

# ---------------- Main Entry ---------------- #
if __name__ == "__main__":
    target_coverage = 0.90
    dataset_list = [
                    # 'basic', 'gaussian', 'uniform', 'outlier', 'edge', 'BA', 'ER', 'grid', 'tree'
                    'education', 'election', 'income', 'unemployment', 'PTBR', 'chameleon', 'crocodile', 'squirrel', 'Anaheim', 'ChicagoSketch'
                    ]
    epochs = 300
    runs = 3

    logs_dict = {}
    result_dict = {}

    os.makedirs("lambda/figs", exist_ok=True)

    for dataset in dataset_list:
        print(f"\n=== Optimizing λ for dataset: {dataset} ===")
        data = load_graph_dataset(dataset)
        result, log = run_lambda_optimization(data, target_coverage, epochs, runs)  # ← 로그 함께 반환받음
        best_lam = result.x[0]
        best_mpiw = result.fun

        result_dict[dataset] = (best_lam, best_mpiw)
        logs_dict[dataset] = log 

        # 시각화 (개별 convergence plot)
        fig = plot_convergence(result)
        plt.title(f"Convergence: {dataset}")
        plt.tight_layout()
        plt.savefig(f"lambda/figs/lambda_convergence_{dataset}.png")
        plt.close()

        print(f">>> Best λ for {dataset}: {best_lam:.5f} (MPIW={best_mpiw:.4f})")

    # 저장
    df = pd.DataFrame([
        {"Dataset": d, "Best Lambda": l, "MPIW": w}
        for d, (l, w) in result_dict.items()
    ])
    df.to_csv("lambda/lambda_optimized_results.csv", index=False)

    # 전체 결과 subplot 시각화
    plot_lambda_sweep_subplots(logs_dict, result_dict, target_coverage)


