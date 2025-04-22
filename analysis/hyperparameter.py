import os
import sys
import torch
import torch.nn.functional as F
import argparse
import numpy as np
import os
import matplotlib.pyplot as plt
import csv

sys.path.append('..')
from utills.model import GQNN, GQNNLoss
from utills.function import (
    generate_graph_data, create_tree_graph_pyg, create_er_graph_pyg, split_graph_data, normalize, 
)

def evaluate_model_performance(preds_low, preds_upper, targets, target=0.9):
    picp = np.mean((targets >= preds_low) & (targets <= preds_upper))
    interval_width = np.mean(preds_upper - preds_low)
    data_range = np.max(targets) - np.min(targets)
    nmpiw = interval_width / data_range if data_range > 0 else interval_width

    gamma = 1
    eta = 10
    penalty = 1 + gamma * np.exp(-eta * (picp - target))
    cwc = nmpiw * penalty

    print(f"종합 - CWC ⬇: {cwc:.4f}")
    print(f"예측 관련 - PICP ⬆: {picp:.4f}")
    print(f"구간 관련 - MPIW ⬇: {interval_width:.4f}")

    return {
        "PICP": picp,
        'MPIW': interval_width,
        "CWC": cwc,
    }

def run_sensitivity_experiment(target_coverages, lambdas, device, args):
    # base_graph = generate_graph_data(num_nodes=1000)
    base_graph = create_er_graph_pyg(n=1000)
    train_data, test_data = split_graph_data(base_graph, test_ratio=0.2)
    in_dim = train_data.x.shape[1]

    # Normalize
    train_min, train_max = train_data.x.min(), train_data.x.max()
    y_min, y_max = train_data.y.min(), train_data.y.max()
    train_data.x = normalize(train_data.x, train_min, train_max)
    test_data.x = normalize(test_data.x, train_min, train_max)
    train_data.y = normalize(train_data.y, y_min, y_max)
    test_data.y = normalize(test_data.y, y_min, y_max)

    results = {}

    for tc in target_coverages:
        for lam in lambdas:
            picps, mpiws = [], []

            for run in range(args.runs):
                model = GQNN(in_dim=in_dim, hidden_dim=64).to(device)
                criterion = GQNNLoss(target_coverage=tc, lambda_factor=lam)
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

                model.train()
                for epoch in range(args.epochs):
                    optimizer.zero_grad()
                    pl, pu = model(train_data.x.to(device), train_data.edge_index.to(device))
                    loss = criterion(pl, pu, train_data.y.to(device))
                    loss.backward()
                    optimizer.step()

                model.eval()
                with torch.no_grad():
                    pl, pu = model(test_data.x.to(device), test_data.edge_index.to(device))
                    metrics = evaluate_model_performance(
                        pl.cpu().numpy(), pu.cpu().numpy(), test_data.y.cpu().numpy(), target=tc)
                    picps.append(metrics['PICP'] * 100)
                    mpiws.append(metrics['MPIW'])

            results[(tc, lam)] = (np.mean(picps), np.mean(mpiws))
            print(f"target={tc}, lambda={lam} → PICP={np.mean(picps):.2f}%, MPIW={np.mean(mpiws):.3f}")

    return results

def save_results_to_csv(results, save_path="sensitivity/sensitivity_results.csv"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["target_coverage", "lambda", "PICP", "MPIW"])
        for (tc, lam), (picp, mpiw) in results.items():
            writer.writerow([tc, lam, picp, mpiw])
    print(f"Saved results to {save_path}")

def plot_sensitivity_matrix(results, target_coverages, lambdas):
    os.makedirs("sensitivity/figs", exist_ok=True)

    picp_matrix = np.array([[results[(tc, lam)][0] for lam in lambdas] for tc in target_coverages])
    mpiw_matrix = np.array([[results[(tc, lam)][1] for lam in lambdas] for tc in target_coverages])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)

    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
    linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]  # 다양한 패턴

    for i, tc in enumerate(target_coverages):
        color = colors[i % len(colors)]
        style = linestyles[i % len(linestyles)]
        
        # Plot PICP
        axes[0].plot(lambdas, picp_matrix[i], marker='o', color=color, linestyle=style, label=f"Target={tc}")
        # Plot MPIW
        axes[1].plot(lambdas, mpiw_matrix[i], marker='s', color=color, linestyle=style, label=f"Target={tc}")

    # PICP subplot
    axes[0].set_ylabel("PICP (%)", fontsize=12)
    axes[0].set_title("PICP vs. Lambda", fontsize=13)

    # MPIW subplot
    axes[1].set_xlabel("Lambda (Width Penalty)", fontsize=12)
    axes[1].set_ylabel("MPIW", fontsize=12)
    axes[1].set_title("MPIW vs. Lambda", fontsize=13)

    # 범례 (하단 subplot에만 넣고 하나로 통합)
    axes[1].legend(title="Target Coverage", fontsize=10, title_fontsize=11)

    # 그리드와 레이아웃 정리
    for ax in axes:
        ax.grid(True)

    fig.tight_layout()
    path = "sensitivity/figs/sensitivity_combined.png"
    plt.savefig(path)
    print(f"Saved: {path}")
    plt.close()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)

    target_coverages = [0.80, 0.85, 0.90, 0.95]
    lambda_factors = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]

    results = run_sensitivity_experiment(target_coverages, lambda_factors, device, args)
    save_results_to_csv(results)
    plot_sensitivity_matrix(results, target_coverages, lambda_factors)
