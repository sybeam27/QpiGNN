import torch
import torch.nn.functional as F
import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import csv

sys.path.append('..')
from utils.model import GQNN, GQNNLoss
from utils.function import create_er_graph_pyg, split_graph_data, normalize

# ---------------- Evaluation ---------------- #
def evaluate_model_performance(preds_low, preds_upper, targets, target=0.9):
    picp = np.mean((targets >= preds_low) & (targets <= preds_upper))
    interval_width = np.mean(preds_upper - preds_low)
    data_range = np.max(targets) - np.min(targets)
    nmpiw = interval_width / data_range if data_range > 0 else interval_width

    gamma, eta = 1, 10
    penalty = 1 + gamma * np.exp(-eta * (picp - target))
    cwc = nmpiw * penalty

    return {"PICP": picp, "MPIW": interval_width, "CWC": cwc}

# ---------------- Perturbations ---------------- #
def add_gaussian_noise(x, std):
    return x + torch.randn_like(x) * std

def edge_dropout(edge_index, p):
    num_edges = edge_index.size(1)
    mask = torch.rand(num_edges) > p
    return edge_index[:, mask]

def add_target_noise(y, std):
    return y + torch.randn_like(y) * std

# ---------------- Main Experiment ---------------- #
def run_robustness_experiment(noise_type, levels, device, args):
    base_graph = create_er_graph_pyg(n=1000)
    train_data, test_data = split_graph_data(base_graph, test_ratio=0.2)
    in_dim = train_data.x.shape[1]
    results = []

    for level in levels:
        train_data_ = train_data.clone()
        test_data_ = test_data.clone()

        if noise_type == "feature":
            train_data_.x = add_gaussian_noise(train_data_.x, std=level)
        elif noise_type == "edge":
            train_data_.edge_index = edge_dropout(train_data.edge_index, p=level)
        elif noise_type == "target":
            train_data_.y = add_target_noise(train_data.y, std=level)

        # Normalize
        train_min, train_max = train_data.x.min(), train_data.x.max()
        y_min, y_max = train_data.y.min(), train_data.y.max()
        train_data.x = normalize(train_data.x, train_min, train_max)
        test_data.x = normalize(test_data.x, train_min, train_max)
        train_data.y = normalize(train_data.y, y_min, y_max)
        test_data.y = normalize(test_data.y, y_min, y_max)

        picps, mpiws = [], []
        for _ in range(args.runs):
            model = GQNN(in_dim=in_dim, hidden_dim=64).to(device)
            criterion = GQNNLoss(target_coverage=0.9, lambda_factor=0.05)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            model.train()
            for _ in range(args.epochs):
                optimizer.zero_grad()
                pl, pu = model(train_data_.x.to(device), train_data_.edge_index.to(device))
                loss = criterion(pl, pu, train_data_.y.to(device))
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                pl, pu = model(test_data_.x.to(device), test_data_.edge_index.to(device))
                metrics = evaluate_model_performance(
                    pl.cpu().numpy(), pu.cpu().numpy(), test_data.y.cpu().numpy(), target=0.9)
                picps.append(metrics['PICP'])
                mpiws.append(metrics['MPIW'])

        results.append((
            level,
            np.mean(picps), np.mean(mpiws),
            np.std(picps), np.std(mpiws)
        ))

    return results

# ---------------- Save CSV ---------------- #
def save_results_csv(results, noise_type):
    os.makedirs("robust", exist_ok=True)
    filepath = f"robust/robustness_{noise_type}.csv"
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Noise_Level", "PICP", "MPIW", "std_PICP", "std_MPIW"])
        for row in results:
            writer.writerow(row)

# ---------------- Plot ---------------- #
NOISE_LABELS = {
    "feature": "Gaussian Noise Std",
    "edge": "Edge Dropout Ratio",
    "target": "Target Noise Std"
}

def plot_tradeoff_scatter(noise_type, levels, results):
    picps = [r[1] for r in results]
    mpiws = [r[2] for r in results]

    plt.figure(figsize=(6, 5))
    scatter = plt.scatter(mpiws, picps, c=levels, cmap='viridis', s=100, edgecolor='k')

    for i, lvl in enumerate(levels):
        plt.text(mpiws[i] + 0.01, picps[i], f"{lvl:.2f}", fontsize=9)

    plt.colorbar(scatter, label=NOISE_LABELS[noise_type])
    plt.axhline(y=90, color='red', linestyle='--', linewidth=1.2, label='Target PICP (90%)')
    plt.xlabel("MPIW (↓)")
    plt.ylabel("PICP (↑)")
    plt.title(f"Trade-off under {noise_type.capitalize()} Noise")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    os.makedirs("robust/figs", exist_ok=True)
    path = f"robust/figs/robustness_tradeoff_{noise_type}.png"
    plt.savefig(path)
    plt.close()

def plot_results_combined(results_dict):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for idx, (noise_type, (levels, results)) in enumerate(results_dict.items()):
        # 0.0 레벨 제거
        levels, results = zip(*[(l, r) for l, r in zip(levels, results) if l != 0.0])
        picps = [r[1] for r in results]
        mpiws = [r[2] for r in results]
        std_picps = [r[3] for r in results]
        std_mpiws = [r[4] for r in results]

        ax1 = axes[idx]
        ax2 = ax1.twinx()

        # PICP (왼쪽 y축)
        l1 = ax1.errorbar(
            levels, picps, yerr=std_picps,
            fmt='o-', color='tab:blue', label="PICP", capsize=3, alpha=0.9
        )
        ax1.set_ylabel("PICP", size = 11)
        ax1.set_ylim(0.85, 1.05)
        ax1.tick_params(axis='y')

        # MPIW (오른쪽 y축)
        l2 = ax2.errorbar(
            levels, mpiws, yerr=std_mpiws,
            fmt='s--', color='tab:red', label="MPIW", capsize=3, alpha=0.9
        )
        ax2.set_ylabel("MPIW", size = 11)
        ax2.set_ylim(0.8, 2.05)
        ax2.tick_params(axis='y')

        # 공통 X축 및 제목
        ax1.set_xlabel(NOISE_LABELS[noise_type])
        ax1.set_title(f"{noise_type.capitalize()} Noise", size=13)
        ax1.grid(True)

        # 범례 추가 (ax1 기준으로만 처리)
        ax1.legend([l1[0], l2[0]], ['PICP', 'MPIW'], loc='upper left', fontsize=11)

    plt.tight_layout()
    os.makedirs("robust/figs", exist_ok=True)
    plt.savefig("robust/figs/robustness_all_dualaxis.png")
    print("Saved: robustness_all_dualaxis.png")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)

    tasks = {
        "feature": [round(v, 2) for v in np.linspace(0.0, 0.3, 10)],
        "edge":    [round(v, 2) for v in np.linspace(0.0, 0.6, 10)],
        "target":  [round(v, 2) for v in np.linspace(0.0, 0.3, 10)],
    }

    results_dict = {}
    for noise_type, levels in tasks.items():
        print(f"\n===== Running robustness test: {noise_type} noise =====")
        results = run_robustness_experiment(noise_type, levels, device, args)
        save_results_csv(results, noise_type)
        # plot_tradeoff_scatter(noise_type, levels, results)
        results_dict[noise_type] = (levels, results)

    plot_results_combined(results_dict)
    print("\n✓ Robustness evaluation complete.")
