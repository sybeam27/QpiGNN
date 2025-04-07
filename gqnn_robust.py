import torch
import torch.nn.functional as F
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import csv

from utills.model import GQNN, GQNNLoss
from utills.function import generate_graph_data, split_graph_data, normalize

# ---------------- Evaluation ---------------- #
def evaluate_model_performance(preds_low, preds_upper, targets, target=0.9):
    picp = np.mean((targets >= preds_low) & (targets <= preds_upper))
    interval_width = np.mean(preds_upper - preds_low)
    data_range = np.max(targets) - np.min(targets)
    nmpiw = interval_width / data_range if data_range > 0 else interval_width

    gamma, eta = 1, 10
    penalty = 1 + gamma * np.exp(-eta * (picp - target))
    cwc = nmpiw * penalty

    print(f"종합 - CWC ⬇: {cwc:.4f} | PICP ⬆: {picp:.4f} | MPIW ⬇: {interval_width:.4f}")
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
    base_graph = generate_graph_data(num_nodes=1000)
    train_data, test_data = split_graph_data(base_graph, test_ratio=0.2)
    in_dim = train_data.x.shape[1]
    results = []

    for level in levels:
        train_data_ = train_data.clone()
        test_data_ = test_data.clone()

        # Apply perturbation
        if noise_type == "feature":
            train_data_.x = add_gaussian_noise(train_data_.x, std=level)
        elif noise_type == "edge":
            train_data_.edge_index = edge_dropout(train_data.edge_index, p=level)
        elif noise_type == "target":
            train_data_.y = add_target_noise(train_data.y, std=level)

        # Normalize
        train_data_.x = normalize(train_data_.x)
        test_data_.x = normalize(test_data_.x)
        train_data_.y = normalize(train_data_.y)
        test_data_.y = normalize(test_data_.y)

        # Run model
        picps, mpiws = [], []
        for run in range(args.runs):
            model = GQNN(in_dim=in_dim, hidden_dim=64).to(device)
            criterion = GQNNLoss(target_coverage=0.9, lambda_factor=0.01)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            model.train()
            for epoch in range(args.epochs):
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
                picps.append(metrics['PICP'] * 100)
                mpiws.append(metrics['MPIW'])

        mean_picp, mean_mpiw = np.mean(picps), np.mean(mpiws)
        results.append((level, mean_picp, mean_mpiw))
        print(f"{noise_type}={level:.2f} → PICP={mean_picp:.2f}%, MPIW={mean_mpiw:.3f}")

    return results

# ---------------- Save CSV ---------------- #
def save_results_csv(results, noise_type):
    os.makedirs("robust", exist_ok=True)
    filepath = f"robust/robustness_{noise_type}.csv"
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Noise_Level", "PICP", "MPIW"])
        for row in results:
            writer.writerow(row)
    print(f"Saved CSV: {filepath}")

# ---------------- Plot ---------------- #
def plot_results(noise_type, levels, results):
    picps = [r[1] for r in results]
    mpiws = [r[2] for r in results]

    plt.figure(figsize=(6, 4))
    plt.plot(levels, picps, marker='o', label="PICP (%)")
    plt.plot(levels, mpiws, marker='s', label="MPIW")
    plt.xlabel({
        "feature": "Gaussian Noise Std",
        "edge": "Edge Dropout Ratio",
        "target": "Target Noise Std"
    }[noise_type])
    plt.ylabel("Metric Value")
    plt.title(f"Robustness to {noise_type.capitalize()} Noise")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    os.makedirs("./figs", exist_ok=True)
    path = f"./figs/robustness_{noise_type}.png"
    plt.savefig(path)
    print(f"Saved Figure: {path}")
    plt.close()

# ---------------- Main Entry ---------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)

    tasks = {
        "feature": [0.0, 0.1, 0.2, 0.3],
        "edge": [0.0, 0.2, 0.4],
        "target": [0.0, 0.1, 0.2],
    }

    for noise_type, levels in tasks.items():
        print(f"\n===== Running: {noise_type} perturbation =====")
        results = run_robustness_experiment(noise_type, levels, device, args)
        save_results_csv(results, noise_type)
        plot_results(noise_type, levels, results)
