import torch
import torch.nn.functional as F
import argparse
from utills.model import GQNN, GQNNLoss  # 너의 기존 GQNN 정의 import
from utills.function import generate_graph_data, split_graph_data, normalize
import numpy as np
import matplotlib.pyplot as plt

def evaluate_model_performance(preds_low, preds_upper, targets, target=0.9):
    # PICP (Prediction Interval Coverage Probability): 커버리지 계산
    picp = np.mean((targets >= preds_low) & (targets <= preds_upper))
    
    # NMPIW (Normalized Mean Prediction Interval Width): 정규화된 구간 너비
    interval_width = np.mean(preds_upper - preds_low)
    data_range = np.max(targets) - np.min(targets)
    nmpiw = interval_width / data_range if data_range > 0 else interval_width  # 범위가 0일 경우 대비
    
    # CWC (Coverage Width-based Criterion): NMPIW와 커버리지 패널티 결합
    mu = target  # 목표 커버리지를 target으로 설정
    # gamma (float): CWC의 패널티 강도 하이퍼파라미터 (기본값: 1.0)
    # eta (float): CWC의 지수 함수 감쇠 속도 하이퍼파라미터 (기본값: 10.0)
    gamma = 1
    eta = 10
    penalty = 1 + gamma * np.exp(-eta * (picp - mu))
    cwc = nmpiw * penalty
    
    print(f"종합 - CWC ⬇: {cwc:.4f}")
    print(f"예측 관련 - PICP ⬆: {picp:.4f}")
    print(f"구간 관련 - MPIW ⬇: {interval_width:.4f}")

    
    return {
        "PCIP": picp,
        'MPIW': interval_width, 
        "CWC": cwc,
    }
    
# ---------- perturbation functions ---------- #
def add_gaussian_noise(x, std):
    return x + torch.randn_like(x) * std

def edge_dropout(edge_index, p):
    num_edges = edge_index.size(1)
    mask = torch.rand(num_edges) > p
    return edge_index[:, mask]

def add_target_noise(y, std):
    return y + torch.randn_like(y) * std

# ---------- experiment loop ---------- #
def run_robustness_experiment(noise_type, levels, device, args):
    base_graph = generate_graph_data(num_nodes=1000)
    train_data, test_data = split_graph_data(base_graph, test_ratio=0.2)
    in_dim = train_data.x.shape[1]
    results = []

    for level in levels:
        train_data_ = train_data.clone()
        test_data_ = test_data.clone()

        # perturb
        if noise_type == "feature":
            train_data_.x = add_gaussian_noise(train_data_.x, std=level)
        elif noise_type == "edge":
            train_data_.edge_index = edge_dropout(train_data.edge_index, p=level)
        elif noise_type == "target":
            train_data_.y = add_target_noise(train_data.y, std=level)

        # normalize
        train_data_.x = normalize(train_data_.x)
        test_data_.x = normalize(test_data_.x)
        train_data_.y = normalize(train_data_.y)
        test_data_.y = normalize(test_data_.y)

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

        results.append((level, np.mean(picps), np.mean(mpiws)))
    return results

# ---------- plot & save ---------- #
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
    print(f"Saved: {path}")
    plt.close()

# ---------- main ---------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)

    # 실험 설정
    tasks = {
        "feature": [0.0, 0.1, 0.2, 0.3],
        "edge": [0.0, 0.2, 0.4],
        "target": [0.0, 0.1, 0.2],
    }

    for noise_type, levels in tasks.items():
        print(f"\nRunning: {noise_type} perturbation...")
        results = run_robustness_experiment(noise_type, levels, device, args)
        for l, p, m in results:
            print(f"{noise_type}={l:.2f} → PICP={p:.2f}%, MPIW={m:.3f}")
        plot_results(noise_type, levels, results)