import torch
import numpy as np
import os
import pandas as pd
from utills.model import GQNN, GQNNLoss
from utills.function import (
    generate_graph_data, create_er_graph_pyg, create_grid_graph_pyg, create_tree_graph_pyg,
    split_graph_data, normalize, evaluate_model_performance
)

def load_graph_by_name(name, num_nodes=1000):
    if name == 'ER':
        return create_er_graph_pyg(n=num_nodes)
    elif name == 'grid':
        return create_grid_graph_pyg()
    elif name == 'tree':
        return create_tree_graph_pyg()
    elif name == 'basic':
        return generate_graph_data(num_nodes)
    else:
        raise ValueError(f"Unknown dataset: {name}")

def preprocess_data(data):
    train_data, test_data = split_graph_data(data, test_ratio=0.2)
    x_min, x_max = train_data.x.min(), train_data.x.max()
    y_min, y_max = train_data.y.min(), train_data.y.max()

    train_data.x = normalize(train_data.x, x_min, x_max)
    test_data.x = normalize(test_data.x, x_min, x_max)
    train_data.y = normalize(train_data.y, y_min, y_max)
    test_data.y = normalize(test_data.y, y_min, y_max)
    return train_data, test_data

def train_model(train_data, in_dim, device, tau=0.9, lam=0.01, hidden_dim=64, epochs=500):
    model = GQNN(in_dim=in_dim, hidden_dim=hidden_dim).to(device)
    criterion = GQNNLoss(target_coverage=tau, lambda_factor=lam)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pl, pu = model(train_data.x.to(device), train_data.edge_index.to(device))
        loss = criterion(pl, pu, train_data.y.to(device))
        loss.backward()
        optimizer.step()

    return model

def evaluate(model, test_data, device, tau):
    model.eval()
    with torch.no_grad():
        pl, pu = model(test_data.x.to(device), test_data.edge_index.to(device))
        metrics = evaluate_model_performance(
            pl.cpu().numpy(), pu.cpu().numpy(), test_data.y.cpu().numpy(), target=tau
        )
    return metrics

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tau = 0.9
    lam = 0.03  # for ER
    runs = 10
    source_dataset = 'ER'
    target_datasets = ['ER', 'grid', 'tree', 'basic']

    # Load and preprocess all test data once
    target_testsets = {}
    for target in target_datasets:
        data = load_graph_by_name(target)
        _, test_data = preprocess_data(data)
        target_testsets[target] = test_data

    all_results = []

    for run in range(1, runs + 1):
        print(f"\n=== Run {run}/{runs}: Training on {source_dataset} ===")
        source_data = load_graph_by_name(source_dataset)
        train_data, _ = preprocess_data(source_data)
        in_dim = train_data.x.shape[1]

        model = train_model(train_data, in_dim, device, tau=tau, lam=lam)

        for target, test_data in target_testsets.items():
            metrics = evaluate(model, test_data, device, tau)
            print(f"[Run {run}] {target}: PICP={metrics['PICP']:.3f}, MPIW={metrics['MPIW']:.3f}")
            all_results.append({
                'Run': run,
                'Target': target,
                'PICP': metrics['PICP'],
                'MPIW': metrics['MPIW']
            })

    os.makedirs("generalization", exist_ok=True)
    df = pd.DataFrame(all_results)
    df.to_csv("generalization/generalization_runwise.csv", index=False)

    # 평균 결과 저장
    summary = df.groupby("Target")[["PICP", "MPIW"]].mean().reset_index()
    summary.to_csv("generalization/generalization_avg.csv", index=False)
    print("\n=== Average Results ===")
    print(summary)
