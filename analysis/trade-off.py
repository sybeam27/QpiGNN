import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

sys.path.append('..')
from utills.model import GQNN, GQNNLoss
from utills.function import (
    create_er_graph_pyg, split_graph_data, normalize, evaluate_model_performance
)

def run_tradeoff_analysis(tau=0.9, lambda_list=None, runs=5, epochs=500, hidden_dim=64):
    if lambda_list is None:
        lambda_list = np.round(np.linspace(0.1, 0.7, 15), 5).tolist()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data = create_er_graph_pyg(n=1000)
    train_data, test_data = split_graph_data(data, test_ratio=0.2)

    x_min, x_max = train_data.x.min(), train_data.x.max()
    y_min, y_max = train_data.y.min(), train_data.y.max()

    train_data.x = normalize(train_data.x, x_min, x_max)
    test_data.x = normalize(test_data.x, x_min, x_max)
    train_data.y = normalize(train_data.y, y_min, y_max)
    test_data.y = normalize(test_data.y, y_min, y_max)

    in_dim = train_data.x.shape[1]
    results = []

    for lam in lambda_list:
        picps, mpiws = [], []

        for _ in range(runs):
            model = GQNN(in_dim=in_dim, hidden_dim=hidden_dim).to(device)
            criterion = GQNNLoss(target_coverage=tau, lambda_factor=lam)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)

            for epoch in range(epochs):
                model.train()
                optimizer.zero_grad()
                pl, pu = model(train_data.x.to(device), train_data.edge_index.to(device))
                loss = criterion(pl, pu, train_data.y.to(device))
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                pl, pu = model(test_data.x.to(device), test_data.edge_index.to(device))
                metrics = evaluate_model_performance(
                    pl.cpu().numpy(), pu.cpu().numpy(), test_data.y.cpu().numpy(), target=tau
                )

            picps.append(metrics["PICP"])
            mpiws.append(metrics["MPIW"])

        results.append({
            "lambda": lam,
            "PICP": np.mean(picps),
            "MPIW": np.mean(mpiws),
            "std_PICP": np.std(picps),
            "std_MPIW": np.std(mpiws)
        })

    return pd.DataFrame(results)

def plot_tradeoff(df, tau):
    plt.figure(figsize=(7, 4))
    plt.plot(df['lambda'], df['PICP'], 'o-', label="PICP", color='tab:blue')
    plt.fill_between(df['lambda'], df['PICP'] - df['std_PICP'], df['PICP'] + df['std_PICP'], color='tab:blue', alpha=0.2)

    plt.plot(df['lambda'], df['MPIW'], 's--', label="MPIW", color='tab:red')
    plt.fill_between(df['lambda'], df['MPIW'] - df['std_MPIW'], df['MPIW'] + df['std_MPIW'], color='tab:red', alpha=0.2)

    # plt.xlabel("Lambda ($\\lambda$)", size = 10)
    plt.ylabel("Metric", size = 11)
    plt.title(f"PCIP-MPIW Trade-off ({tau})", size=13)
    plt.legend(fontsize = 11)
    plt.grid(True)
    plt.tight_layout()

    os.makedirs("tradeoff", exist_ok=True)
    plt.savefig("tradeoff/tradeoff_plot.png")
    print("Saved: tradeoff/tradeoff_plot.png")
    plt.show()

if __name__ == "__main__":
    df = run_tradeoff_analysis()
    os.makedirs("tradeoff", exist_ok=True)
    df.to_csv("tradeoff/tradeoff_results.csv", index=False)
    print("Saved: tradeoff/tradeoff_results.csv")
    plot_tradeoff(df, tau=0.9)
