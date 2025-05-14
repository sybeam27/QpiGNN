import torch
import numpy as np
import os
import sys
import pandas as pd
import torch.optim as optim

sys.path.append('..')
from utils.model import GQNN, GQNNLoss
from utils.function import (
    set_seed, generate_graph_data, create_er_graph_pyg, create_ba_graph_pyg, create_grid_graph_pyg, create_tree_graph_pyg,
    split_graph_data, normalize, evaluate_model_performance
)

set_seed(1127)  

def load_graph_by_name(name, num_nodes=1000):
    if name == 'ER':
        return create_er_graph_pyg(n=num_nodes)
    elif name == 'BA':
        return create_ba_graph_pyg(n=num_nodes)
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

def train_model(train_data, in_dim, device, optimal_lambda, tau=0.9, epochs=500):
    model = GQNN(in_dim=in_dim, hidden_dim=64).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    criterion = GQNNLoss(target_coverage=tau, lambda_factor=optimal_lambda)
        
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

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from math import pi
import os

plt.rcParams["font.family"] = "Times New Roman"


def plot_generalization_combined_with_consistent_colors(df, metric='PICP'):
    targets = df['Target'].unique().tolist()
    sources = df['Source'].unique().tolist()

    # Define a consistent color palette for sources
    palette = sns.color_palette("Set1", n_colors=len(sources))
    color_map = dict(zip(sources, palette))

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # -------- Barplot --------
    df_bar = df[['Source', 'Target', metric]].copy()
    df_bar['Metric'] = metric
    df_bar = df_bar.rename(columns={metric: "Value"})
    
    sns.barplot(
        data=df_bar,
        x='Source', y='Value', hue='Target',
        palette='Set1', ax=axes[0]
    )
    axes[0].set_title(f"{metric} Bar Plot")
    axes[0].set_ylabel(metric)
    axes[0].grid(True)
    axes[0].set_ylim(0, 1.2 if metric == 'PICP' else None)
    axes[0].legend_.remove()

    # -------- Radar Chart --------
    categories = targets
    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    ax_radar = plt.subplot(1, 2, 2, polar=True)

    for source in sources:
        values = df[df['Source'] == source][metric].tolist()
        values += values[:1]
        ax_radar.plot(angles, values, marker='o', label=source, color=color_map[source])
        ax_radar.fill(angles, values, alpha=0.1, color=color_map[source])

    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(categories)
    ax_radar.set_title(f"{metric} Radar Plot")
    ax_radar.set_ylim(0, 1.1 if metric == 'PICP' else None)

    # Legend
    handles = [plt.Line2D([], [], marker='o', color=color_map[src], label=src) for src in sources]
    fig.legend(handles=handles, loc='upper center', ncol=len(sources), bbox_to_anchor=(0.5, 1.05), frameon=False)

    # Save and show
    os.makedirs("general/figs", exist_ok=True)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    filename = f"general/figs/barplusradar_{metric.lower()}_unified.png"
    plt.savefig(filename, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.show()
    plt.close()

def plot_radar_chart_with_values(df, metric='PICP'):
    targets = df['Target'].unique().tolist()
    sources = df['Source'].unique().tolist()

    # Define a consistent color palette for sources
    palette = sns.color_palette("Set1", n_colors=len(sources))
    color_map = dict(zip(sources, palette))

    # Radar chart setup
    categories = targets
    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 6), subplot_kw=dict(polar=True))

    for source in sources:
        values = df[df['Source'] == source][metric].tolist()
        values += values[:1]
        ax.plot(angles, values, marker='o', label=source, color=color_map[source])
        ax.fill(angles, values, alpha=0.1, color=color_map[source])

        # # Annotate values
        # for angle, val in zip(angles, values):
        #     ax.text(angle, val + 0.02, f"{val:.2f}", ha='center', va='bottom', fontsize=9, color=color_map[source])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=15)
    ax.set_title(f"{metric} Radar Chart", fontsize=14, pad=24)
    ax.set_ylim(0, 1.1 if metric == 'PICP' else None)
    ax.grid(True)

    # Legend
    ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1), fontsize=13)

    # Save and show
    os.makedirs("general/figs", exist_ok=True)
    filename = f"general/figs/radar_{metric.lower()}_with_values.png"
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.show()
    plt.close()
    
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tau = 0.9
    runs = 10
    source_datasets = ['ER', 'BA', 'grid', 'tree', 'basic']
    target_datasets = ['ER', 'BA', 'grid', 'tree', 'basic']
    # df = pd.read_csv("./lambda/syn/lambda_optimized_results.csv")

    # 미리 target dataset 전체 준비
    target_testsets = {}
    for target in target_datasets:
        data = load_graph_by_name(target)
        _, test_data = preprocess_data(data)
        target_testsets[target] = test_data

    all_results = []

    for source in source_datasets:
        # optimal_lambda = df[df['Dataset'] == source]['Best Lambda'].values[0]
        
        print(f"\n=== Source Dataset: {source} ===")
        for run in range(1, runs + 1):
            print(f"→ Run {run}/{runs}")
            source_data = load_graph_by_name(source)
            train_data, _ = preprocess_data(source_data)
            in_dim = train_data.x.shape[1]

            model = train_model(train_data, in_dim, device, optimal_lambda=0.5, tau=tau)

            for target, test_data in target_testsets.items():
                metrics = evaluate(model, test_data, device, tau)
                print(f"[{source}→{target}] PICP={metrics['PICP']:.3f}, MPIW={metrics['MPIW']:.3f}")
                all_results.append({
                    'Source': source,
                    'Run': run,
                    'Target': target,
                    'PICP': metrics['PICP'],
                    'MPIW': metrics['MPIW']
                })

    os.makedirs("general", exist_ok=True)
    df = pd.DataFrame(all_results)
    df.to_csv("general/generalization_all_runs.csv", index=False)

    summary = df.groupby(["Source", "Target"])[["PICP", "MPIW"]].mean().reset_index()
    summary.to_csv("general/generalization_avg.csv", index=False)

    print("\n=== Generalization Summary ===")
    print(summary)
    
    # 결과 시각화
    plot_radar_chart_with_values(summary, metric='PICP')
    plot_radar_chart_with_values(summary, metric='MPIW')



