
import numpy as np
import torch
import matplotlib.pyplot as plt
from utills.function import coverage_width, evaluate_model_performance
from utills.model import GQNNLoss

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from utills.function import coverage_width, evaluate_model_performance
from utills.model import GQNNLoss, GQNN

def evaluate_calibration(model, data, tau_list, device):
    """
    Evaluate empirical coverage at different target levels.
    """
    model.eval()
    results = []

    with torch.no_grad():
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        y_true = data.y.cpu().numpy()

        for tau in tau_list:
            # Reuse trained model; we only adjust target in loss for evaluation
            preds_low, preds_up = model(x, edge_index)
            preds_low = preds_low.cpu().numpy()
            preds_up = preds_up.cpu().numpy()

            inside = (y_true >= preds_low) & (y_true <= preds_up)
            empirical_coverage = np.mean(inside)

            results.append((tau, empirical_coverage))

    return results

def plot_calibration_curve(results):
    """
    Plot calibration curve comparing target and empirical coverage.
    """
    taus = [r[0] for r in results]
    empirical = [r[1] for r in results]

    plt.figure(figsize=(5, 5))
    plt.plot(taus, empirical, 'o-', label="Empirical Coverage")
    plt.plot([0, 1], [0, 1], '--', color='gray', label="Ideal Calibration")
    plt.xlabel("Target Coverage (1 - α)")
    plt.ylabel("Empirical Coverage")
    plt.title("Calibration Curve")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    os.makedirs("calibration", exist_ok=True)
    plt.savefig("calibration/calibration_curve.png")
    print("Saved: calibration/calibration_curve.png")
    plt.show()

if __name__ == "__main__":
    # === 필요한 객체 설정 ===
    from utills.function import generate_graph_data, normalize, split_graph_data, create_er_graph_pyg

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load or generate data
    # data = generate_graph_data(num_nodes=1000)
    data = create_er_graph_pyg(n=1000)
    train_data, test_data = split_graph_data(data, test_ratio=0.2)

    # Normalize
    x_min, x_max = train_data.x.min(), train_data.x.max()
    y_min, y_max = train_data.y.min(), train_data.y.max()
    train_data.x = normalize(train_data.x, x_min, x_max)
    test_data.x = normalize(test_data.x, x_min, x_max)
    train_data.y = normalize(train_data.y, y_min, y_max)
    test_data.y = normalize(test_data.y, y_min, y_max)

    # Train a model (optional: replace with pre-trained model)
    in_dim = train_data.x.shape[1]
    model = GQNN(in_dim=in_dim, hidden_dim=64).to(device)
    criterion = GQNNLoss(target_coverage=0.9, lambda_factor=0.03)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(500):
        model.train()
        optimizer.zero_grad()
        pl, pu = model(train_data.x.to(device), train_data.edge_index.to(device))
        loss = criterion(pl, pu, train_data.y.to(device))
        loss.backward()
        optimizer.step()

    # === Calibration Analysis ===
    tau_list = np.linspace(0.5, 0.95, 10).tolist()
    calib_results = evaluate_calibration(model, test_data, tau_list, device)
    plot_calibration_curve(calib_results)