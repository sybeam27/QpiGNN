import os, time, pickle
from datetime import datetime
from itertools import product
import argparse
import seaborn as sns
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PyPDF2 import PdfMerger
import numpy as np
from itertools import product

# 유틸리티 함수 로드
from utills.function import (
    set_seed, generate_graph_data, generate_noisy_graph_data,
    load_county_graph_data, load_twitch_graph_data,
    load_wiki_graph_data, load_trans_graph_data,
    create_ba_graph_pyg, create_er_graph_pyg, create_grid_graph_pyg, create_tree_graph_pyg,
    normalize, split_graph_data, get_gpu_memory, get_cpu_memory, count_parameters,
    evaluate_model_performance, sort_by_y
)

def generate_valid_ablation_configs():
    configs = []
    for dual_output in [True, False]:
        for fixed_margin in [None, 0.05, 0.1]:
            # fixed_margin은 dual_output=True일 때만 의미 있음
            if fixed_margin is not None and not dual_output:
                continue

            for use_sample_loss, use_coverage_loss, use_width_loss in product([True, False], repeat=3):
                # 1. 모든 loss가 꺼져 있으면 스킵
                if not (use_sample_loss or use_coverage_loss or use_width_loss):
                    continue

                # 2. coverage loss만 켜져 있으면 학습 불가 → 스킵
                if use_coverage_loss and not (use_sample_loss or use_width_loss):
                    continue

                config = {
                    'dual_output': dual_output,
                    'fixed_margin': fixed_margin,
                    'use_sample_loss': use_sample_loss,
                    'use_coverage_loss': use_coverage_loss,
                    'use_width_loss': use_width_loss,
                }
                configs.append(config)
    return configs

def evaluate_ablation_performance(preds_low, preds_upper, targets, target=0.9):
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
    
class GQNN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, dual_output=True, fixed_margin=None):
        super().__init__()
        self.dual_output = dual_output
        self.fixed_margin = fixed_margin
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)

        if dual_output:
            self.fc_pred = torch.nn.Linear(hidden_dim, 1)
            if fixed_margin is None:
                self.fc_diff = torch.nn.Linear(hidden_dim, 1)
        else:
            self.fc_low = torch.nn.Linear(hidden_dim, 1)
            self.fc_upper = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        
        if self.dual_output:
            preds = self.fc_pred(x)
            if self.fixed_margin is not None:
                diffs = torch.ones_like(preds) * self.fixed_margin
            else:
                diffs = torch.sigmoid(self.fc_diff(x))
                
            pred_low, pred_upper = preds - diffs, preds + diffs
            return pred_low, pred_upper
        
        else:
            pred_low = self.fc_low(x)
            pred_upper = self.fc_upper(x)
            
            return pred_low, pred_upper
        
class GQNNLoss(torch.nn.Module):
    def __init__(self, target_coverage=0.9, lambda_factor=0.1,
                 use_sample_loss=True, use_coverage_loss=True, use_width_loss=True):
        super().__init__()

        if not (use_sample_loss or use_coverage_loss or use_width_loss):
            raise ValueError("All loss terms are disabled.")

        self.target_coverage = target_coverage
        self.lf = lambda_factor
        self.use_sample_loss = use_sample_loss
        self.use_coverage_loss = use_coverage_loss
        self.use_width_loss = use_width_loss

    def forward(self, preds_low, preds_upper, target):
        diffs = (preds_upper - preds_low) / 2
        loss_terms = []

        sample_loss_val = 0.0
        coverage_penalty_val = 0.0
        width_penalty_val = 0.0

        if self.use_sample_loss:
            below_loss = torch.relu(preds_low - target)
            above_loss = torch.relu(target - preds_upper)
            sample_loss = below_loss + above_loss
            sample_loss_val = sample_loss.mean()
            loss_terms.append(sample_loss_val)

        if self.use_coverage_loss:
            covered = (preds_low <= target) & (target <= preds_upper)
            current_coverage = covered.float().mean()
            coverage_penalty_val = (current_coverage - self.target_coverage) ** 2
            loss_terms.append(coverage_penalty_val)

        if self.use_width_loss:
            width_penalty_val = self.lf * 2 * diffs.mean()
            loss_terms.append(width_penalty_val)
            
        self.last_loss_terms = {
            'sample': sample_loss_val.item() if isinstance(sample_loss_val, torch.Tensor) else 0.0,
            'coverage': coverage_penalty_val.item() if isinstance(coverage_penalty_val, torch.Tensor) else 0.0,
            'width': width_penalty_val.item() if isinstance(width_penalty_val, torch.Tensor) else 0.0,
        }

        return sum(loss_terms)

def save_prediction_plots(phase, x, y, low, upper, true_y, args, color, config_name):
    pdf_dir = args.pdf_dir
    os.makedirs(pdf_dir, exist_ok=True)

    existing_pdf = os.path.join(pdf_dir, f"{args.model}_eval_plots.pdf")
    new_pdf = os.path.join(pdf_dir, f"new_{phase.lower()}_plot.pdf")
    merged_pdf = os.path.join(pdf_dir, "merged_temp.pdf")

    x_st, y_st, (low_r, upper_r) = sort_by_y(x, y, low, upper)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    with PdfPages(new_pdf) as pdf:
        fig, axes = plt.subplots(1, 2, figsize=(16, 4))
        plt.suptitle(f"Ablation: {config_name}, Dataset: {args.dataset}, Time: {timestamp} ({phase})", fontsize=12, fontweight='bold')
        axes[0].scatter(range(len(x_st)), y_st, alpha=0.3, color='blue', s=15)
        axes[0].fill_between(range(len(x_st)), low_r, upper_r, color=color, alpha=0.5)
        axes[1].scatter(range(len(x_st)), true_y, alpha=0.3, color='blue', s=15)
        axes[1].fill_between(range(len(x_st)), low.squeeze(), upper.squeeze(), color=color, alpha=0.5)
        pdf.savefig(fig)
        plt.close(fig)

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

def train_and_evaluate(model, criterion, optimizer, train_data, test_data, args, run, device, results, result_this_run, color, config_name):
    model.train()
    start_time = time.time()
    
    for epoch in range(args.epochs):
        optimizer.zero_grad()
        preds_low, preds_upper = model(train_data.x.to(device), train_data.edge_index.to(device))
        loss = criterion(preds_low, preds_upper, train_data.y.to(device))
        loss.backward()
        optimizer.step()

    # 리소스 및 성능 기록
    training_time = time.time() - start_time
    result_this_run['training_time_sec'] = round(training_time, 2)
    result_this_run['gpu_mem_MB'] = round(get_gpu_memory(), 2)
    result_this_run['cpu_mem_MB'] = round(get_cpu_memory(), 2)
    result_this_run['param_count'] = count_parameters(model)

    model.eval()
    # -- 학습 후
    with torch.no_grad():
        preds_low_train, preds_upper_train = model(train_data.x.to(device), train_data.edge_index.to(device))
        preds_low_test, preds_upper_test = model(test_data.x.to(device), test_data.edge_index.to(device))

    # -- 평가 저장
    result_this_run['train_metrics'] = evaluate_ablation_performance(
        preds_low_train.cpu().numpy(), preds_upper_train.cpu().numpy(), train_data.y.cpu().numpy(), target=args.target_coverage
    )
    result_this_run['test_metrics'] = evaluate_ablation_performance(
        preds_low_test.cpu().numpy(), preds_upper_test.cpu().numpy(), test_data.y.cpu().numpy(), target=args.target_coverage
    )

    if hasattr(criterion, 'last_loss_terms'):
        result_this_run['loss_terms'] = criterion.last_loss_terms
        
    # -- PDF 저장 시도
    if args.pdf:
        save_prediction_plots('Train', train_data.x, train_data.y, preds_low_train.cpu().numpy(), preds_upper_train.cpu().numpy(), train_data.y.cpu().numpy(), args, color, config_name)
        save_prediction_plots('Test', test_data.x, test_data.y, preds_low_test.cpu().numpy(), preds_upper_test.cpu().numpy(), test_data.y.cpu().numpy(), args, color, config_name)


    results[run] = result_this_run
    print(f'Finished training run {run}')


# ----------------------------- 메인 실행 -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='basic')
    parser.add_argument('--model', type=str, default='GQNN')
    parser.add_argument('--pdf', type=bool, default=False)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--lambda_factor', type=float, default=0.01)
    parser.add_argument('--target_coverage', type=float, default=0.9, help='Target coverage level (1 - α)')
    parser.add_argument('--nodes', type=float, default=1000, help='num_nodes')
    parser.add_argument('--noise', type=float, default=0.3, help='noise_level')
    args = parser.parse_args()

    device = torch.device(args.device)
    set_seed(1127)
    
    if args.pdf == True:
        args.pdf_dir = f"./ablation/{args.dataset}/pdf"
        os.makedirs(args.pdf_dir, exist_ok=True)

    # 데이터 로딩
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

    train_data, test_data = split_graph_data(graph_data, test_ratio=0.2)
    train_min, train_max, y_min, y_max = train_data.x.min(), train_data.x.max(), train_data.y.min(), train_data.y.max()
    train_data.x, test_data.x, train_data.y, test_data.y= normalize(train_data.x, train_min, train_max), normalize(test_data.x, train_min, train_max), normalize(train_data.y, y_min, y_max), normalize(test_data.y, y_min, y_max)

    # Ablation 조합 생성
    ablations = generate_valid_ablation_configs()
    print(f"Total valid ablation configs: {len(ablations)}")

    # Ablation 실행
    in_dim = train_data.x.shape[1]
    color_bar = sns.color_palette("Set1", len(ablations))

    for i, config in enumerate(ablations):
        color = color_bar[i]

        # config_name 만들기 (float 값 처리 포함)
        def format_val(v):
            if isinstance(v, bool):
                return int(v)
            elif v is None:
                return "None"
            elif isinstance(v, float):
                return f"{v:.2f}"
            else:
                return str(v)

        config_name = "_".join([f"{k}({format_val(v)})" for k, v in config.items()])
        print(f"\n===== Running: {config_name} =====")

        results = {}
        for run in range(args.runs):
            result_this_run = {}

            model = GQNN(
                in_dim, 64,
                dual_output=config['dual_output'],
                fixed_margin=config['fixed_margin']
            ).to(device)

            criterion = GQNNLoss(
                target_coverage=args.target_coverage,
                lambda_factor=args.lambda_factor,
                use_sample_loss=config['use_sample_loss'],
                use_coverage_loss=config['use_coverage_loss'],
                use_width_loss=config['use_width_loss']
            )

            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)

            train_and_evaluate(
                model, criterion, optimizer,
                train_data, test_data,
                args, run, device,
                results, result_this_run,
                color, config_name
            )

        save_final_results(results, f"./ablation/{args.dataset}/{config_name}.pkl")

