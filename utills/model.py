import sys
sys.path.append('./utils')

import os
import re
import random
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx
import torch.nn.functional as F
import numpy as np
import pandas as pd
import seaborn as sns
import json
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix, coo_matrix
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt
from torch.distributions.normal import Normal
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import from_networkx, to_networkx
from torch_geometric.nn import SAGEConv, GATConv, GCNConv, GraphSAGE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch_geometric.transforms import RandomNodeSplit
from utills.function import augment_features

class GQNN_R(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.conv1 = SAGEConv(in_dim+1, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x, edge_index, tau):
        x = augment_features(x, tau)
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        
        return self.fc(x)
    
class GQNN_N(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, 2)
        
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        
        return self.fc(x)
    
class GNN_CP(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        
        return self.fc(x)
    
class QRLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, y_pred, y_true, tau):
        diff = y_true - y_pred
        loss = torch.where(diff > 0, tau * diff, (tau - 1) * diff)
        
        return torch.mean(loss)
    
class RQRLoss(nn.Module):
    def __init__(self, target=0.9, lambda_factor=1.0, order_penalty=1.0):
        super().__init__()
        self.target = target
        self.lf = lambda_factor
        self.order_penalty = order_penalty
    
    def forward(self, preds, target):
        q1, q2 = preds[:, 0], preds[:, 1]
        diff1 = target - q1
        diff2 = target - q2
        width = q2 - q1
        
        rqr_loss = torch.maximum(diff1 * diff2 * (self.target + 2 * self.lf),
                                diff2 * diff1 * (self.target + 2 * self.lf - 1))
        
        width_loss = self.lf * torch.square(width) * 0.5
        
        # 추가 손실항
        order_penalty_term = self.order_penalty * torch.relu(q1 - q2)
        
        return torch.mean(rqr_loss + width_loss + order_penalty_term)
    
class BayesianLinear(nn.Module):
    """Bayesian Linear Layer"""
    def __init__(self, in_features, out_features, prior_std=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Mean and log variance of weights
        self.w_mu = nn.Parameter(torch.zeros(out_features, in_features))
        self.w_logvar = nn.Parameter(torch.zeros(out_features, in_features))

        # Mean and log variance of biases
        self.b_mu = nn.Parameter(torch.zeros(out_features))
        self.b_logvar = nn.Parameter(torch.zeros(out_features))

        # Prior distribution (Gaussian)
        self.prior = Normal(0, prior_std)

    def forward(self, x):
        # Sample weights and biases
        w_std = torch.exp(0.5 * self.w_logvar)
        b_std = torch.exp(0.5 * self.b_logvar)

        w = self.w_mu + w_std * torch.randn_like(w_std)
        b = self.b_mu + b_std * torch.randn_like(b_std)

        return F.linear(x, w, b)

class BayesianGNN(nn.Module):
    """Bayesian GNN using Bayesian Linear Layer"""
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.fc = BayesianLinear(hidden_dim, 1)  # Bayesian Linear Layer

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return self.fc(x).squeeze()  # (batch_size,)
    
class MCDropoutGNN(nn.Module):
    """MC Dropout 기반 GNN"""
    def __init__(self, in_dim, hidden_dim, dropout=0.2):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)
        self.dropout = dropout

    def forward(self, x, edge_index, training=True):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=training)  # Dropout 유지
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=training)  # Dropout 유지
        return self.fc(x).squeeze()  # (batch_size,)

class GQNN(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.fc_pred = nn.Linear(hidden_dim, 1)
        self.fc_diff = nn.Linear(hidden_dim, 1)
        
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        
        preds = self.fc_pred(x)
        diffs = torch.sigmoid(self.fc_diff(x))
        
        preds_low = preds - diffs
        preds_upper = preds + diffs
        
        return preds_low, preds_upper
    
class GQNNLoss(nn.Module):
    def __init__(self, target_coverage=0.9, lambda_factor=0.1):
        super().__init__()
        self.target_coverage = target_coverage
        self.lf = lambda_factor
    
    def forward(self, preds_low, preds_upper, target):
        # 구간 폭 계산
        diffs = (preds_upper - preds_low) / 2
        
        # 샘플별 커버리지 손실 계산
        below_loss = torch.relu(preds_low - target)  # target < preds_low
        above_loss = torch.relu(target - preds_upper)  # target > preds_upper
        sample_loss = below_loss + above_loss  # 구간 밖일 때 손실
        
        # 커버리지 계산: 구간 안에 있는 샘플 비율
        covered = (preds_low <= target) & (target <= preds_upper)
        current_coverage = covered.float().mean()
        
        # 커버리지 손실: 목표 커버리지와의 차이를 패널티로
        # coverage_penalty = torch.relu(self.target_coverage - current_coverage) ** 2
        coverage_penalty = (self.target_coverage - current_coverage) ** 2
        
        # 폭 패널티
        width_loss = self.lf * 2 * diffs.mean()
        
        # 평균 샘플 손실 (구간 밖 거리 최소화)
        mean_sample_loss = sample_loss.mean()
        
        # 최종 손실: 샘플 손실 + 커버리지 패널티 + 폭 패널티
        return mean_sample_loss + coverage_penalty + width_loss
    
class GQNNLoss2(nn.Module): 
    def __init__(self, lambda_width=0.1, target_coverage=0.9, beta=10.0):
        super().__init__()
        self.target_coverage = target_coverage  # 목표 커버리지
        self.lambda_width = lambda_width        # 구간 폭 패널티 가중치
        self.beta = beta                        # 시그모이드 근사 강도 (클수록 지시 함수에 가까워짐)
    
    def soft_coverage(self, preds_low, preds_upper, target):
        # preds_low <= target <= preds_upper를 시그모이드로 근사
        lower_bound = torch.sigmoid(self.beta * (target - preds_low))  # target >= preds_low
        upper_bound = torch.sigmoid(self.beta * (preds_upper - target))  # target <= preds_upper
        return lower_bound * upper_bound  # 두 조건의 곱으로 커버리지 근사
    
    def forward(self, preds_low, preds_upper, target):
        # 1. 샘플 손실: Softplus로 부드럽게
        below_loss = nn.functional.softplus(preds_low - target)  # target < preds_low
        above_loss = nn.functional.softplus(target - preds_upper)  # target > preds_upper
        sample_loss = (below_loss + above_loss).mean()

        # 2. 커버리지 계산 및 패널티
        soft_covered = self.soft_coverage(preds_low, preds_upper, target)  # 부드러운 커버리지
        current_coverage = soft_covered.mean()
        coverage_penalty = (self.target_coverage - current_coverage) ** 2

        # 3. 구간 폭 패널티
        diffs = preds_upper - preds_low
        width_penalty = self.lambda_width * diffs.mean()

        # 4. 최종 손실
        total_loss = sample_loss + coverage_penalty + width_penalty
        return total_loss