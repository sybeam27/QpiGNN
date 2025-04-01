import os
import sys
import re
import random
import argparse
from tqdm import tqdm
import time
import psutil
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

def get_gpu_memory():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
    return 0

def get_cpu_memory():
    return psutil.Process().memory_info().rss / (1024 ** 2)  # MB

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def generate_graph_data(num_nodes=1000):
    np.random.seed(1127)
    torch.manual_seed(1127)
    
    # 비선형 데이터
    X = np.random.rand(num_nodes, 5)  # 5차원
    y = np.sin(X[:, 0] * 3) + 0.1 * np.random.rand(num_nodes)
    
    edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2))
    
    # PyG 데이터 변환
    return Data(
        x = torch.tensor(X, dtype = torch.float32),
        edge_index= edge_index,
        y = torch.tensor(y, dtype = torch.float32).unsqueeze(1),
    )

def generate_noisy_graph_data(num_nodes=1000, noise_type="gaussian", noise_level=0.1, outlier_ratio=0.05):
    """
    다양한 노이즈를 추가하여 그래프 데이터를 생성하는 함수

    Args:
    - num_nodes (int): 노드 개수
    - noise_type (str): 추가할 노이즈 유형 ("gaussian", "uniform", "outlier", "edge_noise")
    - noise_level (float): 노이즈의 강도 (가우시안 및 유니폼 노이즈)
    - outlier_ratio (float): 이상치(outlier) 비율

    Returns:
    - PyG Data 객체
    """
    np.random.seed(1127)
    torch.manual_seed(1127)
    
    X = np.random.rand(num_nodes, 5)  # 5차원 특징
    y = np.sin(X[:, 0] * 3) + 0.1 * np.random.rand(num_nodes)  # 기본 타겟
    
    if noise_type == "gaussian":
        y += np.random.normal(0, noise_level, size=num_nodes)
    elif noise_type == "uniform":
        y += np.random.uniform(-noise_level, noise_level, size=num_nodes)
    elif noise_type == "outlier":
        num_outliers = int(num_nodes * outlier_ratio)
        outlier_indices = np.random.choice(num_nodes, num_outliers, replace=False)
        y[outlier_indices] += np.random.normal(3, 1.0, size=num_outliers)  # 극단적인 변화

    # 그래프 구조적 노이즈 (엣지 변경)
    edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2))
    if noise_type == "edge":
        # 엣지에 무작위 잡음을 추가하여 구조적 변형 수행
        num_noisy_edges = int(edge_index.shape[1] * noise_level)
        noise_indices = np.random.choice(edge_index.shape[1], num_noisy_edges, replace=False)
        edge_index[:, noise_indices] = torch.randint(0, num_nodes, (2, num_noisy_edges))

    return Data(
        x=torch.tensor(X, dtype=torch.float32),
        edge_index=edge_index,
        y=torch.tensor(y, dtype=torch.float32).unsqueeze(1),
    )

def max_normalize(x):
    return x / np.max(np.abs(x)) if np.max(np.abs(x)) != 0 else x

def std_normalize(x):
    return (x - np.mean(x)) / np.std(x) if np.std(x) != 0 else np.zeros(len(x))

def int_normalize(x):
    return ((x - np.min(x)) / (np.max(x) - np.min(x)) * 2 - 1) if np.std(x) != 0 else np.zeros(len(x))

def normalize(tensor, min_val, max_val):
    return (tensor - min_val) / (max_val - min_val)

def simulate_ising(n, h0, J):
    G = nx.grid_2d_graph(n, n)
    l = np.linspace(-1.0, 1.0, n)
    
    s = np.random.choice([-1, 1], size=(n, n))
    # Placeholder for metropolis algorithm
    y = s.flatten()
    f = [[l[i], l[j]] for j in range(n) for i in range(n)]
    
    return G, [nx.to_scipy_sparse_matrix(G)], y, f

def parse_mean_fill(series, normalize=False):
    series = series.replace({',': ''}, regex=True)
    series = pd.to_numeric(series, errors='coerce')
    mean_val = series.mean()
    series.fillna(mean_val, inplace=True)
    
    if normalize:
        series = (series - mean_val) / series.std()
    
    return series.values

def read_county(prediction, year):
    adj = pd.read_csv("dataset/election/adjacency.txt", header=None, sep="\t", dtype=str, encoding="ISO-8859-1")
    fips2cty = {row[1]: row[0] for _, row in adj.iterrows() if pd.notna(row[1])}
    
    hh = adj.iloc[:, 1].ffill().astype(int)
    tt = adj.iloc[:, 3].astype(int)
    
    fips = sorted(set(hh).union(set(tt)))
    id2num = {id_: num for num, id_ in enumerate(fips)}
    
    G = nx.Graph()
    G.add_nodes_from(range(len(id2num)))
    G.add_edges_from([(id2num[h], id2num[t]) for h, t in zip(hh, tt)])
    
    # Load datasets
    VOT = pd.read_csv("dataset/election/election.csv")
    ICM = pd.read_csv("dataset/election/income.csv")
    POP = pd.read_csv("dataset/election/population.csv")
    EDU = pd.read_csv("dataset/election/education.csv")
    UEP = pd.read_csv("dataset/election/unemployment.csv")
    
    cty = pd.DataFrame({'FIPS': fips, 'County': [fips2cty.get(f, '') for f in fips]})
    vot = VOT[['fips_code', f'dem_{year}', f'gop_{year}']].rename(columns={'fips_code': 'FIPS'})
    icm = ICM[['FIPS', f'MedianIncome{min(max(2011, year), 2018)}']]
    pop = POP[['FIPS', f'R_NET_MIG_{min(max(2011, year), 2018)}', f'R_birth_{min(max(2011, year), 2018)}', f'R_death_{min(max(2011, year), 2018)}']]
    edu = EDU[['FIPS', f'BachelorRate{year}']]
    uep = UEP[['FIPS', f'Unemployment_rate_{min(max(2007, year), 2018)}']]
    
    dat = cty.merge(vot, on='FIPS', how='left')
    dat = dat.merge(icm, on='FIPS', how='left')
    dat = dat.merge(pop, on='FIPS', how='left')
    dat = dat.merge(edu, on='FIPS', how='left')
    dat = dat.merge(uep, on='FIPS', how='left')
    
    # Extract features and labels
    dem = parse_mean_fill(dat.iloc[:, 2])
    gop = parse_mean_fill(dat.iloc[:, 3])
    
    ff = np.zeros((len(dat), 7), dtype=np.float32)
    for i in range(6):
        ff[:, i] = parse_mean_fill(dat.iloc[:, i + 4], normalize=True)
    
    ff[:, 6] = (gop - dem) / (gop + dem)
    
    label_mapping = {
        "income": 0, "migration": 1, "birth": 2, "death": 3,
        "education": 4, "unemployment": 5, "election": 6
    }
    
    if prediction not in label_mapping:
        raise ValueError("Unexpected prediction type")
    
    pos = label_mapping[prediction]
    y = ff[:, pos]
    f = [np.concatenate((ff[i, :pos], ff[i, pos + 1:])) for i in range(len(dat))]
    
    return G, [csr_matrix(nx.adjacency_matrix(G))], y, f

def load_county_graph_data(prediction: str, year: int):
    G, A, labels, feats = read_county(prediction, year)

    mapping = {node: i for i, node in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, mapping)

    pyg_data = from_networkx(G)

    edge_index = pyg_data.edge_index
    sorted_edges = torch.sort(edge_index, dim=0)[0]  # (u, v)와 (v, u)를 정렬
    unique_edges = torch.unique(sorted_edges, dim=1)  # 고유 엣지만 유지
    pyg_data.edge_index = unique_edges  # 중복 제거된 edge_index 적용

    feats_array = np.array(feats)  # 리스트를 하나의 ndarray로 합치기
    pyg_data.x = torch.tensor(feats_array, dtype=torch.float)
    
    # for i, f in enumerate(feats):
        # print(f"Index {i}, shape: {f.shape}")

    pyg_data.y = torch.tensor(labels, dtype=torch.float).view(-1, 1)

    return pyg_data

def read_transportation_network(network_name, net_skips, net_cols, netf_cols, flow_skips, flow_cols, V_range):
    # Load data
    dat_net = pd.read_csv(f"dataset/transportation/{network_name}/{network_name}_net.tntp", 
                           skiprows=net_skips, sep='\s+', usecols=net_cols, header=None).values
    dat_netf = pd.read_csv(f"dataset/transportation/{network_name}/{network_name}_net.tntp", 
                            skiprows=net_skips, sep='\s+', usecols=netf_cols, header=None).values
    dat_flow = pd.read_csv(f"dataset/transportation/{network_name}/{network_name}_flow.tntp", 
                            skiprows=flow_skips, sep='\s+', usecols=flow_cols, header=None).values
    
    # Map node labels to indices
    lb2id = {v: i for i, v in enumerate(V_range, start=1)}
    NV = len(V_range)
    
    # Create directed graph
    g = nx.DiGraph()
    g.add_nodes_from(range(1, NV + 1))
    
    for src, dst in dat_net:
        if src in lb2id and dst in lb2id:
            g.add_edge(lb2id[src], lb2id[dst])
    
    # Edge labels
    flow_dict = {}
    for src, dst, flow in dat_flow:
        if src in lb2id and dst in lb2id:
            flow_dict[(lb2id[src], lb2id[dst])] = flow
    
    y = np.array([flow_dict.get((e[0], e[1]), 0) for e in g.edges()])
    y = (y - np.mean(y)) / np.std(y)  # Standard normalization
    
    # Edge features
    netf_dict = {}
    for i in range(len(dat_net)):
        src, dst = dat_net[i]
        if src in lb2id and dst in lb2id:
            netf_dict[(lb2id[src], lb2id[dst])] = dat_netf[i]
    
    ff = np.array([netf_dict[e] for e in g.edges()])
    mean_ff = np.mean(ff, axis=0)
    std_ff = np.std(ff, axis=0)
    std_ff[std_ff == 0] = 1  # Prevent division by zero
    netf = (ff - mean_ff) / std_ff  
    
    f = list(netf)
    
    # Line graph transformation
    G1 = nx.Graph()
    G2 = nx.Graph()
    sorted_edges = sorted(g.edges())
    tuple2id = {e: i for i, e in enumerate(sorted_edges)}
    
    for u in g.nodes:
        innbrs = list(g.predecessors(u))
        outnbrs = list(g.successors(u))
        
        for v in innbrs:
            for w in outnbrs:
                if (v, u) in tuple2id and (u, w) in tuple2id:
                    G1.add_edge(tuple2id[(v, u)], tuple2id[(u, w)])
        
        for v in innbrs:
            for w in innbrs:
                if w > v and (v, u) in tuple2id and (w, u) in tuple2id:
                    G2.add_edge(tuple2id[(v, u)], tuple2id[(w, u)])
        
        for v in outnbrs:
            for w in outnbrs:
                if w > v and (u, v) in tuple2id and (u, w) in tuple2id:
                    G2.add_edge(tuple2id[(u, v)], tuple2id[(u, w)])
                    
    size = max(len(G1.nodes), len(G2.nodes))
    A1 = np.zeros((size, size))
    A2 = np.zeros((size, size))
    
    A1[:nx.number_of_nodes(G1), :nx.number_of_nodes(G1)] = nx.adjacency_matrix(G1).todense()
    A2[:nx.number_of_nodes(G2), :nx.number_of_nodes(G2)] = nx.adjacency_matrix(G2).todense()
    
    A = A1 + A2
    
    return nx.Graph(A), A, y, f

def load_trans_graph_data(city: str):
    if city == 'Anaheim':
        G, A, labels, feats = read_transportation_network(city, 8, [0, 1], [2, 3, 4, 7], 6, [0, 1, 3], range(1, 417))
    elif city == 'ChicagoSketch':
        G, A, labels, feats = read_transportation_network(city, 7, [0, 1], [2, 3, 4, 7], 1, [0, 1, 2], range(388, 934))

    mapping = {node: i for i, node in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, mapping)

    pyg_data = from_networkx(G)

    edge_index = pyg_data.edge_index
    sorted_edges = torch.sort(edge_index, dim=0)[0]  # (u, v)와 (v, u)를 정렬
    unique_edges = torch.unique(sorted_edges, dim=1)  # 고유 엣지만 유지
    pyg_data.edge_index = unique_edges  # 중복 제거된 edge_index 적용

    feats_array = np.array(feats)  # 리스트를 하나의 ndarray로 합치기
    pyg_data.x = torch.tensor(feats_array, dtype=torch.float)

    pyg_data.y = torch.tensor(labels, dtype=torch.float).view(-1, 1)

    return pyg_data

def read_twitch_network(cnm, dim_reduction=False, dim_embed=8):
    feats_all = []
    countries = ["DE", "ENGB", "ES", "FR", "PTBR", "RU"]
    
    for cn in countries:
        with open(f"dataset/twitch/{cn}/musae_{cn}_features.json", "r") as f:
            feats = json.load(f)
        feats_all.extend(feats.values())

    ndim = max(np.concatenate(feats_all)) + 1

    def feat_encode(feat_list):
        """특징 벡터를 원핫 인코딩 형태로 변환"""
        vv = np.zeros(ndim, dtype=np.float32)
        valid_indices = np.array(feat_list)
        
        if np.any(valid_indices >= ndim):
            raise ValueError(f"Index out of bounds! Max index: {max(valid_indices)}, ndim: {ndim}")
        
        vv[valid_indices] = 1.0
        return vv

    f_all = list(map(feat_encode, feats_all))

    with open(f"dataset/twitch/{cnm}/musae_{cnm}_features.json", "r") as f:
        feats = json.load(f)

    id2ft = {int(k) + 1: v for k, v in feats.items()}
    n = len(id2ft)
    assert min(id2ft.keys()) == 1 and max(id2ft.keys()) == n

    f = [feat_encode(id2ft[i]) for i in sorted(id2ft.keys())]

    if dim_reduction:
        f_matrix = np.stack(f_all, axis=1)
        U, S, Vt = svds(f_matrix, k=dim_embed)
        U *= np.sign(np.sum(U, axis=0))  # sign correction
        f = [U.T @ f_ for f_ in f]

    g = nx.Graph()
    g.add_nodes_from(range(1, len(f) + 1))

    links = pd.read_csv(f"dataset/twitch/{cnm}/musae_{cnm}_edges.csv")
    for _, row in links.iterrows():
        g.add_edge(row["from"] + 1, row["to"] + 1)

    trgts = pd.read_csv(f"dataset/twitch/{cnm}/musae_{cnm}_target.csv")
    nid2views = dict(zip(trgts["new_id"], trgts["views"]))
    y = std_normalize(np.log([nid2views[i - 1] + 1.0 for i in range(1, g.number_of_nodes() + 1)]))

    return g, [csr_matrix(nx.adjacency_matrix(g))], y, f

def load_twitch_graph_data(cnm: str):
    G, A, labels, feats = read_twitch_network(cnm)

    mapping = {node: i for i, node in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, mapping)

    pyg_data = from_networkx(G)

    edge_index = pyg_data.edge_index
    sorted_edges = torch.sort(edge_index, dim=0)[0]  # (u, v)와 (v, u)를 정렬
    unique_edges = torch.unique(sorted_edges, dim=1)  # 고유 엣지만 유지
    pyg_data.edge_index = unique_edges  # 중복 제거된 edge_index 적용

    pyg_data.x = torch.tensor(feats, dtype=torch.float)

    pyg_data.y = torch.tensor(labels, dtype=torch.float).view(-1, 1)

    return pyg_data

def load_wiki_graph_data(category):
    edge_path = f'dataset/wikipedia/{category}/musae_{category}_edges.csv'
    feature_path = f'dataset/wikipedia/{category}/musae_{category}_features.json'
    target_path = f'dataset/wikipedia/{category}/musae_{category}_target.csv'
    
    # 엣지 데이터 로드
    edge_df = pd.read_csv(edge_path)
    edge_index = torch.tensor(edge_df.values.T, dtype=torch.long)
    
    # 피처 데이터 로드
    with open(feature_path, "r") as f:
        features_dict = json.load(f)
    
    node_ids = sorted(map(int, features_dict.keys()))  # 노드 ID 정렬
    node_id_map = {old_id: new_id for new_id, old_id in enumerate(node_ids)}
    
    num_nodes = len(node_ids)
    num_features = max(max(v) for v in features_dict.values()) + 1  # 가장 큰 feature index 찾기
    x = torch.zeros((num_nodes, num_features), dtype=torch.float32)
    
    for node, features in features_dict.items():
        new_id = node_id_map[int(node)]  # 노드 ID 변환
        x[new_id, features] = 1.0  # One-hot 인코딩
    
    # 타겟 데이터 로드
    target_df = pd.read_csv(target_path)
    target_df["id"] = target_df["id"].map(node_id_map)  # 노드 ID 변환
    target_df = target_df.dropna().astype(int)  # 변환되지 않은 노드 제거
    
    # y = torch.zeros(num_nodes, dtype=torch.long)
    y = torch.zeros((num_nodes, 1), dtype=torch.long)  # [, 1] 형태로 변경
    y[target_df["id"].values] = torch.tensor(target_df["target"].values, dtype=torch.long).view(-1, 1)
    
    # PyG Data 객체 생성
    graph_data = Data(x=x, edge_index=edge_index, y=y)
    return graph_data

def set_seed(seed=1127):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def to_numpy(tensor):
    """PyTorch Tensor → NumPy 변환 후 1차원으로 변형"""
    if isinstance(tensor, torch.Tensor):
        return tensor.cpu().numpy().squeeze()
    elif isinstance(tensor, np.ndarray):
        return tensor.squeeze()
    else:
        raise TypeError(f"Unsupported data type: {type(tensor)}")
 
def sort_by_y(x, y, *intervals):
    sort_idx = np.argsort(to_numpy(y).ravel())  # Y값을 기준으로 정렬할 인덱스
    sorted_x = to_numpy(x).ravel()[sort_idx]
    sorted_y = to_numpy(y).ravel()[sort_idx]
    sorted_intervals = [to_numpy(interval).ravel()[sort_idx] for interval in intervals]
    return sorted_x, sorted_y, sorted_intervals

def split_graph_data(graph_data, test_ratio=0.2):
    """
    GNN용 Train-Test Split (edge_index를 올바르게 재매핑하여 유지)
    """
    num_nodes = graph_data.x.shape[0]  # 전체 노드 개수
    num_test = int(num_nodes * test_ratio)  # 테스트 데이터 노드 개수

    # 랜덤하게 Train/Test 노드 인덱스 선택
    indices = torch.randperm(num_nodes)
    test_nodes = indices[:num_test]
    train_nodes = indices[num_test:]

    # 노드 마스크 생성
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_nodes] = True
    test_mask[test_nodes] = True

    # Train/Test용 edge_index 필터링
    train_edge_mask = train_mask[graph_data.edge_index[0]] & train_mask[graph_data.edge_index[1]]
    test_edge_mask = test_mask[graph_data.edge_index[0]] & test_mask[graph_data.edge_index[1]]

    # Train용 edge_index와 노드 인덱스 재매핑
    train_edge_index = graph_data.edge_index[:, train_edge_mask]
    train_node_map = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(train_nodes)}
    train_edge_index = torch.tensor(
        [[train_node_map[idx.item()] for idx in train_edge_index[0]],
         [train_node_map[idx.item()] for idx in train_edge_index[1]]],
        dtype=torch.long
    )

    # Test용 edge_index와 노드 인덱스 재매핑
    test_edge_index = graph_data.edge_index[:, test_edge_mask]
    test_node_map = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(test_nodes)}
    test_edge_index = torch.tensor(
        [[test_node_map[idx.item()] for idx in test_edge_index[0]],
         [test_node_map[idx.item()] for idx in test_edge_index[1]]],
        dtype=torch.long
    )

    # Train 데이터 생성
    train_data = Data(
        x=graph_data.x[train_mask],
        y=graph_data.y[train_mask],
        edge_index=train_edge_index
    )

    # Test 데이터 생성
    test_data = Data(
        x=graph_data.x[test_mask],
        y=graph_data.y[test_mask],
        edge_index=test_edge_index
    )

    # 결과 출력
    # print(f"Train Nodes: {train_data.x.shape[0]}, Train Edges: {train_data.edge_index.shape[1]}")
    # print(f"Train edge_index 최대값: {train_data.edge_index.max().item()}")
    # print(f"Test Nodes: {test_data.x.shape[0]}, Test Edges: {test_data.edge_index.shape[1]}")
    # print(f"Test edge_index 최대값: {test_data.edge_index.max().item()}")

    return train_data, test_data

def split_cp_graph_data(train_data, cali_ratio=0.2):
    cp_train_data, calibration_data = split_graph_data(train_data)
    num_train = train_data.x.shape[0]
    num_calibration = int(num_train * cali_ratio)
    
    indices = torch.randperm(num_train)
    calibration_indices = indices[:num_calibration]
    cp_train_indices = indices[num_calibration:]

    cp_train_mask = torch.zeros(num_train, dtype=torch.bool)
    calibration_mask = torch.zeros(num_train, dtype=torch.bool)
    cp_train_mask[cp_train_indices] = True
    calibration_mask[calibration_indices] = True

    cp_train_edge_mask = cp_train_mask[train_data.edge_index[0]] & cp_train_mask[train_data.edge_index[1]]
    cp_train_edge_index = train_data.edge_index[:, cp_train_edge_mask]
    cp_train_node_map = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(cp_train_indices)}
    cp_train_edge_index = torch.tensor(
        [[cp_train_node_map[idx.item()] for idx in cp_train_edge_index[0]],
        [cp_train_node_map[idx.item()] for idx in cp_train_edge_index[1]]],
        dtype=torch.long
    )

    calibration_edge_mask = calibration_mask[train_data.edge_index[0]] & calibration_mask[train_data.edge_index[1]]
    calibration_edge_index = train_data.edge_index[:, calibration_edge_mask]
    calibration_node_map = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(calibration_indices)}
    calibration_edge_index = torch.tensor(
        [[calibration_node_map[idx.item()] for idx in calibration_edge_index[0]],
        [calibration_node_map[idx.item()] for idx in calibration_edge_index[1]]],
        dtype=torch.long
    )

    # cp_train_data 생성
    cp_train_data = Data(
        x=train_data.x[cp_train_indices],
        y=train_data.y[cp_train_indices],
        edge_index=cp_train_edge_index
    )

    # calibration_data 생성
    calibration_data = Data(
        x=train_data.x[calibration_indices],
        y=train_data.y[calibration_indices],
        edge_index=calibration_edge_index
    )
    
    return cp_train_data, calibration_data
    
def augment_features(x, tau):
    if isinstance(tau, float):  # tau가 float이면 변환
        tau = torch.tensor([tau])
    
    tau = tau.view(-1, 1)
    tau_transformed = (tau - 0.5) * 12 # 분위수 값 변환: 학습 안정성 증가
    
    return torch.cat((x, tau_transformed.expand(x.size(0), -1)), dim = 1)

def coverage_width(y_true, y_low, y_upper):
    coverage = ((y_true >= y_low) & (y_true <= y_upper)).float().mean()
    width = (y_upper - y_low).float().abs().mean()
    
    return coverage, width
    
def evaluate_model_performance(preds_low, preds_upper, targets, target=0.9):
    # PICP (Prediction Interval Coverage Probability): 커버리지 계산
    picp = np.mean((targets >= preds_low) & (targets <= preds_upper))
    
    # NMPIW (Normalized Mean Prediction Interval Width): 정규화된 구간 너비
    interval_width = np.mean(preds_upper - preds_low)
    data_range = np.max(targets) - np.min(targets)
    nmpiw = interval_width / data_range if data_range > 0 else interval_width  # 범위가 0일 경우 대비
    
    # MPE (Mean Prediction Error): 중앙값과 실제 값의 평균 오차
    median_pred = (preds_low + preds_upper) / 2    # 신뢰구간 중앙값
    mpe = np.mean(np.abs(median_pred - targets)) # 예측 구간 중심이 실제값과 얼마나 가까운지
    
    # Sharpness: 예측 구간의 날카로움 (제곱을 사용해 큰 폭에 패널티)
    sharpness = np.mean(np.square(preds_upper - preds_low)) # 예측 구간의 날카로움: 제곱을 사용해 큰 폭일수록 더 강한 패널티
    
    # 5. Winkler Score: 구간 너비 + 커버리지 실패 시 패널티
    alpha = 0.5  # 패널티 가중치
    penalties = np.where(targets < preds_low, preds_low - targets, 
                         np.where(targets > preds_upper, targets - preds_upper, 0))
    winkler = np.mean(interval_width + 2 * alpha * penalties)
    
    # MCT (Modified Coverage Tradeoff): 구간 너비와 커버리지 차이의 곱
    mct = interval_width * abs(picp - target)
    
    # CWC (Coverage Width-based Criterion): NMPIW와 커버리지 패널티 결합
    mu = target  # 목표 커버리지를 target으로 설정
    # gamma (float): CWC의 패널티 강도 하이퍼파라미터 (기본값: 1.0)
    # eta (float): CWC의 지수 함수 감쇠 속도 하이퍼파라미터 (기본값: 10.0)
    gamma = 1
    eta = 10
    penalty = 1 + gamma * np.exp(-eta * (picp - mu))
    cwc = nmpiw * penalty
    
    print(f"종합 - CWC ⬇: {cwc:.4f}, MCT ⬇: {mct:.4f}")
    print(f"예측 관련 - PICP ⬆: {picp:.4f}, MPE ⬇: {mpe:.4f}")
    print(f"구간 관련 - NMPIW ⬇: {nmpiw:.4f}, Sharpness ⬇: {sharpness:.4f}, WS ⬇: {winkler:.4f}")

    
    return {
        "PCIP": picp,
        'MPIW': interval_width, 
        "NMPIW": nmpiw,
        "MCT": mct,
        "CWC": cwc,
        "MPE": mpe,
        "Sharpness": sharpness,
        "WS": winkler
    }
    
def add_node_features_and_target(data: Data):
    g_nx = to_networkx(data, to_undirected=True)

    # 각각 [num_nodes, 1] 크기의 feature tensor 생성
    degrees = torch.tensor([val for _, val in g_nx.degree()], dtype=torch.float).unsqueeze(1)
    clustering = torch.tensor(list(nx.clustering(g_nx).values()), dtype=torch.float).unsqueeze(1)
    betweenness = torch.tensor(list(nx.betweenness_centrality(g_nx).values()), dtype=torch.float).unsqueeze(1)
    closeness = torch.tensor(list(nx.closeness_centrality(g_nx).values()), dtype=torch.float).unsqueeze(1)
    eigenvector = torch.tensor(list(nx.eigenvector_centrality(g_nx, max_iter=1000).values()), dtype=torch.float).unsqueeze(1)

    # 5차원 feature 벡터
    data.x = torch.cat([degrees, clustering, betweenness, closeness, eigenvector], dim=1)

    # 타겟: sin(degree) + noise
    target = torch.sin(degrees * 0.5) + 0.1 * torch.randn_like(degrees)
    data.y = target

    return data

# 1. Barabási–Albert Graph (BA Graph)
def create_ba_graph_pyg(n=1000, m=3):
    g_nx = nx.barabasi_albert_graph(n, m)
    data = from_networkx(g_nx)
    data = add_node_features_and_target(data)
    return data

# 2. Erdős–Rényi Graph (ER Graph)
def create_er_graph_pyg(n=1000, p=0.01):
    g_nx = nx.erdos_renyi_graph(n, p)
    data = from_networkx(g_nx)
    data = add_node_features_and_target(data)
    return data

# 3. Grid Graph
def create_grid_graph_pyg():
    rows, cols = 30, 30  # 30 x 35 = 900
    g_nx = nx.grid_2d_graph(rows, cols)
    g_nx = nx.convert_node_labels_to_integers(g_nx)
    data = from_networkx(g_nx)
    data = add_node_features_and_target(data)
    return data

# 4. Tree Graph 
def create_tree_graph_pyg(r=2, height=8):
    g_nx = nx.balanced_tree(r, height)
    g_nx = g_nx.subgraph(list(g_nx.nodes)[:1000])  # 잘라냄
    data = from_networkx(g_nx)
    data = add_node_features_and_target(data)
    return data