# import os
# import json
# import random
# from datetime import datetime
# from itertools import product
# import pandas as pd
# import numpy as np
# import torch
#
# print(torch.__version__)
#
# import ArgParser
#
# from sklearn.metrics import roc_auc_score
# from sklearn.model_selection import StratifiedKFold, KFold
# # from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
# from torch_geometric.data import Data
# from torch_geometric.loader import DataLoader
# from tqdm import tqdm
#
# from PathAwareGNN import PathAwareGNN
# from SimpleGCN import SimpleGCN
#
# from data_process.Data_Enhance import UltimateTripleStreamAugmentor
# from data_process.Data_Query import Data_Query
# from data_process.Eearly_Stopping import RobustEarlyStopping
# from data_process.Community_Search import select_nodes_via_label
#
# # ==========================================================
# # 使用当前同目录下的 RWR_QDCC
# # ==========================================================
# from RWR_QDCC import (
#     dap_arwr_first_stage_gpu,
#     dap_arwr_single_query_balanced,
#     preprocess_graph_gpu,
# )
#
# # Stage 14 复用之前社区构造对比实验中的 BFS / PPR / RWR
# from qisc_construction_benchmark import (
#     build_graph_from_dataframes,
#     bfs_select,
#     ppr_scores,
#     connected_ranked_expand,
#     rwr_mc_select,
#     torch_graph_cache,
#     clear_torch_graph_cache,
# )
#
# os.environ["OMP_NUM_THREADS"] = "6"
#
#
# def train(model, loader, optimizer, device, augmentor=None):
#     model.train()
#     total_loss = 0
#
#     for data in loader:
#         data = data.to(device, non_blocking=True)
#         optimizer.zero_grad()
#
#         # 🚀 自动判定并执行一致性正则化
#         has_gray_samples = (data.y == -1).any().item()
#         if has_gray_samples and augmentor is not None:
#             loss = model.total_loss(data, augmentor=augmentor)
#         else:
#             loss = model.total_loss(data)
#
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
#
#     return total_loss / len(loader)
#
#
# def test(model, loader, device):
#     model.eval()
#     y_true, y_pred = [], []
#     with torch.no_grad():
#         for data in loader:
#             data = data.to(device)
#
#             # 1. 究极防爆：无限解包取出真实的 Tensor
#             out = model.predict(data)
#             while isinstance(out, (tuple, list)):
#                 out = out[0]
#             pred = out
#
#             # 2. 维度安全保护
#             if pred.dim() == 0:
#                 pred = pred.unsqueeze(0)
#
#             # 3. 过滤掉灰区数据（测试和验证集决不包含 -1 的杂质！）
#             valid_mask = (data.y != -1)
#             if valid_mask.sum() > 0:
#                 y_true.append(data.y[valid_mask].cpu().numpy())
#                 y_pred.append(pred[valid_mask].cpu().numpy())
#
#     if len(y_true) == 0: return 0.0
#
#     y_true = np.concatenate(y_true).ravel()
#     y_pred = np.concatenate(y_pred)
#
#     if y_pred.ndim == 2:
#         y_pred = y_pred[:, 1] if y_pred.shape[1] >= 2 else y_pred.ravel()
#
#     y_true_binary = (y_true > 0).astype(int)
#
#     if len(np.unique(y_true_binary)) < 2:
#         return 0.0
#
#     return roc_auc_score(y_true_binary, y_pred)
#
#
# # # 5折数据划分函数
# def stratified_split(data_list, val_ratio=0.2,
#                      n_splits=5, batch_size=128, seed=42):
#     random.seed(seed)
#     np.random.seed(seed)
#
#     labels = np.array([data.y.item() for data in data_list])
#     fold_loaders = []
#
#     # 🚀 安全机制：检查每个类别的样本数，如果最少类别的数量 < n_splits，退化为普通的 KFold
#     unique_classes, counts = np.unique(labels, return_counts=True)
#     min_count = np.min(counts)
#
#     if min_count < n_splits:
#         tqdm.write(
#             f"⚠️ 警告: 类别 {unique_classes[np.argmin(counts)]} 只有 {min_count} 个样本，无法进行严格的分层划分。自动切换为普通 KFold。")
#         main_kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
#         inner_splitter_class = KFold
#     else:
#         main_kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
#         inner_splitter_class = StratifiedKFold
#
#     for fold, (train_val_idx, test_idx) in enumerate(main_kfold.split(data_list, labels)):
#         train_labels = labels[train_val_idx]
#
#         # 内部验证集划分
#         inner_splitter = inner_splitter_class(n_splits=int(1 / val_ratio), shuffle=True, random_state=seed + fold)
#
#         for train_idx, val_idx in inner_splitter.split(train_val_idx, train_labels):
#             real_train_idx = train_val_idx[train_idx]
#             real_val_idx = train_val_idx[val_idx]
#             break
#
#         train_data = [data_list[i] for i in real_train_idx]
#         val_data = [data_list[i] for i in real_val_idx]
#         test_data = [data_list[i] for i in test_idx]
#
#         train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=False)
#         val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, pin_memory=False)
#         test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, pin_memory=False)
#
#         fold_loaders.append((train_loader, val_loader, test_loader))
#
#     return fold_loaders
#
#
# def compute_val_loss(model, val_loader, device):
#     """计算验证集损失"""
#     model.eval()
#     total_loss = 0
#     with torch.no_grad():
#         for data in val_loader:
#             data = data.to(device)
#             loss = model.total_loss(data)
#             total_loss += loss.item()
#     return total_loss / len(val_loader)
#
#
# def cross_val_train_and_test(data_list, params, device, num_epochs=300, k_folds=5, batch_size=300,
#                              save_best_model_path="best_model.pth"):
#     folds = stratified_split(data_list, n_splits=k_folds, batch_size=batch_size)
#
#     test_aucs = []
#     best_overall_model = None
#     best_overall_test_auc = -float('inf')
#
#     warmup_epochs = 15
#     patience_val = 25
#
#     start_time = datetime.now()
#
#     # 🚀 强力打印实验参数
#     print("\n" + "★" * 80)
#     print(f"★ 启动新实验 | 学习率 LR: {params['learning_rate']} | 隐藏层 Hidden: {params['hidden_dim']}")
#     print("★" * 80)
#
#     global_pbar = tqdm(total=k_folds * num_epochs, desc="🚀 训练进度", ncols=180, unit="step",
#                        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}, {postfix}]")
#
#     augmentor = UltimateTripleStreamAugmentor(feature_noise=0.02, edge_drop_rate=0.15, path_jitter_rate=0.2)
#
#     for fold in range(k_folds):
#         train_loader, val_loader, test_loader = folds[fold]
#
#         # model = PathAwareGNN(feat_dim=data_list[0].x.shape[1], hidden_dim=params['hidden_dim'], dropout_prob=0.6).to(
#         #     device)
#         model = PathAwareGNN(
#             feat_dim=data_list[0].x.shape[1],
#             hidden_dim=params['hidden_dim'],
#             num_layers=params.get('num_layers', 2),
#             dropout_prob=0.6
#         ).to(device)
#         optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=5e-4)
#         scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
#         early_stopping = RobustEarlyStopping(patience=patience_val, min_delta=0.001, smoothing=10,
#                                              restore_best_weights=True, verbose=False)
#
#         fold_best_val_auc = -float('inf')
#         fold_best_epoch = 0
#         stop_training = False
#
#         for epoch in range(num_epochs):
#             train_loss = train(model, train_loader, optimizer, device, augmentor)
#             val_auc = test(model, val_loader, device)
#
#             scheduler.step(val_auc)
#             current_lr = optimizer.param_groups[0]['lr']
#
#             if val_auc > fold_best_val_auc:
#                 fold_best_val_auc = val_auc
#                 fold_best_epoch = epoch + 1
#
#             if epoch >= warmup_epochs:
#                 if early_stopping(val_auc, model):
#                     stop_training = True
#
#             global_pbar.set_postfix({'折数': f'{fold + 1}/{k_folds}', 'Epoch': f'{epoch + 1}/{num_epochs}',
#                                      'Loss': f'{train_loss:.4f}', 'ValAUC': f'{val_auc:.4f}', 'LR': f'{current_lr:.2e}',
#                                      '最佳Epoch': f'{fold_best_epoch}', '最佳ValAUC': f'{fold_best_val_auc:.4f}'})
#             global_pbar.update(1)
#
#             if stop_training:
#                 global_pbar.update(num_epochs - epoch - 1)
#                 break
#
#         test_auc = test(model, test_loader, device)
#         test_aucs.append(test_auc)
#
#         if test_auc > best_overall_test_auc:
#             best_overall_test_auc = test_auc
#             best_overall_model = {k: v.cpu().clone() for k, v in model.state_dict().items()}
#
#         tqdm.write(f"\n✅ Fold {fold + 1} 完成 | Test AUC: {test_auc:.4f} | 最佳Epoch: {fold_best_epoch}")
#
#     global_pbar.close()
#
#     # 🚀 强力汇总当前实验的最终参数与结果
#     res_str = f"🏁 实验结束 -> LR: {params['learning_rate']} | Hidden: {params['hidden_dim']} | 最终AUC: {np.mean(test_aucs):.4f} ± {np.std(test_aucs):.4f}"
#     print("\n" + "=" * len(res_str))
#     print(res_str)
#     print("=" * len(res_str) + "\n")
#
#     if best_overall_model is not None:
#         torch.save(best_overall_model, save_best_model_path)
#     return params, np.mean(test_aucs), np.std(test_aucs)
#
#
# def cross_val_train_and_test_gcn(data_list, params, device, num_epochs=300, k_folds=5, batch_size=300,
#                                  save_best_model_path="best_gcn_model.pth"):
#     """GCN 基线版本的交叉验证训练（与 PathAwareGNN 版本结构完全一致，仅替换模型类）"""
#     folds = stratified_split(data_list, n_splits=k_folds, batch_size=batch_size)
#
#     test_aucs = []
#     best_overall_model = None
#     best_overall_test_auc = -float('inf')
#
#     warmup_epochs = 15
#     patience_val = 25
#
#     start_time = datetime.now()
#
#     print("\n" + "★" * 80)
#     print(f"★ [GCN基线] 启动新实验 | LR: {params['learning_rate']} | Hidden: {params['hidden_dim']}")
#     print("★" * 80)
#
#     global_pbar = tqdm(total=k_folds * num_epochs, desc="🔵 GCN训练进度", ncols=180, unit="step",
#                        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}, {postfix}]")
#
#     for fold in range(k_folds):
#         train_loader, val_loader, test_loader = folds[fold]
#
#         # 🔵 唯一区别：使用 SimpleGCN 模型
#         model = SimpleGCN(
#             feat_dim=data_list[0].x.shape[1],
#             hidden_dim=params['hidden_dim'],
#             num_layers=params.get('num_layers', 2),
#             dropout_prob=0.6
#         ).to(device)
#
#         optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=5e-4)
#         scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
#         early_stopping = RobustEarlyStopping(patience=patience_val, min_delta=0.001, smoothing=10,
#                                              restore_best_weights=True, verbose=False)
#
#         fold_best_val_auc = -float('inf')
#         fold_best_epoch = 0
#         stop_training = False
#
#         for epoch in range(num_epochs):
#             # GCN 不需要 augmentor（因为没有一致性正则化分支）
#             train_loss = train(model, train_loader, optimizer, device, augmentor=None)
#             val_auc = test(model, val_loader, device)
#
#             scheduler.step(val_auc)
#             current_lr = optimizer.param_groups[0]['lr']
#
#             if val_auc > fold_best_val_auc:
#                 fold_best_val_auc = val_auc
#                 fold_best_epoch = epoch + 1
#
#             if epoch >= warmup_epochs:
#                 if early_stopping(val_auc, model):
#                     stop_training = True
#
#             global_pbar.set_postfix({'折数': f'{fold + 1}/{k_folds}', 'Epoch': f'{epoch + 1}/{num_epochs}',
#                                      'Loss': f'{train_loss:.4f}', 'ValAUC': f'{val_auc:.4f}', 'LR': f'{current_lr:.2e}',
#                                      '最佳Epoch': f'{fold_best_epoch}', '最佳ValAUC': f'{fold_best_val_auc:.4f}'})
#             global_pbar.update(1)
#
#             if stop_training:
#                 global_pbar.update(num_epochs - epoch - 1)
#                 break
#
#         test_auc = test(model, test_loader, device)
#         test_aucs.append(test_auc)
#
#         if test_auc > best_overall_test_auc:
#             best_overall_test_auc = test_auc
#             best_overall_model = {k: v.cpu().clone() for k, v in model.state_dict().items()}
#
#         tqdm.write(f"\n✅ GCN Fold {fold + 1} 完成 | Test AUC: {test_auc:.4f} | 最佳Epoch: {fold_best_epoch}")
#
#     global_pbar.close()
#
#     res_str = f"🏁 [GCN基线] 实验结束 -> LR: {params['learning_rate']} | Hidden: {params['hidden_dim']} | 最终AUC: {np.mean(test_aucs):.4f} ± {np.std(test_aucs):.4f}"
#     print("\n" + "=" * len(res_str))
#     print(res_str)
#     print("=" * len(res_str) + "\n")
#
#     if best_overall_model is not None:
#         torch.save(best_overall_model, save_best_model_path)
#     return params, np.mean(test_aucs), np.std(test_aucs)
#
#
# # =====================================================================
# # Stage 13 / 14 数据导出辅助函数
# # =====================================================================
# def reset_all_seeds(seed=42):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(seed)
#         torch.cuda.manual_seed_all(seed)
#
#
# def build_pyg_subgraph_from_indices(selected_nodes, q_idx, all_edge_index, node_feat_tensor, node_label_tensor):
#     """把 BFS/PPR/RWR 选出的全局节点索引打包成与 RWR-QDCC 一致的 PyG Data。"""
#     selected_nodes = [int(v) for v in selected_nodes]
#     if q_idx not in selected_nodes:
#         raise RuntimeError(f"query index {q_idx} 不在 baseline 社区中。")
#
#     device = node_feat_tensor.device
#     n_nodes = node_feat_tensor.size(0)
#     community_idx_tensor = torch.tensor(selected_nodes, dtype=torch.long, device=device)
#
#     x = node_feat_tensor[community_idx_tensor].detach()
#     x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
#     node_labels = node_label_tensor[community_idx_tensor].detach()
#     local_q_idx = selected_nodes.index(q_idx)
#
#     mask = torch.isin(all_edge_index[0], community_idx_tensor) & torch.isin(all_edge_index[1], community_idx_tensor)
#     edge_index_global = all_edge_index[:, mask]
#
#     if edge_index_global.shape[1] > 0:
#         global2local = torch.zeros(n_nodes, dtype=torch.long, device=device)
#         global2local[community_idx_tensor] = torch.arange(len(selected_nodes), device=device)
#         edge_index = global2local[edge_index_global]
#
#         # 与 RWR-QDCC 当前实现统一：community_quality 使用社区内部边两端特征余弦相似度均值
#         src_global = edge_index_global[0]
#         dst_global = edge_index_global[1]
#         feat_src = node_feat_tensor[src_global]
#         feat_dst = node_feat_tensor[dst_global]
#         edge_cos = torch.nn.functional.cosine_similarity(feat_src, feat_dst, dim=1).clamp(min=0.0)
#         avg_quality = torch.mean(edge_cos).item()
#     else:
#         edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
#         avg_quality = 0.0
#
#     return Data(
#         x=x,
#         edge_index=edge_index,
#         node_labels=node_labels,
#         q_id=torch.tensor([local_q_idx], dtype=torch.long, device=device),
#         y=torch.tensor([-1], dtype=torch.long, device=device),
#         community_quality=torch.tensor([avg_quality], dtype=torch.float32, device=device),
#         p_val=None,
#     )
#
#
# def apply_export_labels(data_list, features_df, label_method, label_tau=None):
#     """Stage 14 只使用 Query-label 或 Ratio-based，两者与当前 RWR_QDCC.py 定义保持一致。"""
#     all_labels = features_df['label'].to_numpy(dtype=int)
#     valid_global = all_labels != -1
#     n_star = int(valid_global.sum())
#     k_star = int((all_labels[valid_global] == 1).sum())
#
#     if label_method == 'query_label':
#         for d in data_list:
#             q_local_idx = int(d.q_id.item())
#             query_label = int(d.node_labels[q_local_idx].item())
#             if query_label == 1:
#                 d.y = torch.tensor([1], dtype=torch.long, device=d.x.device)
#                 d.p_val = 0.0
#             elif query_label == 0:
#                 d.y = torch.tensor([0], dtype=torch.long, device=d.x.device)
#                 d.p_val = 1.0
#             else:
#                 d.y = torch.tensor([-1], dtype=torch.long, device=d.x.device)
#                 d.p_val = 2.0
#
#     elif label_method == 'ratio_based':
#         tau = label_tau if label_tau is not None else (k_star / n_star if n_star > 0 else 0.5)
#         for d in data_list:
#             comm_labels = d.node_labels.detach().cpu().numpy()
#             valid = comm_labels != -1
#             n_labeled = int(valid.sum())
#             k_labeled = int((comm_labels[valid] == 1).sum())
#             if n_labeled == 0:
#                 d.y = torch.tensor([-1], dtype=torch.long, device=d.x.device)
#                 d.p_val = 2.0
#                 continue
#             local_ratio = k_labeled / n_labeled
#             d.p_val = float(1.0 - local_ratio)
#             d.y = torch.tensor([1 if local_ratio > tau else 0], dtype=torch.long, device=d.x.device)
#     else:
#         raise ValueError(f"Stage 14 不支持 label_method={label_method}")
#
#     return data_list
#
#
# def save_export_pt(data_list, save_path, metadata):
#     """.pt 保持为纯 data_list，便于其他异常检测方法直接 torch.load；参数另存 JSON。"""
#     os.makedirs(os.path.dirname(save_path), exist_ok=True)
#     cpu_data_list = [d.cpu() for d in data_list]
#     torch.save(cpu_data_list, save_path)
#
#     meta_path = os.path.splitext(save_path)[0] + '.json'
#     with open(meta_path, 'w', encoding='utf-8') as f:
#         json.dump(metadata, f, ensure_ascii=False, indent=2)
#
#     normal_count = sum(1 for d in cpu_data_list if int(d.y.item()) == 0)
#     anomalous_count = sum(1 for d in cpu_data_list if int(d.y.item()) == 1)
#     uncertain_count = sum(1 for d in cpu_data_list if int(d.y.item()) == -1)
#     avg_nodes = float(np.mean([d.x.shape[0] for d in cpu_data_list])) if cpu_data_list else 0.0
#     avg_edges = float(np.mean([d.edge_index.shape[1] for d in cpu_data_list])) if cpu_data_list else 0.0
#
#     print(f"💾 PT 已保存: {save_path}")
#     print(f"   Communities={len(cpu_data_list)} | N={normal_count} | A={anomalous_count} | U={uncertain_count} | "
#           f"AvgNodes={avg_nodes:.2f} | AvgEdges={avg_edges:.2f}")
#     print(f"🧾 Metadata: {meta_path}")
#
#
# def build_stage14_baseline_communities(
#         edges_df,
#         features_df,
#         num_samples,
#         min_community_size,
#         max_community_size,
#         q_quantile,
#         restart_prob,
#         max_rwr_iter,
#         rwr_tol=1e-6,
#         seed=42,
#         mc_walks=4000,
#         mc_length=20,
# ):
#     """
#     Stage 14：对同一批 query，先用 RWR-QDCC 得到逐 query 目标规模 S_q，
#     再分别构造 size-matched BFS / PPR / RWR 社区。
#
#     RWR 沿用此前 qisc_construction_benchmark.py 的 Monte-Carlo connected RWR 定义；
#     PPR 使用 exact personalized PageRank + connected ranked expansion。
#     """
#     reset_all_seeds(seed)
#
#     all_edge_index, node2idx, idx2node, node_feat_tensor, node_label_tensor, ptr, dst = preprocess_graph_gpu(
#         edges_df, features_df
#     )
#
#     clear_torch_graph_cache()
#     _, _, x_np, adjacency = build_graph_from_dataframes(
#         edges_df, features_df, node_id_col='node_id', label_col='label'
#     )
#     torch_graph_cache(x_np, adjacency)
#
#     # 与 dap_arwr_first_stage_gpu 完全相同的 query 选择入口
#     anomaly_nodes, normal_nodes = select_nodes_via_label(features_df, num_samples)
#     query_node_list = anomaly_nodes + normal_nodes
#
#     method_data = {'bfs': [], 'ppr': [], 'rwr': []}
#
#     pbar = tqdm(total=len(query_node_list), desc='📦 Stage14 构造 BFS/PPR/RWR', ncols=180,
#                 bar_format='{l_bar}{bar:60}{r_bar}', colour='cyan')
#
#     for position, q_id in enumerate(query_node_list, start=1):
#         q_id_int = int(q_id)
#         q_idx = node2idx.get(q_id_int, -1)
#         if q_idx < 0:
#             raise RuntimeError(f"Query {q_id_int} 不在 node2idx 中。")
#
#         # 只用于获取该 query 在 RWR-QDCC 下的真实社区规模 S_q
#         qdcc_data = dap_arwr_single_query_balanced(
#             q_id_int, node2idx, idx2node, all_edge_index, ptr, dst,
#             node_feat_tensor, node_label_tensor,
#             min_community_size, max_community_size, q_quantile,
#             restart_prob=restart_prob, max_rwr_iter=max_rwr_iter, rwr_tol=rwr_tol
#         )
#         if qdcc_data is None:
#             raise RuntimeError(f"RWR-QDCC 在 Query {q_id_int} 上返回 None，无法进行成对规模对齐。")
#
#         target_size = int(qdcc_data.x.shape[0])
#
#         bfs_nodes = bfs_select(q_idx, target_size, adjacency)
#         ppr_nodes = connected_ranked_expand(
#             q_idx,
#             ppr_scores(q_idx, adjacency, restart_prob, max_rwr_iter, rwr_tol),
#             target_size,
#             adjacency,
#         )
#         rwr_nodes = rwr_mc_select(
#             q_idx,
#             target_size,
#             adjacency,
#             restart_prob,
#             mc_walks,
#             mc_length,
#             seed + 104729 * position,
#         )
#
#         selected = {'bfs': bfs_nodes, 'ppr': ppr_nodes, 'rwr': rwr_nodes}
#         for method_name, nodes in selected.items():
#             if len(nodes) != target_size:
#                 raise RuntimeError(
#                     f"{method_name.upper()} 在 Query {q_id_int} 上规模未对齐: "
#                     f"target={target_size}, actual={len(nodes)}"
#                 )
#             method_data[method_name].append(
#                 build_pyg_subgraph_from_indices(nodes, q_idx, all_edge_index, node_feat_tensor, node_label_tensor)
#             )
#
#         pbar.update(1)
#         if position % 10 == 0:
#             pbar.set_postfix({'query': position, 'S_q': target_size})
#
#     pbar.close()
#     return query_node_list, method_data
#
#
# def main():
#     torch.autograd.set_detect_anomaly(False)
#     torch.backends.cudnn.benchmark = True
#     torch.backends.cudnn.deterministic = False
#
#     args = ArgParser.parse_args()
#     device = torch.device(args.device if torch.cuda.is_available() else "cpu")
#     now = datetime.now().strftime("%Y%m%d_%H%M%S")
#
#     # =====================================================================
#     # 核心控制台：平时只需要修改下面两行
#     # =====================================================================
#     DATASET_NAME = "Actor"
#     EXPERIMENT_STAGE = 10
#
#     # 可选数据集：
#     # "Elliptic" / "DGraph" / "Actor" / "Arxiv" / "Roman" / "Minesweeper"
#     #
#     # Stage:
#     # 1  -> k_min × k_max
#     # 2  -> q_quantile
#     # 3  -> num_samples
#     # 4  -> restart_prob × max_rwr_iter
#     # 5  -> num_layers
#     # 8  -> confidence_tau
#     # 9  -> GCN baseline
#     # 10 -> PathAwareGNN 最优参数验证 + data_list 导出
#     # 11 -> learning_rate × hidden_dim
#     # 12 -> query_label / ratio_based / fisher_gmm
#     # 13 -> RWR-QDCC + Fisher-GMM：使用最优参数导出 .pt，不训练
#     # 14 -> BFS/PPR/RWR × Query-label/Ratio-based：逐 query 与 RWR-QDCC 等规模导出 .pt，不训练
#
#     # =====================================================================
#     # 各数据集历史最优参数
#     # =====================================================================
#     DATASET_CONFIGS = {
#         "Elliptic": {
#             "BEST_NUM_SAMPLES": 3000, "BEST_K_MIN": 5, "BEST_K_MAX": 90, "BEST_Q": 0.75,
#             "BEST_LR": 0.003, "BEST_HIDDEN": 256, "BEST_RESTART_PROB": 0.15, "BEST_MAX_RWR_ITER": 200, "BEST_TAU": 0.9,
#         },
#         "DGraph": {
#             "BEST_NUM_SAMPLES": 2000, "BEST_K_MIN": 5, "BEST_K_MAX": 30, "BEST_Q": 0.9,
#             "BEST_LR": 0.001, "BEST_HIDDEN": 128, "BEST_RESTART_PROB": 0.15, "BEST_MAX_RWR_ITER": 50, "BEST_TAU": 0.9,
#         },
#         "Actor": {
#             "BEST_NUM_SAMPLES": 2000, "BEST_K_MIN": 20, "BEST_K_MAX": 90, "BEST_Q": 0.85,
#             "BEST_LR": 0.003, "BEST_HIDDEN": 128, "BEST_RESTART_PROB": 0.05, "BEST_MAX_RWR_ITER": 200, "BEST_TAU": 0.9,
#         },
#         "Arxiv": {
#             "BEST_NUM_SAMPLES": 2000, "BEST_K_MIN": 5, "BEST_K_MAX": 90, "BEST_Q": 0.6,
#             "BEST_LR": 0.001, "BEST_HIDDEN": 256, "BEST_RESTART_PROB": 0.15, "BEST_MAX_RWR_ITER": 50, "BEST_TAU": 0.9,
#         },
#         "Roman": {
#             "BEST_NUM_SAMPLES": 1500, "BEST_K_MIN": 5, "BEST_K_MAX": 30, "BEST_Q": 0.75,
#             "BEST_LR": 0.0001, "BEST_HIDDEN": 256, "BEST_RESTART_PROB": 0.05, "BEST_MAX_RWR_ITER": 150, "BEST_TAU": 0.9,
#         },
#         "Minesweeper": {
#             "BEST_NUM_SAMPLES": 2000, "BEST_K_MIN": 20, "BEST_K_MAX": 90, "BEST_Q": 0.5,
#             "BEST_LR": 0.003, "BEST_HIDDEN": 256, "BEST_RESTART_PROB": 0.15, "BEST_MAX_RWR_ITER": 200, "BEST_TAU": 0.9,
#         },
#     }
#
#     # =====================================================================
#     # 敏感性实验候选值，只需要在这里调整
#     # =====================================================================
#     STAGE_GRIDS = {
#         1: {"k_min": [5, 10, 15, 20], "k_max": [30, 50, 70, 90]},
#         2: {"q": [0.5, 0.6, 0.75, 0.85, 0.9]},
#         3: {"num_samples": [1500, 2000, 2500, 3000]},
#         4: {"restart_prob": [0.05, 0.1, 0.15, 0.2, 0.3], "max_rwr_iter": [50, 100, 150, 200]},
#         5: {"num_layers": [2, 3, 4, 5]},
#         8: {"confidence_tau": [0.6, 0.7, 0.8, 0.9]},
#         11: {
#             "learning_rate": [0.0001, 0.0003, 0.001, 0.003],
#             "hidden_dim": [32, 64, 128, 256],
#             # "hidden_dim": [128, 256],
#         },
#         12: {"label_method": ["query_label", "ratio_based", "fisher_gmm"]},
#     }
#
#     SUPPORTED_STAGES = {1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14, 15}
#
#     if DATASET_NAME not in DATASET_CONFIGS:
#         raise ValueError(f"未知数据集: {DATASET_NAME}")
#
#     if EXPERIMENT_STAGE not in SUPPORTED_STAGES:
#         raise ValueError("仅支持 Stage 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14。")
#
#     # =====================================================================
#     # 当前数据集配置
#     # =====================================================================
#     cfg = DATASET_CONFIGS[DATASET_NAME]
#     args.dataset = DATASET_NAME
#     file_path = os.path.join(args.filePath, f"{DATASET_NAME}.npz")
#
#     BEST_NUM_SAMPLES = cfg["BEST_NUM_SAMPLES"]
#     BEST_K_MIN = cfg["BEST_K_MIN"]
#     BEST_K_MAX = cfg["BEST_K_MAX"]
#     BEST_Q = cfg["BEST_Q"]
#     BEST_LR = cfg["BEST_LR"]
#     BEST_HIDDEN = cfg["BEST_HIDDEN"]
#     BEST_RESTART_PROB = cfg["BEST_RESTART_PROB"]
#     BEST_MAX_RWR_ITER = cfg["BEST_MAX_RWR_ITER"]
#     BEST_TAU = cfg["BEST_TAU"]
#
#     # =====================================================================
#     # experiment 结果目录
#     # =====================================================================
#     EXPERIMENT_DIR = "experiment"
#     EXPERIMENT_PT_DIR = "experiment_pt"
#     os.makedirs(EXPERIMENT_DIR, exist_ok=True)
#     os.makedirs(EXPERIMENT_PT_DIR, exist_ok=True)
#
#     print("\n" + "=" * 120)
#     print(f"🚀 自动化实验流水线 | Dataset={DATASET_NAME} | Stage={EXPERIMENT_STAGE} | Device={device}")
#     print(f"📂 Data: {file_path}")
#     print(
#         f"📌 Best Params | samples={BEST_NUM_SAMPLES} | k_min={BEST_K_MIN} | k_max={BEST_K_MAX} | "
#         f"q={BEST_Q} | restart={BEST_RESTART_PROB} | iter={BEST_MAX_RWR_ITER} | "
#         f"lr={BEST_LR} | hidden={BEST_HIDDEN} | tau={BEST_TAU}"
#     )
#     print("=" * 120 + "\n")
#
#     edges_df, features_df = Data_Query(file_path)
#
#     results = []
#     community_cache = {}
#
#     # =====================================================================
#     # 社区构造
#     # =====================================================================
#     def get_communities(**overrides):
#         build_params = {
#             "num_samples": BEST_NUM_SAMPLES,
#             "min_community_size": BEST_K_MIN,
#             "max_community_size": BEST_K_MAX,
#             "q_quantile": BEST_Q,
#             "restart_prob": BEST_RESTART_PROB,
#             "max_rwr_iter": BEST_MAX_RWR_ITER,
#             "rwr_tol": 1e-6,
#             "confidence_tau": BEST_TAU,
#             "method": "hacs",
#             "label_method": "fisher_gmm",
#         }
#
#         build_params.update(overrides)
#         cache_key = tuple(sorted(build_params.items()))
#
#         if cache_key not in community_cache:
#             print(f"\n🔨 构建社区 | {build_params}")
#             community_cache[cache_key] = dap_arwr_first_stage_gpu(edges_df, features_df, **build_params)
#         else:
#             print("\n♻️ 使用已缓存的社区数据")
#
#         return community_cache[cache_key]
#
#     # =====================================================================
#     # 公共训练入口
#     # =====================================================================
#     def run_training_case(
#             case_name,
#             build_overrides=None,
#             train_overrides=None,
#             use_gcn=False,
#             save_data=False,
#             extra_result=None,
#     ):
#         build_overrides = build_overrides or {}
#         train_overrides = train_overrides or {}
#         extra_result = extra_result or {}
#
#         print("\n" + "★" * 100)
#         print(f"★ Case: {case_name}")
#         print("★" * 100)
#
#         data_list = get_communities(**build_overrides)
#
#         if not data_list:
#             print("⚠️ data_list 为空，跳过当前实验。")
#             return
#
#         avg_nodes = float(np.mean([d.x.shape[0] for d in data_list]))
#         avg_quality = float(np.mean([d.community_quality.item() for d in data_list]))
#
#         normal_count = sum(1 for d in data_list if int(d.y.item()) == 0)
#         anomalous_count = sum(1 for d in data_list if int(d.y.item()) == 1)
#         uncertain_count = sum(1 for d in data_list if int(d.y.item()) == -1)
#
#         total_count = len(data_list)
#         supervised_count = normal_count + anomalous_count
#
#         normal_ratio = normal_count / total_count if total_count > 0 else 0.0
#         anomalous_ratio = anomalous_count / total_count if total_count > 0 else 0.0
#         uncertain_ratio = uncertain_count / total_count if total_count > 0 else 0.0
#
#         print(
#             f"📊 Communities={total_count} | Supervised={supervised_count} | "
#             f"N={normal_count} | A={anomalous_count} | U={uncertain_count}"
#         )
#         print(
#             f"📊 Ratio | N={normal_ratio:.4f} | A={anomalous_ratio:.4f} | U={uncertain_ratio:.4f} | "
#             f"|C|_avg={avg_nodes:.2f} | S_avg={avg_quality:.4f}"
#         )
#
#         # Stage 10 保存 data_list
#         if save_data:
#             os.makedirs(args.bestModelPath, exist_ok=True)
#             data_save_path = os.path.join(args.bestModelPath, f"{DATASET_NAME}_data_list_{now}.pt")
#             torch.save(data_list, data_save_path)
#             print(f"💾 data_list 已保存: {data_save_path}")
#
#         params = {"learning_rate": BEST_LR, "hidden_dim": BEST_HIDDEN}
#         params.update(train_overrides)
#
#         os.makedirs(args.bestModelPath, exist_ok=True)
#         model_path = os.path.join(args.bestModelPath, f"{case_name}_{DATASET_NAME}_{now}.pth")
#
#         if use_gcn:
#             _, avg_test_auc, std_test_auc = cross_val_train_and_test_gcn(
#                 data_list,
#                 params,
#                 device,
#                 batch_size=args.batch_size,
#                 save_best_model_path=model_path,
#             )
#             model_name = "GCN"
#         else:
#             _, avg_test_auc, std_test_auc = cross_val_train_and_test(
#                 data_list,
#                 params,
#                 device,
#                 batch_size=args.batch_size,
#                 save_best_model_path=model_path,
#             )
#             model_name = "PathAwareGNN"
#
#         result_row = {
#             "Case": case_name,
#             "Model": model_name,
#             "Community_Count": total_count,
#             "Supervised_Count": supervised_count,
#             "Normal_Count": normal_count,
#             "Anomalous_Count": anomalous_count,
#             "Uncertain_Count": uncertain_count,
#             "Normal_Ratio": normal_ratio,
#             "Anomalous_Ratio": anomalous_ratio,
#             "Uncertain_Ratio": uncertain_ratio,
#             "|C|_avg": avg_nodes,
#             "S_avg": avg_quality,
#             "AUC": avg_test_auc * 100,
#             "STD": std_test_auc * 100,
#             "learning_rate": params["learning_rate"],
#             "hidden_dim": params["hidden_dim"],
#         }
#
#         if "num_layers" in params:
#             result_row["num_layers"] = params["num_layers"]
#
#         result_row.update(build_overrides)
#         result_row.update(extra_result)
#         results.append(result_row)
#
#     # =====================================================================
#     # Stage 1：k_min × k_max
#     # =====================================================================
#     if EXPERIMENT_STAGE == 1:
#         print("\n🔥 Stage 1：k_min × k_max")
#
#         for k_min, k_max in product(STAGE_GRIDS[1]["k_min"], STAGE_GRIDS[1]["k_max"]):
#             if k_min >= k_max:
#                 continue
#
#             run_training_case(
#                 f"stage1_kmin{k_min}_kmax{k_max}",
#                 build_overrides={"min_community_size": k_min, "max_community_size": k_max},
#                 extra_result={"k_min": k_min, "k_max": k_max},
#             )
#
#     # =====================================================================
#     # Stage 2：q_quantile
#     # =====================================================================
#     elif EXPERIMENT_STAGE == 2:
#         print("\n🔥 Stage 2：q_quantile")
#
#         for q in STAGE_GRIDS[2]["q"]:
#             run_training_case(
#                 f"stage2_q{q}",
#                 build_overrides={"q_quantile": q},
#                 extra_result={"q": q},
#             )
#
#     # =====================================================================
#     # Stage 3：num_samples
#     # =====================================================================
#     elif EXPERIMENT_STAGE == 3:
#         print("\n🔥 Stage 3：num_samples")
#
#         for num_samples in STAGE_GRIDS[3]["num_samples"]:
#             run_training_case(
#                 f"stage3_n{num_samples}",
#                 build_overrides={"num_samples": num_samples},
#                 extra_result={"num_samples": num_samples},
#             )
#
#     # =====================================================================
#     # Stage 4：restart_prob × max_rwr_iter
#     # =====================================================================
#     elif EXPERIMENT_STAGE == 4:
#         print("\n🔥 Stage 4：restart_prob × max_rwr_iter")
#
#         for rp, mi in product(STAGE_GRIDS[4]["restart_prob"], STAGE_GRIDS[4]["max_rwr_iter"]):
#             run_training_case(
#                 f"stage4_rp{rp}_mi{mi}",
#                 build_overrides={"restart_prob": rp, "max_rwr_iter": mi},
#                 extra_result={"restart_prob": rp, "max_rwr_iter": mi},
#             )
#
#     # =====================================================================
#     # Stage 5：num_layers
#     # =====================================================================
#     elif EXPERIMENT_STAGE == 5:
#         print("\n🔥 Stage 5：num_layers")
#
#         for num_layers in STAGE_GRIDS[5]["num_layers"]:
#             run_training_case(
#                 f"stage5_layers{num_layers}",
#                 train_overrides={"num_layers": num_layers},
#                 extra_result={"num_layers": num_layers},
#             )
#
#     # =====================================================================
#     # Stage 8：confidence_tau
#     # =====================================================================
#     elif EXPERIMENT_STAGE == 8:
#         print("\n🔥 Stage 8：confidence_tau")
#
#         for tau in STAGE_GRIDS[8]["confidence_tau"]:
#             run_training_case(
#                 f"stage8_tau{tau}",
#                 build_overrides={"confidence_tau": tau, "label_method": "fisher_gmm"},
#                 extra_result={"confidence_tau": tau},
#             )
#
#     # =====================================================================
#     # Stage 9：GCN baseline
#     # =====================================================================
#     elif EXPERIMENT_STAGE == 9:
#         print("\n🔥 Stage 9：GCN baseline")
#
#         run_training_case(
#             "stage9_gcn",
#             use_gcn=True,
#             extra_result={"Baseline": "GCN"},
#         )
#
#     # =====================================================================
#     # Stage 10：PathAwareGNN 最优参数验证 + data_list 导出
#     # =====================================================================
#     elif EXPERIMENT_STAGE == 10:
#         print("\n🔥 Stage 10：PathAwareGNN 最优参数验证 + data_list 导出")
#
#         run_training_case(
#             "stage10_best",
#             save_data=True,
#             extra_result={"Setting": "Best"},
#         )
#
#     # =====================================================================
#     # Stage 11：learning_rate × hidden_dim
#     # =====================================================================
#     elif EXPERIMENT_STAGE == 11:
#         print("\n🔥 Stage 11：learning_rate × hidden_dim")
#
#         for lr, hidden_dim in product(STAGE_GRIDS[11]["learning_rate"], STAGE_GRIDS[11]["hidden_dim"]):
#             run_training_case(
#                 f"stage11_lr{lr}_h{hidden_dim}",
#                 train_overrides={"learning_rate": lr, "hidden_dim": hidden_dim},
#                 extra_result={"learning_rate": lr, "hidden_dim": hidden_dim},
#             )
#
#     # =====================================================================
#     # Stage 12：三种社区伪标签方法
#     # =====================================================================
#     elif EXPERIMENT_STAGE == 12:
#         print("\n🔥 Stage 12：Query-label / Ratio-based / Fisher-GMM")
#
#         for label_method in STAGE_GRIDS[12]["label_method"]:
#             # 保证三种方法从相同随机种子开始
#             random.seed(42)
#             np.random.seed(42)
#             torch.manual_seed(42)
#
#             if torch.cuda.is_available():
#                 torch.cuda.manual_seed_all(42)
#
#             run_training_case(
#                 f"stage12_{label_method}",
#                 build_overrides={"label_method": label_method},
#                 extra_result={"label_method": label_method},
#             )
#
#     # =====================================================================
#     # Stage 13：RWR-QDCC + Fisher-GMM，最优参数导出 .pt
#     # =====================================================================
#     elif EXPERIMENT_STAGE == 13:
#         print("\n🔥 Stage 13：RWR-QDCC + Fisher-GMM 最优参数社区导出")
#         reset_all_seeds(42)
#
#         # 记录与 RWR_QDCC 内部相同的 query 集合，便于追溯；随后重新置种子保证内部选择一致
#         anomaly_nodes, normal_nodes = select_nodes_via_label(features_df, BEST_NUM_SAMPLES)
#         export_queries = [int(v) for v in (anomaly_nodes + normal_nodes)]
#         reset_all_seeds(42)
#
#         data_list = get_communities(
#             method="hacs",
#             label_method="fisher_gmm",
#             confidence_tau=BEST_TAU,
#         )
#
#         dataset_dir = os.path.join(EXPERIMENT_PT_DIR, DATASET_NAME)
#         save_path = os.path.join(dataset_dir, f"{DATASET_NAME}_RWRQDCC_FisherGMM.pt")
#         metadata = {
#             "dataset": DATASET_NAME,
#             "stage": 13,
#             "construction_method": "RWR-QDCC",
#             "label_method": "Fisher-GMM",
#             "query_nodes": export_queries,
#             "num_samples": BEST_NUM_SAMPLES,
#             "k_min": BEST_K_MIN,
#             "k_max": BEST_K_MAX,
#             "q_quantile": BEST_Q,
#             "restart_prob": BEST_RESTART_PROB,
#             "max_rwr_iter": BEST_MAX_RWR_ITER,
#             "rwr_tol": 1e-6,
#             "confidence_tau": BEST_TAU,
#             "generated_at": now,
#         }
#         save_export_pt(data_list, save_path, metadata)
#         print("\n✅ Stage 13 完成：未训练模型，只导出 RWR-QDCC + Fisher-GMM 社区数据。")
#         return
#
#     # =====================================================================
#     # Stage 14：BFS/PPR/RWR × Query-label/Ratio-based，逐 query 等规模导出 .pt
#     # =====================================================================
#     elif EXPERIMENT_STAGE == 14:
#         print("\n🔥 Stage 14：BFS/PPR/RWR × Query-label/Ratio-based 社区导出")
#         print("📐 公平约束：对每个相同 query，BFS/PPR/RWR 的社区大小严格等于 RWR-QDCC 的 S_q。")
#
#         query_nodes, baseline_data = build_stage14_baseline_communities(
#             edges_df=edges_df,
#             features_df=features_df,
#             num_samples=BEST_NUM_SAMPLES,
#             min_community_size=BEST_K_MIN,
#             max_community_size=BEST_K_MAX,
#             q_quantile=BEST_Q,
#             restart_prob=BEST_RESTART_PROB,
#             max_rwr_iter=BEST_MAX_RWR_ITER,
#             rwr_tol=1e-6,
#             seed=42,
#             mc_walks=4000,
#             mc_length=20,
#         )
#
#         dataset_dir = os.path.join(EXPERIMENT_PT_DIR, DATASET_NAME)
#         label_file_names = {"query_label": "QueryLabel", "ratio_based": "RatioBased"}
#         method_file_names = {"bfs": "BFS", "ppr": "PPR", "rwr": "RWR"}
#
#         for method_name in ("bfs", "ppr", "rwr"):
#             base_list = baseline_data[method_name]
#             for label_method in ("query_label", "ratio_based"):
#                 labeled_list = [d.clone() for d in base_list]
#                 apply_export_labels(labeled_list, features_df, label_method)
#
#                 save_name = f"{DATASET_NAME}_{method_file_names[method_name]}_{label_file_names[label_method]}.pt"
#                 save_path = os.path.join(dataset_dir, save_name)
#                 metadata = {
#                     "dataset": DATASET_NAME,
#                     "stage": 14,
#                     "construction_method": method_file_names[method_name],
#                     "label_method": label_file_names[label_method],
#                     "size_reference": "RWR-QDCC per-query S_q",
#                     "query_nodes": [int(v) for v in query_nodes],
#                     "num_samples": BEST_NUM_SAMPLES,
#                     "k_min": BEST_K_MIN,
#                     "k_max": BEST_K_MAX,
#                     "q_quantile_for_size_reference": BEST_Q,
#                     "restart_prob": BEST_RESTART_PROB,
#                     "max_rwr_iter": BEST_MAX_RWR_ITER,
#                     "rwr_tol": 1e-6,
#                     "rwr_mc_walks": 4000 if method_name == "rwr" else None,
#                     "rwr_mc_length": 20 if method_name == "rwr" else None,
#                     "ratio_tau": "global K*/N*" if label_method == "ratio_based" else None,
#                     "generated_at": now,
#                 }
#                 save_export_pt(labeled_list, save_path, metadata)
#
#         print("\n✅ Stage 14 完成：共导出 6 份 .pt，未训练任何模型。")
#         print(f"📂 输出目录: {os.path.abspath(dataset_dir)}")
#         return
#
#     # =====================================================================
#     # Stage 15：跨数据集 × 跨社区构造方法，GCN 全量评测（一个 CSV）
#     # =====================================================================
#     elif EXPERIMENT_STAGE == 15:
#         print("\n🔥 Stage 15：跨数据集 × 跨社区构造方法 GCN 全量评测")
#         print("📐 每个数据集 7 个变体：RWRQDCC_FisherGMM / RWR_RatioBased / RWR_QueryLabel / "
#               "PPR_RatioBased / PPR_QueryLabel / BFS_RatioBased / BFS_QueryLabel")
#         print("📂 从 experiment_pt/{Dataset}/ 下加载 .pt，最后汇总为单一 CSV。")
#         print("📌 每个数据集使用其在 DATASET_CONFIGS 中自己的 BEST_LR / BEST_HIDDEN。")
#
#         ALL_DATASETS_STAGE15 = ["Actor", "Arxiv", "Elliptic", "Roman", "DGraph", "Minesweeper"]
#         PT_VARIANTS_STAGE15 = [
#             "RWRQDCC_FisherGMM",
#             "RWR_RatioBased",
#             "RWR_QueryLabel",
#             "PPR_RatioBased",
#             "PPR_QueryLabel",
#             "BFS_RatioBased",
#             "BFS_QueryLabel",
#         ]
#
#         stage15_results = []
#         stage15_csv = os.path.join(EXPERIMENT_DIR, f"Stage15_GCN_AllDatasets_{now}.csv")
#
#         total_jobs = len(ALL_DATASETS_STAGE15) * len(PT_VARIANTS_STAGE15)
#         job_idx = 0
#
#         for ds_s15 in ALL_DATASETS_STAGE15:
#             # ★ 关键修复：按当前数据集重新读取它自己的最优超参，而不是沿用 main() 顶部的 Elliptic 配置
#             if ds_s15 not in DATASET_CONFIGS:
#                 print(f"⚠️ DATASET_CONFIGS 中没有 {ds_s15} 的配置，跳过整个数据集。")
#                 continue
#             ds_cfg_s15 = DATASET_CONFIGS[ds_s15]
#             params = {
#                 "learning_rate": ds_cfg_s15["BEST_LR"],
#                 "hidden_dim": ds_cfg_s15["BEST_HIDDEN"],
#             }
#             print("\n" + "★" * 100)
#             print(f"★ [Stage15] Dataset={ds_s15} | 使用该数据集自己的最优参数 | "
#                   f"lr={params['learning_rate']} | hidden={params['hidden_dim']}")
#             print("★" * 100)
#
#             for variant in PT_VARIANTS_STAGE15:
#                 job_idx += 1
#                 pt_name = f"{ds_s15}_{variant}.pt"
#                 pt_path = os.path.join(EXPERIMENT_PT_DIR, ds_s15, pt_name)
#
#                 print("\n" + "=" * 120)
#                 print(f"🔥 [Stage15 | {job_idx}/{total_jobs}] Dataset={ds_s15} | Variant={variant}")
#                 print(f"📂 PT 路径: {pt_path}")
#                 print(f"📌 参数: lr={params['learning_rate']} | hidden={params['hidden_dim']}")
#                 print("=" * 120)
#
#                 if not os.path.exists(pt_path):
#                     print(f"⚠️ 文件不存在，跳过: {pt_path}")
#                     continue
#
#                 reset_all_seeds(42)
#
#                 try:
#                     data_list_s15 = torch.load(pt_path, weights_only=False)
#                 except Exception as e:
#                     print(f"❌ 加载失败: {e}")
#                     continue
#
#                 if not data_list_s15:
#                     print(f"⚠️ 数据为空，跳过: {pt_path}")
#                     continue
#
#                 total_count = len(data_list_s15)
#                 normal_count = sum(1 for d in data_list_s15 if int(d.y.item()) == 0)
#                 anomalous_count = sum(1 for d in data_list_s15 if int(d.y.item()) == 1)
#                 uncertain_count = sum(1 for d in data_list_s15 if int(d.y.item()) == -1)
#                 supervised_count = normal_count + anomalous_count
#
#                 avg_nodes = float(np.mean([d.x.shape[0] for d in data_list_s15]))
#                 try:
#                     avg_quality = float(np.mean([d.community_quality.item() for d in data_list_s15]))
#                 except Exception:
#                     avg_quality = float("nan")
#
#                 normal_ratio = normal_count / total_count if total_count > 0 else 0.0
#                 anomalous_ratio = anomalous_count / total_count if total_count > 0 else 0.0
#                 uncertain_ratio = uncertain_count / total_count if total_count > 0 else 0.0
#
#                 print(
#                     f"📊 Communities={total_count} | Supervised={supervised_count} | "
#                     f"N={normal_count} | A={anomalous_count} | U={uncertain_count} | "
#                     f"|C|_avg={avg_nodes:.2f} | S_avg={avg_quality:.4f}"
#                 )
#
#                 # 类别不全会导致 StratifiedKFold 崩，直接跳过
#                 if normal_count == 0 or anomalous_count == 0:
#                     print("⚠️ 正/负类缺一，无法训练，跳过。")
#                     continue
#
#                 os.makedirs(args.bestModelPath, exist_ok=True)
#                 model_path_s15 = os.path.join(
#                     args.bestModelPath,
#                     f"stage15_{ds_s15}_{variant}_{now}.pth"
#                 )
#
#                 try:
#                     _, avg_test_auc, std_test_auc = cross_val_train_and_test_gcn(
#                         data_list_s15,
#                         params,
#                         device,
#                         batch_size=args.batch_size,
#                         save_best_model_path=model_path_s15,
#                     )
#                 except Exception as e:
#                     print(f"❌ 训练失败: {e}")
#                     import traceback
#                     traceback.print_exc()
#                     continue
#
#                 row_s15 = {
#                     "Dataset": ds_s15,
#                     "Variant": variant,
#                     "Model": "GCN",
#                     "Community_Count": total_count,
#                     "Supervised_Count": supervised_count,
#                     "Normal_Count": normal_count,
#                     "Anomalous_Count": anomalous_count,
#                     "Uncertain_Count": uncertain_count,
#                     "Normal_Ratio": normal_ratio,
#                     "Anomalous_Ratio": anomalous_ratio,
#                     "Uncertain_Ratio": uncertain_ratio,
#                     "|C|_avg": avg_nodes,
#                     "S_avg": avg_quality,
#                     "AUC": avg_test_auc * 100,
#                     "STD": std_test_auc * 100,
#                     "learning_rate": params["learning_rate"],
#                     "hidden_dim": params["hidden_dim"],
#                 }
#                 stage15_results.append(row_s15)
#
#                 # 每跑完一个变体立刻落盘，防止中途被打断丢结果
#                 pd.DataFrame(stage15_results).to_csv(
#                     stage15_csv, index=False, encoding="utf-8-sig"
#                 )
#                 print(f"💾 已更新: {stage15_csv} (已完成 {len(stage15_results)} 条)")
#
#         print("\n" + "🏆" * 50)
#         print(f"🏆 Stage 15 全部完成 | 共生成 {len(stage15_results)} 条结果")
#         print("🏆" * 50)
#
#         if stage15_results:
#             stage15_df = pd.DataFrame(stage15_results)
#             stage15_df = stage15_df.sort_values(["Dataset", "AUC"], ascending=[True, False])
#             stage15_df.to_csv(stage15_csv, index=False, encoding="utf-8-sig")
#             print(stage15_df.to_string(index=False))
#             print(f"\n💾 最终 CSV: {stage15_csv}")
#         else:
#             print("⚠️ Stage 15 没有生成任何结果。")
#
#         return
#
#     # =====================================================================
#     # 统一结果汇总
#     # =====================================================================
#     print("\n" + "🏆" * 50)
#     print(f"🏆 Dataset={DATASET_NAME} | Stage={EXPERIMENT_STAGE} 结果汇总")
#     print("🏆" * 50)
#
#     if not results:
#         print("⚠️ 没有生成任何实验结果，不保存 CSV。")
#         return
#
#     results.sort(key=lambda x: x["AUC"], reverse=True)
#
#     metric_keys = {
#         "AUC", "STD", "|C|_avg", "S_avg",
#         "Community_Count", "Supervised_Count",
#         "Normal_Count", "Anomalous_Count", "Uncertain_Count",
#         "Normal_Ratio", "Anomalous_Ratio", "Uncertain_Ratio",
#     }
#
#     for rank, res in enumerate(results, start=1):
#         param_str = " | ".join(f"{k}: {v}" for k, v in res.items() if k not in metric_keys)
#
#         print(
#             f"Rank {rank:2d} | {param_str} | "
#             f"Community={res['Community_Count']} | Supervised={res['Supervised_Count']} | "
#             f"N={res['Normal_Count']} | A={res['Anomalous_Count']} | U={res['Uncertain_Count']} | "
#             f"|C|_avg={res['|C|_avg']:.2f} | S_avg={res['S_avg']:.4f} | "
#             f"AUC={res['AUC']:.4f}% ± {res['STD']:.4f}%"
#         )
#
#     print("🏆" * 50)
#
#     # =====================================================================
#     # 当前 Stage 实验结果保存 CSV
#     # =====================================================================
#     results_df = pd.DataFrame(results)
#
#     # AUC 排序后增加 Rank
#     results_df.insert(0, "Rank", range(1, len(results_df) + 1))
#     results_df.insert(1, "Dataset", DATASET_NAME)
#     results_df.insert(2, "Stage", EXPERIMENT_STAGE)
#
#     # 同时记录本次实验使用的基础最优参数
#     results_df["Best_Num_Samples"] = BEST_NUM_SAMPLES
#     results_df["Best_K_Min"] = BEST_K_MIN
#     results_df["Best_K_Max"] = BEST_K_MAX
#     results_df["Best_Q"] = BEST_Q
#     results_df["Best_Restart_Prob"] = BEST_RESTART_PROB
#     results_df["Best_Max_RWR_Iter"] = BEST_MAX_RWR_ITER
#     results_df["Best_LR"] = BEST_LR
#     results_df["Best_Hidden"] = BEST_HIDDEN
#     results_df["Best_Tau"] = BEST_TAU
#
#     csv_path = os.path.join(
#         EXPERIMENT_DIR,
#         f"{DATASET_NAME}_stage{EXPERIMENT_STAGE}_{now}.csv"
#     )
#
#     results_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
#
#     print(f"\n💾 CSV 已保存: {csv_path}")
#     print(f"📊 共保存 {len(results_df)} 条实验结果")
#     print(f"📂 保存目录: {os.path.abspath(EXPERIMENT_DIR)}\n")
#
#
# if __name__ == "__main__":
#     import time
#
#     start_time = time.time()
#     main()
#     elapsed = time.time() - start_time
#     print(f"\n⏱️  main() 总耗时: {elapsed:.2f} 秒 ({elapsed / 60:.2f} 分钟)")


import os
import json
import random
from datetime import datetime
from itertools import product
import pandas as pd
import numpy as np
import torch

print(torch.__version__)

import ArgParser

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.mixture import GaussianMixture
from scipy.stats import fisher_exact
# from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from PathAwareGNN import PathAwareGNN
from SimpleGCN import SimpleGCN

from data_process.Data_Enhance import UltimateTripleStreamAugmentor
from data_process.Data_Query import Data_Query
from data_process.Eearly_Stopping import RobustEarlyStopping
from data_process.Community_Search import select_nodes_via_label

# ==========================================================
# 使用当前同目录下的 RWR_QDCC
# ==========================================================
from RWR_QDCC import (
    dap_arwr_first_stage_gpu,
    dap_arwr_single_query_balanced,
    preprocess_graph_gpu,
)

# Stage 14 复用之前社区构造对比实验中的 BFS / PPR / RWR
from qisc_construction_benchmark import (
    build_graph_from_dataframes,
    bfs_select,
    ppr_scores,
    connected_ranked_expand,
    rwr_mc_select,
    torch_graph_cache,
    clear_torch_graph_cache,
)

os.environ["OMP_NUM_THREADS"] = "6"


def train(model, loader, optimizer, device, augmentor=None):
    model.train()
    total_loss = 0

    for data in loader:
        data = data.to(device, non_blocking=True)
        optimizer.zero_grad()

        # 🚀 自动判定并执行一致性正则化
        has_gray_samples = (data.y == -1).any().item()
        if has_gray_samples and augmentor is not None:
            loss = model.total_loss(data, augmentor=augmentor)
        else:
            loss = model.total_loss(data)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def test(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)

            # 1. 究极防爆：无限解包取出真实的 Tensor
            out = model.predict(data)
            while isinstance(out, (tuple, list)):
                out = out[0]
            pred = out

            # 2. 维度安全保护
            if pred.dim() == 0:
                pred = pred.unsqueeze(0)

            # 3. 过滤掉灰区数据（测试和验证集决不包含 -1 的杂质！）
            valid_mask = (data.y != -1)
            if valid_mask.sum() > 0:
                y_true.append(data.y[valid_mask].cpu().numpy())
                y_pred.append(pred[valid_mask].cpu().numpy())

    if len(y_true) == 0: return 0.0

    y_true = np.concatenate(y_true).ravel()
    y_pred = np.concatenate(y_pred)

    if y_pred.ndim == 2:
        y_pred = y_pred[:, 1] if y_pred.shape[1] >= 2 else y_pred.ravel()

    y_true_binary = (y_true > 0).astype(int)

    if len(np.unique(y_true_binary)) < 2:
        return 0.0

    return roc_auc_score(y_true_binary, y_pred)


# # 5折数据划分函数
def stratified_split(data_list, val_ratio=0.2,
                     n_splits=5, batch_size=128, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    labels = np.array([data.y.item() for data in data_list])
    fold_loaders = []

    # 🚀 安全机制：检查每个类别的样本数，如果最少类别的数量 < n_splits，退化为普通的 KFold
    unique_classes, counts = np.unique(labels, return_counts=True)
    min_count = np.min(counts)

    if min_count < n_splits:
        tqdm.write(
            f"⚠️ 警告: 类别 {unique_classes[np.argmin(counts)]} 只有 {min_count} 个样本，无法进行严格的分层划分。自动切换为普通 KFold。")
        main_kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        inner_splitter_class = KFold
    else:
        main_kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        inner_splitter_class = StratifiedKFold

    for fold, (train_val_idx, test_idx) in enumerate(main_kfold.split(data_list, labels)):
        train_labels = labels[train_val_idx]

        # 内部验证集划分
        inner_splitter = inner_splitter_class(n_splits=int(1 / val_ratio), shuffle=True, random_state=seed + fold)

        for train_idx, val_idx in inner_splitter.split(train_val_idx, train_labels):
            real_train_idx = train_val_idx[train_idx]
            real_val_idx = train_val_idx[val_idx]
            break

        train_data = [data_list[i] for i in real_train_idx]
        val_data = [data_list[i] for i in real_val_idx]
        test_data = [data_list[i] for i in test_idx]

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=False)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, pin_memory=False)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, pin_memory=False)

        fold_loaders.append((train_loader, val_loader, test_loader))

    return fold_loaders


def compute_val_loss(model, val_loader, device):
    """计算验证集损失"""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)
            loss = model.total_loss(data)
            total_loss += loss.item()
    return total_loss / len(val_loader)


def cross_val_train_and_test(data_list, params, device, num_epochs=300, k_folds=5, batch_size=300,
                             save_best_model_path="best_model.pth"):
    folds = stratified_split(data_list, n_splits=k_folds, batch_size=batch_size)

    test_aucs = []
    best_overall_model = None
    best_overall_test_auc = -float('inf')

    warmup_epochs = 15
    patience_val = 25

    start_time = datetime.now()

    # 🚀 强力打印实验参数
    print("\n" + "★" * 80)
    print(f"★ 启动新实验 | 学习率 LR: {params['learning_rate']} | 隐藏层 Hidden: {params['hidden_dim']}")
    print("★" * 80)

    global_pbar = tqdm(total=k_folds * num_epochs, desc="🚀 训练进度", ncols=180, unit="step",
                       bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}, {postfix}]")

    augmentor = UltimateTripleStreamAugmentor(feature_noise=0.02, edge_drop_rate=0.15, path_jitter_rate=0.2)

    for fold in range(k_folds):
        train_loader, val_loader, test_loader = folds[fold]

        # model = PathAwareGNN(feat_dim=data_list[0].x.shape[1], hidden_dim=params['hidden_dim'], dropout_prob=0.6).to(
        #     device)
        model = PathAwareGNN(
            feat_dim=data_list[0].x.shape[1],
            hidden_dim=params['hidden_dim'],
            num_layers=params.get('num_layers', 2),
            dropout_prob=0.6
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
        early_stopping = RobustEarlyStopping(patience=patience_val, min_delta=0.001, smoothing=10,
                                             restore_best_weights=True, verbose=False)

        fold_best_val_auc = -float('inf')
        fold_best_epoch = 0
        stop_training = False

        for epoch in range(num_epochs):
            train_loss = train(model, train_loader, optimizer, device, augmentor)
            val_auc = test(model, val_loader, device)

            scheduler.step(val_auc)
            current_lr = optimizer.param_groups[0]['lr']

            if val_auc > fold_best_val_auc:
                fold_best_val_auc = val_auc
                fold_best_epoch = epoch + 1

            if epoch >= warmup_epochs:
                if early_stopping(val_auc, model):
                    stop_training = True

            global_pbar.set_postfix({'折数': f'{fold + 1}/{k_folds}', 'Epoch': f'{epoch + 1}/{num_epochs}',
                                     'Loss': f'{train_loss:.4f}', 'ValAUC': f'{val_auc:.4f}', 'LR': f'{current_lr:.2e}',
                                     '最佳Epoch': f'{fold_best_epoch}', '最佳ValAUC': f'{fold_best_val_auc:.4f}'})
            global_pbar.update(1)

            if stop_training:
                global_pbar.update(num_epochs - epoch - 1)
                break

        test_auc = test(model, test_loader, device)
        test_aucs.append(test_auc)

        if test_auc > best_overall_test_auc:
            best_overall_test_auc = test_auc
            best_overall_model = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        tqdm.write(f"\n✅ Fold {fold + 1} 完成 | Test AUC: {test_auc:.4f} | 最佳Epoch: {fold_best_epoch}")

    global_pbar.close()

    # 🚀 强力汇总当前实验的最终参数与结果
    res_str = f"🏁 实验结束 -> LR: {params['learning_rate']} | Hidden: {params['hidden_dim']} | 最终AUC: {np.mean(test_aucs):.4f} ± {np.std(test_aucs):.4f}"
    print("\n" + "=" * len(res_str))
    print(res_str)
    print("=" * len(res_str) + "\n")

    if best_overall_model is not None:
        torch.save(best_overall_model, save_best_model_path)
    return params, np.mean(test_aucs), np.std(test_aucs)


def cross_val_train_and_test_gcn(data_list, params, device, num_epochs=300, k_folds=5, batch_size=300,
                                 save_best_model_path="best_gcn_model.pth"):
    """GCN 基线版本的交叉验证训练（与 PathAwareGNN 版本结构完全一致，仅替换模型类）"""
    folds = stratified_split(data_list, n_splits=k_folds, batch_size=batch_size)

    test_aucs = []
    best_overall_model = None
    best_overall_test_auc = -float('inf')

    warmup_epochs = 15
    patience_val = 25

    start_time = datetime.now()

    print("\n" + "★" * 80)
    print(f"★ [GCN基线] 启动新实验 | LR: {params['learning_rate']} | Hidden: {params['hidden_dim']}")
    print("★" * 80)

    global_pbar = tqdm(total=k_folds * num_epochs, desc="🔵 GCN训练进度", ncols=180, unit="step",
                       bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}, {postfix}]")

    for fold in range(k_folds):
        train_loader, val_loader, test_loader = folds[fold]

        # 🔵 唯一区别：使用 SimpleGCN 模型
        model = SimpleGCN(
            feat_dim=data_list[0].x.shape[1],
            hidden_dim=params['hidden_dim'],
            num_layers=params.get('num_layers', 2),
            dropout_prob=0.6
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
        early_stopping = RobustEarlyStopping(patience=patience_val, min_delta=0.001, smoothing=10,
                                             restore_best_weights=True, verbose=False)

        fold_best_val_auc = -float('inf')
        fold_best_epoch = 0
        stop_training = False

        for epoch in range(num_epochs):
            # GCN 不需要 augmentor（因为没有一致性正则化分支）
            train_loss = train(model, train_loader, optimizer, device, augmentor=None)
            val_auc = test(model, val_loader, device)

            scheduler.step(val_auc)
            current_lr = optimizer.param_groups[0]['lr']

            if val_auc > fold_best_val_auc:
                fold_best_val_auc = val_auc
                fold_best_epoch = epoch + 1

            if epoch >= warmup_epochs:
                if early_stopping(val_auc, model):
                    stop_training = True

            global_pbar.set_postfix({'折数': f'{fold + 1}/{k_folds}', 'Epoch': f'{epoch + 1}/{num_epochs}',
                                     'Loss': f'{train_loss:.4f}', 'ValAUC': f'{val_auc:.4f}', 'LR': f'{current_lr:.2e}',
                                     '最佳Epoch': f'{fold_best_epoch}', '最佳ValAUC': f'{fold_best_val_auc:.4f}'})
            global_pbar.update(1)

            if stop_training:
                global_pbar.update(num_epochs - epoch - 1)
                break

        test_auc = test(model, test_loader, device)
        test_aucs.append(test_auc)

        if test_auc > best_overall_test_auc:
            best_overall_test_auc = test_auc
            best_overall_model = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        tqdm.write(f"\n✅ GCN Fold {fold + 1} 完成 | Test AUC: {test_auc:.4f} | 最佳Epoch: {fold_best_epoch}")

    global_pbar.close()

    res_str = f"🏁 [GCN基线] 实验结束 -> LR: {params['learning_rate']} | Hidden: {params['hidden_dim']} | 最终AUC: {np.mean(test_aucs):.4f} ± {np.std(test_aucs):.4f}"
    print("\n" + "=" * len(res_str))
    print(res_str)
    print("=" * len(res_str) + "\n")

    if best_overall_model is not None:
        torch.save(best_overall_model, save_best_model_path)
    return params, np.mean(test_aucs), np.std(test_aucs)


# =====================================================================
# Stage 13 / 14 数据导出辅助函数
# =====================================================================
def reset_all_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def build_pyg_subgraph_from_indices(selected_nodes, q_idx, all_edge_index, node_feat_tensor, node_label_tensor):
    """把 BFS/PPR/RWR 选出的全局节点索引打包成与 RWR-QDCC 一致的 PyG Data。"""
    selected_nodes = [int(v) for v in selected_nodes]
    if q_idx not in selected_nodes:
        raise RuntimeError(f"query index {q_idx} 不在 baseline 社区中。")

    device = node_feat_tensor.device
    n_nodes = node_feat_tensor.size(0)
    community_idx_tensor = torch.tensor(selected_nodes, dtype=torch.long, device=device)

    x = node_feat_tensor[community_idx_tensor].detach()
    x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
    node_labels = node_label_tensor[community_idx_tensor].detach()
    local_q_idx = selected_nodes.index(q_idx)

    mask = torch.isin(all_edge_index[0], community_idx_tensor) & torch.isin(all_edge_index[1], community_idx_tensor)
    edge_index_global = all_edge_index[:, mask]

    if edge_index_global.shape[1] > 0:
        global2local = torch.zeros(n_nodes, dtype=torch.long, device=device)
        global2local[community_idx_tensor] = torch.arange(len(selected_nodes), device=device)
        edge_index = global2local[edge_index_global]

        # 与 RWR-QDCC 当前实现统一：community_quality 使用社区内部边两端特征余弦相似度均值
        src_global = edge_index_global[0]
        dst_global = edge_index_global[1]
        feat_src = node_feat_tensor[src_global]
        feat_dst = node_feat_tensor[dst_global]
        edge_cos = torch.nn.functional.cosine_similarity(feat_src, feat_dst, dim=1).clamp(min=0.0)
        avg_quality = torch.mean(edge_cos).item()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        avg_quality = 0.0

    return Data(
        x=x,
        edge_index=edge_index,
        node_labels=node_labels,
        q_id=torch.tensor([local_q_idx], dtype=torch.long, device=device),
        y=torch.tensor([-1], dtype=torch.long, device=device),
        community_quality=torch.tensor([avg_quality], dtype=torch.float32, device=device),
        p_val=None,
    )


def apply_export_labels(data_list, features_df, label_method, label_tau=None):
    """Stage 14 只使用 Query-label 或 Ratio-based，两者与当前 RWR_QDCC.py 定义保持一致。"""
    all_labels = features_df['label'].to_numpy(dtype=int)
    valid_global = all_labels != -1
    n_star = int(valid_global.sum())
    k_star = int((all_labels[valid_global] == 1).sum())

    if label_method == 'query_label':
        for d in data_list:
            q_local_idx = int(d.q_id.item())
            query_label = int(d.node_labels[q_local_idx].item())
            if query_label == 1:
                d.y = torch.tensor([1], dtype=torch.long, device=d.x.device)
                d.p_val = 0.0
            elif query_label == 0:
                d.y = torch.tensor([0], dtype=torch.long, device=d.x.device)
                d.p_val = 1.0
            else:
                d.y = torch.tensor([-1], dtype=torch.long, device=d.x.device)
                d.p_val = 2.0

    elif label_method == 'ratio_based':
        tau = label_tau if label_tau is not None else (k_star / n_star if n_star > 0 else 0.5)
        for d in data_list:
            comm_labels = d.node_labels.detach().cpu().numpy()
            valid = comm_labels != -1
            n_labeled = int(valid.sum())
            k_labeled = int((comm_labels[valid] == 1).sum())
            if n_labeled == 0:
                d.y = torch.tensor([-1], dtype=torch.long, device=d.x.device)
                d.p_val = 2.0
                continue
            local_ratio = k_labeled / n_labeled
            d.p_val = float(1.0 - local_ratio)
            d.y = torch.tensor([1 if local_ratio > tau else 0], dtype=torch.long, device=d.x.device)
    else:
        raise ValueError(f"Stage 14 不支持 label_method={label_method}")

    return data_list


def apply_fisher_gmm_labels_stage16(
        data_list,
        features_df,
        confidence_tau=0.90,
        seed=42,
        min_known_nodes_for_confident_label=2,
):
    """Stage 16 Fisher-GMM 适配器：给任意已构造社区列表统一施加最终 Fisher-GMM 标签。

    目的：Stage 14 的 connected Monte-Carlo RWR 只负责社区构造，并不带
    Fisher-GMM 标签。为了做到“只消融 RWR-QDCC，其他组件保持不变”，这里
    对 RWR 社区和 RWR-QDCC 社区使用完全相同的 Fisher-GMM 标签逻辑。

    最终规则采用 post-gate：
      1) 所有 n_L >= 1 的社区先计算 Fisher score 并共同参与 2-GMM 拟合；
      2) GMM 推断完成后，n_L < 2 强制为 -1；
      3) n_L >= 2 时，gamma >= tau -> 1，gamma <= 1-tau -> 0，其他 -> -1。
    """
    if not data_list:
        return data_list

    all_labels = features_df['label'].to_numpy(dtype=int)
    valid_global = all_labels != -1
    n_star = int(valid_global.sum())
    k_star = int((all_labels[valid_global] == 1).sum())
    if n_star <= 0 or k_star <= 0:
        raise RuntimeError("Fisher-GMM 需要全局已知 normal/anomaly 标签。")

    scores = np.full(len(data_list), np.nan, dtype=float)
    p_values = np.full(len(data_list), np.nan, dtype=float)
    known_counts = np.zeros(len(data_list), dtype=int)

    for i, d in enumerate(data_list):
        comm_labels = d.node_labels.detach().cpu().numpy().astype(int).reshape(-1)
        valid = comm_labels != -1
        n_l = int(valid.sum())
        k_l = int((comm_labels[valid] == 1).sum())
        known_counts[i] = n_l

        if n_l <= 0:
            d.p_val = 2.0
            d.y = torch.tensor([-1], dtype=torch.long, device=d.x.device)
            continue

        local_normals = n_l - k_l
        outside_anomalies = k_star - k_l
        outside_normals = (n_star - k_star) - local_normals
        if min(k_l, local_normals, outside_anomalies, outside_normals) < 0:
            raise RuntimeError(
                f"非法 Fisher 列联表: local(k={k_l}, n={n_l}), global(K={k_star}, N={n_star})"
            )

        _, p_val = fisher_exact(
            [[k_l, local_normals], [outside_anomalies, outside_normals]],
            alternative='greater',
        )
        p_val = float(np.clip(p_val, 1e-300, 1.0))
        p_values[i] = p_val
        scores[i] = -np.log(p_val)
        d.p_val = p_val

    fit_mask = np.isfinite(scores) & (known_counts >= 1)
    fit_scores = scores[fit_mask]

    # 默认全部为 uncertain，再由 GMM posterior + post-gate 写回。
    for d in data_list:
        d.y = torch.tensor([-1], dtype=torch.long, device=d.x.device)

    if len(fit_scores) < 2 or np.unique(fit_scores).size < 2:
        print("⚠️ Stage16 Fisher-GMM: 可用 Fisher score 不足，全部保持 -1。")
        return data_list

    gmm = GaussianMixture(
        n_components=2,
        covariance_type='full',
        reg_covar=1e-6,
        n_init=10,
        random_state=seed,
    ).fit(fit_scores.reshape(-1, 1))

    means = gmm.means_.reshape(-1)
    anomaly_component = int(np.argmax(means))
    posterior = gmm.predict_proba(fit_scores.reshape(-1, 1))[:, anomaly_component]

    fit_indices = np.where(fit_mask)[0]
    pos_count = neg_count = uncertain_count = 0
    insufficient_count = ambiguous_count = 0

    for local_idx, data_idx in enumerate(fit_indices):
        d = data_list[data_idx]
        gamma = float(posterior[local_idx])

        if known_counts[data_idx] < min_known_nodes_for_confident_label:
            d.y = torch.tensor([-1], dtype=torch.long, device=d.x.device)
            d.p_val = 2.0
            uncertain_count += 1
            insufficient_count += 1
        elif gamma >= confidence_tau:
            d.y = torch.tensor([1], dtype=torch.long, device=d.x.device)
            pos_count += 1
        elif gamma <= 1.0 - confidence_tau:
            d.y = torch.tensor([0], dtype=torch.long, device=d.x.device)
            neg_count += 1
        else:
            d.y = torch.tensor([-1], dtype=torch.long, device=d.x.device)
            uncertain_count += 1
            ambiguous_count += 1

    # n_L == 0 不在 fit_indices 中，也属于 uncertain。
    zero_known_count = int((known_counts == 0).sum())
    uncertain_count += zero_known_count
    insufficient_count += zero_known_count

    print(
        f"🏷️ Stage16 Fisher-GMM relabel | N={neg_count} | A={pos_count} | U={uncertain_count} | "
        f"insufficient={insufficient_count} | ambiguous={ambiguous_count} | "
        f"GMM means={means.tolist()} | tau={confidence_tau}"
    )
    return data_list


def save_export_pt(data_list, save_path, metadata):
    """.pt 保持为纯 data_list，便于其他异常检测方法直接 torch.load；参数另存 JSON。"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cpu_data_list = [d.cpu() for d in data_list]
    torch.save(cpu_data_list, save_path)

    meta_path = os.path.splitext(save_path)[0] + '.json'
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    normal_count = sum(1 for d in cpu_data_list if int(d.y.item()) == 0)
    anomalous_count = sum(1 for d in cpu_data_list if int(d.y.item()) == 1)
    uncertain_count = sum(1 for d in cpu_data_list if int(d.y.item()) == -1)
    avg_nodes = float(np.mean([d.x.shape[0] for d in cpu_data_list])) if cpu_data_list else 0.0
    avg_edges = float(np.mean([d.edge_index.shape[1] for d in cpu_data_list])) if cpu_data_list else 0.0

    print(f"💾 PT 已保存: {save_path}")
    print(f"   Communities={len(cpu_data_list)} | N={normal_count} | A={anomalous_count} | U={uncertain_count} | "
          f"AvgNodes={avg_nodes:.2f} | AvgEdges={avg_edges:.2f}")
    print(f"🧾 Metadata: {meta_path}")


def build_stage14_baseline_communities(
        edges_df,
        features_df,
        num_samples,
        min_community_size,
        max_community_size,
        q_quantile,
        restart_prob,
        max_rwr_iter,
        rwr_tol=1e-6,
        seed=42,
        mc_walks=4000,
        mc_length=20,
        requested_methods=("bfs", "ppr", "rwr"),
):
    """
    Stage 14/16 共用：对同一批 query，先用 RWR-QDCC 得到逐 query 目标规模 S_q，
    再构造 size-matched baseline 社区。

    requested_methods 默认保持 Stage 14 原行为 (BFS/PPR/RWR 全部构造)；
    Stage 16 仅传入 ("rwr",)，从而直接复用同一个 connected Monte-Carlo RWR
    实现，而不额外计算 BFS/PPR。
    """
    requested_methods = tuple(requested_methods)
    valid_methods = {"bfs", "ppr", "rwr"}
    unknown = set(requested_methods) - valid_methods
    if unknown:
        raise ValueError(f"未知 baseline method: {sorted(unknown)}")

    reset_all_seeds(seed)

    all_edge_index, node2idx, idx2node, node_feat_tensor, node_label_tensor, ptr, dst = preprocess_graph_gpu(
        edges_df, features_df
    )

    clear_torch_graph_cache()
    _, _, x_np, adjacency = build_graph_from_dataframes(
        edges_df, features_df, node_id_col='node_id', label_col='label'
    )
    torch_graph_cache(x_np, adjacency)

    # 与 dap_arwr_first_stage_gpu 完全相同的 query 选择入口。
    anomaly_nodes, normal_nodes = select_nodes_via_label(features_df, num_samples)
    query_node_list = anomaly_nodes + normal_nodes

    method_data = {name: [] for name in requested_methods}
    method_desc = "/".join(name.upper() for name in requested_methods)
    pbar = tqdm(
        total=len(query_node_list),
        desc=f'📦 构造 {method_desc}',
        ncols=180,
        bar_format='{l_bar}{bar:60}{r_bar}',
        colour='cyan',
    )

    for position, q_id in enumerate(query_node_list, start=1):
        q_id_int = int(q_id)
        q_idx = node2idx.get(q_id_int, -1)
        if q_idx < 0:
            raise RuntimeError(f"Query {q_id_int} 不在 node2idx 中。")

        # 只用于获取该 query 在 RWR-QDCC 下的真实社区规模 S_q。
        qdcc_data = dap_arwr_single_query_balanced(
            q_id_int, node2idx, idx2node, all_edge_index, ptr, dst,
            node_feat_tensor, node_label_tensor,
            min_community_size, max_community_size, q_quantile,
            restart_prob=restart_prob, max_rwr_iter=max_rwr_iter, rwr_tol=rwr_tol
        )
        if qdcc_data is None:
            raise RuntimeError(
                f"RWR-QDCC 在 Query {q_id_int} 上返回 None，无法进行成对规模对齐。"
            )

        target_size = int(qdcc_data.x.shape[0])
        selected = {}

        if "bfs" in requested_methods:
            selected["bfs"] = bfs_select(q_idx, target_size, adjacency)

        if "ppr" in requested_methods:
            selected["ppr"] = connected_ranked_expand(
                q_idx,
                ppr_scores(q_idx, adjacency, restart_prob, max_rwr_iter, rwr_tol),
                target_size,
                adjacency,
            )

        if "rwr" in requested_methods:
            selected["rwr"] = rwr_mc_select(
                q_idx,
                target_size,
                adjacency,
                restart_prob,
                mc_walks,
                mc_length,
                seed + 104729 * position,
            )

        for method_name, nodes in selected.items():
            if len(nodes) != target_size:
                raise RuntimeError(
                    f"{method_name.upper()} 在 Query {q_id_int} 上规模未对齐: "
                    f"target={target_size}, actual={len(nodes)}"
                )
            method_data[method_name].append(
                build_pyg_subgraph_from_indices(
                    nodes,
                    q_idx,
                    all_edge_index,
                    node_feat_tensor,
                    node_label_tensor,
                )
            )

        pbar.update(1)
        if position % 10 == 0:
            pbar.set_postfix({'query': position, 'S_q': target_size})

    pbar.close()
    return query_node_list, method_data


def main():
    torch.autograd.set_detect_anomaly(False)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    args = ArgParser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    now = datetime.now().strftime("%Y%m%d_%H%M%S")

    # =====================================================================
    # 核心控制台：平时只需要修改下面两行
    # =====================================================================
    DATASET_NAME = "Minesweeper"
    EXPERIMENT_STAGE = 5

    # 可选数据集：
    # "Elliptic" / "DGraph" / "Actor" / "Arxiv" / "Roman" / "Minesweeper"
    #
    # Stage:
    # 1  -> k_min × k_max
    # 2  -> q_quantile
    # 3  -> num_samples
    # 4  -> restart_prob × max_rwr_iter
    # 5  -> num_layers
    # 8  -> confidence_tau
    # 9  -> GCN baseline
    # 10 -> PathAwareGNN 最优参数验证 + data_list 导出
    # 11 -> learning_rate × hidden_dim
    # 12 -> query_label / ratio_based / fisher_gmm
    # 13 -> RWR-QDCC + Fisher-GMM：使用最优参数导出 .pt，不训练
    # 14 -> BFS/PPR/RWR × Query-label/Ratio-based：逐 query 与 RWR-QDCC 等规模导出 .pt，不训练
    # 15 -> 六数据集 × 7 个导出变体：GCN 全量评测
    # 16 -> 六数据集三组件单独消融：RWR / Ratio-based / GCN，汇总一个 CSV

    # =====================================================================
    # 各数据集历史最优参数
    # =====================================================================
    DATASET_CONFIGS = {
        "Elliptic": {
            "BEST_NUM_SAMPLES": 3000, "BEST_K_MIN": 5, "BEST_K_MAX": 90, "BEST_Q": 0.75,
            "BEST_LR": 0.003, "BEST_HIDDEN": 256, "BEST_RESTART_PROB": 0.15, "BEST_MAX_RWR_ITER": 200, "BEST_TAU": 0.9,
        },
        "DGraph": {
            "BEST_NUM_SAMPLES": 2000, "BEST_K_MIN": 5, "BEST_K_MAX": 30, "BEST_Q": 0.9,
            "BEST_LR": 0.001, "BEST_HIDDEN": 128, "BEST_RESTART_PROB": 0.15, "BEST_MAX_RWR_ITER": 50, "BEST_TAU": 0.9,
        },
        "Actor": {
            "BEST_NUM_SAMPLES": 2000, "BEST_K_MIN": 20, "BEST_K_MAX": 90, "BEST_Q": 0.85,
            "BEST_LR": 0.003, "BEST_HIDDEN": 128, "BEST_RESTART_PROB": 0.05, "BEST_MAX_RWR_ITER": 200, "BEST_TAU": 0.9,
        },
        "Arxiv": {
            "BEST_NUM_SAMPLES": 2000, "BEST_K_MIN": 5, "BEST_K_MAX": 90, "BEST_Q": 0.6,
            "BEST_LR": 0.001, "BEST_HIDDEN": 256, "BEST_RESTART_PROB": 0.15, "BEST_MAX_RWR_ITER": 50, "BEST_TAU": 0.9,
        },
        "Roman": {
            "BEST_NUM_SAMPLES": 1500, "BEST_K_MIN": 5, "BEST_K_MAX": 30, "BEST_Q": 0.75,
            "BEST_LR": 0.0001, "BEST_HIDDEN": 256, "BEST_RESTART_PROB": 0.05, "BEST_MAX_RWR_ITER": 150, "BEST_TAU": 0.9,
        },
        "Minesweeper": {
            "BEST_NUM_SAMPLES": 2000, "BEST_K_MIN": 20, "BEST_K_MAX": 90, "BEST_Q": 0.5,
            "BEST_LR": 0.003, "BEST_HIDDEN": 256, "BEST_RESTART_PROB": 0.15, "BEST_MAX_RWR_ITER": 200, "BEST_TAU": 0.9,
        },
    }

    # =====================================================================
    # 敏感性实验候选值，只需要在这里调整
    # =====================================================================
    STAGE_GRIDS = {
        1: {"k_min": [5, 10, 15, 20], "k_max": [30, 50, 70, 90]},
        2: {"q": [0.5, 0.6, 0.75, 0.85, 0.9]},
        3: {"num_samples": [1500, 2000, 2500, 3000]},
        4: {"restart_prob": [0.05, 0.1, 0.15, 0.2, 0.3], "max_rwr_iter": [50, 100, 150, 200]},
        5: {"num_layers": [2, 3, 4, 5]},
        8: {"confidence_tau": [0.6, 0.7, 0.8, 0.9]},
        11: {
            "learning_rate": [0.0001, 0.0003, 0.001, 0.003],
            "hidden_dim": [32, 64, 128, 256],
            # "hidden_dim": [128, 256],
        },
        12: {"label_method": ["query_label", "ratio_based", "fisher_gmm"]},
    }

    SUPPORTED_STAGES = {1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16}

    if DATASET_NAME not in DATASET_CONFIGS:
        raise ValueError(f"未知数据集: {DATASET_NAME}")

    if EXPERIMENT_STAGE not in SUPPORTED_STAGES:
        raise ValueError("仅支持 Stage 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16。")

    # =====================================================================
    # 当前数据集配置
    # =====================================================================
    cfg = DATASET_CONFIGS[DATASET_NAME]
    args.dataset = DATASET_NAME
    file_path = os.path.join(args.filePath, f"{DATASET_NAME}.npz")

    BEST_NUM_SAMPLES = cfg["BEST_NUM_SAMPLES"]
    BEST_K_MIN = cfg["BEST_K_MIN"]
    BEST_K_MAX = cfg["BEST_K_MAX"]
    BEST_Q = cfg["BEST_Q"]
    BEST_LR = cfg["BEST_LR"]
    BEST_HIDDEN = cfg["BEST_HIDDEN"]
    BEST_RESTART_PROB = cfg["BEST_RESTART_PROB"]
    BEST_MAX_RWR_ITER = cfg["BEST_MAX_RWR_ITER"]
    BEST_TAU = cfg["BEST_TAU"]

    # =====================================================================
    # experiment 结果目录
    # =====================================================================
    EXPERIMENT_DIR = "experiment"
    EXPERIMENT_PT_DIR = "experiment_pt"
    os.makedirs(EXPERIMENT_DIR, exist_ok=True)
    os.makedirs(EXPERIMENT_PT_DIR, exist_ok=True)

    print("\n" + "=" * 120)
    print(f"🚀 自动化实验流水线 | Dataset={DATASET_NAME} | Stage={EXPERIMENT_STAGE} | Device={device}")
    print(f"📂 Data: {file_path}")
    print(
        f"📌 Best Params | samples={BEST_NUM_SAMPLES} | k_min={BEST_K_MIN} | k_max={BEST_K_MAX} | "
        f"q={BEST_Q} | restart={BEST_RESTART_PROB} | iter={BEST_MAX_RWR_ITER} | "
        f"lr={BEST_LR} | hidden={BEST_HIDDEN} | tau={BEST_TAU}"
    )
    print("=" * 120 + "\n")

    # Stage 16 会在内部逐个加载六个数据集；其余 Stage 仍使用顶部 DATASET_NAME。
    if EXPERIMENT_STAGE == 16:
        edges_df, features_df = None, None
    else:
        edges_df, features_df = Data_Query(file_path)

    results = []
    community_cache = {}

    # =====================================================================
    # 社区构造
    # =====================================================================
    def get_communities(**overrides):
        build_params = {
            "num_samples": BEST_NUM_SAMPLES,
            "min_community_size": BEST_K_MIN,
            "max_community_size": BEST_K_MAX,
            "q_quantile": BEST_Q,
            "restart_prob": BEST_RESTART_PROB,
            "max_rwr_iter": BEST_MAX_RWR_ITER,
            "rwr_tol": 1e-6,
            "confidence_tau": BEST_TAU,
            "method": "hacs",
            "label_method": "fisher_gmm",
        }

        build_params.update(overrides)
        cache_key = tuple(sorted(build_params.items()))

        if cache_key not in community_cache:
            print(f"\n🔨 构建社区 | {build_params}")
            community_cache[cache_key] = dap_arwr_first_stage_gpu(edges_df, features_df, **build_params)
        else:
            print("\n♻️ 使用已缓存的社区数据")

        return community_cache[cache_key]

    # =====================================================================
    # 公共训练入口
    # =====================================================================
    def run_training_case(
            case_name,
            build_overrides=None,
            train_overrides=None,
            use_gcn=False,
            save_data=False,
            extra_result=None,
    ):
        build_overrides = build_overrides or {}
        train_overrides = train_overrides or {}
        extra_result = extra_result or {}

        print("\n" + "★" * 100)
        print(f"★ Case: {case_name}")
        print("★" * 100)

        data_list = get_communities(**build_overrides)

        if not data_list:
            print("⚠️ data_list 为空，跳过当前实验。")
            return

        avg_nodes = float(np.mean([d.x.shape[0] for d in data_list]))
        avg_quality = float(np.mean([d.community_quality.item() for d in data_list]))

        normal_count = sum(1 for d in data_list if int(d.y.item()) == 0)
        anomalous_count = sum(1 for d in data_list if int(d.y.item()) == 1)
        uncertain_count = sum(1 for d in data_list if int(d.y.item()) == -1)

        total_count = len(data_list)
        supervised_count = normal_count + anomalous_count

        normal_ratio = normal_count / total_count if total_count > 0 else 0.0
        anomalous_ratio = anomalous_count / total_count if total_count > 0 else 0.0
        uncertain_ratio = uncertain_count / total_count if total_count > 0 else 0.0

        print(
            f"📊 Communities={total_count} | Supervised={supervised_count} | "
            f"N={normal_count} | A={anomalous_count} | U={uncertain_count}"
        )
        print(
            f"📊 Ratio | N={normal_ratio:.4f} | A={anomalous_ratio:.4f} | U={uncertain_ratio:.4f} | "
            f"|C|_avg={avg_nodes:.2f} | S_avg={avg_quality:.4f}"
        )

        # Stage 10 保存 data_list
        if save_data:
            os.makedirs(args.bestModelPath, exist_ok=True)
            data_save_path = os.path.join(args.bestModelPath, f"{DATASET_NAME}_data_list_{now}.pt")
            torch.save(data_list, data_save_path)
            print(f"💾 data_list 已保存: {data_save_path}")

        params = {"learning_rate": BEST_LR, "hidden_dim": BEST_HIDDEN}
        params.update(train_overrides)

        os.makedirs(args.bestModelPath, exist_ok=True)
        model_path = os.path.join(args.bestModelPath, f"{case_name}_{DATASET_NAME}_{now}.pth")

        if use_gcn:
            _, avg_test_auc, std_test_auc = cross_val_train_and_test_gcn(
                data_list,
                params,
                device,
                batch_size=args.batch_size,
                save_best_model_path=model_path,
            )
            model_name = "GCN"
        else:
            _, avg_test_auc, std_test_auc = cross_val_train_and_test(
                data_list,
                params,
                device,
                batch_size=args.batch_size,
                save_best_model_path=model_path,
            )
            model_name = "PathAwareGNN"

        result_row = {
            "Case": case_name,
            "Model": model_name,
            "Community_Count": total_count,
            "Supervised_Count": supervised_count,
            "Normal_Count": normal_count,
            "Anomalous_Count": anomalous_count,
            "Uncertain_Count": uncertain_count,
            "Normal_Ratio": normal_ratio,
            "Anomalous_Ratio": anomalous_ratio,
            "Uncertain_Ratio": uncertain_ratio,
            "|C|_avg": avg_nodes,
            "S_avg": avg_quality,
            "AUC": avg_test_auc * 100,
            "STD": std_test_auc * 100,
            "learning_rate": params["learning_rate"],
            "hidden_dim": params["hidden_dim"],
        }

        if "num_layers" in params:
            result_row["num_layers"] = params["num_layers"]

        result_row.update(build_overrides)
        result_row.update(extra_result)
        results.append(result_row)

    # =====================================================================
    # Stage 1：k_min × k_max
    # =====================================================================
    if EXPERIMENT_STAGE == 1:
        print("\n🔥 Stage 1：k_min × k_max")

        for k_min, k_max in product(STAGE_GRIDS[1]["k_min"], STAGE_GRIDS[1]["k_max"]):
            if k_min >= k_max:
                continue

            run_training_case(
                f"stage1_kmin{k_min}_kmax{k_max}",
                build_overrides={"min_community_size": k_min, "max_community_size": k_max},
                extra_result={"k_min": k_min, "k_max": k_max},
            )

    # =====================================================================
    # Stage 2：q_quantile
    # =====================================================================
    elif EXPERIMENT_STAGE == 2:
        print("\n🔥 Stage 2：q_quantile")

        for q in STAGE_GRIDS[2]["q"]:
            run_training_case(
                f"stage2_q{q}",
                build_overrides={"q_quantile": q},
                extra_result={"q": q},
            )

    # =====================================================================
    # Stage 3：num_samples
    # =====================================================================
    elif EXPERIMENT_STAGE == 3:
        print("\n🔥 Stage 3：num_samples")

        for num_samples in STAGE_GRIDS[3]["num_samples"]:
            run_training_case(
                f"stage3_n{num_samples}",
                build_overrides={"num_samples": num_samples},
                extra_result={"num_samples": num_samples},
            )

    # =====================================================================
    # Stage 4：restart_prob × max_rwr_iter
    # =====================================================================
    elif EXPERIMENT_STAGE == 4:
        print("\n🔥 Stage 4：restart_prob × max_rwr_iter")

        for rp, mi in product(STAGE_GRIDS[4]["restart_prob"], STAGE_GRIDS[4]["max_rwr_iter"]):
            run_training_case(
                f"stage4_rp{rp}_mi{mi}",
                build_overrides={"restart_prob": rp, "max_rwr_iter": mi},
                extra_result={"restart_prob": rp, "max_rwr_iter": mi},
            )

    # =====================================================================
    # Stage 5：num_layers
    # =====================================================================
    elif EXPERIMENT_STAGE == 5:
        print("\n🔥 Stage 5：num_layers")

        for num_layers in STAGE_GRIDS[5]["num_layers"]:
            run_training_case(
                f"stage5_layers{num_layers}",
                train_overrides={"num_layers": num_layers},
                extra_result={"num_layers": num_layers},
            )

    # =====================================================================
    # Stage 8：confidence_tau
    # =====================================================================
    elif EXPERIMENT_STAGE == 8:
        print("\n🔥 Stage 8：confidence_tau")

        for tau in STAGE_GRIDS[8]["confidence_tau"]:
            run_training_case(
                f"stage8_tau{tau}",
                build_overrides={"confidence_tau": tau, "label_method": "fisher_gmm"},
                extra_result={"confidence_tau": tau},
            )

    # =====================================================================
    # Stage 9：GCN baseline
    # =====================================================================
    elif EXPERIMENT_STAGE == 9:
        print("\n🔥 Stage 9：GCN baseline")

        run_training_case(
            "stage9_gcn",
            use_gcn=True,
            extra_result={"Baseline": "GCN"},
        )

    # =====================================================================
    # Stage 10：PathAwareGNN 最优参数验证 + data_list 导出
    # =====================================================================
    elif EXPERIMENT_STAGE == 10:
        print("\n🔥 Stage 10：PathAwareGNN 最优参数验证 + data_list 导出")

        run_training_case(
            "stage10_best",
            save_data=True,
            extra_result={"Setting": "Best"},
        )

    # =====================================================================
    # Stage 11：learning_rate × hidden_dim
    # =====================================================================
    elif EXPERIMENT_STAGE == 11:
        print("\n🔥 Stage 11：learning_rate × hidden_dim")

        for lr, hidden_dim in product(STAGE_GRIDS[11]["learning_rate"], STAGE_GRIDS[11]["hidden_dim"]):
            run_training_case(
                f"stage11_lr{lr}_h{hidden_dim}",
                train_overrides={"learning_rate": lr, "hidden_dim": hidden_dim},
                extra_result={"learning_rate": lr, "hidden_dim": hidden_dim},
            )

    # =====================================================================
    # Stage 12：三种社区伪标签方法
    # =====================================================================
    elif EXPERIMENT_STAGE == 12:
        print("\n🔥 Stage 12：Query-label / Ratio-based / Fisher-GMM")

        for label_method in STAGE_GRIDS[12]["label_method"]:
            # 保证三种方法从相同随机种子开始
            random.seed(42)
            np.random.seed(42)
            torch.manual_seed(42)

            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(42)

            run_training_case(
                f"stage12_{label_method}",
                build_overrides={"label_method": label_method},
                extra_result={"label_method": label_method},
            )

    # =====================================================================
    # Stage 13：RWR-QDCC + Fisher-GMM，最优参数导出 .pt
    # =====================================================================
    elif EXPERIMENT_STAGE == 13:
        print("\n🔥 Stage 13：RWR-QDCC + Fisher-GMM 最优参数社区导出")
        reset_all_seeds(42)

        # 记录与 RWR_QDCC 内部相同的 query 集合，便于追溯；随后重新置种子保证内部选择一致
        anomaly_nodes, normal_nodes = select_nodes_via_label(features_df, BEST_NUM_SAMPLES)
        export_queries = [int(v) for v in (anomaly_nodes + normal_nodes)]
        reset_all_seeds(42)

        data_list = get_communities(
            method="hacs",
            label_method="fisher_gmm",
            confidence_tau=BEST_TAU,
        )

        dataset_dir = os.path.join(EXPERIMENT_PT_DIR, DATASET_NAME)
        save_path = os.path.join(dataset_dir, f"{DATASET_NAME}_RWRQDCC_FisherGMM.pt")
        metadata = {
            "dataset": DATASET_NAME,
            "stage": 13,
            "construction_method": "RWR-QDCC",
            "label_method": "Fisher-GMM",
            "query_nodes": export_queries,
            "num_samples": BEST_NUM_SAMPLES,
            "k_min": BEST_K_MIN,
            "k_max": BEST_K_MAX,
            "q_quantile": BEST_Q,
            "restart_prob": BEST_RESTART_PROB,
            "max_rwr_iter": BEST_MAX_RWR_ITER,
            "rwr_tol": 1e-6,
            "confidence_tau": BEST_TAU,
            "generated_at": now,
        }
        save_export_pt(data_list, save_path, metadata)
        print("\n✅ Stage 13 完成：未训练模型，只导出 RWR-QDCC + Fisher-GMM 社区数据。")
        return

    # =====================================================================
    # Stage 14：BFS/PPR/RWR × Query-label/Ratio-based，逐 query 等规模导出 .pt
    # =====================================================================
    elif EXPERIMENT_STAGE == 14:
        print("\n🔥 Stage 14：BFS/PPR/RWR × Query-label/Ratio-based 社区导出")
        print("📐 公平约束：对每个相同 query，BFS/PPR/RWR 的社区大小严格等于 RWR-QDCC 的 S_q。")

        query_nodes, baseline_data = build_stage14_baseline_communities(
            edges_df=edges_df,
            features_df=features_df,
            num_samples=BEST_NUM_SAMPLES,
            min_community_size=BEST_K_MIN,
            max_community_size=BEST_K_MAX,
            q_quantile=BEST_Q,
            restart_prob=BEST_RESTART_PROB,
            max_rwr_iter=BEST_MAX_RWR_ITER,
            rwr_tol=1e-6,
            seed=42,
            mc_walks=4000,
            mc_length=20,
        )

        dataset_dir = os.path.join(EXPERIMENT_PT_DIR, DATASET_NAME)
        label_file_names = {"query_label": "QueryLabel", "ratio_based": "RatioBased"}
        method_file_names = {"bfs": "BFS", "ppr": "PPR", "rwr": "RWR"}

        for method_name in ("bfs", "ppr", "rwr"):
            base_list = baseline_data[method_name]
            for label_method in ("query_label", "ratio_based"):
                labeled_list = [d.clone() for d in base_list]
                apply_export_labels(labeled_list, features_df, label_method)

                save_name = f"{DATASET_NAME}_{method_file_names[method_name]}_{label_file_names[label_method]}.pt"
                save_path = os.path.join(dataset_dir, save_name)
                metadata = {
                    "dataset": DATASET_NAME,
                    "stage": 14,
                    "construction_method": method_file_names[method_name],
                    "label_method": label_file_names[label_method],
                    "size_reference": "RWR-QDCC per-query S_q",
                    "query_nodes": [int(v) for v in query_nodes],
                    "num_samples": BEST_NUM_SAMPLES,
                    "k_min": BEST_K_MIN,
                    "k_max": BEST_K_MAX,
                    "q_quantile_for_size_reference": BEST_Q,
                    "restart_prob": BEST_RESTART_PROB,
                    "max_rwr_iter": BEST_MAX_RWR_ITER,
                    "rwr_tol": 1e-6,
                    "rwr_mc_walks": 4000 if method_name == "rwr" else None,
                    "rwr_mc_length": 20 if method_name == "rwr" else None,
                    "ratio_tau": "global K*/N*" if label_method == "ratio_based" else None,
                    "generated_at": now,
                }
                save_export_pt(labeled_list, save_path, metadata)

        print("\n✅ Stage 14 完成：共导出 6 份 .pt，未训练任何模型。")
        print(f"📂 输出目录: {os.path.abspath(dataset_dir)}")
        return

    # =====================================================================
    # Stage 15：跨数据集 × 跨社区构造方法，GCN 全量评测（一个 CSV）
    # =====================================================================
    elif EXPERIMENT_STAGE == 15:
        print("\n🔥 Stage 15：跨数据集 × 跨社区构造方法 GCN 全量评测")
        print("📐 每个数据集 7 个变体：RWRQDCC_FisherGMM / RWR_RatioBased / RWR_QueryLabel / "
              "PPR_RatioBased / PPR_QueryLabel / BFS_RatioBased / BFS_QueryLabel")
        print("📂 从 experiment_pt/{Dataset}/ 下加载 .pt，最后汇总为单一 CSV。")
        print("📌 每个数据集使用其在 DATASET_CONFIGS 中自己的 BEST_LR / BEST_HIDDEN。")

        ALL_DATASETS_STAGE15 = ["Actor", "Arxiv", "Elliptic", "Roman", "DGraph", "Minesweeper"]
        PT_VARIANTS_STAGE15 = [
            "RWRQDCC_FisherGMM",
            "RWR_RatioBased",
            "RWR_QueryLabel",
            "PPR_RatioBased",
            "PPR_QueryLabel",
            "BFS_RatioBased",
            "BFS_QueryLabel",
        ]

        stage15_results = []
        stage15_csv = os.path.join(EXPERIMENT_DIR, f"Stage15_GCN_AllDatasets_{now}.csv")

        total_jobs = len(ALL_DATASETS_STAGE15) * len(PT_VARIANTS_STAGE15)
        job_idx = 0

        for ds_s15 in ALL_DATASETS_STAGE15:
            # ★ 关键修复：按当前数据集重新读取它自己的最优超参，而不是沿用 main() 顶部的 Elliptic 配置
            if ds_s15 not in DATASET_CONFIGS:
                print(f"⚠️ DATASET_CONFIGS 中没有 {ds_s15} 的配置，跳过整个数据集。")
                continue
            ds_cfg_s15 = DATASET_CONFIGS[ds_s15]
            params = {
                "learning_rate": ds_cfg_s15["BEST_LR"],
                "hidden_dim": ds_cfg_s15["BEST_HIDDEN"],
            }
            print("\n" + "★" * 100)
            print(f"★ [Stage15] Dataset={ds_s15} | 使用该数据集自己的最优参数 | "
                  f"lr={params['learning_rate']} | hidden={params['hidden_dim']}")
            print("★" * 100)

            for variant in PT_VARIANTS_STAGE15:
                job_idx += 1
                pt_name = f"{ds_s15}_{variant}.pt"
                pt_path = os.path.join(EXPERIMENT_PT_DIR, ds_s15, pt_name)

                print("\n" + "=" * 120)
                print(f"🔥 [Stage15 | {job_idx}/{total_jobs}] Dataset={ds_s15} | Variant={variant}")
                print(f"📂 PT 路径: {pt_path}")
                print(f"📌 参数: lr={params['learning_rate']} | hidden={params['hidden_dim']}")
                print("=" * 120)

                if not os.path.exists(pt_path):
                    print(f"⚠️ 文件不存在，跳过: {pt_path}")
                    continue

                reset_all_seeds(42)

                try:
                    data_list_s15 = torch.load(pt_path, weights_only=False)
                except Exception as e:
                    print(f"❌ 加载失败: {e}")
                    continue

                if not data_list_s15:
                    print(f"⚠️ 数据为空，跳过: {pt_path}")
                    continue

                total_count = len(data_list_s15)
                normal_count = sum(1 for d in data_list_s15 if int(d.y.item()) == 0)
                anomalous_count = sum(1 for d in data_list_s15 if int(d.y.item()) == 1)
                uncertain_count = sum(1 for d in data_list_s15 if int(d.y.item()) == -1)
                supervised_count = normal_count + anomalous_count

                avg_nodes = float(np.mean([d.x.shape[0] for d in data_list_s15]))
                try:
                    avg_quality = float(np.mean([d.community_quality.item() for d in data_list_s15]))
                except Exception:
                    avg_quality = float("nan")

                normal_ratio = normal_count / total_count if total_count > 0 else 0.0
                anomalous_ratio = anomalous_count / total_count if total_count > 0 else 0.0
                uncertain_ratio = uncertain_count / total_count if total_count > 0 else 0.0

                print(
                    f"📊 Communities={total_count} | Supervised={supervised_count} | "
                    f"N={normal_count} | A={anomalous_count} | U={uncertain_count} | "
                    f"|C|_avg={avg_nodes:.2f} | S_avg={avg_quality:.4f}"
                )

                # 类别不全会导致 StratifiedKFold 崩，直接跳过
                if normal_count == 0 or anomalous_count == 0:
                    print("⚠️ 正/负类缺一，无法训练，跳过。")
                    continue

                os.makedirs(args.bestModelPath, exist_ok=True)
                model_path_s15 = os.path.join(
                    args.bestModelPath,
                    f"stage15_{ds_s15}_{variant}_{now}.pth"
                )

                try:
                    _, avg_test_auc, std_test_auc = cross_val_train_and_test_gcn(
                        data_list_s15,
                        params,
                        device,
                        batch_size=args.batch_size,
                        save_best_model_path=model_path_s15,
                    )
                except Exception as e:
                    print(f"❌ 训练失败: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

                row_s15 = {
                    "Dataset": ds_s15,
                    "Variant": variant,
                    "Model": "GCN",
                    "Community_Count": total_count,
                    "Supervised_Count": supervised_count,
                    "Normal_Count": normal_count,
                    "Anomalous_Count": anomalous_count,
                    "Uncertain_Count": uncertain_count,
                    "Normal_Ratio": normal_ratio,
                    "Anomalous_Ratio": anomalous_ratio,
                    "Uncertain_Ratio": uncertain_ratio,
                    "|C|_avg": avg_nodes,
                    "S_avg": avg_quality,
                    "AUC": avg_test_auc * 100,
                    "STD": std_test_auc * 100,
                    "learning_rate": params["learning_rate"],
                    "hidden_dim": params["hidden_dim"],
                }
                stage15_results.append(row_s15)

                # 每跑完一个变体立刻落盘，防止中途被打断丢结果
                pd.DataFrame(stage15_results).to_csv(
                    stage15_csv, index=False, encoding="utf-8-sig"
                )
                print(f"💾 已更新: {stage15_csv} (已完成 {len(stage15_results)} 条)")

        print("\n" + "🏆" * 50)
        print(f"🏆 Stage 15 全部完成 | 共生成 {len(stage15_results)} 条结果")
        print("🏆" * 50)

        if stage15_results:
            stage15_df = pd.DataFrame(stage15_results)
            stage15_df = stage15_df.sort_values(["Dataset", "AUC"], ascending=[True, False])
            stage15_df.to_csv(stage15_csv, index=False, encoding="utf-8-sig")
            print(stage15_df.to_string(index=False))
            print(f"\n💾 最终 CSV: {stage15_csv}")
        else:
            print("⚠️ Stage 15 没有生成任何结果。")

        return

    # =====================================================================
    # Stage 16：六数据集三组件单独消融（一个 CSV）
    #   A. w/o RWR-QDCC : RWR + Fisher-GMM + HD-GNN
    #   B. w/o Fisher-GMM: RWR-QDCC + Ratio-based + HD-GNN
    #   C. w/o HD-GNN    : RWR-QDCC + Fisher-GMM + GCN
    # =====================================================================
    elif EXPERIMENT_STAGE == 16:
        print("\n🔥 Stage 16：六数据集三组件单独消融")
        print("   A) w/o RWR-QDCC  -> connected Monte-Carlo RWR + Fisher-GMM + HD-GNN")
        print("   B) w/o Fisher-GMM -> RWR-QDCC + Ratio-based + HD-GNN")
        print("   C) w/o HD-GNN     -> RWR-QDCC + Fisher-GMM + GCN")
        print("📌 每个数据集均使用 DATASET_CONFIGS 中自己的最优超参数。")
        print("📌 三种消融均沿用同一 5-fold 训练/验证/测试协议。")
        print("📌 每完成 1 个 case 立即写入 CSV，防止中途终止丢失结果。")

        ALL_DATASETS_STAGE16 = [
            "Elliptic", "DGraph", "Actor", "Arxiv", "Roman", "Minesweeper"
        ]

        stage16_results = []
        stage16_failures = []
        stage16_csv = os.path.join(
            EXPERIMENT_DIR, f"Stage16_Ablation_AllDatasets_{now}.csv"
        )
        stage16_failure_csv = os.path.join(
            EXPERIMENT_DIR, f"Stage16_Ablation_Failures_{now}.csv"
        )

        def summarize_stage16_data(data_list):
            total_count = len(data_list)
            normal_count = sum(1 for d in data_list if int(d.y.item()) == 0)
            anomalous_count = sum(1 for d in data_list if int(d.y.item()) == 1)
            uncertain_count = sum(1 for d in data_list if int(d.y.item()) == -1)
            supervised_count = normal_count + anomalous_count

            avg_nodes = float(np.mean([d.x.shape[0] for d in data_list])) if data_list else 0.0
            try:
                avg_quality = float(np.mean([d.community_quality.item() for d in data_list])) if data_list else float("nan")
            except Exception:
                avg_quality = float("nan")

            return {
                "Community_Count": total_count,
                "Supervised_Count": supervised_count,
                "Normal_Count": normal_count,
                "Anomalous_Count": anomalous_count,
                "Uncertain_Count": uncertain_count,
                "Normal_Ratio": normal_count / total_count if total_count else 0.0,
                "Anomalous_Ratio": anomalous_count / total_count if total_count else 0.0,
                "Uncertain_Ratio": uncertain_count / total_count if total_count else 0.0,
                "|C|_avg": avg_nodes,
                "S_avg": avg_quality,
            }

        def save_stage16_progress():
            if stage16_results:
                pd.DataFrame(stage16_results).to_csv(
                    stage16_csv, index=False, encoding="utf-8-sig"
                )
            if stage16_failures:
                pd.DataFrame(stage16_failures).to_csv(
                    stage16_failure_csv, index=False, encoding="utf-8-sig"
                )

        def train_stage16_case(
                ds_name,
                case_name,
                ablated_component,
                replacement,
                construction_method,
                label_method_name,
                detector_name,
                data_list_case,
                params_case,
                ds_cfg_case,
                use_gcn=False,
        ):
            print("\n" + "★" * 110)
            print(
                f"★ [Stage16] Dataset={ds_name} | Case={case_name} | "
                f"Ablated={ablated_component} -> {replacement}"
            )
            print(
                f"★ Pipeline: {construction_method} + {label_method_name} + {detector_name}"
            )
            print(
                f"★ Params: lr={params_case['learning_rate']} | "
                f"hidden={params_case['hidden_dim']}"
            )
            print("★" * 110)

            if not data_list_case:
                raise RuntimeError(f"{ds_name}/{case_name}: data_list 为空")

            stats = summarize_stage16_data(data_list_case)
            print(
                f"📊 Communities={stats['Community_Count']} | "
                f"Supervised={stats['Supervised_Count']} | "
                f"N={stats['Normal_Count']} | A={stats['Anomalous_Count']} | "
                f"U={stats['Uncertain_Count']} | "
                f"|C|_avg={stats['|C|_avg']:.2f} | S_avg={stats['S_avg']:.4f}"
            )

            if stats["Normal_Count"] == 0 or stats["Anomalous_Count"] == 0:
                raise RuntimeError(
                    f"{ds_name}/{case_name}: 正/负伪标签类别不完整，无法进行 AUC 训练。"
                )

            reset_all_seeds(42)
            os.makedirs(args.bestModelPath, exist_ok=True)
            model_path = os.path.join(
                args.bestModelPath,
                f"stage16_{ds_name}_{case_name}_{now}.pth"
            )

            if use_gcn:
                _, avg_test_auc, std_test_auc = cross_val_train_and_test_gcn(
                    data_list_case,
                    params_case,
                    device,
                    batch_size=args.batch_size,
                    save_best_model_path=model_path,
                )
            else:
                _, avg_test_auc, std_test_auc = cross_val_train_and_test(
                    data_list_case,
                    params_case,
                    device,
                    batch_size=args.batch_size,
                    save_best_model_path=model_path,
                )

            row = {
                "Dataset": ds_name,
                "Stage": 16,
                "Case": case_name,
                "Ablated_Component": ablated_component,
                "Replacement": replacement,
                "Construction": construction_method,
                "Pseudo_Labeling": label_method_name,
                "Detector": detector_name,
                **stats,
                "AUC": avg_test_auc * 100,
                "STD": std_test_auc * 100,
                "learning_rate": params_case["learning_rate"],
                "hidden_dim": params_case["hidden_dim"],
                "Best_Num_Samples": ds_cfg_case["BEST_NUM_SAMPLES"],
                "Best_K_Min": ds_cfg_case["BEST_K_MIN"],
                "Best_K_Max": ds_cfg_case["BEST_K_MAX"],
                "Best_Q": ds_cfg_case["BEST_Q"],
                "Best_Restart_Prob": ds_cfg_case["BEST_RESTART_PROB"],
                "Best_Max_RWR_Iter": ds_cfg_case["BEST_MAX_RWR_ITER"],
                "Best_Tau": ds_cfg_case["BEST_TAU"],
            }
            stage16_results.append(row)
            save_stage16_progress()

            print(
                f"✅ [Stage16] {ds_name} | {case_name} | "
                f"AUC={row['AUC']:.4f}% ± {row['STD']:.4f}%"
            )
            print(f"💾 已更新 CSV: {stage16_csv}")

        total_jobs_s16 = len(ALL_DATASETS_STAGE16) * 3
        completed_jobs_s16 = 0

        for ds_s16 in ALL_DATASETS_STAGE16:
            print("\n" + "#" * 120)
            print(f"# [Stage16] 开始数据集: {ds_s16}")
            print("#" * 120)

            if ds_s16 not in DATASET_CONFIGS:
                stage16_failures.append({
                    "Dataset": ds_s16,
                    "Case": "ALL",
                    "Error": "DATASET_CONFIGS 中缺少该数据集配置",
                })
                save_stage16_progress()
                continue

            ds_cfg = DATASET_CONFIGS[ds_s16]
            params_s16 = {
                "learning_rate": ds_cfg["BEST_LR"],
                "hidden_dim": ds_cfg["BEST_HIDDEN"],
            }
            file_path_s16 = os.path.join(args.filePath, f"{ds_s16}.npz")
            args.dataset = ds_s16

            try:
                edges_s16, features_s16 = Data_Query(file_path_s16)
            except Exception as e:
                stage16_failures.append({
                    "Dataset": ds_s16,
                    "Case": "ALL",
                    "Error": f"Data_Query failed: {repr(e)}",
                })
                save_stage16_progress()
                print(f"❌ {ds_s16} 数据加载失败: {e}")
                continue

            common_build = {
                "num_samples": ds_cfg["BEST_NUM_SAMPLES"],
                "min_community_size": ds_cfg["BEST_K_MIN"],
                "max_community_size": ds_cfg["BEST_K_MAX"],
                "q_quantile": ds_cfg["BEST_Q"],
                "restart_prob": ds_cfg["BEST_RESTART_PROB"],
                "max_rwr_iter": ds_cfg["BEST_MAX_RWR_ITER"],
                "rwr_tol": 1e-6,
                "confidence_tau": ds_cfg["BEST_TAU"],
            }

            # -------------------------------------------------------------
            # 先构建完整 RWR-QDCC 社区。随后 Stage16 统一重做 Fisher-GMM 标签，
            # 保证 RWR 与 RWR-QDCC 两种构造使用完全相同的 post-gate Fisher-GMM。
            # -------------------------------------------------------------
            full_fisher_data = None
            ratio_data = None
            try:
                reset_all_seeds(42)
                full_raw_data = dap_arwr_first_stage_gpu(
                    edges_s16,
                    features_s16,
                    **common_build,
                    method="hacs",
                    label_method="fisher_gmm",
                )
                full_fisher_data = [d.clone() for d in full_raw_data]
                apply_fisher_gmm_labels_stage16(
                    full_fisher_data,
                    features_s16,
                    confidence_tau=ds_cfg["BEST_TAU"],
                    seed=42,
                )

                # 只消融 Fisher-GMM：社区结构完全不变，仅改为 Ratio-based。
                ratio_data = [d.clone() for d in full_raw_data]
                apply_export_labels(
                    ratio_data,
                    features_s16,
                    label_method="ratio_based",
                )
                del full_raw_data
            except Exception as e:
                import traceback
                traceback.print_exc()
                stage16_failures.append({
                    "Dataset": ds_s16,
                    "Case": "Build_RWRQDCC_FisherGMM_or_RatioBased",
                    "Error": repr(e),
                })
                save_stage16_progress()

            # -------------------------------------------------------------
            # Ablation A: only remove RWR-QDCC.
            # 直接复用 Stage14/RQ2 的 connected Monte-Carlo RWR，并逐 query
            # 严格匹配 RWR-QDCC 的 S_q；随后施加与完整社区相同的 Fisher-GMM。
            # -------------------------------------------------------------
            rwr_fisher_data = None
            try:
                reset_all_seeds(42)
                _, rwr_baseline_data = build_stage14_baseline_communities(
                    edges_df=edges_s16,
                    features_df=features_s16,
                    num_samples=ds_cfg["BEST_NUM_SAMPLES"],
                    min_community_size=ds_cfg["BEST_K_MIN"],
                    max_community_size=ds_cfg["BEST_K_MAX"],
                    q_quantile=ds_cfg["BEST_Q"],
                    restart_prob=ds_cfg["BEST_RESTART_PROB"],
                    max_rwr_iter=ds_cfg["BEST_MAX_RWR_ITER"],
                    rwr_tol=1e-6,
                    seed=42,
                    mc_walks=4000,
                    mc_length=20,
                    requested_methods=("rwr",),
                )
                rwr_fisher_data = rwr_baseline_data["rwr"]
                apply_fisher_gmm_labels_stage16(
                    rwr_fisher_data,
                    features_s16,
                    confidence_tau=ds_cfg["BEST_TAU"],
                    seed=42,
                )
            except Exception as e:
                import traceback
                traceback.print_exc()
                stage16_failures.append({
                    "Dataset": ds_s16,
                    "Case": "Ablate_RWR_QDCC",
                    "Error": f"Connected RWR construction/Fisher-GMM failed: {repr(e)}",
                })
                save_stage16_progress()

            cases_s16 = [
                {
                    "case_name": "Ablate_RWR_QDCC",
                    "ablated_component": "RWR-QDCC",
                    "replacement": "RWR",
                    "construction_method": "RWR",
                    "label_method_name": "Fisher-GMM",
                    "detector_name": "HD-GNN",
                    "data_list": rwr_fisher_data,
                    "use_gcn": False,
                },
                {
                    "case_name": "Ablate_Fisher_GMM",
                    "ablated_component": "Fisher-GMM",
                    "replacement": "Ratio-based",
                    "construction_method": "RWR-QDCC",
                    "label_method_name": "Ratio-based",
                    "detector_name": "HD-GNN",
                    "data_list": ratio_data,
                    "use_gcn": False,
                },
                {
                    "case_name": "Ablate_HD_GNN",
                    "ablated_component": "HD-GNN",
                    "replacement": "GCN",
                    "construction_method": "RWR-QDCC",
                    "label_method_name": "Fisher-GMM",
                    "detector_name": "GCN",
                    "data_list": full_fisher_data,
                    "use_gcn": True,
                },
            ]

            for case in cases_s16:
                completed_jobs_s16 += 1
                print(
                    f"\n🚀 Stage16 Job {completed_jobs_s16}/{total_jobs_s16} | "
                    f"Dataset={ds_s16} | {case['case_name']}"
                )

                if case["data_list"] is None:
                    stage16_failures.append({
                        "Dataset": ds_s16,
                        "Case": case["case_name"],
                        "Error": "对应 data_list 构建失败，未进入训练",
                    })
                    save_stage16_progress()
                    print("⚠️ data_list 不可用，跳过该 case。")
                    continue

                try:
                    train_stage16_case(
                        ds_name=ds_s16,
                        case_name=case["case_name"],
                        ablated_component=case["ablated_component"],
                        replacement=case["replacement"],
                        construction_method=case["construction_method"],
                        label_method_name=case["label_method_name"],
                        detector_name=case["detector_name"],
                        data_list_case=case["data_list"],
                        params_case=params_s16,
                        ds_cfg_case=ds_cfg,
                        use_gcn=case["use_gcn"],
                    )
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    stage16_failures.append({
                        "Dataset": ds_s16,
                        "Case": case["case_name"],
                        "Error": repr(e),
                    })
                    save_stage16_progress()
                    print(f"❌ 训练失败: {e}")

            # 释放当前数据集图与社区，防止六个数据集连续运行时显存/内存累积。
            del edges_s16, features_s16
            if full_fisher_data is not None:
                del full_fisher_data
            if ratio_data is not None:
                del ratio_data
            if rwr_fisher_data is not None:
                del rwr_fisher_data
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print("\n" + "🏆" * 55)
        print(
            f"🏆 Stage 16 全部结束 | 计划 {total_jobs_s16} 条 | "
            f"成功 {len(stage16_results)} 条 | 失败 {len(stage16_failures)} 条"
        )
        print("🏆" * 55)

        if stage16_results:
            stage16_df = pd.DataFrame(stage16_results)
            # 保留论文常用顺序：每个数据集依次 RWR-QDCC / Fisher-GMM / HD-GNN 三个单独消融。
            dataset_order = {name: i for i, name in enumerate(ALL_DATASETS_STAGE16)}
            case_order = {
                "Ablate_RWR_QDCC": 0,
                "Ablate_Fisher_GMM": 1,
                "Ablate_HD_GNN": 2,
            }
            stage16_df["__dataset_order"] = stage16_df["Dataset"].map(dataset_order)
            stage16_df["__case_order"] = stage16_df["Case"].map(case_order)
            stage16_df = stage16_df.sort_values(
                ["__dataset_order", "__case_order"]
            ).drop(columns=["__dataset_order", "__case_order"])
            stage16_df.to_csv(stage16_csv, index=False, encoding="utf-8-sig")

            print("\nFINAL STAGE 16 ABLATION RESULTS")
            print(
                stage16_df[
                    [
                        "Dataset", "Ablated_Component", "Replacement",
                        "Construction", "Pseudo_Labeling", "Detector",
                        "AUC", "STD", "Community_Count", "Supervised_Count",
                        "Normal_Count", "Anomalous_Count", "Uncertain_Count",
                    ]
                ].to_string(
                    index=False,
                    float_format=lambda x: f"{x:.4f}",
                )
            )
            print(f"\n💾 Stage16 最终 CSV: {os.path.abspath(stage16_csv)}")
        else:
            print("⚠️ Stage 16 没有成功生成任何训练结果。")

        if stage16_failures:
            pd.DataFrame(stage16_failures).to_csv(
                stage16_failure_csv, index=False, encoding="utf-8-sig"
            )
            print(f"⚠️ 失败记录: {os.path.abspath(stage16_failure_csv)}")

        return

    # =====================================================================
    # 统一结果汇总
    # =====================================================================
    print("\n" + "🏆" * 50)
    print(f"🏆 Dataset={DATASET_NAME} | Stage={EXPERIMENT_STAGE} 结果汇总")
    print("🏆" * 50)

    if not results:
        print("⚠️ 没有生成任何实验结果，不保存 CSV。")
        return

    results.sort(key=lambda x: x["AUC"], reverse=True)

    metric_keys = {
        "AUC", "STD", "|C|_avg", "S_avg",
        "Community_Count", "Supervised_Count",
        "Normal_Count", "Anomalous_Count", "Uncertain_Count",
        "Normal_Ratio", "Anomalous_Ratio", "Uncertain_Ratio",
    }

    for rank, res in enumerate(results, start=1):
        param_str = " | ".join(f"{k}: {v}" for k, v in res.items() if k not in metric_keys)

        print(
            f"Rank {rank:2d} | {param_str} | "
            f"Community={res['Community_Count']} | Supervised={res['Supervised_Count']} | "
            f"N={res['Normal_Count']} | A={res['Anomalous_Count']} | U={res['Uncertain_Count']} | "
            f"|C|_avg={res['|C|_avg']:.2f} | S_avg={res['S_avg']:.4f} | "
            f"AUC={res['AUC']:.4f}% ± {res['STD']:.4f}%"
        )

    print("🏆" * 50)

    # =====================================================================
    # 当前 Stage 实验结果保存 CSV
    # =====================================================================
    results_df = pd.DataFrame(results)

    # AUC 排序后增加 Rank
    results_df.insert(0, "Rank", range(1, len(results_df) + 1))
    results_df.insert(1, "Dataset", DATASET_NAME)
    results_df.insert(2, "Stage", EXPERIMENT_STAGE)

    # 同时记录本次实验使用的基础最优参数
    results_df["Best_Num_Samples"] = BEST_NUM_SAMPLES
    results_df["Best_K_Min"] = BEST_K_MIN
    results_df["Best_K_Max"] = BEST_K_MAX
    results_df["Best_Q"] = BEST_Q
    results_df["Best_Restart_Prob"] = BEST_RESTART_PROB
    results_df["Best_Max_RWR_Iter"] = BEST_MAX_RWR_ITER
    results_df["Best_LR"] = BEST_LR
    results_df["Best_Hidden"] = BEST_HIDDEN
    results_df["Best_Tau"] = BEST_TAU

    csv_path = os.path.join(
        EXPERIMENT_DIR,
        f"{DATASET_NAME}_stage{EXPERIMENT_STAGE}_{now}.csv"
    )

    results_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print(f"\n💾 CSV 已保存: {csv_path}")
    print(f"📊 共保存 {len(results_df)} 条实验结果")
    print(f"📂 保存目录: {os.path.abspath(EXPERIMENT_DIR)}\n")


if __name__ == "__main__":
    import time

    start_time = time.time()
    main()
    elapsed = time.time() - start_time
    print(f"\n⏱️  main() 总耗时: {elapsed:.2f} 秒 ({elapsed / 60:.2f} 分钟)")
