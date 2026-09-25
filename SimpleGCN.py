import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
import torch


class SimpleGCN(nn.Module):
    """
    简单的两层 GCN 基线模型
    接口与 PathAwareGNN 保持一致，可直接复用训练函数
    """
    def __init__(self, feat_dim, hidden_dim, num_layers=2, dropout_prob=0.5):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout_prob)

        # GCN 卷积层
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(feat_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim // 2, 1)
        )

        # 损失函数（仅图级别分类损失，保持接口兼容）
        self.graph_focal_loss = nn.BCEWithLogitsLoss()

    def forward(self, x, edge_index, q_indices_global=None, batch=None):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        if x.numel() == 0:
            return torch.zeros(0, 1, device=x.device), torch.zeros(0, device=x.device)

        # 逐层卷积
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = F.gelu(x)
            if i < self.num_layers - 1:
                x = self.dropout(x)

        # 全局平均池化
        graph_repr = global_mean_pool(x, batch)

        # 分类
        logits = self.classifier(graph_repr)

        # 第二个返回值占位（兼容 PathAwareGNN 的 node_rel_score 接口）
        node_rel_score = torch.zeros(x.size(0), device=x.device)

        return logits, node_rel_score

    def total_loss(self, data, augmentor=None):
        """计算图级别分类损失（兼容原训练函数接口）"""
        batch = data.batch if hasattr(data, 'batch') and data.batch is not None else torch.zeros(
            data.x.size(0), dtype=torch.long, device=data.x.device
        )

        logits, _ = self(data.x, data.edge_index, None, batch)
        labels = data.y.float()

        # 只计算有标签样本的损失
        labeled_mask = (labels != -1)
        if labeled_mask.any():
            loss = F.binary_cross_entropy_with_logits(
                logits.squeeze(-1)[labeled_mask], labels[labeled_mask]
            )
        else:
            loss = (logits.sum() * 0.0)

        return loss

    def predict(self, data):
        """预测接口，与 PathAwareGNN 保持一致"""
        with torch.no_grad():
            batch = data.batch if hasattr(data, 'batch') and data.batch is not None else torch.zeros(
                data.x.size(0), dtype=torch.long, device=data.x.device
            )

            logits, _ = self(data.x, data.edge_index, None, batch)

            if logits.numel() == 0:
                return torch.empty(0, dtype=torch.float, device=data.x.device)

            probs = torch.sigmoid(logits.squeeze(-1))
            return probs

    def extract_graph_repr(self, data):
        """提取图级别特征（用于可视化，兼容接口）"""
        self.eval()
        with torch.no_grad():
            batch = data.batch if hasattr(data, 'batch') and data.batch is not None else torch.zeros(
                data.x.size(0), dtype=torch.long, device=data.x.device
            )

            x = data.x
            edge_index = data.edge_index
            for i, conv in enumerate(self.convs):
                x = conv(x, edge_index)
                x = F.gelu(x)
                if i < self.num_layers - 1:
                    x = self.dropout(x)

            graph_repr = global_mean_pool(x, batch)
            return graph_repr


