"""Small dependency-light PyTorch GNNs for the NCG STRING graph."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def normalized_sparse_adjacency(edge_index, edge_weight, num_nodes):
    """Symmetrically normalize a weighted adjacency after adding self loops."""
    device = edge_index.device
    loops = torch.arange(num_nodes, device=device)
    idx = torch.cat([edge_index, torch.stack([loops, loops])], dim=1)
    weights = torch.cat([edge_weight, torch.ones(num_nodes, device=device)])
    adj = torch.sparse_coo_tensor(idx, weights, (num_nodes, num_nodes)).coalesce()
    idx, weights = adj.indices(), adj.values()
    deg = torch.zeros(num_nodes, device=device).index_add_(0, idx[0], weights)
    norm = weights * deg[idx[0]].clamp_min(1e-12).rsqrt() * deg[idx[1]].clamp_min(1e-12).rsqrt()
    return torch.sparse_coo_tensor(idx, norm, (num_nodes, num_nodes)).coalesce()


class GCN(nn.Module):
    def __init__(self, in_dim, hidden=32, dropout=0.3):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden)
        self.norm1 = nn.LayerNorm(hidden)
        self.lin2 = nn.Linear(hidden, 1)
        self.dropout = dropout

    def forward(self, x, adj, edge_index=None, edge_weight=None):
        h = torch.sparse.mm(adj, self.lin1(x))
        h = F.gelu(self.norm1(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return torch.sparse.mm(adj, self.lin2(h)).squeeze(-1)


class SAGEConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.root = nn.Linear(in_dim, out_dim, bias=True)
        self.neigh = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x, edge_index, edge_weight, num_nodes):
        src, dst = edge_index
        weighted = x[src] * edge_weight[:, None]
        agg = torch.zeros_like(x).index_add_(0, dst, weighted)
        denom = torch.zeros(num_nodes, device=x.device).index_add_(0, dst, edge_weight).clamp_min(1e-8)
        agg = agg / denom[:, None]
        return self.root(x) + self.neigh(agg)


class GraphSAGE(nn.Module):
    def __init__(self, in_dim, hidden=32, dropout=0.3):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden)
        self.norm1 = nn.LayerNorm(hidden)
        self.conv2 = SAGEConv(hidden, 1)
        self.dropout = dropout

    def forward(self, x, adj, edge_index, edge_weight):
        h = F.gelu(self.norm1(self.conv1(x, edge_index, edge_weight, x.shape[0])))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.conv2(h, edge_index, edge_weight, x.shape[0]).squeeze(-1)


class GATConv(nn.Module):
    def __init__(self, in_dim, out_dim, heads=4, concat=True, dropout=0.2, edge_bias=True):
        super().__init__()
        self.heads, self.out_dim, self.concat = heads, out_dim, concat
        self.proj = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.att_src = nn.Parameter(torch.empty(heads, out_dim))
        self.att_dst = nn.Parameter(torch.empty(heads, out_dim))
        self.edge_scale = nn.Parameter(torch.ones(heads)) if edge_bias else None
        self.bias = nn.Parameter(torch.zeros(heads * out_dim if concat else out_dim))
        self.dropout = dropout
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)

    def forward(self, x, edge_index, edge_weight):
        n = x.shape[0]
        h = self.proj(x).view(n, self.heads, self.out_dim)
        src, dst = edge_index
        logits = (h[src] * self.att_src).sum(-1) + (h[dst] * self.att_dst).sum(-1)
        logits = F.leaky_relu(logits, negative_slope=0.2)
        if self.edge_scale is not None:
            logits = logits + (edge_weight.clamp_min(1e-6).log()[:, None] * self.edge_scale[None, :])
        group = dst[:, None].expand(-1, self.heads)
        maxima = torch.full((n, self.heads), -torch.inf, device=x.device)
        maxima.scatter_reduce_(0, group, logits, reduce="amax", include_self=True)
        exp = torch.exp(logits - maxima[dst])
        denom = torch.zeros((n, self.heads), device=x.device).index_add_(0, dst, exp).clamp_min(1e-12)
        alpha = F.dropout(exp / denom[dst], p=self.dropout, training=self.training)
        messages = h[src] * alpha[..., None]
        out = torch.zeros((n, self.heads, self.out_dim), device=x.device).index_add_(0, dst, messages)
        if self.concat:
            return out.reshape(n, self.heads * self.out_dim) + self.bias
        return out.mean(dim=1) + self.bias


class GAT(nn.Module):
    def __init__(self, in_dim, hidden=8, heads=4, dropout=0.3):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden, heads=heads, concat=True, dropout=dropout)
        self.norm1 = nn.LayerNorm(hidden * heads)
        self.conv2 = GATConv(hidden * heads, 1, heads=heads, concat=False, dropout=dropout)
        self.dropout = dropout

    def forward(self, x, adj, edge_index, edge_weight):
        h = self.conv1(F.dropout(x, p=self.dropout, training=self.training), edge_index, edge_weight)
        h = F.elu(self.norm1(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.conv2(h, edge_index, edge_weight).squeeze(-1)


class ResidualMLP(nn.Module):
    """Topology-only no-message-passing ablation/reference model."""
    def __init__(self, in_dim, hidden=32, dropout=0.3):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.block1 = nn.Linear(hidden, hidden)
        self.block2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, 1)
        self.dropout = dropout

    def forward(self, x, adj=None, edge_index=None, edge_weight=None):
        h = F.gelu(self.norm(self.in_proj(x)))
        residual = h
        h = F.gelu(self.block1(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.block2(h) + residual
        h = F.gelu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.out(h).squeeze(-1)


class ResidualExpressionEncoder(nn.Module):
    def __init__(self, in_dim=3, hidden=64, dropout=0.25):
        super().__init__()
        self.input = nn.Linear(in_dim, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.fc1 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.dropout = dropout

    def forward(self, x):
        h = F.gelu(self.norm(self.input(x)))
        z = F.gelu(self.fc1(h))
        z = F.dropout(z, p=self.dropout, training=self.training)
        return F.gelu(self.fc2(z) + h)


class TransformerExpressionEncoder(nn.Module):
    """Self-attention across the three typed summary statistics, not a sequence."""
    def __init__(self, hidden=64, token_dim=32, dropout=0.2):
        super().__init__()
        self.value_proj = nn.Linear(1, token_dim)
        self.feature_id = nn.Parameter(torch.randn(1, 3, token_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model=token_dim, nhead=4,
            dim_feedforward=token_dim * 2, dropout=dropout, activation="gelu",
            batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=1, enable_nested_tensor=False)
        self.project = nn.Linear(token_dim, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.res1 = nn.Linear(hidden, hidden)
        self.res2 = nn.Linear(hidden, hidden)
        self.dropout = dropout

    def forward(self, x):
        tokens = self.value_proj(x.unsqueeze(-1)) + self.feature_id
        h = self.project(self.transformer(tokens).mean(dim=1))
        h = F.gelu(self.norm(h))
        z = F.gelu(self.res1(h))
        z = F.dropout(z, p=self.dropout, training=self.training)
        return F.gelu(self.res2(z) + h)


class ResidualPredictionHead(nn.Module):
    def __init__(self, in_dim, hidden=64, dropout=0.3):
        super().__init__()
        self.project = nn.Linear(in_dim, hidden)
        self.norm1 = nn.LayerNorm(hidden)
        self.fc1 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.dropout = dropout
        self.out = nn.Linear(hidden, 1)

    def forward(self, x):
        h = F.gelu(self.norm1(self.project(x)))
        z = F.gelu(self.fc1(h))
        z = F.dropout(z, p=self.dropout, training=self.training)
        return self.out(F.gelu(self.norm2(self.fc2(z) + h))).squeeze(-1)


class GCNEncoder(nn.Module):
    def __init__(self, in_dim, hidden=64, dropout=0.3):
        super().__init__()
        self.lin1, self.lin2 = nn.Linear(in_dim, hidden), nn.Linear(hidden, hidden)
        self.norm1, self.norm2 = nn.LayerNorm(hidden), nn.LayerNorm(hidden)
        self.dropout = dropout

    def forward(self, x, adj, edge_index, edge_weight):
        h = F.gelu(self.norm1(torch.sparse.mm(adj, self.lin1(x))))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = torch.sparse.mm(adj, self.lin2(h))
        return F.gelu(self.norm2(h))


class SAGEEncoder(nn.Module):
    def __init__(self, in_dim, hidden=64, dropout=0.3):
        super().__init__()
        self.conv1, self.conv2 = SAGEConv(in_dim, hidden), SAGEConv(hidden, hidden)
        self.norm1, self.norm2 = nn.LayerNorm(hidden), nn.LayerNorm(hidden)
        self.dropout = dropout

    def forward(self, x, adj, edge_index, edge_weight):
        h = F.gelu(self.norm1(self.conv1(x, edge_index, edge_weight, x.shape[0])))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h, edge_index, edge_weight, x.shape[0])
        return F.gelu(self.norm2(h))


class GATEncoder(nn.Module):
    def __init__(self, in_dim, hidden=16, heads=4, dropout=0.3):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden, heads=heads, concat=True, dropout=dropout)
        self.norm1 = nn.LayerNorm(hidden * heads)
        self.conv2 = GATConv(hidden * heads, hidden, heads=heads, concat=True, dropout=dropout)
        self.norm2 = nn.LayerNorm(hidden * heads)
        self.dropout = dropout

    def forward(self, x, adj, edge_index, edge_weight):
        h = self.conv1(F.dropout(x, p=self.dropout, training=self.training), edge_index, edge_weight)
        h = F.elu(self.norm1(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h, edge_index, edge_weight)
        return F.elu(self.norm2(h))


class NodeClassifier(nn.Module):
    """Shared factory for graph-only, expression-only, hybrid, and fusion ablations."""
    def __init__(self, name, topo_dim=3, dropout=0.3):
        super().__init__()
        self.name = name
        self.dropout = dropout
        if name in {"graph_gcn", "graph_sage", "graph_gat"}:
            if name == "graph_gcn":
                self.graph = GCNEncoder(topo_dim, hidden=32, dropout=dropout)
                graph_dim = 32
            elif name == "graph_sage":
                self.graph = SAGEEncoder(topo_dim, hidden=32, dropout=dropout)
                graph_dim = 32
            else:
                self.graph = GATEncoder(topo_dim, hidden=8, heads=4, dropout=dropout)
                graph_dim = 32
            self.head = nn.Sequential(nn.Linear(graph_dim, 32), nn.GELU(), nn.Dropout(dropout), nn.Linear(32, 1))
        elif name == "expression_residual":
            self.expression = ResidualExpressionEncoder(3, 64, dropout)
            self.head = nn.Linear(64, 1)
        elif name == "expression_transformer":
            self.expression = TransformerExpressionEncoder(64, 32, dropout)
            self.head = nn.Linear(64, 1)
        elif name in {"expression_gat_concat", "expression_sage_concat", "hmgat_et_gated", "hmgat_et_concat"}:
            self.expression = (TransformerExpressionEncoder(64, 32, dropout) if name in {"hmgat_et_gated", "hmgat_et_concat"}
                else ResidualExpressionEncoder(3, 64, dropout))
            graph_input = 64 + topo_dim
            if name == "expression_sage_concat":
                self.graph = SAGEEncoder(graph_input, hidden=64, dropout=dropout)
            else:
                self.graph = GATEncoder(graph_input, hidden=16, heads=4, dropout=dropout)
            if name == "hmgat_et_gated":
                self.gate = nn.Linear(128, 64)
                self.head = ResidualPredictionHead(64, 64, dropout)
            else:
                self.head = ResidualPredictionHead(128, 64, dropout)
        else:
            raise KeyError(name)

    def forward(self, expression, topology, adj, edge_index, edge_weight):
        if self.name.startswith("graph_"):
            h = self.graph(topology, adj, edge_index, edge_weight)
            return self.head(h).squeeze(-1)
        h_expr = self.expression(expression)
        if self.name in {"expression_residual", "expression_transformer"}:
            return self.head(h_expr).squeeze(-1)
        h_graph = self.graph(torch.cat([h_expr, topology], dim=1), adj, edge_index, edge_weight)
        if self.name == "hmgat_et_gated":
            gate = torch.sigmoid(self.gate(torch.cat([h_expr, h_graph], dim=1)))
            fused = gate * h_expr + (1.0 - gate) * h_graph
        else:
            fused = torch.cat([h_expr, h_graph], dim=1)
        return self.head(fused).squeeze(-1)


def make_model(name, in_dim, hidden=32, dropout=0.3):
    if name == "gcn":
        return GCN(in_dim, hidden, dropout)
    if name == "sage":
        return GraphSAGE(in_dim, hidden, dropout)
    if name == "gat":
        return GAT(in_dim, hidden=max(4, hidden // 4), heads=4, dropout=dropout)
    if name == "topology_residual_mlp":
        return ResidualMLP(in_dim, hidden, dropout)
    raise KeyError(name)
