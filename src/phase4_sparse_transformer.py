"""Small sparse, edge-aware graph Transformer blocks for Phase 4 experiments."""
import torch
from torch import nn
from torch.nn import functional as F


class SparseTransformerBlock(nn.Module):
    """Multi-head dot-product attention restricted to STRING edges."""
    def __init__(self, dim=32, heads=4, dropout=0.25):
        super().__init__()
        assert dim % heads == 0
        self.dim, self.heads, self.dk = dim, heads, dim // heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.edge_bias = nn.Parameter(torch.ones(heads))
        self.out = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, 2*dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(2*dim,dim))
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight):
        n = x.shape[0]
        q, k, v = self.qkv(self.norm1(x)).view(n, 3, self.heads, self.dk).unbind(1)
        src, dst = edge_index
        logits = (q[dst] * k[src]).sum(-1) / (self.dk ** 0.5)
        logits = logits + edge_weight.clamp_min(1e-6).log()[:, None] * self.edge_bias
        groups = dst[:, None].expand(-1, self.heads)
        maxima = torch.full((n, self.heads), -torch.inf, device=x.device)
        maxima.scatter_reduce_(0, groups, logits, reduce="amax", include_self=True)
        weights = torch.exp(logits - maxima[dst])
        denom = torch.zeros((n, self.heads), device=x.device).index_add_(0, dst, weights).clamp_min(1e-12)
        alpha = F.dropout(weights / denom[dst], p=self.dropout, training=self.training)
        msg = v[src] * alpha[..., None]
        agg = torch.zeros((n, self.heads, self.dk), device=x.device).index_add_(0, dst, msg).reshape(n, self.dim)
        h = x + F.dropout(self.out(agg), p=self.dropout, training=self.training)
        h = h + F.dropout(self.ff(self.norm2(h)), p=self.dropout, training=self.training)
        return h


class SparseGraphTransformer(nn.Module):
    def __init__(self, in_dim, hidden=32, dropout=0.25):
        super().__init__()
        self.input = nn.Linear(in_dim, hidden)
        self.blocks = nn.ModuleList([SparseTransformerBlock(hidden, 4, dropout) for _ in range(2)])
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 24), nn.GELU(), nn.Dropout(dropout), nn.Linear(24,1))
    def embed(self, x, edge_index, edge_weight):
        h = F.gelu(self.input(x))
        for block in self.blocks:
            h = block(h, edge_index, edge_weight)
        return h
    def forward(self, x, edge_index, edge_weight):
        return self.head(self.embed(x, edge_index, edge_weight)).squeeze(-1)


class SparseTransformerFusion(nn.Module):
    """Expression MLP and edge-aware graph Transformer with gated fusion."""
    def __init__(self, expr_dim=3, topo_dim=3, hidden=32, dropout=0.25):
        super().__init__()
        self.expr = nn.Sequential(nn.Linear(expr_dim,hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden,hidden), nn.GELU())
        self.graph = SparseGraphTransformer(expr_dim+topo_dim, hidden, dropout)
        self.gate = nn.Linear(2*hidden,hidden)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden,24), nn.GELU(), nn.Dropout(dropout), nn.Linear(24,1))
    def forward(self, expr, topo, edge_index, edge_weight):
        e = self.expr(expr)
        g = self.graph.embed(torch.cat([expr,topo],dim=1),edge_index,edge_weight)
        gate = torch.sigmoid(self.gate(torch.cat([e,g],dim=1)))
        return self.head(gate*e + (1-gate)*g).squeeze(-1)
