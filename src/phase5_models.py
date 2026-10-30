"""Small gene-level models for the expanded Phase 5 universe."""
import torch
from torch import nn
from torch.nn import functional as F

class ExpandedGCN(nn.Module):
    def __init__(self,in_dim=2,hidden=32,layers=2,dropout=.3):
        super().__init__();self.dropout=dropout
        dims=[in_dim]+[hidden]*layers
        self.layers=nn.ModuleList(nn.Linear(dims[i],dims[i+1]) for i in range(layers))
        self.norms=nn.ModuleList(nn.LayerNorm(hidden) for _ in range(layers))
        self.head=nn.Sequential(nn.Linear(hidden,32),nn.GELU(),nn.Dropout(dropout),nn.Linear(32,1))
    def forward(self,x,adj):
        h=x
        for i,layer in enumerate(self.layers):
            h=torch.sparse.mm(adj,layer(h));h=F.gelu(self.norms[i](h))
            if i<len(self.layers)-1: h=F.dropout(h,p=self.dropout,training=self.training)
        return self.head(h).squeeze(-1)
