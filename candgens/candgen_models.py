# candgen_models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class MeanUserEncoder(nn.Module):
    def __init__(self, n_items: int, emb_dim: int):
        super().__init__()
        self.item_embeddings = nn.Embedding(n_items, emb_dim)

    def forward(self, history: Dict[str, torch.Tensor]):
        item_idx = history["item_id"]
        item_embs = self.item_embeddings(item_idx)

        length = history["length"]
        batch_size = length.shape[0]

        user_indices = torch.repeat_interleave(
            torch.arange(batch_size, device=item_idx.device),
            length
        )

        emb_dim = self.item_embeddings.embedding_dim
        user_emb = torch.zeros(batch_size, emb_dim, device=item_idx.device)
        user_emb.scatter_add_(
            0,
            user_indices.unsqueeze(1).expand(-1, emb_dim),
            item_embs
        )
        user_emb = user_emb / length.unsqueeze(1)
        return user_emb


class BaseTwoTowerModel(nn.Module):
    def __init__(self, n_users: int, n_items: int, emb_dim: int):
        super().__init__()
        self.user_embeds = nn.Embedding(n_users, emb_dim)
        self.item_embeds = nn.Embedding(n_items, emb_dim)


class BaseMeanTwoTowerModel(nn.Module):
    def __init__(self, n_items, emb_dim):
        super().__init__()
        self.encoder = MeanUserEncoder(n_items, emb_dim)


class InBatchLogQMeanTwoTowerModel(BaseMeanTwoTowerModel):
    def __init__(self, n_items, emb_dim, item_counts):
        super().__init__(n_items, emb_dim)
        item_probs = item_counts / item_counts.sum()
        log_q = torch.log(item_probs + 1e-10)
        self.register_buffer("log_q", log_q)


class InBatchLogQMF(BaseTwoTowerModel):
    def __init__(self, n_users, n_items, emb_dim, item_counts):
        super().__init__(n_users, n_items, emb_dim)
        item_probs = item_counts / item_counts.sum()
        log_q = torch.log(item_probs + 1e-10)
        self.register_buffer("log_q", log_q)
