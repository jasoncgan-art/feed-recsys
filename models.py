"""Model definitions.

1. TwoTower -- stage-1 retrieval model (same design as the ML-100K project):
       user tower:  [user_id emb | age emb | gender emb | occupation emb]
                    -> MLP -> L2-normalized user vector
       item tower:  [item_id emb | genre multi-hot proj | year emb]
                    -> MLP -> L2-normalized item vector
       score(u, i) = <user_vec, item_vec> / temperature
   Trained with IN-BATCH SAMPLED SOFTMAX: within a batch of B (user, item)
   pairs, each user's item is the positive and the other B-1 items in the
   batch act as negatives. Loss = cross-entropy over the B x B score matrix.

   Two corrections carry over from the ML-100K project (both matter):
     * logQ (Yi et al., 2019): in-batch negatives oversample popular items,
       so we subtract log Q(item) from each logit before softmax.
     * neg_mask: a user's OTHER liked items showing up as in-batch negatives
       are masked to -inf -- they are false negatives, not real negatives.

2. Ranker -- stage-2 scoring model. It does NOT learn its own embeddings.
   Instead it reuses the frozen stage-1 vectors and adds three hand-crafted
   cross features the dot product cannot express:
       genre_match : how much the item's genres overlap the user's taste
                     profile (mean genre vector over their train positives)
       log_pop     : log-scaled item popularity (train positive count)
       log_activity: log-scaled user activity (train positive count)
   Input = [user_vec (64) | item_vec (64) | 3 cross features] -> MLP -> score.
   Trained with pointwise BCE on (positive, sampled negatives). This is the
   standard "retrieval embeddings + ranker" split: stage 1 learns good
   representations fast, stage 2 learns a flexible scoring function on top.
   Freezing the towers keeps ranker training cheap (a tiny MLP on 131-dim
   inputs) and keeps the two stages' responsibilities cleanly separated.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class UserTower(nn.Module):
    def __init__(self, n_users, n_age, n_occ, id_dim=32, out_dim=64):
        super().__init__()
        self.emb_user = nn.Embedding(n_users, id_dim)
        self.emb_age = nn.Embedding(n_age, 8)
        self.emb_gender = nn.Embedding(2, 4)
        self.emb_occ = nn.Embedding(n_occ, 8)
        self.mlp = nn.Sequential(
            nn.Linear(id_dim + 8 + 4 + 8, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim),
        )

    def forward(self, user_idx, age, gender, occ):
        x = torch.cat(
            [self.emb_user(user_idx), self.emb_age(age),
             self.emb_gender(gender), self.emb_occ(occ)],
            dim=-1,
        )
        return F.normalize(self.mlp(x), p=2, dim=-1)  # unit vectors -> cosine sim


class ItemTower(nn.Module):
    def __init__(self, n_items, n_genres, n_year, id_dim=32, out_dim=64):
        super().__init__()
        self.emb_item = nn.Embedding(n_items, id_dim)
        self.genre_proj = nn.Linear(n_genres, 16)  # multi-hot -> dense
        self.emb_year = nn.Embedding(n_year, 8)
        self.mlp = nn.Sequential(
            nn.Linear(id_dim + 16 + 8, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim),
        )

    def forward(self, item_idx, genres, year):
        x = torch.cat(
            [self.emb_item(item_idx), self.genre_proj(genres),
             self.emb_year(year)],
            dim=-1,
        )
        return F.normalize(self.mlp(x), p=2, dim=-1)


class TwoTower(nn.Module):
    def __init__(self, user_tower: UserTower, item_tower: ItemTower,
                 temperature: float = 0.1):
        super().__init__()
        self.user_tower = user_tower
        self.item_tower = item_tower
        self.temperature = temperature

    def forward(self, u_idx, age, gender, occ, i_idx, genres, year):
        u = self.user_tower(u_idx, age, gender, occ)   # (B, d)
        v = self.item_tower(i_idx, genres, year)      # (B, d)
        return u, v

    def inbatch_softmax_loss(self, u, v, neg_mask=None, log_q=None):
        """u, v: (B, d) L2-normalized. Positive for row b is column b."""
        logits = (u @ v.t()) / self.temperature       # (B, B)
        if log_q is not None:
            logits = logits - log_q.unsqueeze(0)      # sampled-softmax correction
        if neg_mask is not None:
            logits = logits.masked_fill(neg_mask, float("-inf"))
        labels = torch.arange(u.size(0), device=u.device)
        return F.cross_entropy(logits, labels)

    @torch.no_grad()
    def encode_users(self, u_idx, age, gender, occ):
        return self.user_tower(u_idx, age, gender, occ)

    @torch.no_grad()
    def encode_items(self, i_idx, genres, year):
        return self.item_tower(i_idx, genres, year)


class Ranker(nn.Module):
    """Stage-2 scorer: MLP over [user_vec | item_vec | cross features].

    Cross features (3 scalars, all precomputed by the caller):
        [0] genre_match  -- dot(user genre profile, item genre multi-hot)
        [1] log_pop      -- log1p(item train count), scaled to ~[0, 1]
        [2] log_activity -- log1p(user train count), scaled to ~[0, 1]
    """

    N_CROSS = 3

    def __init__(self, vec_dim: int = 64):
        super().__init__()
        in_dim = vec_dim * 2 + self.N_CROSS
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, u_vec, v_vec, cross):
        x = torch.cat([u_vec, v_vec, cross], dim=-1)
        return self.mlp(x).squeeze(-1)  # (B,) raw scores (logits for BCE)
