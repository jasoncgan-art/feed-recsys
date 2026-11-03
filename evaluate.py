"""Evaluate the two-stage pipeline and the single-stage baseline.

All metrics are on the held-out last interaction per test user (one
relevant item per user), on the SAME ML-1M split. Train-seen items are
always masked out -- recommending something the user already watched
would inflate every number below.

Metrics reported:
    Retrieval Recall@200 -- held-out item in the retriever's top-200
        candidates (all test users). This is the ceiling for the whole
        pipeline: if retrieval misses, the ranker never sees the item.
    End-to-end Recall@10 -- held-out item in the ranker's top-10 of those
        200 candidates (all test users; a retrieval miss counts as 0).
    NDCG@10 (end-to-end) -- 1/log2(rank+1) if the ranker puts the held-out
        item at rank <= 10, else 0 (all test users).
    NDCG@10 (ranker-only) -- same, but averaged only over users whose
        held-out item survived retrieval. This isolates ranker quality
        from retrieval quality.
    AUC (ranker-only) -- for retrieval-hit users, the fraction of the
        other 199 candidates the ranker scored BELOW the held-out item.
        0.5 = random, 1.0 = perfect.
    Baseline Recall@10 -- the single-stage two-tower (separate model,
        same config, seed 123) scoring ALL unseen items directly, top-10.
        The comparison that matters: end-to-end Recall@10 vs this number.
        The only difference between the two systems is the stage-2 ranker.
"""

import json
import os
import time

import numpy as np
import torch

from models import ItemTower, Ranker, TwoTower, UserTower
from train import build_cross_tables, load_twotower

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ART_DIR = os.path.join(BASE_DIR, "artifacts")
TOP_CAND = 200


def ranks_and_scores(data, retriever, ranker):
    """Run retrieval + ranking for all test users.

    Returns (hit200, e2e_rank, ranker_scores) where:
        hit200: (T,) bool -- held-out item in retrieval top-200
        e2e_rank: (T,) int, 1-based ranker rank of the held-out item
            among the 200 candidates, or -1 when retrieval missed
        ranker_scores: (T, 200) ranker scores over the candidates
    """
    test_u, test_i = data["test_u"], data["test_i"]
    retriever.eval()
    ranker.eval()
    genre_profile, log_pop, log_activity = build_cross_tables(data)
    with torch.no_grad():
        all_i = torch.arange(len(data["item_ids"]))
        v_all = retriever.encode_items(all_i, data["i_genres"], data["i_year"])
        u_test = retriever.encode_users(test_u, data["u_age"][test_u],
                                        data["u_gender"][test_u],
                                        data["u_occ"][test_u])
        # Retrieval: exact dot-product scores over ALL unseen items.
        scores = u_test @ v_all.t()
        scores[data["seen"][test_u]] = float("-inf")
        cand = scores.topk(TOP_CAND, dim=1).indices            # (T, 200)
        hit200 = (cand == test_i.unsqueeze(1)).any(dim=1)      # (T,)

        # Ranking: score the 200 candidates per user with the ranker.
        ranker_scores = torch.empty(len(test_u), TOP_CAND)
        for s in range(0, len(test_u), 512):
            chunk = slice(s, min(s + 512, len(test_u)))
            cu = test_u[chunk]
            cc = cand[chunk]                                   # (C, 200)
            u_rep = u_test[chunk].unsqueeze(1).expand(-1, TOP_CAND, -1)
            v_cand = v_all[cc]                                 # (C, 200, d)
            genre_match = (genre_profile[cu].unsqueeze(1)
                           * data["i_genres"][cc]).sum(-1)     # (C, 200)
            cross = torch.stack([genre_match, log_pop[cc],
                                 log_activity[cu].unsqueeze(1).expand(-1, TOP_CAND)],
                                dim=-1)
            ranker_scores[chunk] = ranker(
                u_rep.reshape(-1, u_rep.size(-1)),
                v_cand.reshape(-1, v_cand.size(-1)),
                cross.reshape(-1, cross.size(-1))).reshape(-1, TOP_CAND)

        # Rank of the held-out item inside the ranker's ordering.
        order = ranker_scores.argsort(dim=1, descending=True)  # (T, 200)
        # Column where the held-out item sits in the candidate list.
        col = (cand == test_i.unsqueeze(1)).nonzero()
        e2e_rank = torch.full((len(test_u),), -1, dtype=torch.long)
        rows = col[:, 0]
        pos_in_order = (order[rows] == col[:, 1].unsqueeze(1)).nonzero()[:, 1] + 1
        e2e_rank[rows] = pos_in_order
    return hit200.numpy(), e2e_rank.numpy(), ranker_scores.numpy(), cand.numpy()


def main():
    t0 = time.time()
    data = torch.load(os.path.join(ART_DIR, "dataset.pt"), weights_only=False)
    retriever = load_twotower(data, "retriever.pt")
    baseline = load_twotower(data, "baseline.pt")
    rart = torch.load(os.path.join(ART_DIR, "ranker.pt"), weights_only=True)
    ranker = Ranker(vec_dim=rart["vec_dim"])
    ranker.load_state_dict(rart["state_dict"])
    ranker.eval()

    hit200, e2e_rank, ranker_scores, cand = ranks_and_scores(data, retriever, ranker)
    test_i = data["test_i"].numpy()
    n_test = len(test_i)

    recall200 = hit200.mean()
    # End-to-end: retrieval miss counts as rank = infinity -> 0.
    valid_rank = np.where(e2e_rank > 0, e2e_rank, 10 ** 9)
    recall10_e2e = (valid_rank <= 10).mean()
    ndcg10_e2e = np.where(valid_rank <= 10, 1.0 / np.log2(valid_rank + 1.0), 0.0).mean()
    # Ranker-only: only users whose held-out item survived retrieval.
    hit_idx = np.where(hit200)[0]
    ranks_hit = e2e_rank[hit_idx]
    ndcg10_ranker = np.where(ranks_hit <= 10,
                             1.0 / np.log2(ranks_hit + 1.0), 0.0).mean()
    # AUC: fraction of the other 199 candidates scored below the held-out item.
    aucs = []
    for r in hit_idx:
        col = np.where(cand[r] == test_i[r])[0][0]
        s = ranker_scores[r]
        aucs.append((s < s[col]).mean())
    auc = float(np.mean(aucs))

    # Baseline: single-stage two-tower scores ALL unseen items directly.
    baseline.eval()
    with torch.no_grad():
        all_i = torch.arange(len(data["item_ids"]))
        v_base = baseline.encode_items(all_i, data["i_genres"], data["i_year"])
        u_base = baseline.encode_users(data["test_u"], data["u_age"][data["test_u"]],
                                       data["u_gender"][data["test_u"]],
                                       data["u_occ"][data["test_u"]])
        scores = u_base @ v_base.t()
        scores[data["seen"][data["test_u"]]] = float("-inf")
        top10 = scores.topk(10, dim=1).indices
        baseline_recall10 = (top10 == data["test_i"].unsqueeze(1)).any(dim=1).float().mean().item()

    metrics = {
        "n_test_users": n_test,
        "retrieval_recall@200": float(recall200),
        "end_to_end_recall@10": float(recall10_e2e),
        "ndcg@10_end_to_end": float(ndcg10_e2e),
        "ndcg@10_ranker_only": float(ndcg10_ranker),
        "auc_ranker_only": auc,
        "n_ranker_only_users": int(len(hit_idx)),
        "baseline_recall@10": float(baseline_recall10),
    }
    with open(os.path.join(ART_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print()
    print(f"{'metric':<26} {'value':>10}")
    print("-" * 38)
    print(f"{'test users':<26} {n_test:>10d}")
    print(f"{'retrieval Recall@200':<26} {recall200:>10.4f}")
    print(f"{'end-to-end Recall@10':<26} {recall10_e2e:>10.4f}")
    print(f"{'NDCG@10 (end-to-end)':<26} {ndcg10_e2e:>10.4f}")
    print(f"{'NDCG@10 (ranker-only)':<26} {ndcg10_ranker:>10.4f}")
    print(f"{'AUC (ranker-only)':<26} {auc:>10.4f}")
    print(f"{'baseline Recall@10':<26} {baseline_recall10:>10.4f}")
    print(f"\n[eval] finished in {time.time() - t0:.1f}s, "
          f"metrics saved to artifacts/metrics.json")


if __name__ == "__main__":
    main()
