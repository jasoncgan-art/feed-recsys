"""Train the two-stage feed recommender on MovieLens 1M.

Usage:
    python train.py                                     # full pipeline
    python train.py --retriever-epochs 2 --ranker-epochs 1 --baseline-epochs 2
    python train.py --stages retriever,ranker,faiss     # skip the baseline

Stages (all artifacts under ./artifacts/, gitignored):
    prepare()   shared split + features + index maps   -> dataset.pt
    retriever   two-tower, in-batch softmax + logQ     -> retriever.pt
    ranker      MLP over [user_vec | item_vec | cross] -> ranker.pt
    baseline    second two-tower, same config, seed 123 -> baseline.pt
                (the single-stage system we compare the two-stage
                pipeline against -- same data, same split, only the
                stage-2 ranker differs)
    faiss       IndexFlatIP over normalized item vecs  -> item_index.faiss

Determinism: seeds fixed for random/numpy/torch; DataLoader shuffling uses
a seeded torch.Generator. Re-running on the same machine gives the same
numbers. (The ML-100K project's train.py once called np.random.seed
without importing numpy -- that import is present here, line 1 of imports.)
"""

import argparse
import os
import random
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from dataset import (
    build_item_features,
    build_user_features,
    load_raw,
    sample_negatives_fast,
    temporal_leave_last_out,
)
from models import ItemTower, Ranker, TwoTower, UserTower

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ART_DIR = os.path.join(BASE_DIR, "artifacts")

SEED = 42
BASELINE_SEED = 123
DEVICE = torch.device("cpu")


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# --------------------------------------------------------------------------- #
# shared data prep
# --------------------------------------------------------------------------- #
def prepare():
    """Build the split + features once; save everything downstream needs."""
    os.makedirs(ART_DIR, exist_ok=True)
    interactions, movies, users = load_raw()
    train_df, test_df = temporal_leave_last_out(interactions)
    ufeat = build_user_features(users)
    ifeat = build_item_features(movies)

    print(f"[data] train positives: {len(train_df)}, test users: {len(test_df)}, "
          f"users: {len(ufeat['user_ids'])}, items: {len(ifeat['item_ids'])}")

    def to_idx(df):
        return (
            torch.tensor([ufeat["user_idx"][u] for u in df["user_id"]], dtype=torch.long),
            torch.tensor([ifeat["item_idx"][i] for i in df["item_id"]], dtype=torch.long),
        )

    tr_u, tr_i = to_idx(train_df)
    te_u, te_i = to_idx(test_df)

    # seen mask: (n_users, n_items) bool -- train interactions to exclude at eval
    seen = torch.zeros(len(ufeat["user_ids"]), len(ifeat["item_ids"]), dtype=torch.bool)
    seen[tr_u, tr_i] = True

    # item popularity counts (train positives) -- the ranker's log_pop feature
    counts = torch.zeros(len(ifeat["item_ids"]))
    counts.index_add_(0, tr_i, torch.ones(len(tr_i)))

    bundle = {
        "user_ids": ufeat["user_ids"], "item_ids": ifeat["item_ids"],
        "user_idx": ufeat["user_idx"], "item_idx": ifeat["item_idx"],
        "titles": ifeat["titles"],
        "train_u": tr_u, "train_i": tr_i, "test_u": te_u, "test_i": te_i,
        "seen": seen, "counts": counts,
        "u_age": torch.tensor(ufeat["age"]), "u_gender": torch.tensor(ufeat["gender"]),
        "u_occ": torch.tensor(ufeat["occupation"]),
        "n_age": ufeat["n_age_buckets"], "n_occ": ufeat["n_occupations"],
        "i_genres": torch.tensor(ifeat["genres"]), "i_year": torch.tensor(ifeat["year"]),
        "n_year": ifeat["n_year_buckets"], "n_genres": ifeat["genres"].shape[1],
    }
    torch.save(bundle, os.path.join(ART_DIR, "dataset.pt"))
    return bundle


# --------------------------------------------------------------------------- #
# stage 1: two-tower retrieval (shared by retriever + baseline)
# --------------------------------------------------------------------------- #
def build_pos_padded(train_u, train_i, n_users):
    """(n_users, max_pos) LongTensor of each user's train-positive item indices,
    padded with -1. Used to mask false negatives out of the in-batch softmax."""
    pos = defaultdict(list)
    for u, i in zip(train_u.tolist(), train_i.tolist()):
        pos[u].append(i)
    max_p = max(len(v) for v in pos.values())
    padded = torch.full((n_users, max_p), -1, dtype=torch.long)
    for u, lst in pos.items():
        padded[u, :len(lst)] = torch.tensor(lst, dtype=torch.long)
    return padded


def inbatch_neg_mask(bu, bi, pos_padded, n_items):
    """(B, B) bool mask: True where batch item j is a known train positive of
    batch user b (but not the pair's own positive on the diagonal)."""
    B = len(bu)
    col_of = torch.full((n_items,), -1, dtype=torch.long)
    col_of[bi] = torch.arange(B)                       # item idx -> batch column
    cand = pos_padded[bu].clamp(min=0)                 # (B, P) item indices
    valid = (pos_padded[bu] >= 0) & (col_of[cand] >= 0)
    rows = torch.arange(B).unsqueeze(1).expand_as(cand)
    mask = torch.zeros(B, B, dtype=torch.bool)
    mask[rows[valid], col_of[cand][valid]] = True
    mask.fill_diagonal_(False)                         # keep the true positive
    return mask


def train_twotower(data, out_name, seed=SEED, id_dim=32, out_dim=64, epochs=25,
                   batch_size=1024, lr=5e-3, temperature=0.1):
    """Train a TwoTower and save it to artifacts/<out_name>.

    out_name/seed are parameters (not hardcoded) so the baseline run can
    reuse this exact function with a different seed and output file.
    """
    set_seed(seed)  # re-seed: this model's init must not depend on what ran before
    n_users, n_items = len(data["user_ids"]), len(data["item_ids"])
    loader = DataLoader(
        TensorDataset(data["train_u"], data["train_i"]), batch_size=batch_size,
        shuffle=True, generator=torch.Generator().manual_seed(seed),
    )
    pos_padded = build_pos_padded(data["train_u"], data["train_i"], n_users)
    # logQ correction (Yi et al., 2019): in-batch negatives are sampled
    # roughly proportional to item popularity Q(i); precompute log Q(i).
    counts = torch.zeros(n_items)
    counts.index_add_(0, data["train_i"], torch.ones(len(data["train_i"])))
    logQ = (counts / counts.sum()).clamp(min=1e-12).log()
    model = TwoTower(
        UserTower(n_users, data["n_age"], data["n_occ"], id_dim, out_dim),
        ItemTower(n_items, data["n_genres"], data["n_year"], id_dim, out_dim),
        temperature=temperature,
    ).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(1, epochs + 1):
        model.train()
        total, n = 0.0, 0
        for bu, bi in loader:
            opt.zero_grad()
            u = model.user_tower(bu, data["u_age"][bu], data["u_gender"][bu], data["u_occ"][bu])
            v = model.item_tower(bi, data["i_genres"][bi], data["i_year"][bi])
            mask = inbatch_neg_mask(bu, bi, pos_padded, n_items)
            loss = model.inbatch_softmax_loss(u, v, neg_mask=mask,
                                              log_q=logQ[bi])
            loss.backward()
            opt.step()
            total += loss.item() * len(bu)
            n += len(bu)
        print(f"[{out_name}] epoch {ep:02d}/{epochs}  inbatch_softmax={total / n:.4f}")
    torch.save(
        {"state_dict": model.state_dict(), "id_dim": id_dim, "out_dim": out_dim,
         "temperature": temperature},
        os.path.join(ART_DIR, out_name),
    )
    return model


# --------------------------------------------------------------------------- #
# stage 2: ranker (MLP over frozen retrieval vectors + cross features)
# --------------------------------------------------------------------------- #
def build_cross_tables(data):
    """Precompute the ranker's cross-feature tables from TRAIN data only.

    genre_profile (U, G): each user's taste = mean genre multi-hot over
        their train positives. (Users with no train positives get zeros.)
    log_pop (I,): log1p(item train count), scaled to ~[0, 1].
    log_activity (U,): log1p(user train count), scaled to ~[0, 1].
    Using train data only is what keeps eval honest: the ranker never sees
    test interactions through its features.
    """
    n_users, n_items = len(data["user_ids"]), len(data["item_ids"])
    genre_profile = torch.zeros(n_users, data["n_genres"])
    genre_profile.index_add_(0, data["train_u"], data["i_genres"][data["train_i"]])
    user_counts = data["seen"].sum(dim=1).clamp(min=1).unsqueeze(1).float()
    genre_profile = genre_profile / user_counts
    log_pop = torch.log1p(data["counts"]) / torch.log1p(data["counts"].max())
    log_activity = (torch.log1p(data["seen"].sum(dim=1).float())
                    / torch.log1p(torch.tensor(float(len(data["train_u"])))))
    return genre_profile, log_pop, log_activity


def train_ranker(data, retriever, epochs=5, batch_size=4096, n_neg=4, lr=1e-3):
    set_seed()  # re-seed: ranker training must not depend on what ran before
    n_items = len(data["item_ids"])
    retriever.eval()
    with torch.no_grad():
        all_u = torch.arange(len(data["user_ids"]))
        all_i = torch.arange(n_items)
        u_vec = retriever.encode_users(all_u, data["u_age"], data["u_gender"],
                                       data["u_occ"])          # (U, d)
        v_vec = retriever.encode_items(all_i, data["i_genres"],
                                       data["i_year"])         # (I, d)
    genre_profile, log_pop, log_activity = build_cross_tables(data)

    # BCE dataset: every train positive (label 1) + n_neg sampled negatives
    # (label 0) per positive. Negatives come from the fast vectorized
    # sampler -- the ML-100K project's per-positive Python scan does not
    # scale to ~600k positives.
    neg_u, neg_i = sample_negatives_fast(
        data["train_u"], data["train_i"], n_items, data["seen"],
        n_neg=n_neg, seed=SEED)
    ru = torch.cat([data["train_u"], torch.tensor(neg_u)])
    ri = torch.cat([data["train_i"], torch.tensor(neg_i)])
    ry = torch.cat([torch.ones(len(data["train_u"])),
                    torch.zeros(len(neg_u))]).float()
    print(f"[ranker] training rows: {len(ry)} "
          f"({len(data['train_u'])} pos + {len(neg_u)} neg)")
    loader = DataLoader(
        TensorDataset(ru, ri, ry), batch_size=batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    model = Ranker(vec_dim=u_vec.size(1)).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    for ep in range(1, epochs + 1):
        model.train()
        total, n = 0.0, 0
        for bu, bi, by in loader:
            # Cross features for this batch. genre_match is the dot product
            # of the user's taste profile with the item's genre multi-hot:
            # high when the item's genres are ones this user likes.
            genre_match = (genre_profile[bu] * data["i_genres"][bi]).sum(-1)
            cross = torch.stack([genre_match, log_pop[bi], log_activity[bu]],
                                dim=-1)
            opt.zero_grad()
            loss = loss_fn(model(u_vec[bu], v_vec[bi], cross), by)
            loss.backward()
            opt.step()
            total += loss.item() * len(by)
            n += len(by)
        print(f"[ranker] epoch {ep:02d}/{epochs}  bce={total / n:.4f}")
    torch.save({"state_dict": model.state_dict(), "vec_dim": u_vec.size(1)},
               os.path.join(ART_DIR, "ranker.pt"))
    return model


# --------------------------------------------------------------------------- #
# faiss index over normalized item vectors + exactness sanity check
# --------------------------------------------------------------------------- #
def build_faiss_index(data, retriever, sample_users=500, over_fetch=1000):
    import faiss

    retriever.eval()
    with torch.no_grad():
        all_i = torch.arange(len(data["item_ids"]))
        v_all = retriever.encode_items(all_i, data["i_genres"],
                                       data["i_year"]).numpy().astype("float32")
    d = v_all.shape[1]
    index = faiss.IndexFlatIP(d)  # exact inner product (vectors are unit-norm,
    index.add(v_all)              # so this IS cosine similarity, not approx.)
    faiss.write_index(index, os.path.join(ART_DIR, "item_index.faiss"))
    print(f"[faiss] index built: {index.ntotal} vectors, dim {d}")

    # Sanity check: FAISS top-200 (after filtering train-seen items) must
    # match brute-force top-200. IndexFlatIP is exact, so agreement should
    # be 100% -- anything less means the index/serving wiring is wrong.
    # (We sample users with <= 500 train positives so that over-fetching
    # 1000 candidates always leaves >= 200 unseen ones.)
    rng = np.random.default_rng(SEED)
    train_counts = data["seen"].sum(dim=1)
    eligible = [int(u) for u in data["test_u"].tolist() if train_counts[u] <= 500]
    sample = rng.choice(eligible, size=min(sample_users, len(eligible)),
                        replace=False)
    with torch.no_grad():
        u_samp = retriever.encode_users(
            torch.tensor(sample), data["u_age"][sample],
            data["u_gender"][sample], data["u_occ"][sample]).numpy().astype("float32")
    _, faiss_idx = index.search(u_samp, over_fetch)
    seen_np = data["seen"].numpy()
    overlaps = []
    for row, u in enumerate(sample):
        cand = [int(j) for j in faiss_idx[row] if not seen_np[u, j]][:200]
        scores = torch.tensor(u_samp[row]) @ torch.tensor(v_all).t()
        scores_np = scores.numpy().copy()
        scores_np[seen_np[u]] = -np.inf
        exact = np.argsort(-scores_np)[:200].tolist()
        overlaps.append(len(set(cand) & set(exact)) / 200.0)
    print(f"[faiss] top-200 agreement vs brute force: {np.mean(overlaps):.4f} "
          f"over {len(sample)} sampled test users")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="retriever,ranker,baseline,faiss",
                    help="comma-separated subset of {retriever,ranker,baseline,faiss}")
    ap.add_argument("--retriever-epochs", type=int, default=25)
    ap.add_argument("--ranker-epochs", type=int, default=5)
    ap.add_argument("--baseline-epochs", type=int, default=25)
    args = ap.parse_args()
    which = {s.strip() for s in args.stages.split(",")}

    set_seed()
    data = prepare()
    retriever = None
    if "retriever" in which:
        retriever = train_twotower(data, "retriever.pt", seed=SEED,
                                   epochs=args.retriever_epochs)
    if "ranker" in which:
        if retriever is None:
            retriever = load_twotower(data, "retriever.pt")
        train_ranker(data, retriever, epochs=args.ranker_epochs)
    if "baseline" in which:
        # Same config as the retriever, different seed: the comparison
        # isolates the stage-2 ranker, not a lucky/unlucky init.
        train_twotower(data, "baseline.pt", seed=BASELINE_SEED,
                       epochs=args.baseline_epochs)
    if "faiss" in which:
        if retriever is None:
            retriever = load_twotower(data, "retriever.pt")
        build_faiss_index(data, retriever)
    print("[done] artifacts in", ART_DIR)


def load_twotower(data, name):
    """Rebuild a TwoTower from a saved artifact (used when stages run apart)."""
    art = torch.load(os.path.join(ART_DIR, name), weights_only=True)
    model = TwoTower(
        UserTower(len(data["user_ids"]), data["n_age"], data["n_occ"],
                  art["id_dim"], art["out_dim"]),
        ItemTower(len(data["item_ids"]), data["n_genres"], data["n_year"],
                  art["id_dim"], art["out_dim"]),
        temperature=art["temperature"],
    )
    model.load_state_dict(art["state_dict"])
    model.eval()
    return model


if __name__ == "__main__":
    main()
