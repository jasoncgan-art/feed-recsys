"""Shared data loading: features, temporal leave-last-out split, negative sampling.

Split protocol -- temporal leave-last-out (a.k.a. leave-one-out by time):
    * For each user, sort their positive interactions by timestamp.
    * Hold out the LAST interaction as the test target.
    * Everything earlier is training data.
  This mimics the real retrieval task: "given what the user liked in the past,
  retrieve what they will like next" -- no future leakage.

Users with fewer than 2 positive interactions are dropped (can't split them).

Why sample_negatives_fast exists: the ML-100K project's sampler scanned the
whole item catalog in Python once per positive (a list comprehension over
~4k items x ~55k positives -- fine there). On ML-1M that would be ~600k
positives x ~4k items = ~2.4 billion Python-level checks. The fast sampler
below works per user with vectorized numpy instead: draw a block of random
candidates, drop the ones the user already saw, keep the first n_neg per
row. Same semantics (negatives the user never saw), ~1000x less work.
"""

import os
from collections import defaultdict

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

GENRE_NAMES = [
    "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
    "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]
N_GENRES = len(GENRE_NAMES)

# ML-1M age codes -> bucket index. The codes already describe the same seven
# ranges the ML-100K project used (0-17, 18-24, 25-34, 35-44, 45-49, 50-55,
# 56+), so this is a lookup, not a computation.
AGE_CODE_TO_BUCKET = {1: 0, 18: 1, 25: 2, 35: 3, 45: 4, 50: 5, 56: 6}
N_AGE_BUCKETS = 7

# ML-1M occupation codes are already 0..20, used directly as embedding ids.
N_OCCUPATIONS = 21


def year_bucket(year: int) -> int:
    """Bucket release year into ~decade-ish buckets (missing year -> 0)."""
    if year <= 0:
        return 0
    return min(1 + (year - 1900) // 10, 11)  # 0=unknown, 1..11 = 1900s..2000s+


def load_raw():
    """Load the processed CSVs written by download.py."""
    interactions = pd.read_csv(os.path.join(DATA_DIR, "interactions.csv"))
    movies = pd.read_csv(os.path.join(DATA_DIR, "movies.csv"))
    users = pd.read_csv(os.path.join(DATA_DIR, "users.csv"))
    return interactions, movies, users


def temporal_leave_last_out(interactions: pd.DataFrame, min_interactions: int = 2):
    """Split into train/test by holding out each user's last interaction by time.

    Returns (train_df, test_df). Both have columns [user_id, item_id, timestamp].
    """
    counts = interactions["user_id"].value_counts()
    keep = counts[counts >= min_interactions].index
    df = interactions[interactions["user_id"].isin(keep)].copy()
    df = df.sort_values(["user_id", "timestamp"])
    # The last row per user (by timestamp) becomes the test target.
    is_last = df.groupby("user_id").cumcount(ascending=False) == 0
    test_df = df[is_last].reset_index(drop=True)
    train_df = df[~is_last].reset_index(drop=True)
    return train_df, test_df


def build_user_features(users: pd.DataFrame):
    """Encode user side features for the user tower.

    Returns a dict with:
        user_ids   : array of raw user ids (index i -> user id)
        user_idx   : raw user id -> row index
        age        : int array of age-bucket ids (from ML-1M age codes)
        gender     : int array (0=F, 1=M)
        occupation : int array of ML-1M occupation codes (0..20)
    """
    users = users.sort_values("user_id").reset_index(drop=True)
    gender_to_id = {"F": 0, "M": 1}
    feats = {
        "user_ids": users["user_id"].to_numpy(),
        "user_idx": {u: i for i, u in enumerate(users["user_id"].to_numpy())},
        "age": np.array(
            [AGE_CODE_TO_BUCKET.get(int(a), 0) for a in users["age_code"]],
            dtype=np.int64,
        ),
        "gender": np.array([gender_to_id[g] for g in users["gender"]], dtype=np.int64),
        "occupation": np.array(users["occupation_code"].to_numpy(), dtype=np.int64),
        "n_age_buckets": N_AGE_BUCKETS,
        "n_occupations": N_OCCUPATIONS,
    }
    return feats


def build_item_features(movies: pd.DataFrame):
    """Encode item side features for the item tower.

    Returns a dict with:
        item_ids : array of raw item ids (index i -> item id)
        item_idx : raw item id -> row index
        genres   : float32 multi-hot matrix (n_items, N_GENRES)
        year     : int array of year-bucket ids
    """
    movies = movies.sort_values("item_id").reset_index(drop=True)
    genre_to_id = {g: i for i, g in enumerate(GENRE_NAMES)}
    n = len(movies)
    genres = np.zeros((n, N_GENRES), dtype=np.float32)
    years = np.zeros(n, dtype=np.int64)
    for i, row in movies.iterrows():
        if isinstance(row["genres"], str) and row["genres"]:
            for g in row["genres"].split("|"):
                if g in genre_to_id:
                    genres[i, genre_to_id[g]] = 1.0
        try:
            years[i] = year_bucket(int(row["year"]))
        except (ValueError, TypeError):
            years[i] = 0
    return {
        "item_ids": movies["item_id"].to_numpy(),
        "item_idx": {m: i for i, m in enumerate(movies["item_id"].to_numpy())},
        "titles": movies["title"].to_numpy(),
        "genres": genres,
        "year": years,
        "n_year_buckets": 12,
    }


def user_seen_items(train_df: pd.DataFrame):
    """Map raw user_id -> set of raw item_ids the user interacted with in train."""
    seen = defaultdict(set)
    for u, i in zip(train_df["user_id"], train_df["item_id"]):
        seen[u].add(i)
    return seen


def sample_negatives_fast(train_u, train_i, n_items, seen_mask, n_neg: int,
                          seed: int = 42):
    """Sample `n_neg` negatives per train positive, vectorized per user.

    Arguments are index-space (not raw ids): train_u / train_i are LongTensors
    of row indices, seen_mask is a (n_users, n_items) bool tensor marking
    train interactions. Returns (neg_u, neg_i) LongTensors with one row per
    (positive, negative) pair, plus a label tensor of zeros -- the caller
    concatenates positives (label 1) itself.

    Method: for each user, draw a block of random item indices (2x what we
    need, to absorb collisions), zero out the ones the user already saw,
    and keep the first n_neg valid draws per positive. Rows that come up
    short (rare -- only when a user has seen almost the whole catalog)
    are topped up with a second draw. No Python loop over items, ever.
    """
    rng = np.random.default_rng(seed)
    train_u = train_u.numpy()
    train_i = train_i.numpy()
    seen = seen_mask.numpy()
    neg_u_list, neg_i_list = [], []
    # Group row positions by user so each user's draws are one numpy block.
    order = np.argsort(train_u, kind="stable")
    u_sorted = train_u[order]
    uniq_users, starts = np.unique(u_sorted, return_index=True)
    ends = np.append(starts[1:], len(u_sorted))
    for u, s, e in zip(uniq_users, starts, ends):
        n_pos = e - s
        seen_u = seen[u]
        # Draw 2x candidates; collisions with seen items are dropped.
        cand = rng.integers(0, n_items, size=(n_pos, n_neg * 2))
        cand[seen_u[cand]] = -1
        # Keep the first n_neg valid draws per row, in row order.
        valid = cand >= 0
        # Rows are almost always full; top up the rare short rows.
        short = np.where(valid.sum(axis=1) < n_neg)[0]
        for r in short:
            # Keep drawing until the row has n_neg valid negatives. (Only
            # reachable for users who have seen almost the whole catalog;
            # on ML-1M the first extra draw is virtually always enough.)
            fill = list(cand[r][valid[r]])
            while len(fill) < n_neg:
                extra = rng.integers(0, n_items, size=n_neg * 4)
                fill.extend(extra[~seen_u[extra]].tolist())
            cand[r, :n_neg] = np.array(fill[:n_neg])
            valid[r, :n_neg] = True
        picked = np.stack([cand[r][valid[r]][:n_neg] for r in range(n_pos)])
        neg_u_list.append(np.full((n_pos, n_neg), u, dtype=np.int64))
        neg_i_list.append(picked)
    neg_u = np.concatenate([a.reshape(-1) for a in neg_u_list])
    neg_i = np.concatenate([a.reshape(-1) for a in neg_i_list])
    return neg_u, neg_i
