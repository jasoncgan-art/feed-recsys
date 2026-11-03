"""Sample recommendations for one user (same pipeline as serving/app.py).

Usage:  python recommend.py --user_id 1 --top_n 10
Prints the user's recent likes, then the two-stage top-N.
"""

import argparse

import numpy as np
import torch

from serving.app import DATA, INDEX, LOG_ACTIVITY, LOG_POP, GENRE_PROFILE, \
    RANKER, RETRIEVER, TOP_CAND, V_ALL, u_vec_t


def recommend(user_id, top_n=10):
    u = DATA["user_idx"][user_id]
    with torch.no_grad():
        u_vec = RETRIEVER.encode_users(
            torch.tensor([u]), DATA["u_age"][[u]], DATA["u_gender"][[u]],
            DATA["u_occ"][[u]]).numpy().astype("float32")
    _, faiss_idx = INDEX.search(u_vec, 1000)
    seen = DATA["seen"][u].numpy()
    cand = [int(j) for j in faiss_idx[0] if not seen[j]][:TOP_CAND]
    with torch.no_grad():
        cand_t = torch.tensor(cand)
        genre_match = (GENRE_PROFILE[u].unsqueeze(0)
                       * DATA["i_genres"][cand_t]).sum(-1)
        cross = torch.stack([genre_match, LOG_POP[cand_t],
                             LOG_ACTIVITY[u].expand(len(cand))], dim=-1)
        scores = RANKER(u_vec_t(u_vec, len(cand)), V_ALL[cand_t],
                        cross).numpy()
    order = np.argsort(-scores)[:top_n]
    return [(int(DATA["item_ids"][cand[j]]), str(DATA["titles"][cand[j]]),
             float(scores[j])) for j in order]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user_id", type=int, default=1)
    ap.add_argument("--top_n", type=int, default=10)
    args = ap.parse_args()
    if args.user_id not in DATA["user_idx"]:
        raise SystemExit(f"unknown user_id {args.user_id}")
    u = DATA["user_idx"][args.user_id]
    seen_idx = np.where(DATA["seen"][u].numpy())[0]
    print(f"[user {args.user_id}] train positives: {len(seen_idx)}")
    print(f"[recommend] top-{args.top_n} for user {args.user_id}:")
    for rank, (mid, title, score) in enumerate(
            recommend(args.user_id, args.top_n), 1):
        print(f"  {rank:2d}. {title} (id={mid}, score={score:.3f})")


if __name__ == "__main__":
    main()
