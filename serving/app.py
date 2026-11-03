"""FastAPI serving for the two-stage feed recommender.

Run:  bash run_serving.sh        (uvicorn on port 8000)
Then: curl "http://localhost:8000/recommend/1?top_n=10"

Pipeline per request (the same two stages as offline eval):
    1. Encode the user with the stage-1 user tower.
    2. FAISS top-200 candidates (over-fetch, then drop train-seen items).
       Train-seen items are never recommended -- same honesty rule as eval.
    3. Score the candidates with the stage-2 ranker, return the top-N.

Everything is loaded once at startup; per-request work is one tower
forward + one FAISS lookup + one small ranker forward.
"""

import os

import faiss
import numpy as np
import torch
from fastapi import FastAPI, HTTPException

from models import Ranker
from train import build_cross_tables, load_twotower

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART_DIR = os.path.join(BASE_DIR, "artifacts")
TOP_CAND = 200
OVER_FETCH = 1000

app = FastAPI(title="Two-Stage Feed Recommender")


def _load():
    data = torch.load(os.path.join(ART_DIR, "dataset.pt"), weights_only=False)
    retriever = load_twotower(data, "retriever.pt")
    rart = torch.load(os.path.join(ART_DIR, "ranker.pt"), weights_only=True)
    ranker = Ranker(vec_dim=rart["vec_dim"])
    ranker.load_state_dict(rart["state_dict"])
    ranker.eval()
    index = faiss.read_index(os.path.join(ART_DIR, "item_index.faiss"))
    genre_profile, log_pop, log_activity = build_cross_tables(data)
    with torch.no_grad():
        all_i = torch.arange(len(data["item_ids"]))
        v_all = retriever.encode_items(all_i, data["i_genres"],
                                       data["i_year"])
    return data, retriever, ranker, index, v_all, (genre_profile, log_pop, log_activity)


DATA, RETRIEVER, RANKER, INDEX, V_ALL, CROSS = _load()
GENRE_PROFILE, LOG_POP, LOG_ACTIVITY = CROSS


@app.get("/health")
def health():
    return {"status": "ok", "items_indexed": INDEX.ntotal}


@app.get("/recommend/{user_id}")
def recommend(user_id: int, top_n: int = 10):
    if user_id not in DATA["user_idx"]:
        raise HTTPException(status_code=404, detail=f"unknown user_id {user_id}")
    u = DATA["user_idx"][user_id]
    with torch.no_grad():
        u_vec = RETRIEVER.encode_users(
            torch.tensor([u]), DATA["u_age"][[u]], DATA["u_gender"][[u]],
            DATA["u_occ"][[u]]).numpy().astype("float32")
    # FAISS candidates, then drop anything the user already saw in training.
    _, faiss_idx = INDEX.search(u_vec, OVER_FETCH)
    seen = DATA["seen"][u].numpy()
    cand = [int(j) for j in faiss_idx[0] if not seen[j]][:TOP_CAND]
    if not cand:
        return {"user_id": user_id, "recommendations": []}
    with torch.no_grad():
        cand_t = torch.tensor(cand)
        genre_match = (GENRE_PROFILE[u].unsqueeze(0)
                       * DATA["i_genres"][cand_t]).sum(-1)
        cross = torch.stack([genre_match, LOG_POP[cand_t],
                             LOG_ACTIVITY[u].expand(len(cand))], dim=-1)
        scores = RANKER(u_vec_t(u_vec, len(cand)), V_ALL[cand_t],
                        cross).numpy()
    order = np.argsort(-scores)[:top_n]
    recs = [{"rank": r + 1,
             "item_id": int(DATA["item_ids"][cand[j]]),
             "title": str(DATA["titles"][cand[j]]),
             "score": float(scores[j])} for r, j in enumerate(order)]
    return {"user_id": user_id, "recommendations": recs}


def u_vec_t(u_vec, n):
    """Repeat one user vector n times (ranker scores one row per candidate)."""
    return torch.tensor(u_vec).expand(n, -1)
