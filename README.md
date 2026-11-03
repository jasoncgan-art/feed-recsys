# Two-Stage Feed Recommender (MovieLens 1M)

A TikTok-style feed recommendation pipeline in two stages, built as the
upgrade to the [recsys-tower](../recsys-tower) single-stage two-tower
project. Same data discipline (temporal leave-last-out, train-seen items
always masked at eval), one new architectural idea: **retrieval and
ranking are different jobs, so they get different models.**

```
user ──► UserTower ──┐
                      ├─► FAISS top-200 ──► Ranker MLP ──► top-10
item ──► ItemTower ──┘   (retrieval)         (ranking)
```

## Why two stages?

One model cannot be both fast and smart. Scoring every item in the
catalog with a rich ranker costs O(catalog) per request -- fine for
3,883 movies, impossible for 100 million videos. Production systems
split the job:

- **Stage 1, retrieval**: a cheap model (two-tower dot product) narrows
  the whole catalog to a few hundred candidates. It only needs to be
  *good enough not to lose the right item* -- measured by Recall@200.
- **Stage 2, ranking**: an expressive model (MLP over retrieval vectors
  + cross features) scores just those candidates. It only needs to put
  the best ones on top -- measured by NDCG@10 and AUC.

The two stages are evaluated separately (so you can see which one is
the bottleneck) and end-to-end (so you can see the system users get).

## Honesty note (read this before quoting numbers)

At MovieLens 1M scale, brute-force scoring of all 3,883 items still
works fine. The FAISS index here uses `IndexFlatIP`, which is *exact*
search, not approximate -- we verified 100% top-200 agreement with
brute force on 500 sampled test users. FAISS in this project
demonstrates the **architecture** (encode once, index, query) that
becomes *necessary* at real catalog scale, where IVF/PQ approximation
trades a little recall for orders-of-magnitude speed. Claiming "ANN
made this fast" at 3,883 items would be dishonest; claiming "the
pipeline is built so ANN can slot in" is the true statement.

## Data

MovieLens 1M: 6,040 users, 3,883 movies (IDs up to 3952), 1,000,209
ratings. Ratings >= 4 stars are positives (575,281); ratings 1-3 are
dropped, not treated as negatives (a low rating still means the user
*saw* the item). Split: temporal leave-last-out per user -- each
user's last positive interaction is the test target (6,037 test users,
569,243 train positives).

ML-1M quirks handled in code: age and occupation are *codes*, not raw
values (age codes 1/18/25/35/45/50/56 map to the same seven buckets the
ML-100K project used; occupation codes 0-20 are used directly as
embedding ids). Movie titles carry the year in parentheses
("Toy Story (1995)"), split off in `download.py`.

## Models

**Stage 1 -- TwoTower** (`models.py`): user tower [user_id | age | gender
| occupation] and item tower [item_id | genre multi-hot | year], each an
MLP to a 64-dim L2-normalized vector. Trained with in-batch sampled
softmax (batch 1024, temperature 0.1, Adam lr 5e-3, 25 epochs) with two
corrections: logQ (Yi et al., 2019) for in-batch popularity bias, and a
neg_mask that removes false negatives (a user's other liked items
appearing as in-batch negatives) from the softmax denominator.

**Stage 2 -- Ranker** (`models.py`): an MLP over [user_vec (64) |
item_vec (64) | 3 cross features] -> score. The towers are *frozen* --
the ranker learns a scoring function on top of retrieval
representations, it does not re-learn embeddings. Cross features:
genre_match (dot of the user's taste profile with the item's genres),
log_pop (item popularity), log_activity (user activity). Trained with
BCE on train positives + 4 sampled negatives each (Adam lr 1e-3,
5 epochs).

**Baseline**: a second two-tower with identical config and a different
seed (123), used as a *single-stage* system (dot-product top-10 over
all unseen items). Same data, same split -- the only difference from
the two-stage pipeline is the stage-2 ranker, so the Recall@10 gap
isolates the ranker's contribution (plus ordinary training noise).

## Results (measured on the ML-1M split, 6,037 test users)

All numbers below are from a full `run_all.sh` run on this machine
(2-CPU, torch CPU); `artifacts/metrics.json` holds the same values.
Training: retriever 25 epochs (in-batch softmax 6.88 -> 5.97), ranker
5 epochs (BCE 0.29 -> 0.23), baseline 25 epochs (6.85 -> 5.97).

| Metric | Value |
|---|---|
| Retrieval Recall@200 | 0.5455 |
| End-to-end Recall@10 (two-stage) | 0.0762 |
| NDCG@10 (end-to-end) | 0.0361 |
| NDCG@10 (ranker-only) | 0.0662 |
| AUC (ranker-only) | 0.6673 |
| Baseline Recall@10 (single-stage two-tower) | 0.0730 |

How to read this table: Recall@200 is the pipeline's ceiling -- every
held-out item retrieval misses is invisible to the ranker. NDCG@10
(ranker-only) vs NDCG@10 (end-to-end) shows how much the retrieval
bottleneck costs. Baseline Recall@10 vs end-to-end Recall@10 shows
whether the ranker earned its keep: here the two-stage pipeline beats
the single-stage baseline 0.0762 vs 0.0730 (+4.4% relative) -- a modest,
real gain, not a blowout. The honest headline is that **retrieval is
the bottleneck** (0.5455 recall lets 45% of held-out items never reach
the ranker); improving retrieval (more epochs, hard-negative mining)
is the highest-leverage next step, and hard-negative mining is already
the stated v2 plan. These numbers are on ML-1M and are NOT comparable
to ML-100K figures (different catalog size, different user activity);
no ML-100K number is quoted anywhere in this repo.

## How to run

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
bash run_all.sh          # download -> train -> evaluate -> sample recs
bash run_serving.sh      # FastAPI on port 8000
curl "http://localhost:8000/recommend/1?top_n=10"
```

Note on download: `download.py` tries GroupLens first. GroupLens
refuses connections from some datacenter IPs; on failure it falls back
to the same canonical files mirrored on GitHub, then verifies line
counts (1,000,209 / 6,040 / 3,883) before proceeding.

## Files

| File | What it does |
|---|---|
| `download.py` | Fetch ML-1M, write processed CSVs |
| `dataset.py` | Temporal split, features, fast negative sampler |
| `models.py` | UserTower / ItemTower / TwoTower / Ranker |
| `train.py` | Retriever, ranker, baseline, FAISS index |
| `evaluate.py` | Retrieval, ranking, end-to-end, baseline metrics |
| `serving/app.py` | FastAPI top-10 endpoint (retrieval + ranking) |
| `recommend.py` | CLI sample recommendations for one user |
| `run_all.sh` | Full pipeline end to end |
