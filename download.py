"""Download MovieLens 1M and write small processed CSVs.

Source: https://files.grouplens.org/datasets/movielens/ml-1m.zip
(6,040 users, 3,883 movies with IDs up to 3952, ~1M ratings -- big enough
to feel real, small enough to train on a laptop CPU.)

Download reliability: GroupLens refuses connections from some datacenter
IPs (empty reply). If the GroupLens download fails, we fall back to the
same canonical files mirrored on GitHub (see MIRROR_BASE), then verify
line counts against EXPECTED_LINES before proceeding -- a wrong file
fails loudly here instead of silently corrupting training later.

ML-1M's raw files are "::" separated and latin-1 encoded:
    ratings.dat : UserID::MovieID::Rating::Timestamp
    users.dat   : UserID::Gender::Age::Occupation::Zip-code
    movies.dat  : MovieID::Title (Year)::Genres

Two ML-1M quirks worth knowing (they change how dataset.py encodes features):
    * Age is already a CODE, not a year count: 1=Under 18, 18=18-24,
      25=25-34, 35=35-44, 45=45-49, 50=50-55, 56=56+. These seven codes line
      up exactly with the seven age buckets used by the ML-100K project, so
      dataset.py maps each code to a bucket index 0..6.
    * Occupation is also a CODE (0-20), not a string. We keep the code as
      the embedding id (21 distinct occupations). The model is trained from
      scratch on ML-1M, so the code's meaning only has to be consistent
      inside this dataset -- it does not have to match ML-100K's strings.

Design choice (same as the ML-100K project): keep only ratings >= 4 stars
and treat them as positive implicit feedback ("the user liked this").
Ratings 1-3 are dropped, not used as negatives -- a low rating still means
the user *saw* the item, which is a different signal from "never
interacted". Dropping them keeps the semantics clean: every row in
interactions.csv is a positive example.

Outputs (all under ./data/):
    data/ml-1m/            extracted raw files (ratings.dat, users.dat, ...)
    data/interactions.csv  user_id,item_id,rating,timestamp  (ratings >= 4 only)
    data/movies.csv        item_id,title,year,genres  (genres pipe-separated)
    data/users.csv         user_id,age_code,gender,occupation_code
"""

import csv
import io
import os
import re
import urllib.request
import zipfile

DATA_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
# Same canonical files, mirrored on GitHub (fallback when GroupLens refuses).
MIRROR_BASE = ("https://raw.githubusercontent.com/qiongwu86/"
                "Edge-Caching-Based-on-Multi-Agent-Deep-Reinforcement-"
                "Learning-and-Federated-Learning/main/ml-1m")
# Canonical ML-1M sizes: 1,000,209 ratings, 6,040 users, 3,883 movies.
EXPECTED_LINES = {"ratings.dat": 1000209, "movies.dat": 3883, "users.dat": 6040}
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "ml-1m")

TITLE_YEAR_RE = re.compile(r"^(.*) \((\d{4})\)\s*$")


def _count_lines(path):
    with open(path, encoding="latin-1") as f:
        return sum(1 for _ in f)


def verify_raw(dest=RAW_DIR):
    """Fail loudly if any raw file is missing or has the wrong line count."""
    for name, expected in EXPECTED_LINES.items():
        path = os.path.join(dest, name)
        if not os.path.exists(path):
            raise SystemExit(f"[download] missing {path}")
        got = _count_lines(path)
        if got != expected:
            raise SystemExit(
                f"[download] {name}: expected {expected} lines, got {got} "
                f"-- file is corrupt or not canonical ML-1M, refusing to continue")
    print("[download] raw files verified against canonical ML-1M line counts")


def download_mirror(dest=RAW_DIR):
    """Download the three .dat files individually from the GitHub mirror."""
    os.makedirs(dest, exist_ok=True)
    for name in EXPECTED_LINES:
        url = f"{MIRROR_BASE}/{name}"
        print(f"[download] mirror fetching {url} ...")
        with urllib.request.urlopen(url) as resp:
            blob = resp.read()
        with open(os.path.join(dest, name), "wb") as f:
            f.write(blob)
    print(f"[download] mirror files written to {dest}")


def download_and_extract(url=DATA_URL, dest=RAW_DIR):
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(os.path.join(dest, "ratings.dat")):
        try:
            verify_raw(dest)
            print(f"[download] raw data already present at {dest}, skipping download")
            return
        except SystemExit:
            print("[download] present raw files failed verification, re-downloading")
    print(f"[download] fetching {url} ...")
    try:
        with urllib.request.urlopen(url) as resp:
            blob = resp.read()
        print(f"[download] got {len(blob) / 1e6:.1f} MB, extracting ...")
        with zipfile.ZipFile(io.BytesIO(blob)) as zf:
            zf.extractall(DATA_DIR)
        print(f"[download] extracted to {dest}")
    except Exception as e:  # GroupLens often refuses datacenter IPs
        print(f"[download] GroupLens failed ({e}); falling back to GitHub mirror")
        download_mirror(dest)
    verify_raw(dest)


def write_interactions(raw=RAW_DIR, out=os.path.join(DATA_DIR, "interactions.csv")):
    """ratings.dat is UserID::MovieID::Rating::Timestamp. Keep rating >= 4."""
    n_all, n_kept = 0, 0
    with open(os.path.join(raw, "ratings.dat"), encoding="latin-1") as f, \
            open(out, "w", newline="") as g:
        w = csv.writer(g)
        w.writerow(["user_id", "item_id", "rating", "timestamp"])
        for line in f:
            user_id, item_id, rating, timestamp = line.rstrip("\n").split("::")
            n_all += 1
            if int(rating) >= 4:  # positive implicit feedback
                w.writerow([user_id, item_id, rating, timestamp])
                n_kept += 1
    print(f"[process] interactions.csv: kept {n_kept}/{n_all} ratings >= 4")


def write_movies(raw=RAW_DIR, out=os.path.join(DATA_DIR, "movies.csv")):
    """movies.dat is MovieID::Title (Year)::Genres. Split the year off the title."""
    with open(os.path.join(raw, "movies.dat"), encoding="latin-1") as f, \
            open(out, "w", newline="", encoding="utf-8") as g:
        w = csv.writer(g)
        w.writerow(["item_id", "title", "year", "genres"])
        for line in f:
            item_id, title_year, genres = line.rstrip("\n").split("::")
            m = TITLE_YEAR_RE.match(title_year)
            title, year = (m.group(1), m.group(2)) if m else (title_year, "")
            w.writerow([item_id, title, year, genres])
    print("[process] movies.csv written")


def write_users(raw=RAW_DIR, out=os.path.join(DATA_DIR, "users.csv")):
    """users.dat is UserID::Gender::AgeCode::OccupationCode::Zip. Drop zip."""
    with open(os.path.join(raw, "users.dat"), encoding="latin-1") as f, \
            open(out, "w", newline="") as g:
        w = csv.writer(g)
        w.writerow(["user_id", "age_code", "gender", "occupation_code"])
        for line in f:
            user_id, gender, age_code, occ_code, _zip = line.rstrip("\n").split("::")
            w.writerow([user_id, age_code, gender, occ_code])
    print("[process] users.csv written")


def main():
    download_and_extract()
    write_interactions()
    write_movies()
    write_users()
    print("[done] data ready in", DATA_DIR)


if __name__ == "__main__":
    main()
