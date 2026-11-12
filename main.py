#!/usr/bin/env python3
"""Roblox Up-and-Coming top-10 forecaster (program/v1, stdlib only).

Reads /input/history.ndjson + task.json, writes /output/forecast.json.

Model: empirical rank-transition distributions.  For every pair of reads in the
history we record how a game's rank moved over that gap.  To forecast, for each
candidate game and each horizon h in [0, 72]h we take a kernel-weighted histogram
of the historical rank moves from a similar starting rank over a similar gap
(a heavy-tailed, data-driven diffusion: ranks random-walk, no momentum), with a
small mean-reversion drift toward the rank its playing level implies and a small
drift from its recent playing-growth rate.  The per-horizon distributions are
averaged uniformly over the window (the score is the mean over hourly reads),
leaving an unnamed share per place for games never seen (new entrants).  Games
whose 28-day eligibility expires inside the window (from details.created when
present) lose mass after expiry.  Fully deterministic.
"""
import json
import math
import os
import sys
from datetime import datetime, timezone

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

# ---- tunables (fitted on within-history backtests) -------------------------
HORIZONS = [1, 3, 6, 12, 18, 24, 36, 48, 60, 72]   # hours, averaged uniformly
RANK_KERNEL = 8.0     # rank distance scale for the transition kernel
TIME_KERNEL = 24.0    # hours, gap distance scale for the transition kernel
PRIOR_ALPHA = 2.0     # pseudocount weight of the marginal prior per place
UNNAMED = 0.20        # share of each place left unnamed (new entrants)
MAX_ROWS_PER_PLACE = 60
REVERSION = 0.0       # mean-reversion strength toward playing-implied rank
GROWTH_DRIFT = 1.0    # rank places gained per unit log-playing growth per hour * h
MAX_DRIFT = 15.0
RECENT_WINDOW_H = 72.0  # candidate games: seen within this many hours


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def load_history(input_dir):
    with open(os.path.join(input_dir, "history.ndjson"), "r", encoding="utf-8") as f:
        reads = [json.loads(line) for line in f if line.strip()]
    with open(os.path.join(input_dir, "task.json"), "r", encoding="utf-8") as f:
        task = json.load(f)
    reads.sort(key=lambda r: parse_time(r["at"]))
    return reads, task


def chart_of(read):
    """rank -> (universe_id, playing) sorted by rank."""
    rows = sorted(read.get("up_and_coming") or [], key=lambda g: g["rank"])
    return rows


def build_pairs(reads):
    """All (r1, dt_hours, r2) triples for games present at both reads of a pair,
    plus (r1, dt, None) when the game left the chart.  Only pairs <= 76h apart."""
    pairs = []  # (r1, dt, r2 or None)
    n = len(reads)
    maps = []
    for r in reads:
        m = {}
        for g in chart_of(r):
            m[g["universe_id"]] = (g["rank"], g.get("playing"))
        maps.append(m)
    times = [parse_time(r["at"]) for r in reads]
    for i in range(n):
        for j in range(i + 1, n):
            dt = (times[j] - times[i]) / 3600.0
            if dt <= 0 or dt > 76.0:
                continue
            a, b = maps[i], maps[j]
            for uid, (r1, _p1) in a.items():
                r2 = b[uid][0] if uid in b else None
                pairs.append((r1, dt, r2))
    return pairs


def game_features(reads):
    """Per game: last rank/playing/time, recent log-playing growth per hour,
    playing-implied rank at the last read, expiry timestamp if details exist."""
    times = [parse_time(r["at"]) for r in reads]
    t0 = times[-1]
    last = {}
    series = {}  # uid -> list of (t, rank, playing)
    created = {}
    for idx, r in enumerate(reads):
        for g in chart_of(r):
            uid = g["universe_id"]
            series.setdefault(uid, []).append((times[idx], g["rank"], g.get("playing")))
            last[uid] = (times[idx], g["rank"], g.get("playing"))
        det = r.get("details") or {}
        for uid_s, d in det.items():
            if d and d.get("created"):
                try:
                    created[int(uid_s)] = parse_time(d["created"])
                except Exception:
                    pass
    feats = {}
    # playing-implied rank at the last read (rank by playing, ties by uid)
    last_rows = chart_of(reads[-1])
    by_play = sorted(last_rows, key=lambda g: (-(g.get("playing") or 0), g["universe_id"]))
    implied = {g["universe_id"]: k + 1 for k, g in enumerate(by_play)}
    for uid, (t_last, r_last, p_last) in last.items():
        if t0 - t_last > RECENT_WINDOW_H * 3600:
            continue
        # log-playing growth per hour over the last ~48h of observations
        pts = [(t, p) for (t, _r, p) in series[uid] if p and p > 0 and t >= t0 - 48 * 3600]
        g_rate = 0.0
        if len(pts) >= 2:
            (t_a, p_a), (t_b, p_b) = pts[0], pts[-1]
            dt = (t_b - t_a) / 3600.0
            if dt >= 3.0:
                g_rate = (math.log(p_b) - math.log(p_a)) / dt
        g_rate = max(-0.2, min(0.2, g_rate))
        expiry = None
        if uid in created:
            expiry = created[uid] + 28 * 86400.0
        feats[uid] = {
            "r0": r_last,
            "t_last": t_last,
            "stale_h": (t0 - t_last) / 3600.0,
            "implied": implied.get(uid, r_last),
            "growth": g_rate,
            "expiry": expiry,
        }
    return feats, t0


def forecast(reads, task):
    places = int(task.get("places", 10))
    pairs = build_pairs(reads)
    feats, t0 = game_features(reads)
    max_rank = 0
    for (r1, _dt, _r2) in pairs:
        if r1 > max_rank:
            max_rank = r1
    max_rank = max(max_rank, 100)
    # Per horizon: A[r1][k] = time-weighted count of moves r1 -> place k,
    # D[r1] = time-weighted count of all pairs starting at r1 (incl. off-chart).
    horizons = []
    for h in HORIZONS:
        A = [[0.0] * places for _ in range(max_rank + 1)]
        D = [0.0] * (max_rank + 1)
        for (r1, dt, r2) in pairs:
            w = math.exp(-((dt - h) / TIME_KERNEL) ** 2)
            if w < 1e-6:
                continue
            D[r1] += w
            if r2 is not None and 1 <= r2 <= places:
                A[r1][r2 - 1] += w
        prior = [0.0] * places
        tot = sum(D)
        if tot > 0:
            for k in range(places):
                prior[k] = sum(A[r1][k] for r1 in range(max_rank + 1)) / tot
        horizons.append((h, A, D, prior))
    prob = {uid: [0.0] * places for uid in feats}
    n_h = len(HORIZONS)
    for (h, A, D, prior) in horizons:
        for uid, f in feats.items():
            drift = -REVERSION * (f["r0"] - f["implied"]) * (1.0 - math.exp(-h / 48.0))
            drift += -GROWTH_DRIFT * f["growth"] * h
            drift = max(-MAX_DRIFT, min(MAX_DRIFT, drift))
            r_eff = f["r0"] + drift
            h_eff = h + f["stale_h"]
            if f["expiry"] is not None and t0 + h * 3600.0 >= f["expiry"]:
                continue
            # stale games use the matrix of the nearest horizon to h_eff
            if f["stale_h"] > 0:
                hi_idx = min(range(n_h), key=lambda i: abs(HORIZONS[i] - h_eff))
                _h2, A, D, prior = horizons[hi_idx]
            # kernel over starting ranks, reusing the aggregated matrices
            num = [0.0] * places
            den = 0.0
            lo = max(1, int(r_eff - 3 * RANK_KERNEL))
            hi = min(max_rank, int(r_eff + 3 * RANK_KERNEL) + 1)
            for r1 in range(lo, hi + 1):
                kw = math.exp(-((r1 - r_eff) / RANK_KERNEL) ** 2)
                d = D[r1]
                if d <= 0:
                    continue
                den += kw * d
                row = A[r1]
                for k in range(places):
                    if row[k]:
                        num[k] += kw * row[k]
            if den <= 0:
                continue
            for k in range(places):
                p = (num[k] + PRIOR_ALPHA * prior[k]) / (den + PRIOR_ALPHA)
                prob[uid][k] += p / n_h
    # per-place rows, scaled so each place sums to 1 - UNNAMED
    probabilities = []
    for k in range(places):
        cands = sorted(
            ((prob[uid][k], uid) for uid in prob if prob[uid][k] > 0.0005),
            key=lambda x: (-x[0], x[1]),
        )[:MAX_ROWS_PER_PLACE]
        total = sum(p for p, _ in cands)
        target = 1.0 - UNNAMED
        scale = target / total if total > target else 1.0
        for p, uid in cands:
            q = p * scale
            if q >= 0.001:
                probabilities.append({"universe_id": uid, "place": k + 1, "p": round(q, 6)})
    # top10: highest total top-10 probability
    totals = sorted(
        ((sum(prob[uid]), uid) for uid in prob),
        key=lambda x: (-x[0], x[1]),
    )
    top10 = [uid for _s, uid in totals[:places]]
    # pad top10 from last chart order if short (should not happen)
    if len(top10) < places:
        seen = set(top10)
        for g in chart_of(reads[-1]):
            if g["universe_id"] not in seen:
                top10.append(g["universe_id"])
                seen.add(g["universe_id"])
            if len(top10) >= places:
                break
    probabilities.sort(key=lambda r: (r["place"], -r["p"], r["universe_id"]))
    return {"probabilities": probabilities, "top10": top10[:places]}


def main():
    reads, task = load_history(INPUT_DIR)
    out = forecast(reads, task)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, task.get("output", "forecast.json")), "w", encoding="utf-8") as f:
        f.write(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()
