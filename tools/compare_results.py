"""Summarize archived results and compare paired episodes (standard library only)."""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def wilson(k, n, z=1.96):
    if not 0 <= k <= n or n <= 0:
        raise ValueError("Require 0 <= successes <= episodes, episodes > 0")
    p = k / n
    center = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    denom = 1 + z * z / n
    return (center - half) / denom, (center + half) / denom


def mcnemar_exact(gained, lost):
    n = gained + lost
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(gained, lost) + 1)) / 2**n)


def paired_outcomes(a, b):
    if a["split"] != b["split"] or not a["dataset_sha256_at_export"] or a["dataset_sha256_at_export"] != b["dataset_sha256_at_export"]:
        raise ValueError("Paired reports must share a split and dataset hash")
    maps = []
    for run in (a, b):
        rows = run["results"]
        mapping = {r["episode"]: r["success"] for r in rows}
        if len(mapping) != len(rows) or len(rows) != run["episodes"] or not rows:
            raise ValueError("Duplicate, missing, or empty episodes")
        if any(type(v) is not bool for v in mapping.values()):
            raise ValueError("Success values must be booleans")
        maps.append(mapping)
    if maps[0].keys() != maps[1].keys():
        raise ValueError("Paired reports must contain identical episode IDs")
    ids = sorted(maps[0])
    return [maps[0][i] for i in ids], [maps[1][i] for i in ids]


def compare(a, b, n_boot=5000):
    if n_boot <= 0:
        raise ValueError("Bootstrap count must be positive")
    sa, sb = paired_outcomes(a, b)
    diffs = [int(y) - int(x) for x, y in zip(sa, sb)]
    gained, lost = diffs.count(1), diffs.count(-1)
    rng = random.Random(0)
    means = sorted(sum(rng.choices(diffs, k=len(diffs))) / len(diffs) for _ in range(n_boot))
    return {"n": len(diffs), "delta_pp": 100 * sum(diffs) / len(diffs),
            "gained": gained, "lost": lost, "mcnemar_p": mcnemar_exact(gained, lost),
            "bootstrap_95_pp": [100 * means[int(q * n_boot)] for q in (0.025, 0.975)]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--snapshot", type=Path, default=ROOT / "docs/results/evaluations.json")
    ap.add_argument("--bootstrap", type=int, default=5000)
    args = ap.parse_args()
    runs = json.loads(args.snapshot.read_text(encoding="utf-8"))["runs"]
    index = {(r["name"], r["split"]): r for r in runs}
    print("| Run | Split | Success | Wilson 95% CI | Grasp |")
    print("| --- | --- | --- | --- | --- |")
    for run in runs:
        n = run["episodes"]
        k = sum(r["success"] for r in run["results"])
        lo, hi = wilson(k, n)
        print(f"| {run['name']} | {run['split']} | {k}/{n} ({100*k/n:.1f}%) | {100*lo:.1f}-{100*hi:.1f}% | {100*run['grasp_rate']:.1f}% |")
    print("\n| Comparison (B - A) | Split | Delta (pp) | Gained / lost | McNemar p | Bootstrap 95% CI (pp) |")
    print("| --- | --- | --- | --- | --- | --- |")
    for split in ("ID", "OOD"):
        for a, b in (("SFT", "GRPO20"), ("SFT", "G1"), ("SFT", "G3"), ("G1", "G3")):
            d = compare(index[a, split], index[b, split], args.bootstrap)
            lo, hi = d["bootstrap_95_pp"]
            print(f"| {b} - {a} | {split} | {d['delta_pp']:+.1f} | {d['gained']} / {d['lost']} | {d['mcnemar_p']:.4f} | [{lo:+.1f}, {hi:+.1f}] |")
    print("\nPairing assumes the saved reports replayed the same scenarios. Export-time hashes are not historical provenance. CIs cover scenario sampling, not training-seed variation; p-values are unadjusted exploratory comparisons.")


if __name__ == "__main__":
    main()
