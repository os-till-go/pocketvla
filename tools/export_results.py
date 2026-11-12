"""Export existing local evaluations; never runs training or inference.

Run from any directory: python tools/export_results.py
Hashes describe files present at export time, not historical training provenance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = {
    "SFT": ("outputs_sft_3m", "configs/sft_3m.yaml", "train/pocketvla-3m"),
    "GRPO20": ("outputs_grpo_3m", "configs/grpo_3m.yaml", "pocketvla-3m-grpo"),
    "G1": ("outputs_grpo_3m_g1", "configs/grpo_3m_g1.yaml", "pocketvla-3m-grpo"),
    "G3": ("outputs_grpo_3m_g3", "configs/grpo_3m_g3.yaml", "pocketvla-3m-grpo"),
}


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def build_snapshot(root):
    runs = []
    for name, (directory, config, checkpoint) in RUNS.items():
        weights = root / directory / checkpoint / "model.safetensors"
        log_path = root / directory / "grpo_log.json"
        log = json.loads(log_path.read_text(encoding="utf-8")) if log_path.exists() else []
        for split, folder in (("ID", "eval"), ("OOD", "eval_ood")):
            path = root / directory / folder / "eval_report.json"
            report = json.loads(path.read_text(encoding="utf-8"))
            dataset = root / report["test_set"].replace("\\", "/")
            rows = report["results"]
            if len(rows) != report["episodes"]:
                raise ValueError(f"Episode count mismatch: {path}")
            if len({r['episode'] for r in rows}) != len(rows):
                raise ValueError(f"Duplicate episodes: {path}")
            rate = sum(r["success"] for r in rows) / len(rows)
            if abs(rate - report["success_rate"]) > 1e-12:
                raise ValueError(f"Success rate mismatch: {path}")
            runs.append({
                "name": name, "split": split,
                "source_report": path.relative_to(root).as_posix(),
                "source_sha256": digest(path),
                "dataset": dataset.relative_to(root).as_posix(),
                "dataset_sha256_at_export": digest(dataset),
                "config": config, "config_sha256_at_export": digest(root / config),
                "weights_sha256_at_export": digest(weights) if weights.exists() else None,
                "logged_iterations": len(log) if log else None,
                "best_reward_iteration": max(log, key=lambda r: r["mean_reward"])["iteration"] if log else None,
                "episodes": len(rows), "success_rate": rate,
                "grasp_rate": report["grasp_rate"],
                "failure_taxonomy": report["failure_taxonomy"],
                "results": rows,
            })
    return {"schema_version": 1,
            "provenance": "Existing local reports. File hashes taken at export; original runtime and training commit were not embedded in reports. Matching current dataset hashes alone cannot prove historical scenario identity.",
            "runs": runs}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--output", type=Path, default=ROOT / "docs/results/evaluations.json")
    args = ap.parse_args()
    snapshot = build_snapshot(args.root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Exported {len(snapshot['runs'])} reports to {args.output}")


if __name__ == "__main__":
    main()
