"""Compare SFT/GRPO process metrics on fixed scenarios. See tools/README.md.

Reversals, reentries and grip changes are descriptive proxies, not proof of recovery.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from math import comb

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pvla.config import Config, get_device            # noqa: E402
from pvla.data.collect import scenario_at             # noqa: E402
from pvla.evaluate import policy_action               # noqa: E402
from pvla.model.vla import PocketVLA                  # noqa: E402
from pvla.sim.arm2d import Arm2DPickPlace             # noqa: E402

SCEN = "dataset/test.npz"
CKPTS = {
    "SFT": "outputs_sft_3m/train/pocketvla-3m",
    "GRPO": "outputs_grpo_3m/pocketvla-3m-grpo",
}
EPS_NEAR = 0.005     # 5mm 以内的距离变化视为"没动"，滤抖动
BACKTRACK_CM = 0.03  # 累计后退 3cm 才算一次有效回溯


def rollout(model, env, scen, ep, dev):
    """闭环重放一个场景，记录每步 到目标球距离 / 夹爪指令。"""
    obs = env.reset_scenario(scenario_at(scen, ep))
    ds, grips = [], []
    while True:
        a = policy_action(model, obs, dev)
        ds.append(float(np.hypot(*(env.ee - env.target_pos))))
        grips.append(float(a[2]))
        obs, _, done, info = env.step(a)
        if done:
            break
    ds.append(float(np.hypot(*(env.ee - env.target_pos))))
    return np.asarray(ds), np.asarray(grips), info


def behavior(ds, grips, thr):
    r = np.diff(ds)
    near, far = r < -EPS_NEAR, r > EPS_NEAR

    rev = backtrack = 0
    i = 0
    while i < len(r):
        if near[i]:
            j = i
            while j < len(r) and near[j]:
                j += 1
            k, amp = j, 0.0
            while k < len(r) and far[k]:
                amp += r[k]
                k += 1
            if k > j:                       # 接近段之后确实出现了远离段
                rev += 1
                if amp > BACKTRACK_CM:
                    backtrack += 1
            i = max(k, j)
        else:
            i += 1

    closed = grips > 0
    retries = int((closed[:-1] & ~closed[1:]).sum())
    inwin = ds <= thr
    entries = int((~inwin[:-1] & inwin[1:]).sum()) + int(inwin[0])

    return dict(rev=rev, backtrack=backtrack, retries=retries, entries=entries,
                min_d=float(ds.min()), n_steps=len(ds) - 1,
                tmin_frac=float(np.argmin(ds)) / max(len(ds) - 1, 1))


def sign_test(a, b):
    """配对符号检验（双侧，精确二项）。返回 (n_pos, n_neg, p)。"""
    a = np.asarray(a).astype(float)
    b = np.asarray(b).astype(float)
    d = b - a
    pos, neg = int((d > 0).sum()), int((d < 0).sum())
    n = pos + neg
    if n == 0:
        return pos, neg, 1.0
    k = min(pos, neg)
    p = 2.0 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return pos, neg, min(p, 1.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "configs/sft_3m.yaml"))
    ap.add_argument("--scenarios", default=str(ROOT / SCEN))
    ap.add_argument("--sft", default=str(ROOT / CKPTS["SFT"]))
    ap.add_argument("--grpo", default=str(ROOT / CKPTS["GRPO"]))
    ap.add_argument("--output", type=Path, default=ROOT / "analysis/behavior_diff.json")
    args = ap.parse_args()
    cfg = Config.from_yaml(args.config)
    dev = get_device()
    scen = dict(np.load(args.scenarios, allow_pickle=True))
    n = len(scen["q0"])
    thr = cfg.grasp_thresh
    print(f"[cfg] grasp_thresh={thr:.3f}  scenarios={n}  device={dev}")

    runs = {}
    for name, ckpt in {"SFT": args.sft, "GRPO": args.grpo}.items():
        model, _ = PocketVLA.load_hf(ckpt, dev)
        env = Arm2DPickPlace(cfg, seed=cfg.seed)
        rows = []
        for ep in range(n):
            ds, gr, info = rollout(model, env, scen, ep, dev)
            m = behavior(ds, gr, thr)
            m["success"] = bool(info["success"])
            m["grasped_right"] = bool(info["grasped_right"])
            rows.append(m)
            if (ep + 1) % 100 == 0:
                print(f"  [{name}] {ep + 1}/{n}")
        runs[name] = rows
        sr = np.mean([r["success"] for r in rows])
        print(f"[{name}] success = {sr:.1%}")

    KEYS = ["rev", "backtrack", "retries", "entries", "min_d", "tmin_frac"]
    sft = runs["SFT"]
    grpo = runs["GRPO"]

    def table(idx, title):
        if not idx:
            print(f"\n{title}: 空")
            return
        print(f"\n=== {title} (n={len(idx)}) ===")
        print(f"{'metric':<12}{'SFT mean':>10}{'GRPO mean':>11}"
              f"{'SFT med':>9}{'GRPO med':>10}{'sign p':>9}")
        for k in KEYS:
            a = [sft[i][k] for i in idx]
            b = [grpo[i][k] for i in idx]
            _, _, p = sign_test(a, b)
            print(f"{k:<12}{np.mean(a):>10.3f}{np.mean(b):>11.3f}"
                  f"{np.median(a):>9.2f}{np.median(b):>10.2f}{p:>9.3f}")

        for k in ("rev", "backtrack", "retries"):
            any_a = np.mean([sft[i][k] > 0 for i in idx])
            any_b = np.mean([grpo[i][k] > 0 for i in idx])
            print(f"  {k:<10} 出现比例  SFT {any_a:5.1%}  ->  GRPO {any_b:5.1%}")
        re_a = np.mean([sft[i]["entries"] >= 2 for i in idx])
        re_b = np.mean([grpo[i]["entries"] >= 2 for i in idx])
        print(f"  {'entries':<10} 重入(>=2)  SFT {re_a:5.1%}  ->  GRPO {re_b:5.1%}")

    all_idx = list(range(n))
    table(all_idx, "全部场景")

    sft_fail = [i for i in all_idx if not sft[i]["success"]]
    rescued = [i for i in sft_fail if grpo[i]["success"]]
    lost = [i for i in all_idx if sft[i]["success"] and not grpo[i]["success"]]
    both_fail = [i for i in sft_fail if not grpo[i]["success"]]

    table(sft_fail, "SFT 失败的场景")
    table(rescued, "救回组：SFT 失败 -> GRPO 成功")
    table(lost, "丢掉组：SFT 成功 -> GRPO 失败")
    table(both_fail, "共同失败")

    print("\n=== 汇总 ===")
    print(f"SFT 失败 {len(sft_fail)}  救回 {len(rescued)}  "
          f"丢掉 {len(lost)}  共同失败 {len(both_fail)}")
    for k in ("backtrack", "retries"):
        a = [sft[i][k] > 0 for i in all_idx]
        b = [grpo[i][k] > 0 for i in all_idx]
        pos, neg, p = sign_test(a, b)
        print(f"  {k} 出现率 {np.mean(a):.1%} -> {np.mean(b):.1%}  "
              f"新增 {pos} 消失 {neg}  sign p={p:.3f}")

    out = {"thr": thr, "n": n,
           "SFT": sft, "GRPO": grpo,
           "rescued": rescued, "lost": lost}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"\n[saved] {args.output}")


if __name__ == "__main__":
    main()
