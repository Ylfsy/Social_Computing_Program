"""run_mc_experiments.py

用于“每组参数重复跑 N 次（30/50/100 次）并输出均值±95%CI”的脚本。

做法（保证策略对比公平）：
- 每个重复(rep)先生成同一张 SBM 图
- 在这张图上跑社区发现，得到 node->community
- 根据社区划分识别关键社区
- 固定同一个初始感染者（该 rep 内三种策略共用）
- 分别运行 random / equal / community_priority
- 收集指标并对每个 (beta,gamma,strategy) 做均值与 95% 置信区间

输出：
- --out_raw:    每次模拟的原始结果（便于追溯）
- --out_summary:均值 + 95%CI 汇总表（直接用于论文表格）

"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

# SciPy 有则用 t 分布（小样本更稳），没有则退化为正态近似 1.96
try:
    from scipy import stats  # type: ignore

    def _t_critical(df: int, confidence: float) -> float:
        return float(stats.t.ppf((1.0 + confidence) / 2.0, df=df))

except Exception:  # pragma: no cover

    def _t_critical(df: int, confidence: float) -> float:
        # df>=30 时，t 分布与正态非常接近；这里用 1.96 近似 95% CI
        return 1.96 if df > 0 else 0.0

from community_detection import detect_communities, find_critical_communities
from generate_lfr import generate_sbm_graph
from sir_model import SIRMetrics, run_sir


def _parse_float_list(s: str) -> List[float]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return [float(p) for p in parts]


def _mean_ci(x: Sequence[float], confidence: float = 0.95) -> Dict[str, float]:
    arr = np.asarray(list(x), dtype=float)
    n = int(arr.size)
    if n == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    mean = float(arr.mean())
    if n == 1:
        return {"mean": mean, "ci_low": mean, "ci_high": mean}
    se = float(arr.std(ddof=1) / np.sqrt(n))
    t_crit = _t_critical(df=n - 1, confidence=confidence)
    half = t_crit * se
    return {"mean": mean, "ci_low": mean - half, "ci_high": mean + half}


def main() -> None:
    parser = argparse.ArgumentParser(description="Monte-Carlo batch runner with mean±95%CI outputs")

    # Monte-Carlo
    parser.add_argument("--reps", type=int, default=30, help="repetitions per (beta,gamma)")
    parser.add_argument("--seed", type=int, default=12345, help="base seed")

    # Graph (SBM)
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--p_in", type=float, default=0.05)
    parser.add_argument("--p_out", type=float, default=0.005)

    # Community detection
    parser.add_argument("--cd_method", type=str, default="louvain", help="louvain or lpa")
    parser.add_argument("--top_k", type=int, default=1, help="number of critical communities")

    # SIR
    parser.add_argument("--betas", type=str, default="0.05,0.1,0.2")
    parser.add_argument("--gammas", type=str, default="0.05,0.1,0.2")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--tests", type=int, default=1000, help="total_detection_capacity per step")
    parser.add_argument("--beds", type=int, default=500, help="total_bed_capacity per step")
    parser.add_argument("--priority_fraction", type=float, default=0.7)
    parser.add_argument("--treatment_multiplier", type=float, default=2.0)

    # Output
    parser.add_argument("--out_raw", type=str, default="sir_raw_runs.csv")
    parser.add_argument("--out_summary", type=str, default="sir_summary_ci.csv")

    args = parser.parse_args()

    betas = _parse_float_list(args.betas)
    gammas = _parse_float_list(args.gammas)

    strategies = ["random", "equal", "community_priority"]

    raw_rows = []

    for beta in betas:
        for gamma in gammas:
            for rep in range(int(args.reps)):
                rep_seed = int(args.seed + rep + int(beta * 10_000) + int(gamma * 100_000))

                # 1) same graph for all strategies in this rep
                G, _true_map = generate_sbm_graph(
                    n=args.n,
                    n_communities=args.k,
                    p_in=args.p_in,
                    p_out=args.p_out,
                    seed=rep_seed,
                )

                # 2) community detection (for community_priority)
                _communities, pred_map = detect_communities(G, method=args.cd_method, seed=rep_seed)
                critical = find_critical_communities(G, pred_map, top_k=args.top_k)

                # 3) fixed initial infected for this rep
                rng0 = np.random.default_rng(rep_seed)
                initial_infected = [int(rng0.choice(list(G.nodes())))]

                for s_idx, strategy in enumerate(strategies):
                    # derived seed to keep runs reproducible and independent
                    sim_seed = rep_seed * 1000 + s_idx * 97

                    metrics: SIRMetrics = run_sir(
                        G=G,
                        beta=beta,
                        gamma=gamma,
                        strategy=strategy,
                        total_detection_capacity=int(args.tests),
                        total_bed_capacity=int(args.beds),
                        steps=int(args.steps),
                        seed=sim_seed,
                        initial_infected=initial_infected,
                        node_to_comm=pred_map,
                        critical_comms=critical,
                        priority_fraction=float(args.priority_fraction),
                        treatment_multiplier=float(args.treatment_multiplier),
                    )

                    row = {
                        "beta": beta,
                        "gamma": gamma,
                        "rep": rep,
                        "rep_seed": rep_seed,
                        "strategy": strategy,
                        "cd_method": args.cd_method,
                        "top_k": args.top_k,
                        "critical_comms": ";".join(map(str, critical)),
                        "initial_infected": ";".join(map(str, initial_infected)),
                        "n": G.number_of_nodes(),
                        "m": G.number_of_edges(),
                        "p_in": args.p_in,
                        "p_out": args.p_out,
                        "tests_per_step": int(args.tests),
                        "beds_per_step": int(args.beds),
                        "steps": int(args.steps),
                        "priority_fraction": float(args.priority_fraction),
                        "treatment_multiplier": float(args.treatment_multiplier),
                    }
                    row.update(
                        {
                            "total_infected": metrics.total_infected,
                            "max_infected": metrics.max_infected,
                            "transmission_duration": metrics.transmission_duration,
                            "detection_utilization": metrics.detection_utilization,
                            "bed_utilization": metrics.bed_utilization,
                        }
                    )
                    raw_rows.append(row)

    raw_df = pd.DataFrame(raw_rows)
    raw_df.to_csv(args.out_raw, index=False)

    # 汇总：每个 (beta,gamma,strategy) 的均值与 95%CI
    metric_cols = [
        "total_infected",
        "max_infected",
        "transmission_duration",
        "detection_utilization",
        "bed_utilization",
    ]

    summary_rows = []
    group_keys = ["beta", "gamma", "strategy"]

    bounds = {
        "total_infected": (0.0, float(args.n)),
        "max_infected": (0.0, float(args.n)),
        "transmission_duration": (0.0, float(args.steps)),
        "detection_utilization": (0.0, 1.0),
        "bed_utilization": (0.0, 1.0),
    }

    for (beta, gamma, strategy), g in raw_df.groupby(group_keys, sort=True):
        out = {
            "beta": beta,
            "gamma": gamma,
            "strategy": strategy,
            "n_runs": int(len(g)),
            "cd_method": str(g["cd_method"].iloc[0]),
            "top_k": int(g["top_k"].iloc[0]),
            "p_in": float(g["p_in"].iloc[0]),
            "p_out": float(g["p_out"].iloc[0]),
            "tests_per_step": int(g["tests_per_step"].iloc[0]),
            "beds_per_step": int(g["beds_per_step"].iloc[0]),
            "steps": int(g["steps"].iloc[0]),
        }

        for col in metric_cols:
            stat = _mean_ci(g[col].astype(float).tolist(), confidence=0.95)
            lo, hi = bounds.get(col, (-float("inf"), float("inf")))
            out[f"{col}_mean"] = min(max(stat["mean"], lo), hi)
            out[f"{col}_ci_low"] = min(max(stat["ci_low"], lo), hi)
            out[f"{col}_ci_high"] = min(max(stat["ci_high"], lo), hi)
        summary_rows.append(out)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.sort_values(["beta", "gamma", "strategy"], inplace=True)
    summary_df.to_csv(args.out_summary, index=False)

    print("Saved:")
    print(f"  raw    -> {args.out_raw}")
    print(f"  summary-> {args.out_summary}")


if __name__ == "__main__":
    main()
