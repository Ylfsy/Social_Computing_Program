"""sir_model.py

改造后的 SIR + 资源约束仿真模块：
- 三种策略：random / equal / community_priority
- 检测资源：每步最多测试 total_detection_capacity 人；检测到的感染者进入隔离(不再传播)
- 床位资源：每步最多给 total_bed_capacity 个感染者提供治疗；治疗者恢复概率提升
- 资源分配通过“节点权重”实现，且总量守恒（权重归一化）

该文件只提供函数，不会在 import 时自动跑实验。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import networkx as nx


@dataclass(frozen=True)
class SIRMetrics:
    infected_history: List[int]
    total_infected: int
    max_infected: int
    transmission_duration: int
    detection_utilization: float  # positive detections / (tests capacity * steps)
    bed_utilization: float        # treated infections / (beds capacity * steps)


def _normalize_weights(w: np.ndarray) -> np.ndarray:
    s = float(w.sum())
    if s <= 0:
        # 全 0 或非法时退化为均匀
        return np.ones_like(w, dtype=float) / len(w)
    return w / s


def _weighted_sample_without_replacement(
    rng: np.random.Generator,
    items: Sequence[int],
    weights: np.ndarray,
    k: int,
) -> List[int]:
    """按权重无放回抽样。"""
    if k <= 0:
        return []
    k = min(k, len(items))
    p = _normalize_weights(weights.astype(float))
    chosen = rng.choice(np.asarray(items), size=k, replace=False, p=p)
    return [int(x) for x in chosen]


def build_node_weights(
    G: nx.Graph,
    strategy: str,
    seed: Optional[int] = None,
    node_to_comm: Optional[Dict[int, int]] = None,
    critical_comms: Optional[Sequence[int]] = None,
    priority_fraction: float = 0.7,
) -> Tuple[Dict[int, float], Dict[int, float]]:
    """构造检测/床位的“节点权重”。权重用于决定资源更偏向哪些节点。

    - equal: 所有节点权重相等
    - random: 给每个节点随机权重（一次性生成），再归一化
    - community_priority: 把 priority_fraction 的权重给关键社区，其余给非关键社区

    Returns:
        detection_weight, bed_weight: dict[node]=weight（两者目前一致，便于解释）
    """
    rng = np.random.default_rng(seed)
    nodes = list(G.nodes())
    n = len(nodes)

    strategy = strategy.lower().strip()
    if strategy == "equal":
        w = np.ones(n, dtype=float)

    elif strategy == "random":
        # 用指数分布生成正权重，更容易形成“资源不均匀”的随机基线
        w = rng.exponential(scale=1.0, size=n)

    elif strategy == "community_priority":
        if node_to_comm is None:
            raise ValueError("community_priority requires node_to_comm")
        if not critical_comms:
            raise ValueError("community_priority requires non-empty critical_comms")

        critical_set = set(int(c) for c in critical_comms)
        in_critical = np.array([1 if int(node_to_comm[int(node)]) in critical_set else 0 for node in nodes])

        n_critical = int(in_critical.sum())
        n_other = n - n_critical
        if n_critical == 0:
            # 没识别到关键社区时退化为均匀
            w = np.ones(n, dtype=float)
        else:
            pf = float(priority_fraction)
            pf = max(0.0, min(1.0, pf))

            w = np.zeros(n, dtype=float)
            # 关键社区内均分 priority_fraction
            w[in_critical == 1] = pf / n_critical
            # 非关键社区均分剩余
            if n_other > 0:
                w[in_critical == 0] = (1.0 - pf) / n_other

            # 这里 w 已经归一化，无需再 normalize
            det_map = {int(nodes[i]): float(w[i]) for i in range(n)}
            return det_map, det_map.copy()

    else:
        raise ValueError("strategy must be one of: random, equal, community_priority")

    w = _normalize_weights(w)
    det_map = {int(nodes[i]): float(w[i]) for i in range(n)}
    return det_map, det_map.copy()


def run_sir(
    G: nx.Graph,
    beta: float,
    gamma: float,
    strategy: str,
    total_detection_capacity: int,
    total_bed_capacity: int,
    steps: int = 200,
    seed: Optional[int] = None,
    initial_infected: Optional[Sequence[int]] = None,
    node_to_comm: Optional[Dict[int, int]] = None,
    critical_comms: Optional[Sequence[int]] = None,
    priority_fraction: float = 0.7,
    treatment_multiplier: float = 2.0,
) -> SIRMetrics:
    """在给定图上运行带资源约束的离散时间 SIR。

    Args:
        total_detection_capacity: 每一步最多测试人数（整数）
        total_bed_capacity: 每一步最多可治疗的床位数（整数）

    Returns:
        SIRMetrics
    """
    rng = np.random.default_rng(seed)

    nodes = list(G.nodes())
    if len(nodes) == 0:
        raise ValueError("Graph has no nodes")

    if initial_infected is None:
        initial_infected = [int(rng.choice(nodes))]
    initial_infected = [int(x) for x in initial_infected]

    infected = set(initial_infected)
    recovered: set[int] = set()
    susceptible = set(int(n) for n in nodes) - infected

    ever_infected = set(infected)
    isolated: set[int] = set()  # 被检测到的感染者（隔离，不再传播）

    det_w, bed_w = build_node_weights(
        G,
        strategy=strategy,
        seed=seed,
        node_to_comm=node_to_comm,
        critical_comms=critical_comms,
        priority_fraction=priority_fraction,
    )

    infected_history = [len(infected)]
    max_infected = len(infected)

    positive_detections_total = 0
    treated_total = 0

    steps_run = 0
    for _ in range(int(steps)):
        steps_run += 1

        # 1) 检测：按 det_w 抽样测试，阳性（感染且未隔离）进入隔离
        if total_detection_capacity > 0:
            k_test = min(int(total_detection_capacity), len(nodes))
            test_nodes = _weighted_sample_without_replacement(
                rng,
                items=[int(n) for n in nodes],
                weights=np.array([det_w[int(n)] for n in nodes], dtype=float),
                k=k_test,
            )
            newly_detected = [n for n in test_nodes if (n in infected and n not in isolated)]
            if newly_detected:
                isolated.update(newly_detected)
                positive_detections_total += len(newly_detected)

        # 2) 传播：只有“未隔离”的感染者参与传播
        new_infected: set[int] = set()
        for u in infected:
            if u in isolated:
                continue
            # 对感染者的邻居尝试感染
            for v in G.neighbors(u):
                v = int(v)
                if v in susceptible and rng.random() < beta:
                    new_infected.add(v)

        # 3) 床位治疗：对当前感染者（含隔离者）选最多 total_bed_capacity 个提高恢复率
        treated: set[int] = set()
        if total_bed_capacity > 0 and len(infected) > 0:
            k_bed = min(int(total_bed_capacity), len(infected))
            infected_list = sorted(infected)
            w_inf = np.array([bed_w[n] for n in infected_list], dtype=float)
            if float(w_inf.sum()) <= 0:
                # 退化为均匀
                treated_nodes = rng.choice(infected_list, size=k_bed, replace=False)
            else:
                treated_nodes = rng.choice(infected_list, size=k_bed, replace=False, p=_normalize_weights(w_inf))
            treated = set(int(x) for x in treated_nodes)
            treated_total += len(treated)

        # 4) 恢复
        new_recovered: set[int] = set()
        for u in infected:
            p = gamma * (treatment_multiplier if u in treated else 1.0)
            if rng.random() < min(1.0, p):
                new_recovered.add(u)

        # 5) 状态更新
        if new_infected:
            susceptible -= new_infected
            infected |= new_infected
            ever_infected |= new_infected

        if new_recovered:
            infected -= new_recovered
            recovered |= new_recovered
            # 恢复后就不需要隔离标记了
            isolated -= new_recovered

        infected_history.append(len(infected))
        max_infected = max(max_infected, len(infected))

        if len(infected) == 0:
            break

    total_infected = len(ever_infected)
    transmission_duration = steps_run

    # 指标：用“有效使用量”做分母，避免 >1
    det_denom = float(total_detection_capacity) * steps_run
    bed_denom = float(total_bed_capacity) * steps_run

    detection_utilization = (positive_detections_total / det_denom) if det_denom > 0 else 0.0
    bed_utilization = (treated_total / bed_denom) if bed_denom > 0 else 0.0

    return SIRMetrics(
        infected_history=infected_history,
        total_infected=total_infected,
        max_infected=max_infected,
        transmission_duration=transmission_duration,
        detection_utilization=float(detection_utilization),
        bed_utilization=float(bed_utilization),
    )
