"""community_detection.py

社区发现模块（不依赖 cdlib，直接使用 NetworkX 内置算法）：
- Louvain (networkx.algorithms.community.louvain_communities)
- 标签传播：异步 LPA (networkx.algorithms.community.asyn_lpa_communities)

同时提供：
- 与 SBM 真值标签对比的 NMI/ARI
- 模块度 modularity
- 关键社区识别：按“跨社区边/社区规模”得分，选 top-k

该文件既可被 import（给 SIR 仿真调用），也可作为脚本单独运行。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import networkx as nx
import pandas as pd
from networkx.algorithms.community import asyn_lpa_communities, louvain_communities
from networkx.algorithms.community.quality import modularity
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from generate_lfr import generate_sbm_graph, save_labels_csv


def communities_to_node_map(communities: Sequence[Sequence[int]]) -> Dict[int, int]:
    """把 [set(nodes), ...] 转成 node->community_id 映射。"""
    node_to_comm: Dict[int, int] = {}
    for cid, comm in enumerate(communities):
        for node in comm:
            node_to_comm[int(node)] = int(cid)
    return node_to_comm


def detect_communities(
    G: nx.Graph,
    method: str = "louvain",
    seed: Optional[int] = None,
) -> Tuple[List[set], Dict[int, int]]:
    """对图做社区发现。

    Args:
        G: networkx 图
        method: "louvain" or "lpa"(label propagation)
        seed: 随机种子（保证可复现）

    Returns:
        communities: list[set]
        node_to_comm: dict
    """
    method = method.lower().strip()
    if method == "louvain":
        communities = list(louvain_communities(G, seed=seed))
    elif method in {"lpa", "label", "label_propagation", "asyn_lpa"}:
        communities = list(asyn_lpa_communities(G, seed=seed))
    else:
        raise ValueError("method must be 'louvain' or 'lpa'")

    node_to_comm = communities_to_node_map(communities)
    return communities, node_to_comm


def evaluate_partition(
    G: nx.Graph,
    true_map: Dict[int, int],
    pred_map: Dict[int, int],
) -> Dict[str, float]:
    """计算模块度与 NMI/ARI。"""
    nodes = sorted(G.nodes())
    true_labels = [true_map[int(n)] for n in nodes]
    pred_labels = [pred_map.get(int(n), -1) for n in nodes]

    # modularity 需要 communities(list of sets)
    # 这里从 pred_map 重建 communities
    comm_to_nodes: Dict[int, set] = {}
    for n, c in pred_map.items():
        comm_to_nodes.setdefault(int(c), set()).add(int(n))
    pred_communities = list(comm_to_nodes.values())

    return {
        "modularity": float(modularity(G, pred_communities)),
        "nmi": float(normalized_mutual_info_score(true_labels, pred_labels)),
        "ari": float(adjusted_rand_score(true_labels, pred_labels)),
    }


def find_critical_communities(
    G: nx.Graph,
    node_to_comm: Dict[int, int],
    top_k: int = 1,
) -> List[int]:
    """识别关键社区。

    采用简单可解释的指标：
        score(comm) = 跨社区边条数 / 社区规模
    得分越高，越“桥接/外溢”，越可能成为扩散关键社区。

    Returns:
        community_id 列表（按得分降序）。
    """
    if top_k <= 0:
        return []

    comm_ids = set(node_to_comm.values())
    size = {cid: 0 for cid in comm_ids}
    boundary = {cid: 0 for cid in comm_ids}

    for node, cid in node_to_comm.items():
        size[cid] += 1

    for u, v in G.edges():
        cu = node_to_comm[int(u)]
        cv = node_to_comm[int(v)]
        if cu != cv:
            boundary[cu] += 1
            boundary[cv] += 1

    scores = {cid: (boundary[cid] / size[cid]) if size[cid] > 0 else 0.0 for cid in comm_ids}
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [cid for cid, _ in ranked[: min(top_k, len(ranked))]]


def save_node_to_comm_csv(node_to_comm: Dict[int, int], path: Union[str, Path]) -> None:
    path = Path(path)
    df = pd.DataFrame({"node": list(node_to_comm.keys()), "community": list(node_to_comm.values())})
    df.sort_values("node", inplace=True)
    df.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Community detection on SBM graphs.")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--p_in", type=float, default=0.05)
    parser.add_argument("--p_out", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--method", type=str, default="louvain", help="louvain or lpa")
    parser.add_argument("--top_k", type=int, default=1, help="how many critical communities")
    parser.add_argument("--out_pred", type=str, default="", help="save predicted node->community CSV")
    parser.add_argument("--out_true", type=str, default="", help="save true node->community CSV")
    args = parser.parse_args()

    G, true_map = generate_sbm_graph(
        n=args.n, n_communities=args.k, p_in=args.p_in, p_out=args.p_out, seed=args.seed
    )

    communities, pred_map = detect_communities(G, method=args.method, seed=args.seed)

    metrics = evaluate_partition(G, true_map, pred_map)
    critical = find_critical_communities(G, pred_map, top_k=args.top_k)

    print(
        f"method={args.method}  modularity={metrics['modularity']:.4f}  "
        f"NMI={metrics['nmi']:.4f}  ARI={metrics['ari']:.4f}"
    )
    print(f"critical communities (top {args.top_k}): {critical}")

    if args.out_pred:
        save_node_to_comm_csv(pred_map, args.out_pred)

    if args.out_true:
        save_labels_csv(true_map, args.out_true)


if __name__ == "__main__":
    main()
