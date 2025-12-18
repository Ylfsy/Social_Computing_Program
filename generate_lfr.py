"""generate_lfr.py

⚠️ 文件名沿用 generate_lfr.py，但默认实现 **SBM(随机块模型)** 网络生成。
原因：你的开题 PPT 里写的是“采用随机块模型(SBM)通过调节组内/组间连接概率重现不同社区强度”。
SBM 同时能提供“真实社区标签”，便于后续计算 NMI/ARI，并支持关键社区识别与干预策略评估。

主要函数
- generate_sbm_graph: 生成 SBM 图 + node->true_community 映射
- save_labels_csv:   保存 node->community_id 到 CSV

"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

import networkx as nx
import pandas as pd


def _default_sizes(n: int, k: int) -> Sequence[int]:
    """将 n 个节点尽量均匀分到 k 个社区。"""
    if k <= 0:
        raise ValueError("n_communities must be >= 1")
    base = n // k
    rem = n % k
    return [base + (1 if i < rem else 0) for i in range(k)]


def generate_sbm_graph(
    n: int = 1000,
    n_communities: int = 5,
    p_in: float = 0.05,
    p_out: float = 0.005,
    sizes: Optional[Sequence[int]] = None,
    seed: Optional[int] = None,
) -> Tuple[nx.Graph, Dict[int, int]]:
    """生成 SBM 图。

    Args:
        n: 总节点数。
        n_communities: 社区数（sizes 未提供时生效）。
        p_in: 同社区内两点连边概率。
        p_out: 不同社区两点连边概率。
        sizes: 每个社区的规模列表，例如 [200,200,200,200,200]。
        seed: 随机种子。

    Returns:
        G: networkx.Graph
        node_to_true_comm: 节点到“真实社区 id”的映射。
    """
    if sizes is None:
        sizes = _default_sizes(n, n_communities)
    else:
        sizes = list(sizes)
        if sum(sizes) != n:
            raise ValueError(f"sum(sizes) must equal n (got sum={sum(sizes)} vs n={n})")
        n_communities = len(sizes)

    if not (0.0 <= p_out <= 1.0 and 0.0 <= p_in <= 1.0):
        raise ValueError("p_in and p_out must be in [0, 1]")

    # 概率矩阵：对角为 p_in，非对角为 p_out
    probs = [[p_in if i == j else p_out for j in range(n_communities)] for i in range(n_communities)]

    G = nx.stochastic_block_model(sizes, probs, seed=seed)

    # 建立真值社区映射（SBM 默认节点编号 0..n-1，且按 sizes 顺序分块）
    node_to_true_comm: Dict[int, int] = {}
    node = 0
    for comm_id, size in enumerate(sizes):
        for _ in range(size):
            node_to_true_comm[node] = comm_id
            node += 1

    return G, node_to_true_comm


def save_labels_csv(node_to_comm: Dict[int, int], path: Union[str, Path]) -> None:
    """保存 node->community 的映射到 CSV。"""
    path = Path(path)
    df = pd.DataFrame({"node": list(node_to_comm.keys()), "community": list(node_to_comm.values())})
    df.sort_values("node", inplace=True)
    df.to_csv(path, index=False)


def _parse_sizes(s: str) -> Sequence[int]:
    # 形如 "200,200,200,200,200"
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return [int(p) for p in parts]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SBM graph (file name kept as generate_lfr.py).")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--k", type=int, default=5, help="number of communities (ignored if --sizes provided)")
    parser.add_argument("--p_in", type=float, default=0.05)
    parser.add_argument("--p_out", type=float, default=0.005)
    parser.add_argument("--sizes", type=str, default="", help="comma-separated sizes, e.g. 200,200,200,200,200")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_gpickle", type=str, default="", help="save graph to .gpickle")
    parser.add_argument("--out_labels", type=str, default="", help="save true labels to csv")
    args = parser.parse_args()

    sizes = _parse_sizes(args.sizes) if args.sizes else None
    G, true_map = generate_sbm_graph(
        n=args.n,
        n_communities=args.k,
        p_in=args.p_in,
        p_out=args.p_out,
        sizes=sizes,
        seed=args.seed,
    )

    if args.out_gpickle:
        nx.write_gpickle(G, args.out_gpickle)

    if args.out_labels:
        save_labels_csv(true_map, args.out_labels)

    print(
        f"Generated SBM graph: n={G.number_of_nodes()}, m={G.number_of_edges()}, "
        f"k={len(set(true_map.values()))}, p_in={args.p_in}, p_out={args.p_out}"
    )


if __name__ == "__main__":
    main()
