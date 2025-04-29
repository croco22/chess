import os
from datetime import datetime
from functools import lru_cache

import chess
import chess.engine
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

# ─── Configuration ───────────────────────────────────────────────────────────
PARQUET_PATH = "data/moves_2025_01.parquet"
ENGINE_PATH = "stockfish/stockfish-windows-x86-64-avx2.exe"
MAX_DEPTH = 8
SAMPLE_SIZE = 600 # about 1 %
FIG_DPI = 500
MIN_NODE_SIZE = 100
MAX_NODE_SIZE = 500


# ─── Cached Engine Analysis: Evaluate only Top-K Best Moves ───────────────────
@lru_cache(maxsize=None)
def evaluate_move_complexity(fen: str, threshold_cp: int = 50):
    board = chess.Board(fen)
    info = engine.analyse(board, chess.engine.Limit(depth=MAX_DEPTH))
    best_cp = info["score"].white().score(mate_score=10000)

    good_moves = 0
    for move in board.legal_moves:
        board.push(move)
        move_info = engine.analyse(board, chess.engine.Limit(depth=MAX_DEPTH))
        move_cp = move_info["score"].white().score(mate_score=10000)
        board.pop()

        if abs(best_cp - move_cp) <= threshold_cp:
            good_moves += 1

    if good_moves == 0:
        return 100
    else:
        return 1 / good_moves


# ─── Node Factory ─────────────────────────────────────────────────────────────
def make_node(fen, move=None):
    return {
        "fen": fen,
        "move": move,
        "children": {},
        "count": 0,
        "wins": 0,
        "winrate": None,
        "complexity": None,
    }


# ─── Build Move Tree ──────────────────────────────────────────────────────────
def build_tree(df_tree):
    root_tree = make_node(chess.Board().fen())
    for _, row in df_tree.iterrows():
        board = chess.Board()
        node = root_tree
        for ply, uci in enumerate(row["moves"].split()):
            if ply >= MAX_DEPTH:
                break
            mv = chess.Move.from_uci(uci)
            if mv not in board.legal_moves:
                break
            board.push(mv)
            fen = board.fen()
            if uci not in node["children"]:
                node["children"][uci] = make_node(fen, move=uci)
            node = node["children"][uci]
            node["count"] += 1
            if row["winner"] == 1:
                node["wins"] += 1
    return root_tree


# ─── Flatten Tree ────────────────────────────────────────────────────────────
def collect_nodes(root_nodes):
    out = []

    def recurse(n_nodes):
        for c in n_nodes["children"].values():
            out.append(c)
            recurse(c)

    recurse(root_nodes)
    return out


# ─── Build Graph (skip root) ─────────────────────────────────────────────────
def tree_to_graph(root_graph):
    graph = nx.DiGraph()

    def recurse(n_graph, parent=None):
        if n_graph["move"] is not None:
            graph.add_node(
                n_graph["fen"],
                move=n_graph["move"],
                complexity=n_graph["complexity"],
                winrate=n_graph["winrate"],
            )
            if parent is not None:
                graph.add_edge(parent, n_graph["fen"])
            key = n_graph["fen"]
        else:
            key = None
        for c in n_graph["children"].values():
            recurse(c, key)

    recurse(root_graph, None)
    return graph


# ─── Layout ──────────────────────────────────────────────────────────────────
def hierarchical_forest_layout(tree_graph):
    roots = [a for a, d in tree_graph.in_degree() if d == 0] or list(tree_graph.nodes())
    position = {}

    def _place(key, left, w, vert):
        position[key] = (left + w / 2, vert)
        kids = list(tree_graph.successors(key))
        if not kids:
            return
        dx = w / len(kids)
        for j, k in enumerate(kids):
            _place(k, left + j * dx, dx, vert - 0.2)

    seg = 1 / len(roots)
    for i, r in enumerate(roots):
        _place(r, i * seg, seg, 0)
    return position


# ─── Main & Visualization ────────────────────────────────────────────────────
if __name__ == "__main__":
    df = pd.read_parquet(PARQUET_PATH).sample(SAMPLE_SIZE)
    engine = chess.engine.SimpleEngine.popen_uci(ENGINE_PATH)

    root = build_tree(df)
    nodes = collect_nodes(root)

    # --- Calculate Winrate from Data (no Engine) ---
    for n in nodes:
        if n["count"] > 0:
            n["winrate"] = n["wins"] / n["count"]
        else:
            n["winrate"] = None

    # --- Calculate Move Complexity (Standard Deviation of Scores) ---
    for n in tqdm(nodes, desc="Evaluating Move Complexity"):
        n["complexity"] = evaluate_move_complexity(n["fen"], threshold_cp=50)

    engine.quit()

    G = tree_to_graph(root)
    pos = hierarchical_forest_layout(G)

    comp = np.array([G.nodes[n]["complexity"] if G.nodes[n]["complexity"] is not None else 0 for n in G.nodes()])
    norm = (comp - comp.min()) / (comp.max() - comp.min() + 1e-9)
    scaled = np.sqrt(norm)
    sizes = MIN_NODE_SIZE + scaled * (MAX_NODE_SIZE - MIN_NODE_SIZE)

    winrates = [G.nodes[n]["winrate"] for n in G.nodes()]
    cmap = mpl.colors.LinearSegmentedColormap.from_list("RG", ["red", "orange", "green"])

    fig, ax = plt.subplots(figsize=(16, 8), dpi=FIG_DPI)

    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=sizes,
        node_color=winrates,  # type: ignore
        cmap=cmap,
        vmin=0.35,
        vmax=0.65,
        alpha=0.8,
        ax=ax,
    )
    nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle="-|>", ax=ax)

    labels = {
        n: f"{G.nodes[n]['move']}\n{G.nodes[n]['complexity']:.1f}" if G.nodes[n]['complexity'] is not None else
        G.nodes[n]['move']
        for n in G.nodes()
    }
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=6, font_family="monospace", ax=ax)

    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(vmin=0.35, vmax=0.65))
    sm.set_array(winrates)
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical")
    cbar.set_label("Winrate", fontsize=10)

    fig.suptitle(
        "Move Tree: Winrate (color) & Move Complexity (size)",
        fontsize=24,
        fontweight="bold",
    )
    ax.set_title(
        f"Opening: Scandinavian Defense: Mieses–Kotroc Variation | Depth: {MAX_DEPTH} | Samples: {SAMPLE_SIZE}",
        fontsize=16,
        fontstyle="italic",
    )
    ax.axis("off")
    plt.tight_layout()

    os.makedirs("images", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"images/move_tree_{timestamp}.png"
    fig.savefig(filename, dpi=FIG_DPI, bbox_inches="tight")

    plt.show()
