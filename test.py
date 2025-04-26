from functools import lru_cache

import chess
import chess.engine
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np  # for size normalization
import pandas as pd
from tqdm import tqdm

# ─── Configuration ───────────────────────────────────────────────────────────
PARQUET_PATH = "data/moves_2025_01.parquet"
ENGINE_PATH = "stockfish/stockfish-windows-x86-64-avx2.exe"
MAX_DEPTH = 10
ANALYSIS_TIME = 0.5
FIG_DPI = 200
MIN_NODE_SIZE = 100
MAX_NODE_SIZE = 1000  # increased max size for more contrast


# ─── Cached Engine Analysis ─────────────────────────────────────────────────
@lru_cache(maxsize=None)
def analyse_position(fen: str):
    board = chess.Board(fen)
    return engine.analyse(board, chess.engine.Limit(time=ANALYSIS_TIME))


# ─── Node Factory ─────────────────────────────────────────────────────────────
def make_node(fen, move=None):
    return {"fen": fen, "move": move, "children": {}, "count": 0,
            "wins": 0, "winrate": None, "complexity": None}


# ─── Build Move Tree ──────────────────────────────────────────────────────────
def build_tree(df):
    root = make_node(chess.Board().fen())
    for _, row in df.iterrows():
        board = chess.Board()
        node = root
        for ply, uci in enumerate(row["moves"].split()):
            if ply >= MAX_DEPTH: break
            mv = chess.Move.from_uci(uci)
            if mv not in board.legal_moves: break
            board.push(mv)
            fen = board.fen()
            if uci not in node["children"]:
                node["children"][uci] = make_node(fen, move=uci)
            node = node["children"][uci]
            node["count"] += 1
            if row["winner"] == 1:
                node["wins"] += 1
    return root


# ─── Flatten Tree ────────────────────────────────────────────────────────────
def collect_nodes(root):
    out = []

    def recurse(n):
        for c in n["children"].values():
            out.append(c)
            recurse(c)

    recurse(root)
    return out


# ─── Build Graph (skip root) ─────────────────────────────────────────────────
def tree_to_graph(root):
    G = nx.DiGraph()

    def recurse(n, parent=None):
        if n["move"] is not None:
            G.add_node(n["fen"],
                       move=n["move"],
                       complexity=n["complexity"],
                       winrate=n["winrate"])
            if parent is not None:
                G.add_edge(parent, n["fen"])
            key = n["fen"]
        else:
            key = None
        for c in n["children"].values():
            recurse(c, key)

    recurse(root, None)
    return G


# ─── Layout ──────────────────────────────────────────────────────────────────
def hierarchical_forest_layout(G, width=1.0, vert_gap=0.2, vert_loc=0):
    roots = [n for n, d in G.in_degree() if d == 0] or list(G.nodes())
    pos = {}

    def _place(key, left, w, vert):
        pos[key] = (left + w / 2, vert)
        kids = list(G.successors(key))
        if not kids: return
        dx = w / len(kids)
        for i, k in enumerate(kids):
            _place(k, left + i * dx, dx, vert - vert_gap)

    seg = width / len(roots)
    for i, r in enumerate(roots):
        _place(r, i * seg, seg, vert_loc)
    return pos


# ─── Main & Visualization ────────────────────────────────────────────────────
if __name__ == "__main__":
    df = pd.read_parquet(PARQUET_PATH).head(100)
    engine = chess.engine.SimpleEngine.popen_uci(ENGINE_PATH)

    root = build_tree(df)
    nodes = collect_nodes(root)

    for n in nodes:
        n["complexity"] = chess.Board(n["fen"]).legal_moves.count()

    for n in tqdm(nodes, desc="Engine eval"):
        info = analyse_position(n["fen"])
        cp = info["score"].white().score(mate_score=10000)
        n["winrate"] = 1 / (1 + 10 ** (-cp / 400))

    engine.quit()

    G = tree_to_graph(root)
    pos = hierarchical_forest_layout(G)

    # Size scaling: use square-root of normalized complexity for stronger contrast
    comp = np.array([G.nodes[n]["complexity"] for n in G.nodes()])
    norm = (comp - comp.min()) / (comp.max() - comp.min())
    # apply sqrt to exaggerate differences
    scaled = np.sqrt(norm)
    sizes = MIN_NODE_SIZE + scaled * (MAX_NODE_SIZE - MIN_NODE_SIZE)

    # Color mapping for winrate
    winrates = [G.nodes[n]["winrate"] for n in G.nodes()]
    cmap = mpl.colors.LinearSegmentedColormap.from_list("RG", ["red", "green"])

    # Wider figure to reduce overlap
    fig, ax = plt.subplots(figsize=(32, 8), dpi=FIG_DPI)

    nx.draw_networkx_nodes(
        G, pos,
        node_size=sizes,
        node_color=winrates,
        cmap=cmap,
        vmin=0.4, vmax=0.6,
        alpha=0.9,
        ax=ax
    )
    nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle='-|>', ax=ax)

    labels = {n: G.nodes[n]["move"] for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels,
                            font_size=6, font_family="monospace", ax=ax)

    sm = mpl.cm.ScalarMappable(
        cmap=cmap,
        norm=mpl.colors.Normalize(vmin=0.4, vmax=0.6)
    )
    sm.set_array(winrates)
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical", shrink=0.7)
    cbar.set_label("Winrate", fontsize=10)

    ax.set_title("Move Tree: Winrate (color) & Complexity (size)", fontsize=14)
    ax.axis("off")
    plt.tight_layout()
    plt.show()
