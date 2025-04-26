from functools import lru_cache

import chess
import chess.engine
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from tqdm import tqdm

# ─── Configuration ───────────────────────────────────────────────────────────
PARQUET_PATH = "data/moves_2025_01.parquet"
ENGINE_PATH = "stockfish/stockfish-windows-x86-64-avx2.exe"
MAX_DEPTH = 10
ANALYSIS_TIME = 0.5
FIG_DPI = 200


# ─── Cached Engine Analysis ─────────────────────────────────────────────────
@lru_cache(maxsize=None)
def analyse_position(fen: str):
    """
    Analyze a FEN once and cache the result.
    """
    board = chess.Board(fen)
    return engine.analyse(board, chess.engine.Limit(time=ANALYSIS_TIME))


# ─── Node Factory ─────────────────────────────────────────────────────────────
def make_node(fen=None):
    """
    Create a node; optionally store its FEN here.
    """
    return {
        "fen": fen,
        "count": 0,
        "wins": 0,
        "children": {},
        "winrate": None,
        "complexity": None
    }


# ─── Build Move Tree ──────────────────────────────────────────────────────────
def build_tree(df: pd.DataFrame):
    """
    Build a move-tree and record the FEN at each node.
    """
    root = make_node(chess.Board().fen())
    for _, row in df.iterrows():
        board = chess.Board()
        node = root
        for ply, uci in enumerate(row["moves"].split()):
            if ply >= MAX_DEPTH:
                break
            move = chess.Move.from_uci(uci)
            if move not in board.legal_moves:
                break
            board.push(move)
            fen = board.fen()
            if uci not in node["children"]:
                node["children"][uci] = make_node(fen)
            node = node["children"][uci]
            node["count"] += 1
            if row["winner"] == 1:
                node["wins"] += 1
    return root


# ─── Collect All Nodes ────────────────────────────────────────────────────────
def collect_nodes(root):
    """
    Flatten the tree into a list of all nodes (excluding the root if you like).
    """
    nodes = []

    def recurse(n):
        for child in n["children"].values():
            nodes.append(child)
            recurse(child)

    recurse(root)
    return nodes


# ─── Convert to NetworkX Graph ───────────────────────────────────────────────
def tree_to_graph(root):
    G = nx.DiGraph()

    def recurse(n, parent_label=None):
        for move, child in n["children"].items():
            label = (
                f"{move}\n"
                f"WR={child['winrate'] * 100:4.1f}%  "
                f"C={child['complexity']}"
            )
            G.add_node(label, complexity=child["complexity"])
            if parent_label:
                G.add_edge(parent_label, label)
            recurse(child, label)

    recurse(root, None)
    return G


# ─── Pure-Python Hierarchical Layout ─────────────────────────────────────────
def hierarchical_forest_layout(G, width=1.0, vert_gap=0.2, vert_loc=0):
    roots = [n for n, d in G.in_degree() if d == 0] or list(G.nodes())
    pos = {}

    def _place(node, left, w, vert):
        pos[node] = (left + w / 2, vert)
        children = list(G.successors(node))
        if not children:
            return
        dx = w / len(children)
        for i, c in enumerate(children):
            _place(c, left + i * dx, dx, vert - vert_gap)

    seg = width / len(roots)
    for i, r in enumerate(roots):
        _place(r, i * seg, seg, vert_loc)
    return pos


# ─── Main & Visualization ────────────────────────────────────────────────────
if __name__ == "__main__":
    # 1) Load Data
    df = pd.read_parquet(PARQUET_PATH).head(100)
    engine = chess.engine.SimpleEngine.popen_uci(ENGINE_PATH)

    # 2) Build tree & collect nodes
    root = build_tree(df)
    nodes = collect_nodes(root)

    # 3) Compute complexity (cheap) and winrates with progress bar
    for node in nodes:
        # branch factor
        node["complexity"] = chess.Board(node["fen"]).legal_moves.count()

    for node in tqdm(nodes, desc="Evaluating Winrates"):
        info = analyse_position(node["fen"])
        cp = info["score"].white().score(mate_score=10000)
        node["winrate"] = 1 / (1 + 10 ** (-cp / 400))

    engine.quit()

    # 4) Build graph & layout
    G = tree_to_graph(root)
    pos = hierarchical_forest_layout(G)

    # 5) Draw
    vals = [d["complexity"] for _, d in G.nodes(data=True)]
    fig, ax = plt.subplots(figsize=(16, 12), dpi=FIG_DPI)
    nx.draw_networkx_nodes(
        G, pos,
        node_size=300,
        node_color=vals,
        cmap=plt.cm.Reds,
        vmin=min(vals),
        vmax=max(vals),
        alpha=0.9, ax=ax
    )
    nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle='-|>', ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=6, font_family="monospace", ax=ax)

    sm = mpl.cm.ScalarMappable(
        cmap=plt.cm.Reds,
        norm=mpl.colors.Normalize(vmin=min(vals), vmax=max(vals))
    )
    sm.set_array(vals)
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical", shrink=0.7)
    cbar.set_label("Branching Factor (Complexity)", fontsize=10)

    ax.set_title("Move Tree: Winrate & Complexity", fontsize=14)
    ax.axis("off")
    plt.tight_layout()
    plt.show()
