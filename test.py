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
MAX_DEPTH = 10  # How many plies deep to build
ANALYSIS_DEPTH = None  # Use None to fall back to time limit
ANALYSIS_TIME = 0.5  # Seconds per engine eval if no depth limit
FIG_DPI = 200  # High-res output
SIZE_MULTIPLIER = 50  # Scale for node sizes


# ─── Cached Engine Analysis ─────────────────────────────────────────────────
@lru_cache(maxsize=None)
def analyse_position(fen: str):
    """
    Analyze a FEN once; cache by FEN to avoid redundant calls.
    Returns the best-line InfoDict.
    """
    board = chess.Board(fen)
    limit = (chess.engine.Limit(depth=ANALYSIS_DEPTH)
             if ANALYSIS_DEPTH else
             chess.engine.Limit(time=ANALYSIS_TIME))
    return engine.analyse(board, limit)


# ─── Node Factory ─────────────────────────────────────────────────────────────
def make_node(fen, move=None):
    """
    Node template with FEN, the UCI move that led here, counters,
    plus placeholders for winrate and complexity.
    """
    return {
        "fen": fen,
        "move": move,
        "count": 0,
        "wins": 0,
        "children": {},
        "winrate": None,
        "complexity": None
    }


# ─── Build Move Tree ──────────────────────────────────────────────────────────
def build_tree(df: pd.DataFrame):
    """
    Construct a prefix tree of positions from each game's move list.
    Store FEN at each node for caching.
    """
    root = make_node(chess.Board().fen())
    for _, row in df.iterrows():
        board = chess.Board()
        node = root
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
    return root


# ─── Flatten Tree to List ────────────────────────────────────────────────────
def collect_nodes(root):
    """
    Return a flat list of every node except the root.
    """
    out = []

    def recurse(n):
        for child in n["children"].values():
            out.append(child)
            recurse(child)

    recurse(root)
    return out


# ─── Build Graph ─────────────────────────────────────────────────────────────
def tree_to_graph(root):
    """
    Convert nodes into a NetworkX DiGraph keyed by FEN.
    Attach move, complexity, winrate as attributes.
    """
    G = nx.DiGraph()

    def recurse(parent_key, node):
        G.add_node(node["fen"],
                   move=node["move"],
                   complexity=node["complexity"],
                   winrate=node["winrate"])
        if parent_key is not None:
            G.add_edge(parent_key, node["fen"])
        for child in node["children"].values():
            recurse(node["fen"], child)

    recurse(None, root)
    return G


# ─── Hierarchical Layout ─────────────────────────────────────────────────────
def hierarchical_forest_layout(G, width=1.0, vert_gap=0.2, vert_loc=0):
    """
    Place each root (in-degree=0) in its own horizontal segment.
    """
    roots = [n for n, d in G.in_degree() if d == 0] or list(G.nodes())
    pos = {}

    def _place(key, left, w, vert):
        pos[key] = (left + w / 2, vert)
        kids = list(G.successors(key))
        if not kids:
            return
        dx = w / len(kids)
        for i, k in enumerate(kids):
            _place(k, left + i * dx, dx, vert - vert_gap)

    seg = width / len(roots)
    for i, r in enumerate(roots):
        _place(r, i * seg, seg, vert_loc)
    return pos


# ─── Main & Visualization ────────────────────────────────────────────────────
if __name__ == "__main__":
    # Load data and start engine
    df = pd.read_parquet(PARQUET_PATH).head(100)
    engine = chess.engine.SimpleEngine.popen_uci(ENGINE_PATH)

    # Build tree
    root = build_tree(df)

    # Collect all nodes including root
    nodes = [root] + collect_nodes(root)

    # Compute complexity (branching factor) for every node
    for node in nodes:
        node["complexity"] = chess.Board(node["fen"]).legal_moves.count()

    # Compute winrate for every node with progress bar
    for node in tqdm(nodes, desc="Engine eval"):
        info = analyse_position(node["fen"])
        cp = info["score"].white().score(mate_score=10000)
        node["winrate"] = 1 / (1 + 10 ** (-cp / 400))

    engine.quit()

    # Build graph and compute layout
    G = tree_to_graph(root)
    pos = hierarchical_forest_layout(G)

    # Prepare drawing params
    sizes = [G.nodes[n]["complexity"] * SIZE_MULTIPLIER for n in G.nodes()]
    winrates = [G.nodes[n]["winrate"] for n in G.nodes()]
    cmap = mpl.colors.LinearSegmentedColormap.from_list("RG", ["red", "green"])

    # Plot
    fig, ax = plt.subplots(figsize=(16, 12), dpi=FIG_DPI)
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

    # Label only with UCI move
    labels = {n: G.nodes[n]["move"] or "" for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels,
                            font_size=6, font_family="monospace", ax=ax)

    # Colorbar for winrate
    sm = mpl.cm.ScalarMappable(cmap=cmap,
                               norm=mpl.colors.Normalize(vmin=0.4, vmax=0.6))
    sm.set_array(winrates)
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical", shrink=0.7)
    cbar.set_label("Winrate", fontsize=10)

    # Legend for complexity sizes
    for comp in [5, 20, 50]:
        ax.scatter([], [], s=comp * SIZE_MULTIPLIER,
                   c='gray', alpha=0.6,
                   label=f"Branch factor = {comp}")
    ax.legend(scatterpoints=1, frameon=False,
              labelspacing=1, title="Complexity", loc="lower left")

    ax.set_title("Move Tree: Winrate (color) & Complexity (size)", fontsize=14)
    ax.axis("off")
    plt.tight_layout()
    plt.show()
