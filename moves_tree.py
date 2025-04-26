import math
from functools import lru_cache

import chess
import chess.engine
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

# ─── Settings ─────────────────────────────────────────────────────────────
PARQUET_PATH = "data/data_2025_01.parquet"
ENGINE_PATH = "stockfish/stockfish-windows-x86-64-avx2.exe"
MAX_DEPTH = 3  # Maximum ply depth for the tree
ANALYSIS_TIME = 0.5  # Seconds per engine analysis
MULTIPV = 5  # Number of principal variations for entropy


# ─── Memoized engine analysis ──────────────────────────────────────────────
@lru_cache(maxsize=None)
def analyse_position(fen: str, multipv: int):
    """
    Analyze a FEN with the engine, caching by (fen, multipv).
    """
    board = chess.Board(fen)
    return engine.analyse(
        board,
        chess.engine.Limit(time=ANALYSIS_TIME),
        multipv=multipv
    )


# ─── Create a new tree node ────────────────────────────────────────────────
def make_node():
    return {
        "count": 0,
        "wins": 0,
        "children": {},
        "winrate": None,
        "complexity": None
    }


# ─── Build the move tree from the first 5 games ───────────────────────────
def build_tree(df: pd.DataFrame):
    root = make_node()
    # Only take first 5 rows for speed
    for _, row in df.head(20).iterrows():
        winner = row["winner"]
        board = chess.Board()
        node = root
        for ply, uci in enumerate(row["moves"].split()):
            if ply >= MAX_DEPTH:
                break
            move = chess.Move.from_uci(uci)
            if move not in board.legal_moves:
                break
            board.push(move)
            if uci not in node["children"]:
                node["children"][uci] = make_node()
            node = node["children"][uci]
            node["count"] += 1
            if winner == 1:
                node["wins"] += 1
    return root


# ─── Annotate winrates using engine scores ────────────────────────────────
def annotate_winrates(node, board=None, depth=0):
    if board is None:
        board = chess.Board()
    if depth >= MAX_DEPTH:
        return
    for uci, child in node["children"].items():
        board.push(chess.Move.from_uci(uci))
        info = analyse_position(board.fen(), multipv=1)[0]
        score_obj = info["score"].white()
        # Convert Cp or Mate to integer; mate_score used for mate distances
        cp = score_obj.score(mate_score=10000)
        child["winrate"] = 1 / (1 + 10 ** (-cp / 400))
        annotate_winrates(child, board, depth + 1)
        board.pop()


# ─── Annotate complexity as Shannon entropy ────────────────────────────────
def annotate_complexity(node, board=None, depth=0):
    if board is None:
        board = chess.Board()
    infos = analyse_position(board.fen(), multipv=MULTIPV)
    cps = [info["score"].white().score(mate_score=10000) for info in infos]
    exps = [math.exp(cp / 400) for cp in cps]
    ps = [e / sum(exps) for e in exps]
    # Shannon entropy in bits
    node["complexity"] = -sum(p * math.log2(p) for p in ps if p > 0)
    if depth >= MAX_DEPTH:
        return
    for uci, child in node["children"].items():
        board.push(chess.Move.from_uci(uci))
        annotate_complexity(child, board, depth + 1)
        board.pop()


# ─── Convert tree to NetworkX DiGraph ────────────────────────────────────
def tree_to_graph(root):
    G = nx.DiGraph()

    def recurse(node, parent=None, board=chess.Board(), depth=0):
        if depth >= MAX_DEPTH:
            return
        for uci, child in node["children"].items():
            board.push(chess.Move.from_uci(uci))
            wr = child["winrate"] if child["winrate"] is not None else (
                child["wins"] / child["count"] if child["count"] > 0 else 0)
            comp = child["complexity"]
            label = f"{uci}\\nWR={wr * 100:4.1f}%  C={comp:.2f}"
            G.add_node(label, complexity=comp)
            if parent is not None:
                G.add_edge(parent, label)
            recurse(child, label, board, depth + 1)
            board.pop()

    recurse(root)
    return G


# ─── Improved hierarchical layout handling multiple roots ────────────────
def hierarchy_pos(G, width=1.0, vert_gap=0.2, vert_loc=0, xcenter=0.5):
    """
    Assign positions in a forest layout: each root (in-degree 0) gets its own segment.
    """
    # Find all root nodes (no incoming edges) :contentReference[oaicite:7]{index=7}
    roots = [n for n, d in G.in_degree() if d == 0]
    if not roots:
        roots = list(G.nodes())
    pos = {}

    def _pos(node, left, w, vert):
        # Center node in its horizontal span
        pos[node] = (left + w / 2, vert)
        children = list(G.successors(node))
        if not children:
            return
        dx = w / len(children)
        for i, child in enumerate(children):
            _pos(child, left + i * dx, dx, vert - vert_gap)

    # Divide total width among roots :contentReference[oaicite:8]{index=8}
    segment = width / len(roots)
    for i, r in enumerate(roots):
        _pos(r, i * segment, segment, vert_loc)
    return pos


# ─── Main execution & plotting ────────────────────────────────────────────
if __name__ == "__main__":
    # Read Parquet and take only 5 games
    df = pd.read_parquet(PARQUET_PATH)
    engine = chess.engine.SimpleEngine.popen_uci(ENGINE_PATH)

    root = build_tree(df)
    annotate_winrates(root)
    annotate_complexity(root)

    engine.quit()

    # Build graph and compute positions
    G = tree_to_graph(root)
    pos = hierarchy_pos(G)

    # Extract complexity values for coloring
    vals = list(nx.get_node_attributes(G, "complexity").values())

    # Draw nodes colored by complexity with vmin/vmax :contentReference[oaicite:9]{index=9}
    fig, ax = plt.subplots(figsize=(14, 10))
    nx.draw_networkx_nodes(
        G, pos,
        node_color=vals,
        cmap=plt.cm.Reds,
        vmin=min(vals),
        vmax=max(vals),
        node_size=600
    )
    nx.draw_networkx_edges(G, pos, arrows=True, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)

    # Add colorbar :contentReference[oaicite:10]{index=10}
    sm = mpl.cm.ScalarMappable(
        norm=mpl.colors.Normalize(vmin=min(vals), vmax=max(vals)),
        cmap=plt.cm.Reds
    )
    sm.set_array(vals)
    fig.colorbar(sm, ax=ax, label="Complexity (bits)")

    plt.axis("off")
    plt.tight_layout()
    plt.show()
