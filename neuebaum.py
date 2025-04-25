import pandas as pd
import chess
import chess.engine
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import math
import matplotlib as mpl

# ─── Einstellungen ─────────────────────────────────────────────────────────────
PARQUET_PATH  = "data/data_2025_01.parquet"
ENGINE_PATH   = "stockfish/stockfish-windows-x86-64-avx2.exe"
MAX_DEPTH     = 3    # Maximale Baumtiefe
ANALYSIS_TIME = 0.5  # Zeit pro Stellung in Sekunden

# ─── Baumknoten ────────────────────────────────────────────────────────────────
def make_node():
    return {
        "count": 0,
        "wins": 0,
        "children": defaultdict(make_node),
        "winrate": None,
        "complexity": None
    }

# ─── Baum aufbauen ──────────────────────────────────────────────────────────────
def build_tree(df):
    root = make_node()
    for _, row in df.iterrows():
        winner = row["winner"]
        board  = chess.Board()
        node   = root
        for ply, uci in enumerate(row["moves"].split()):
            if ply >= MAX_DEPTH:
                break
            move = chess.Move.from_uci(uci)
            if move not in board.legal_moves:
                break
            board.push(move)
            node = node["children"][uci]
            node["count"] += 1
            if winner == 1:
                node["wins"] += 1
    return root

# ─── Winrates annotieren ───────────────────────────────────────────────────────
def annotate_winrates(root, engine):
    def recurse(node, board, depth):
        if depth >= MAX_DEPTH:
            return
        for uci, child in node["children"].items():
            board.push(chess.Move.from_uci(uci))
            info  = engine.analyse(board, chess.engine.Limit(time=ANALYSIS_TIME))
            score = info["score"].white()
            cp    = (1000 if score.is_mate() and score.mate()>0
                     else -1000 if score.is_mate()
                     else score.score())
            child["winrate"] = 1 / (1 + 10 ** (-cp / 400))
            recurse(child, board, depth+1)
            board.pop()
    recurse(root, chess.Board(), 0)

# ─── Komplexität annotieren ────────────────────────────────────────────────────
def annotate_complexity(root):
    def recurse(node, board, depth):
        node["complexity"] = board.legal_moves.count()
        if depth >= MAX_DEPTH:
            return
        for uci, child in node["children"].items():
            board.push(chess.Move.from_uci(uci))
            recurse(child, board, depth+1)
            board.pop()
    recurse(root, chess.Board(), 0)

# ─── Baum → networkx.DiGraph ───────────────────────────────────────────────────
def tree_to_graph(root):
    G = nx.DiGraph()
    def recurse(node, parent=None, board=chess.Board(), depth=0):
        if depth >= MAX_DEPTH:
            return
        for uci, child in node["children"].items():
            board.push(chess.Move.from_uci(uci))
            wr   = (child["winrate"]
                    if child["winrate"] is not None
                    else (child["wins"]/child["count"] if child["count"]>0 else 0))
            comp = child["complexity"]
            label = f"{uci}\nWR={wr*100:4.1f}%  C={comp}"
            G.add_node(label, complexity=comp)
            if parent:
                G.add_edge(parent, label)
            recurse(child, label, board, depth+1)
            board.pop()
    recurse(root)
    return G

# ─── Hierarchisches Layout ────────────────────────────────────────────────────
def hierarchy_pos(G, root=None, width=1.0, vert_gap=0.2, vert_loc=0, xcenter=0.5):
    """
    From Joel's answer at https://stackoverflow.com/a/29597209
    Positional layout for a tree: each level down is vert_gap lower.
    """
    if root is None:
        root = next(iter(nx.topological_sort(G)))
    def _hierarchy_pos(G, node, width, vert_gap, vert_loc, xcenter, pos, parent=None):
        pos[node] = (xcenter, vert_loc)
        children = list(G.successors(node))
        if not children:
            return
        dx = width / len(children)
        nextx = xcenter - width/2 - dx/2
        for child in children:
            nextx += dx
            _hierarchy_pos(G, child, dx, vert_gap, vert_loc-vert_gap, nextx, pos, node)
    pos = {}
    _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter, pos)
    return pos

# ─── Hauptprogramm & Plot ─────────────────────────────────────────────────────
if __name__ == "__main__":
    # Daten einlesen (zum Test auf 10 Partien beschränkt)
    df = pd.read_parquet(PARQUET_PATH).head(10)

    # Engine starten
    engine = chess.engine.SimpleEngine.popen_uci(ENGINE_PATH)

    # Baum aufbauen
    root = build_tree(df)

    # Winrates und Komplexität annotieren
    annotate_winrates(root, engine)
    annotate_complexity(root)

    # Engine beenden
    engine.quit()

    # Graph erzeugen
    G = tree_to_graph(root)

    # Komplexitätswerte normieren für Farbe
    comps = nx.get_node_attributes(G, "complexity")
    vals  = list(comps.values())
    min_c, max_c = min(vals), max(vals)
    norm = [ (c-min_c)/(max_c-min_c) if max_c>min_c else 0.0 for c in vals ]

    # Layout berechnen
    pos = hierarchy_pos(G)

    # Plot
    fig, ax = plt.subplots(figsize=(14, 10))
    nodes = nx.draw_networkx_nodes(
        G, pos,
        node_size=800,
        node_color=norm,
        cmap=plt.cm.Reds,
        ax=ax
    )
    nx.draw_networkx_edges(G, pos, arrows=True, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)

    # Colorbar
    sm = mpl.cm.ScalarMappable(cmap=plt.cm.Reds,
                               norm=mpl.colors.Normalize(vmin=min_c, vmax=max_c))
    sm.set_array(vals)
    fig.colorbar(sm, ax=ax, label="Komplexität (Branching Factor)")

    ax.set_title(f"Zugbaum mit Winrate & Komplexität (Tiefe ≤ {MAX_DEPTH})")
    plt.axis("off")
    plt.tight_layout()
    plt.show()
