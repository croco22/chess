import pandas as pd
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt

# ========== Teil 1: Daten laden und Zugbaum aufbauen ==========
parquet_path = "data-2025-01.parquet"
df = pd.read_parquet(parquet_path)

def make_node():
    return {"count": 0, "wins": 0, "children": defaultdict(make_node)}

root = make_node()
max_depth = 10  # 🔻 Begrenzung auf 10 Ebenen

for _, row in df.iterrows():
    winner = row["winner"]
    moves = row["moves"].split()
    node = root
    depth = 0

    for move in moves:
        if depth >= max_depth:
            break  # 🔻 Ab hier abgeschnitten (Tiefe 10 erreicht)
        node = node["children"][move]
        node["count"] += 1
        if winner == 1:  # Weiß gewinnt
            node["wins"] += 1
        depth += 1

# ========== Teil 2: Graph erstellen ==========

G = nx.DiGraph()

def add_to_graph(node, parent_name="", depth=0, max_depth=10):
    if depth >= max_depth:
        return
    for move, child in node["children"].items():
        winrate = (child["wins"] / child["count"]) * 100 if child["count"] > 0 else 0
        node_name = f"{move} ({winrate:.1f}%)"
        G.add_node(node_name)
        if parent_name:
            G.add_edge(parent_name, node_name)
        add_to_graph(child, node_name, depth + 1, max_depth)

add_to_graph(root)

# ========== Teil 3: Graph plotten ==========

plt.figure(figsize=(20, 12))
pos = nx.nx_pydot.graphviz_layout(G, prog="dot")  # hierarchisch (top-down)

nx.draw(
    G,
    pos,
    with_labels=True,
    node_size=2000,
    node_color="#ddddff",
    font_size=8,
    font_family="monospace",
    arrows=False
)
plt.title("Zugbaum mit Winrates (bis Tiefe 10)", fontsize=14)
plt.tight_layout()
plt.show()
