import ast
import json
import sys

import chess.engine
import networkx as nx


def compute_fragility_score(b):
    graph = nx.DiGraph()
    piece_map = b.piece_map()
    for square, piece in piece_map.items():
        graph.add_node(square, piece=piece)

    for square, piece in piece_map.items():
        attacks = b.attacks(square)
        for target in attacks:
            if target in piece_map:
                graph.add_edge(square, target)

    centrality = nx.betweenness_centrality(graph)
    fragility = sum(centrality.values())
    return fragility


# Passed Parameters
board = chess.Board(sys.argv[1])
moves = ast.literal_eval(sys.argv[2])

# Stockfish Engine
engine_path = "stockfish/stockfish-windows-x86-64-avx2.exe"
engine = chess.engine.SimpleEngine.popen_uci(engine_path)

# Baseline
base_info = engine.analyse(board, chess.engine.Limit(time=1.0))
base_score = base_info['score'].white().score(mate_score=10_000)

results = []
for mv in moves:
    board.push_uci(mv)
    info = engine.analyse(board, chess.engine.Limit(time=1.0))
    score = info['score'].white().score(mate_score=10_000)
    complexity = compute_fragility_score(board)
    board.pop()
    results.append({
        'next_move': mv,
        'delta': score,
        'complexity': complexity
    })

engine.quit()
print(json.dumps(results))
