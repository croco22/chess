import statistics
from functools import lru_cache

import chess
import chess.engine
import networkx as nx
import pandas as pd
from tqdm import tqdm

PARQUET_PATH = "../data/positions_2025_01.parquet"
ENGINE_PATH = "../stockfish/stockfish-windows-x86-64-avx2.exe"
DEPTH_LEVELS_VARIANCE = 5   # Number of levels in the variance tree
VARIANCE_N_BEST_NODES = 3   # How many best continuations to evaluate per node
MATE_SCORE = 10_000         # Score used to normalize mate evaluations
LRU_CACHE_SIZE = None       # Unlimited cache size
ENGINE_LIMIT = 20           # Stockfish search depth


# Cached function to get best move from the engine for a given FEN
@lru_cache(maxsize=LRU_CACHE_SIZE)
def engine_play(fen: str) -> str:
    b = chess.Board(fen)
    result = engine.play(b, chess.engine.Limit(depth=ENGINE_LIMIT))
    return str(result.move)


# Cached function to get the evaluation (in centipawns) of a position
@lru_cache(maxsize=LRU_CACHE_SIZE)
def analyse_fen(fen: str) -> int:
    b = chess.Board(fen)
    info = engine.analyse(b, chess.engine.Limit(depth=ENGINE_LIMIT))
    return info["score"].white().score(mate_score=MATE_SCORE)


# Cached function to compute a "fragility score" using piece interaction graph
@lru_cache(maxsize=LRU_CACHE_SIZE)
def compute_fragility_score(fen: str) -> float:
    b = chess.Board(fen)
    graph = nx.DiGraph()
    piece_map = b.piece_map()

    # Add all pieces as nodes
    for square, piece in piece_map.items():
        graph.add_node(square, piece=piece)

    # Add edges for direct attacks
    for square, piece in piece_map.items():
        attacks = b.attacks(square)
        for target in attacks:
            if target in piece_map:
                graph.add_edge(square, target)

    # Use betweenness centrality as a proxy for fragility
    centrality = nx.betweenness_centrality(graph)
    fragility = sum(centrality.values())
    return fragility


# Cached function to build a variance tree and calculate evaluation variance across branches
@lru_cache(maxsize=LRU_CACHE_SIZE)
def build_variance_tree(fen: str) -> float:
    b = chess.Board(fen)
    level_nodes = [b.copy()]
    level_variances = []

    # Explore multiple levels of move continuations
    for depth in range(DEPTH_LEVELS_VARIANCE):
        evals: list[int] = []
        next_nodes: list[chess.Board] = []

        for node in level_nodes:
            infos = engine.analyse(
                node,
                chess.engine.Limit(depth=ENGINE_LIMIT),
                multipv=VARIANCE_N_BEST_NODES
            )

            for info in infos:
                if "pv" not in info or not info["pv"]:
                    continue
                move = info["pv"][0]
                child = node.copy()
                child.push(move)

                sc = info["score"].white().score(mate_score=MATE_SCORE)
                evals.append(sc)
                next_nodes.append(child)

        # Store variance of scores at this level
        if len(evals) > 1:
            level_variances.append(statistics.pvariance(evals))
        else:
            level_variances.append(0.0)

        level_nodes = next_nodes
        if not level_nodes:
            break

    # Return the average variance over all levels
    return statistics.mean(level_variances) if level_variances else 0.0


df = pd.read_parquet(PARQUET_PATH)
engine = chess.engine.SimpleEngine.popen_uci(ENGINE_PATH)

# Evaluate each position in the dataset
for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating positions"):
    board = chess.Board(row["fen"])
    base_cp_white = analyse_fen(board.fen())
    side_sign = 1 if board.turn else -1  # Adjust perspective based on turn

    # Store best move from engine
    df.at[idx, "engine_move"] = engine_play(board.fen())

    # Evaluate the actual move played
    board_after = board.copy()
    board_after.push_uci(row["next_move"])
    cp_white = analyse_fen(board_after.fen())

    # Compute evaluation difference from engine POV
    df.at[idx, "delta"] = (cp_white - base_cp_white) * side_sign

    # Compute fragility and variance features for the resulting position
    df.at[idx, "fragility_score"] = compute_fragility_score(board_after.fen())
    df.at[idx, "variance"] = build_variance_tree(board_after.fen())

engine.quit()

df.to_parquet("../data/score_dataset.parquet", index=False)
print("✅ Engine evaluation results saved to 'data/score_dataset.parquet'")
