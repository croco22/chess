"""
Create a supervised‑learning dataset that links the engine‑based factors (delta,
complexity, avg_variance) to the empirical best move for players around a given
Elo.  For every random position and several randomly drawn Elos, the script:

1.  Builds a candidate list of next moves that occur at least MIN_CANDIDATE_COUNT
    times in the historical games file.
2.  Calls evaluate_engine_variance.py once per position (MultiPV) to obtain the
    engine factors for all candidates in one shot (delta, complexity,
    avg_variance).
3.  Scores every candidate with the weighted formula from the research code and
    selects the engine‑recommended move.
4.  Compares that recommendation against the historically most successful move
    among players of similar strength (±ELO_DEV) and records whether the engine
    was *correct*.
5.  Stores one row per (position, Elo) with all factors and the binary label
    `is_correct`.

The resulting Parquet file `data/recommendation_dataset.parquet` is ready for
exploratory analysis / modelling in a notebook.
"""

import statistics
from functools import lru_cache

import chess
import chess.engine
import networkx as nx
import pandas as pd
from tqdm import tqdm

PARQUET_PATH = "data/positions_2025_01.parquet"
ENGINE_PATH = "stockfish/stockfish-windows-x86-64-avx2.exe"
ELO_MIN, ELO_MAX = 500, 2500
ELO_DEV = 250
SAMPLE_SIZE = 1000
DEPTH_LEVELS = 3
WIDTH = 3
MATE_SCORE = 10_000
LRU_CACHE_SIZE = 10_000
ENGINE_LIMIT = 0.1


@lru_cache(maxsize=LRU_CACHE_SIZE)
def engine_play(fen: str) -> str:
    b = chess.Board(fen)
    result = engine.play(b, chess.engine.Limit(time=ENGINE_LIMIT))
    return str(result.move)


@lru_cache(maxsize=LRU_CACHE_SIZE)
def analyse_fen(fen: str):
    b = chess.Board(fen)
    info = engine.analyse(b, chess.engine.Limit(time=ENGINE_LIMIT))
    return info["score"].white().score(mate_score=MATE_SCORE)


def compute_fragility_score(b: chess.Board) -> float:
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


def build_variance_tree(b, sign):
    level_nodes = [b.copy()]
    level_variances = []

    for depth in range(DEPTH_LEVELS):
        evals: list[int] = []
        next_nodes: list[chess.Board] = []

        for node in level_nodes:
            infos = engine.analyse(node, chess.engine.Limit(time=ENGINE_LIMIT), multipv=WIDTH)

            for info in infos:
                move = info["pv"][0]
                child = node.copy()
                child.push(move)

                score = info['score'].white().score(mate_score=MATE_SCORE)
                sc = score * sign
                evals.append(sc)
                next_nodes.append(child)

        if len(evals) > 1:
            level_variances.append(statistics.pvariance(evals))
        else:
            level_variances.append(0.0)

        level_nodes = next_nodes
        if not level_nodes:
            break

    return statistics.mean(level_variances) if level_variances else 0.0


df = pd.read_parquet(PARQUET_PATH).sample(SAMPLE_SIZE)
engine = chess.engine.SimpleEngine.popen_uci(ENGINE_PATH)

for idx, row in tqdm(df.iterrows(), total=len(df), desc="analysing"):
    board = chess.Board(row["fen"])
    side_sign = 1 if board.turn else -1

    df.at[idx, "engine_move"] = engine_play(board.fen())

    base_cp_white = analyse_fen(board.fen())

    board_after = board.copy()
    board_after.push_uci(row["next_move"])

    df.at[idx, "fragility_score"] = compute_fragility_score(board_after)

    cp_white = analyse_fen(board_after.fen())

    df.at[idx, "delta"] = (cp_white - base_cp_white) * side_sign
    df.at[idx, "variance"] = build_variance_tree(board_after, side_sign)

df["winrate_white"] = df.groupby(["fen", "next_move"])["win_pov"].transform("mean")
idx_best = df.groupby("fen")["winrate_white"].idxmax()
best_moves = df.loc[idx_best, ["fen", "next_move"]].set_index("fen")["next_move"]
df["historical_best"] = df["fen"].map(best_moves)

df["score_test"] = df["fragility_score"] + df["delta"] + df["variance"]
idx_best = df.groupby("fen")["score_test"].idxmax()
best_moves = df.loc[idx_best, ["fen", "next_move"]].set_index("fen")["next_move"]
df["recommended_move"] = df["fen"].map(best_moves)

df["is_best"] = df["recommended_move"].eq(df["historical_best"])

engine.quit()
df.to_parquet("data/stats_dataset.parquet", index=False)
print("saved")
