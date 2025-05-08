import statistics
from functools import lru_cache

import chess
import chess.engine
import networkx as nx
import pandas as pd
from tqdm import tqdm

PARQUET_PATH = "../data/positions_2025_01.parquet"
ENGINE_PATH = "../stockfish/stockfish-windows-x86-64-avx2.exe"
SAMPLE_SIZE = 1000
DEPTH_LEVELS_VARIANCE = 3
VARIANCE_N_BEST_NODES = 3
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


def build_variance_tree(b):
    level_nodes = [b.copy()]
    level_variances = []

    for depth in range(DEPTH_LEVELS_VARIANCE):
        evals: list[int] = []
        next_nodes: list[chess.Board] = []

        for node in level_nodes:
            infos = engine.analyse(
                node,
                chess.engine.Limit(time=ENGINE_LIMIT),
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

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Calculating scores"):
    board = chess.Board(row["fen"])
    side_sign = 1 if board.turn else -1

    df.at[idx, "engine_move"] = engine_play(board.fen())

    base_cp_white = analyse_fen(board.fen())

    board_after = board.copy()
    board_after.push_uci(row["next_move"])

    df.at[idx, "fragility_score"] = compute_fragility_score(board_after)

    cp_white = analyse_fen(board_after.fen())

    df.at[idx, "delta"] = (cp_white - base_cp_white) * side_sign
    df.at[idx, "variance"] = build_variance_tree(board_after)

df.to_parquet("../data/score_dataset.parquet", index=False)
print("✅ Saved stats dataset to 'data/score_dataset.parquet'")

engine.quit()
