import random
import chess
import chess.engine
import pandas as pd

PARQUET_PATH = "data/moves_2025_01.parquet"
ENGINE_PATH = "stockfish/stockfish-windows-x86-64-avx2.exe"
ELO_LEVELS = None  # e.g. [323, 2353, 563, 1382]


def get_random_position(data):
    row = data.sample(1).iloc[0]
    moves = row["moves"].split()
    if len(moves) < 2:
        return chess.STARTING_FEN
    cutoff = random.randint(1, len(moves) - 1)
    board = chess.Board()
    for uci in moves[:cutoff]:
        board.push_uci(uci)
    return board.fen()


def get_global_best_move(fen, stockfish):
    board = chess.Board(fen)
    result = stockfish.play(board, chess.engine.Limit(time=1.0))
    return result.move


def get_historical_best_move(fen, data, e):
    low, high = 0.9 * elo, 1.1 * elo
    subset = data[(data['avg_elo'] >= low) & (data['avg_elo'] <= high)]
    if subset.empty:
        return None

    target_parts = fen.split()[:4]
    move_counts = {}

    for moves_str in subset['moves']:
        moves = moves_str.split()
        board = chess.Board()
        for idx, uci in enumerate(moves[:-1]):
            board.push_uci(uci)
            if board.fen().split()[:4] == target_parts:
                next_uci = moves[idx + 1]
                move_counts[next_uci] = move_counts.get(next_uci, 0) + 1
                break

    if not move_counts:
        return None

    best_uci = max(move_counts, key=move_counts.get)
    return best_uci


if __name__ == "__main__":
    df = pd.read_parquet(PARQUET_PATH)
    engine = chess.engine.SimpleEngine.popen_uci(ENGINE_PATH)

    fen_position = get_random_position(df)
    print(f"Position (FEN): {fen_position}")

    best_global = get_global_best_move(fen_position, engine)
    print(f"\t* Global best move: {best_global}")

    if not ELO_LEVELS:
        min_elo = int(df["avg_elo"].min())
        max_elo = int(df["avg_elo"].max())
        ELO_LEVELS = random.sample(range(min_elo, max_elo + 1), 5)

    for elo in ELO_LEVELS:
        best_historical = get_historical_best_move(fen_position, df, elo)
        if best_historical:
            print(f"\t* ELO {elo}: Historical best move: {best_historical}")
        else:
            print(f"\t* ELO {elo}: No data for this position and elo segment")

    engine.quit()
