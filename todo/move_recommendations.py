import random
import chess
import chess.engine
import chess.engine
import matplotlib.pyplot as plt
import pandas as pd

PARQUET_PATH = "../data/moves_2025_01.parquet"
ENGINE_PATH = "../stockfish/stockfish-windows-x86-64-avx2.exe"
ELO_LEVELS = None  # e.g. [323, 2353, 563, 1382]
ELO_DEVIATION = 0.10

RATING_WEIGHT = 0.25
COMPLEXITY_WEIGHT = 0.25
HISTORICAL_WEIGHT = 1 - RATING_WEIGHT - COMPLEXITY_WEIGHT


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


def get_global_best_move(fen, e):
    board = chess.Board(fen)
    result = e.play(board, chess.engine.Limit(time=1.0))
    return result.move


def analyze_moves(data, stockfish, fen, e):
    board = chess.Board(fen)

    low, high = e * (1 - ELO_DEVIATION), e * (1 + ELO_DEVIATION)
    subset = data[(data['avg_elo'] >= low) & (data['avg_elo'] <= high)]
    if subset.empty:
        return None

    data = []
    moves_series = subset['moves']
    winners = subset['winner']

    for move in board.legal_moves:
        board.push(move)
        # Rating
        info = stockfish.analyse(board, chess.engine.Limit(time=0.5))
        score = info['score'].white().score(mate_score=10_000) or 0
        # Complexity
        complexity = board.legal_moves.count()
        board.pop()

        # Historical Winrate
        uci = move.uci()
        mask = moves_series.str.contains(uci)
        count = mask.sum()
        if count > 0:
            wins = winners[mask].sum()
            hist_winrate = wins / count
        else:
            hist_winrate = 0.0

        data.append({
            'move': move,
            'score': score,
            'complexity': complexity,
            'hist_winrate': hist_winrate
        })

    def normalize(arr):
        mn, mx = min(arr), max(arr)
        return [(x - mn) / (mx - mn) if mx > mn else 0.0 for x in arr]

    scores_norm = normalize([d['score'] for d in data])
    comps_norm = normalize([d['complexity'] for d in data])

    for i, d in enumerate(data):
        d['combined'] = (
                RATING_WEIGHT * scores_norm[i]
                + COMPLEXITY_WEIGHT * (1 - comps_norm[i])
                + HISTORICAL_WEIGHT * d['hist_winrate']
        )

    best = max(data, key=lambda d: d['combined'], default=None)
    return best['move'] if best else None


def plot_recommendations(df, engine, fen, elos):
    xs, ys, labels = [], [], []
    for elo in elos:
        mv = analyze_moves(df, engine, fen, elo)
        if mv is None:
            continue
        board = chess.Board(fen)
        board.push(mv)
        info = engine.analyse(board, chess.engine.Limit(depth=8))
        score = info['score'].white().score(mate_score=10_000) or 0
        complexity = board.legal_moves.count()
        xs.append(complexity)
        ys.append(score)
        labels.append(str(elo))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(xs, ys)
    for x, y, l in zip(xs, ys, labels):
        ax.annotate(l, (x, y))
    ax.set_xlabel('Komplexität')
    ax.set_ylabel('Engine-Bewertung')
    ax.set_title('Best Moves nach ELO: Komplexität vs Bewertung')
    plt.tight_layout()
    plt.show()


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

    for elo in sorted(ELO_LEVELS):
        best_move = analyze_moves(df, engine, fen_position, elo)
        if best_move:
            print(f"\t* ELO {elo}: {best_move}")
        else:
            print(f"\t* ELO {elo}: No data for this position and elo segment")

    plot_recommendations(df, engine, fen_position, ELO_LEVELS)

    engine.quit()
