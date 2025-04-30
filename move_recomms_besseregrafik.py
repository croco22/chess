#%%
import chess
import chess.engine
import pandas as pd
import numpy as np
from functools import lru_cache

# Konfiguration
ENGINE_PATH = "stockfish/stockfish-windows-x86-64-avx2.exe"
MAX_DEPTH = 8
THRESHOLD_CP = 50  # Schwellenwert für gleichwertige Züge

# Engine initialisieren
engine = chess.engine.SimpleEngine.popen_uci(ENGINE_PATH)

#%%
# Beispiel: Laden eines DataFrames mit den Spieldaten
# Die Daten sollten die Spalten 'moves', 'winner' und 'avg_elo' enthalten
df = pd.read_parquet("data/moves_2025_01.parquet")

#%%
@lru_cache(maxsize=None)
def evaluate_move_complexity(fen: str, threshold_cp: int = THRESHOLD_CP):
    board = chess.Board(fen)
    info = engine.analyse(board, chess.engine.Limit(depth=MAX_DEPTH))
    best_cp = info["score"].white().score(mate_score=10000)

    good_moves = 0
    for move in board.legal_moves:
        board.push(move)
        move_info = engine.analyse(board, chess.engine.Limit(depth=MAX_DEPTH))
        move_cp = move_info["score"].white().score(mate_score=10000)
        board.pop()

        if abs(best_cp - move_cp) <= threshold_cp:
            good_moves += 1

    return good_moves

#%%
def recommend_moves(fen: str, player_elo: int, df: pd.DataFrame, top_n: int = 3):
    board = chess.Board(fen)
    legal_moves = list(board.legal_moves)
    recommendations = []

    for move in legal_moves:
        board.push(move)
        new_fen = board.fen()
        board.pop()

        # Filtern der Spiele, die mit dem aktuellen Zug beginnen
        matching_games = df[df['moves'].str.startswith(move.uci())]

        if matching_games.empty:
            continue

        # Berechnung der Gewinnrate
        wins = matching_games[matching_games['winner'] == 1].shape[0]
        total = matching_games.shape[0]
        winrate = wins / total if total > 0 else 0

        # Berechnung der durchschnittlichen Elo
        avg_elo = matching_games['avg_elo'].mean()

        # Bewertung der Zugkomplexität
        complexity = evaluate_move_complexity(new_fen)

        # Bewertung basierend auf Spieler-Elo
        elo_diff = abs(avg_elo - player_elo)
        elo_score = max(0, 1 - elo_diff / 1000)  # Einfaches Modell

        # Gesamtscore berechnen
        score = (winrate * 0.5) + (elo_score * 0.3) + ((1 / (1 + complexity)) * 0.2)

        recommendations.append({
            'move': move.uci(),
            'winrate': winrate,
            'avg_elo': avg_elo,
            'complexity': complexity,
            'score': score
        })

    # Sortieren der Empfehlungen nach Score
    recommendations.sort(key=lambda x: x['score'], reverse=True)

    return recommendations[:top_n]

#%%
# Beispiel-FEN für die Startposition
fen = chess.STARTING_FEN
player_elo = 1500  # Beispiel-Elo

recommendations = recommend_moves(fen, player_elo, df)

for rec in recommendations:
    print(f"Zug: {rec['move']}, Gewinnrate: {rec['winrate']:.2f}, "
          f"Durchschnittliche Elo: {rec['avg_elo']:.0f}, "
          f"Komplexität: {rec['complexity']}, Score: {rec['score']:.2f}")

#%%
elos = [800, 1200, 1600, 2000]

for elo in elos:
    print(f"\nEmpfehlungen für Spieler mit Elo {elo}:")
    recs = recommend_moves(fen, elo, df)
    for rec in recs:
        print(f"Zug: {rec['move']}, Gewinnrate: {rec['winrate']:.2f}, "
              f"Durchschnittliche Elo: {rec['avg_elo']:.0f}, "
              f"Komplexität: {rec['complexity']}, Score: {rec['score']:.2f}")

#%%
import matplotlib.pyplot as plt

def plot_recommendations(recommendations):
    moves = [rec['move'] for rec in recommendations]
    winrates = [rec['winrate'] for rec in recommendations]
    complexities = [rec['complexity'] for rec in recommendations]

    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    ax1.set_xlabel('Zug')
    ax1.set_ylabel('Gewinnrate', color=color)
    ax1.bar(moves, winrates, color=color, alpha=0.6)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Komplexität', color=color)
    ax2.plot(moves, complexities, color=color, marker='o')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.title('Zugempfehlungen: Gewinnrate und Komplexität')
    plt.show()

# Beispielaufruf
plot_recommendations(recommendations)
