import os
import random
import chess
import chess.engine
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Konfiguration
PARQUET_PATH = "data/moves_2025_01.parquet"
ENGINE_PATH = "stockfish/stockfish-windows-x86-64-avx2.exe"
MAX_DEPTH = 8
ELO_LEVELS = [800, 1200, 1600, 2000, 2400]

# Engine initialisieren
engine = chess.engine.SimpleEngine.popen_uci(ENGINE_PATH)

def get_random_position(df):
    """
    Wählt eine zufällige Stellung aus dem Datensatz aus.
    """
    row = df.sample(1).iloc[0]
    moves = row["moves"].split()
    if len(moves) < 2:
        return chess.STARTING_FEN
    cutoff = random.randint(1, len(moves) - 1)
    board = chess.Board()
    for move in moves[:cutoff]:
        try:
            board.push_uci(move)
        except:
            break
    return board.fen()

def evaluate_moves(fen):
    """
    Bewertet alle legalen Züge in der gegebenen Stellung.
    """
    board = chess.Board(fen)
    move_scores = []
    for move in board.legal_moves:
        board.push(move)
        info = engine.analyse(board, chess.engine.Limit(depth=MAX_DEPTH))
        score = info["score"].white().score(mate_score=10000)
        board.pop()
        move_scores.append((move, score))
    return move_scores

def suggest_move(move_scores, player_elo):
    """
    Gibt den empfohlenen Zug basierend auf der Spieler-ELO zurück.
    """
    if player_elo < 1200:
        # Anfänger: Züge mit geringer Bewertung bevorzugen
        move_scores.sort(key=lambda x: abs(x[1]))
    elif player_elo < 1800:
        # Fortgeschrittene: Balance zwischen Bewertung und Komplexität
        move_scores.sort(key=lambda x: abs(x[1] - 50))
    else:
        # Experten: Beste Bewertungen bevorzugen
        move_scores.sort(key=lambda x: -x[1])
    return move_scores[0]

def plot_recommendations(recommendations):
    """
    Visualisiert die Bewertungen der empfohlenen Züge für verschiedene ELO-Stufen.
    """
    elos = [rec[0] for rec in recommendations]
    scores = [rec[2] for rec in recommendations]
    moves = [rec[1] for rec in recommendations]

    plt.figure(figsize=(10, 6))
    bars = plt.bar([str(elo) for elo in elos], scores, color='skyblue')
    plt.xlabel("Spieler-ELO")
    plt.ylabel("Bewertung (Centipawns)")
    plt.title("Empfohlene Züge für verschiedene ELO-Stufen")
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Beschriftungen hinzufügen
    for bar, move in zip(bars, moves):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 5, move, ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

def main():
    # Datensatz laden
    if not os.path.exists(PARQUET_PATH):
        print(f"Datei {PARQUET_PATH} nicht gefunden.")
        return
    df = pd.read_parquet(PARQUET_PATH)

    # Zufällige Stellung auswählen
    fen = get_random_position(df)
    print(f"Ausgewählte Stellung (FEN): {fen}")

    # Züge bewerten
    move_scores = evaluate_moves(fen)

    # Empfehlungen für verschiedene ELO-Stufen
    recommendations = []
    for elo in ELO_LEVELS:
        move, score = suggest_move(move_scores, elo)
        board = chess.Board(fen)
        san_move = board.san(move)
        print(f"ELO {elo}: Empfohlener Zug: {san_move}, Bewertung: {score}")
        recommendations.append((elo, san_move, score))

    # Visualisierung
    plot_recommendations(recommendations)

    # Engine beenden
    engine.quit()

if __name__ == "__main__":
    main()
