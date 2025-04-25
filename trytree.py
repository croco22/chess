import chess
import chess.engine
import matplotlib.pyplot as plt

# Deine Züge als String
moves_str = "b1c3 e7e5 e2e4 f8c5 d1h5 g8f6 h5e5 c5e7 d2d3 d7d6 e5f4 f6h5 f4f3 h5f6 f3g3 f6g4 f1e2 h7h5 e2g4 c8g4 f2f3 h5h4 g3g4 g7g6 c3d5 e7f8 c1g5 d8d7 d5f6"
moves = moves_str.split()

# Starte ein Board
board = chess.Board()

# Starte Stockfish Engine (Pfad anpassen!)
engine = chess.engine.SimpleEngine.popen_uci("stockfish/stockfish-windows-x86-64-avx2.exe")  # z.B. Windows: "C:/path/to/stockfish.exe"

evals = []
positions = []

for i, move in enumerate(moves):
    board.push_uci(move)

    # Bewertung nach jedem Zug
    info = engine.analyse(board, chess.engine.Limit(depth=15))
    score = info["score"].white()

    if score.is_mate():
        # Mate wird als +/-1000 gesetzt
        eval_cp = 1000 if score.mate() > 0 else -1000
    else:
        eval_cp = score.score()

    # In Wahrscheinlichkeit umrechnen (simple sigmoid auf 0-1 skaliert)
    win_prob = 1 / (1 + 10 ** (-eval_cp / 400))
    evals.append(win_prob)
    positions.append(i + 1)

engine.quit()

# Plotten
plt.figure(figsize=(12, 6))
plt.plot(positions, evals, marker='o', label="Weiß gewinnt-Wahrscheinlichkeit")
plt.axhline(0.5, color='gray', linestyle='--')
plt.title("Gewinnwahrscheinlichkeit für Weiß nach jedem Zug")
plt.xlabel("Zugnummer")
plt.ylabel("Wahrscheinlichkeit")
plt.ylim(0, 1)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
