import sys
import chess.engine

# Passed Parameter
board = chess.Board(sys.argv[1])

# Stockfish Engine
engine_path = "stockfish/stockfish-windows-x86-64-avx2.exe"
engine = chess.engine.SimpleEngine.popen_uci(engine_path)

result = engine.play(board, chess.engine.Limit(time=5.0))
engine.quit()

print(result.move)
