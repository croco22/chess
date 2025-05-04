import sys
import chess.engine

board = chess.Board(sys.argv[1])
engine = chess.engine.SimpleEngine.popen_uci("stockfish/stockfish-windows-x86-64-avx2.exe")
result = engine.play(board, chess.engine.Limit(time=1.0))
engine.quit()
print(result.move)
