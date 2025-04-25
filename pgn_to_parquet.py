import zstandard as zstd
import chess.pgn
import pandas as pd
import io
from tqdm import tqdm
import os

zst_path = "data/lichess_db_standard_rated_2025-01.pgn.zst"
parquet_path = "data-2025_01.parquet"
max_games = 1_000_000
batch_size = 100

games_data = []

if os.path.exists(parquet_path):
    df_exist     = pd.read_parquet(parquet_path, columns=["game_id"])
    game_id = int(df_exist["game_id"].max())
else:
    game_id = 1

with open(zst_path, 'rb') as compressed:
    decompressor = zstd.ZstdDecompressor()
    with decompressor.stream_reader(compressed) as reader:
        text_stream = io.TextIOWrapper(reader, encoding='utf-8', errors='ignore')

        if game_id > 1:
            with tqdm(total=game_id, desc="Skipping parsed games") as pbar:
                for _ in range(game_id):
                    chess.pgn.skip_game(text_stream)
                    pbar.update(1)

        with tqdm(initial=game_id, total=max_games, desc="Processing games") as pbar:
            while game_id <= max_games:
                game = chess.pgn.read_game(text_stream)
                if game is None:
                    break  # EOF

                game_id += 1
                headers = game.headers

                if headers.get("Event", "") != "Rated Classical game":
                    continue
                if headers.get("WhiteTitle", "") == "BOT":
                    continue
                if headers.get("BlackTitle", "") == "BOT":
                    continue

                result = headers.get("Result")
                if result == "1-0":
                    winner = 1
                elif result == "0-1":
                    winner = 2
                else:
                    winner = 0

                # Züge extrahieren
                moves = []
                node = game
                while not node.is_end():
                    node = node.variation(0)
                    moves.append(node.move.uci())

                games_data.append({
                    "game_id": game_id,
                    "white_elo": int(headers.get("WhiteElo", 0)),
                    "black_elo": int(headers.get("BlackElo", 0)),
                    "opening": headers.get("Opening", ""),
                    "winner": winner,
                    "moves": " ".join(moves)
                })

                pbar.update(1)

                if game_id % batch_size == 0:
                    df = pd.DataFrame(games_data)
                    if os.path.exists(parquet_path):
                        df.to_parquet(parquet_path, index=False, engine="fastparquet", append=True)
                    else:
                        df.to_parquet(parquet_path, index=False)
                    games_data = []
