import io
import os

import chess.pgn
import pandas as pd
import zstandard as zstd
from tqdm import tqdm

# Paths to the input PGN.zst file and the output Parquet file
zst_path = "../data/lichess_db_standard_rated_2025-01.pgn.zst"
parquet_path = "../data/data_2025_01.parquet"

# Temporary storage for parsed games
games_data = []
batch_size = 10_000  # Number of games to store before writing to disk
max_games = 10_000_000  # Total number of games to process

# Check if a Parquet file already exists to determine where to resume
if os.path.exists(parquet_path):
    df_exist = pd.read_parquet(parquet_path, columns=["game_id"])
    game_id = int(df_exist["game_id"].max())
else:
    game_id = 1

# Open and decompress the PGN.zst file
with open(zst_path, 'rb') as compressed:
    decompressor = zstd.ZstdDecompressor()
    with decompressor.stream_reader(compressed) as reader:
        text_stream = io.TextIOWrapper(reader, encoding='utf-8', errors='ignore')

        # Skip already processed games
        if game_id > 1:
            with tqdm(total=game_id, desc="Skipping parsed games") as pbar:
                for _ in range(game_id):
                    chess.pgn.skip_game(text_stream)
                    pbar.update(1)

        # Start parsing new games
        with tqdm(initial=game_id, total=max_games, desc="Processing games") as pbar:
            while game_id <= max_games:
                game = chess.pgn.read_game(text_stream)
                if game is None:
                    break  # End of file

                game_id += 1
                headers = game.headers

                # Skip games involving bots
                if headers.get("WhiteTitle", "") == "BOT" or headers.get("BlackTitle", "") == "BOT":
                    continue

                # Determine winner
                result = headers.get("Result", "")
                winner = {"1-0": 1, "0-1": 2}.get(result, 0)

                # Collect all moves in UCI notation
                moves = []
                node = game
                while not node.is_end():
                    node = node.variation(0)
                    moves.append(node.move.uci())

                # Append game data to list
                games_data.append({
                    "game_id": game_id,
                    "event": str(headers.get("Event", "")),
                    "white_elo": int(headers.get("WhiteElo", 0)),
                    "black_elo": int(headers.get("BlackElo", 0)),
                    "opening": str(headers.get("Opening", "")),
                    "winner": winner,
                    "moves": " ".join(moves)
                })

                pbar.update(1)

                # Write batch to Parquet
                if game_id % batch_size == 0:
                    df = pd.DataFrame(games_data)
                    if os.path.exists(parquet_path):
                        df.to_parquet(parquet_path, engine="fastparquet", append=True)
                    else:
                        df.to_parquet(parquet_path)
                    games_data = []
