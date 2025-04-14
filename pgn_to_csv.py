import zstandard as zstd
import chess.pgn
import pandas as pd
import io
from tqdm import tqdm

zst_path = "data/lichess_db_standard_rated_2025-01.pgn.zst"
csv_path = "data-2025-01.csv"
max_games = 20_000

games_data = []
game_id = 1

with open(zst_path, 'rb') as compressed:
    decompressor = zstd.ZstdDecompressor()
    with decompressor.stream_reader(compressed) as reader:
        text_stream = io.TextIOWrapper(reader, encoding='utf-8', errors='ignore')

        with tqdm(total=max_games, desc="Processing games") as pbar:
            while game_id <= max_games:
                game = chess.pgn.read_game(text_stream)
                if game is None:
                    break

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

                games_data.append({
                    "game_id": game_id,
                    "white_elo": headers.get("WhiteElo", ""),
                    "black_elo": headers.get("BlackElo", ""),
                    "opening": headers.get("Opening", ""),
                    "winner": winner
                })

                game_id += 1
                pbar.update(1)

df = pd.DataFrame(games_data)
df.to_csv(csv_path, index=False)
print(f"✅ CSV saved to: {csv_path}")
