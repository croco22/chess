import zstandard as zstd
import chess.pgn
import pandas as pd
import multiprocessing as mp
import io
import os
from tqdm import tqdm

zst_path = "data/lichess_db_standard_rated_2025-03.pgn.zst"
parquet_path = "data-2025-03.parquet"
batch_size = 1000
max_games = 1_000_000
num_workers = mp.cpu_count()

existing_ids = set()
if os.path.exists(parquet_path):
    existing_df = pd.read_parquet(parquet_path, columns=["game_id"])
    existing_ids = set(existing_df["game_id"])


def parse_games_batch(batch_bytes, start_id):
    games = []
    text_stream = io.TextIOWrapper(io.BytesIO(batch_bytes), encoding='utf-8', errors='ignore')
    for i in range(batch_size):
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

        moves = []
        node = game
        while not node.is_end():
            node = node.variation(0)
            moves.append(node.move.uci())

        game_id = start_id + i
        if game_id in existing_ids:
            continue

        games.append({
            "game_id": game_id,
            "white_elo": int(headers.get("WhiteElo", 0)),
            "black_elo": int(headers.get("BlackElo", 0)),
            "opening": headers.get("Opening", ""),
            "winner": winner,
            "termination": headers.get("Termination", ""),
            "moves": " ".join(moves)
        })

    return games


def stream_pgn_batches():
    with open(zst_path, 'rb') as compressed:
        decompressor = zstd.ZstdDecompressor()
        with decompressor.stream_reader(compressed) as reader:
            buf = b""
            game_count = 0
            batch = []
            while game_count < max_games:
                chunk = reader.read(65536)
                if not chunk:
                    break
                buf += chunk
                while b"\n\n\n" in buf and game_count < max_games:
                    split_idx = buf.find(b"\n\n\n") + 3
                    game_pgn = buf[:split_idx]
                    batch.append(game_pgn)
                    buf = buf[split_idx:]
                    game_count += 1
                    if len(batch) == batch_size:
                        yield b"".join(batch), game_count - batch_size
                        batch = []
            if batch:
                yield b"".join(batch), game_count - len(batch)


with mp.Pool(num_workers) as pool:
    for batch_bytes_o, start_id_o in tqdm(stream_pgn_batches(), total=max_games // batch_size):
        f_result = pool.apply_async(parse_games_batch, args=(batch_bytes_o, start_id_o))
        games_data = f_result.get()
        if games_data:
            df = pd.DataFrame(games_data)
            df.to_parquet(parquet_path, index=False, compression="snappy", append=True)
