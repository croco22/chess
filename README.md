# Elo-Dependent Move Recommendations in Chess

A concise pipeline to generate Elo-dependent move recommendations in chess by processing raw Lichess data, computing position‐move features, training simple models per Elo bracket, and producing visualizations.

## Purpose

- Convert Lichess PGN data into Parquet tables of positions and moves.
- Compute three core features for each (FEN, move) pair:
  1. **Engine‐Delta**: Centipawn change when a move is played.
  2. **Fragility Score**: Sum of betweenness centralities in an attack/defense graph.
  3. **Variance Score**: Average variance of top‐3 continuation evaluations over several plies.
- Label moves as “optimal” if their empirical win‐rate (within an Elo window) is within 1% of the top move.
- Train and evaluate simple models (e.g., logistic regression) per Elo bracket to predict which moves are optimal.
- Generate static visualizations (feature importance plots, win‐rate curves, move trees).

## Key Scripts & Notebooks

- **p1_pgn_to_parquet.py**  
  Reads `lichess_db_standard_rated_2025-01.pgn.zst`, extracts plies 10–25, and writes Parquet tables for positions (`positions_2025_01.parquet`) and moves (`moves_2025_01.parquet`).

- **p2_parse_positions.ipynb**  
  Loads and filters position/move tables to standard‐rated human games and top openings; saves filtered data (e.g., `data_2025_01.parquet`).

- **p3_calculate_scores.py**  
  For each (FEN, move) with frequency ≥ 100:
  - Runs Stockfish (depth 20) before and after the move → computes Engine‐Delta.
  - Builds an attack/defense graph via NetworkX → computes Fragility Score (sum of centralities).
  - Examines the top 3 Stockfish candidates and their 5‐ply evaluations → computes Variance Score.
  Outputs `score_dataset.parquet` and a smaller sample `score_dataset_48h.parquet`.

- **p4_modified_preprocessing.py**  
  Merges `score_dataset.parquet` with raw win flags from `moves_2025_01.parquet` and `positions_2025_01.parquet`, filters out low‐frequency pairs, computes empirical win‐rates per Elo bucket (±50 Elo smoothing), labels “optimal” moves, and writes:
  - `recommendation_dataset.parquet` (features + labels)
  - `stats_dataset.parquet` (summary statistics per (FEN, Elo bucket)).

- **p5_feature_importance.ipynb**  
  Loads `recommendation_dataset.parquet`, trains separate logistic regression models for each Elo bracket, and produces:
  - `feature_importance.svg` (standardized importance of Engine‐Delta, Fragility, Variance)
  - `feature_coefficients.svg` (raw coefficient values).

- **p6_result_analysis.ipynb**  
  Evaluates model accuracy versus baseline (Engine move), generates:
  - `prediction_accuracy.svg`
  - `total_performance.svg`
  - `elo_distribution.svg`
  - `winrate_example.svg`

- **generate_move_tree.py**  
  Takes a FEN string and Elo bracket, reads a Parquet slice (e.g., from `recommendation_dataset.parquet`), and draws a directed move tree annotated by frequency and win‐rate. Saved as PNG under `images/`.

## Data Files

- **lichess_db_standard_rated_2025-01.pgn.zst**  
  Raw Lichess January 2025 dump.

- **Parquet Tables** (all under `data/`):
  - `positions_2025_01.parquet` (filtered plies 10–25; columns: FEN, halfmove index, Elo, result).
  - `moves_2025_01.parquet` (FEN, UCI move, Elo, win flag).
  - `data_2025_01.parquet` (filtered subset of positions & moves).
  - `score_dataset.parquet` (Engine‐Delta, Fragility, Variance per (FEN, move), plus raw win flags).
  - `score_dataset_48h.parquet` (48-hour sample for quick prototyping).
  - `recommendation_dataset.parquet` (merged features, labels, win rates).
  - `stats_dataset.parquet` (summary stats per (FEN, Elo bucket)).

## Visualizations

Final static plots live in `images/`:
- **elo_distribution.svg**  
- **feature_coefficients.svg**  
- **feature_importance.svg**  
- **prediction_accuracy.svg**  
- **selection_decision.svg**  
- **total_performance.svg**  
- **winrate_example.svg**  
- **move_tree_<timestamp>.png** (one or more examples).

## Stockfish Integration

- Requires Stockfish v15+ accessible via a local binary under `stockfish/` or defined by the `STOCKFISH_PATH` environment variable.
- Scripts in **p3_calculate_scores.py** call Stockfish at depth 20 using a Python UCI‐wrapper to obtain centipawn evaluations.
