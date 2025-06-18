# Elo-Dependent Move Recommendations in Chess

This project provides a pipeline for generating chess move recommendations that are tailored to specific Elo ratings.  
It processes raw game data from Lichess, evaluates each position using Stockfish, derives statistical features based on
positions and moves, trains models for different Elo groups and creates high-quality visualizations for
analysis and presentation.

## Quick Setup

1. **Clone this repository** and set up a Python environment with version 3.8 or higher.
2. **Install all required dependencies** by running `pip install -r requirements.txt`.
3. **Download the data**: Choose any `.pgn.zst` file from the
   official [Lichess database](https://database.lichess.org) and save it in the `data/` folder.
4. **Execute the pipeline**: Run the scripts and notebooks in the correct order to process the data step by step.
5. **Adjust configuration settings** in `pipeline/config.py` as needed. You can:
    * Define which Elos should be analyzed
    * Set the Elo span around each target rating
    * Specify the minimum number of samples required per group for analysis
    * Adjust the winrate threshold used to identify the best move in a given position

## Stockfish

* The project requires **Stockfish version 15 or newer** for position evaluation.
* You can either place the Stockfish binary in the `stockfish/` directory or set the path using the `STOCKFISH_PATH`
  environment variable.
* During evaluation, each position is analyzed using Stockfish at depth 20 to ensure consistent and meaningful centipawn
  scores.

## Research Context

This repository was created as part of my seminar paper in the master's program *Information Systems* at the Chair of
Business Analytics, University of Würzburg.  
The paper establishes the theoretical foundation for Elo-specific move recommendations and positions the empirical
results within the broader context of research on chess analytics.

## Contact

If you have any questions or would like to receive a copy of the seminar paper as a PDF, feel free to reach out to me
via email: [philipp.landeck@stud-mail.uni-wuerzburg.de](mailto:philipp.landeck@stud-mail.uni-wuerzburg.de)
