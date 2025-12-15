# vgc-lab

Python wrapper and utilities for Pokemon Showdown CLI, focused on Gen 9 VGC Reg F battle simulation and data collection.

## Overview

`vgc-lab` is a helper project that wraps the sibling `pokemon-showdown/` folder (a fork of smogon/pokemon-showdown). It provides Python utilities to:

- Generate random teams for VGC formats
- Validate teams against format rules
- Run CLI simulated battles (AI vs AI) without requiring the web server
- Save battle logs and structured metadata under `vgc-lab/data/`

## Project Structure

- `src/vgc_lab/` - Main Python package (see [Core Source Modules](#core-source-modules-simplified-architecture) below)
- `scripts/cli.py` - Unified CLI entrypoint for all commands
- `data/` - Generated data (teams, battle logs, JSON results)

## Team Preview Dataset

`vgc-lab` can generate structured team preview datasets from simulated battles:

- **TeamPreviewSnapshot**: Represents the open team sheet state at the start of a Reg F battle, including both players' full teams (6 Pokémon each) with moves, items, abilities, and Tera types
- **Generation method**: Uses `generate_random_team` + `simulate_battle` to create battle logs, then parses team preview requests from the logs
- **Dataset location**: `data/datasets/team_preview/team_preview.jsonl` (JSON Lines format, one snapshot per line)
- **Usage example**:
  ```bash
  cd Pokai/vgc-lab
  python scripts/cli.py dataset team-preview --n 50
  ```

This dataset is intended as the input space for future agents making team-preview decisions (which 4 mons to bring, BO3 planning, etc.), but RL/ML implementation is out of scope for this phase.

## Full Battle Dataset (Random Self-Play)

`vgc-lab` can generate datasets of completed battles using Showdown's internal random AI:

- **Method**: Uses Showdown's `BattleStream` and `RandomPlayerAI` via a Node.js script (`js/random_selfplay.js`)
- **Generation**: The CLI command `dataset full-battles` runs complete battles and collects results
- **Output**: JSONL dataset with battle logs, winners, turn counts, and metadata
- **Usage example**:
  ```bash
  cd Pokai/vgc-lab
  python scripts/cli.py dataset full-battles --n 5
  ```
- **Data locations**:
  - Raw logs: `data/battles_raw/`
  - Dataset: `data/datasets/full_battles/full_battles.jsonl`

The Node script `js/random_selfplay.js` serves as the bridge to Showdown's battle engine, running battles to completion and outputting structured JSON results.

## Preview + Outcome Dataset (Random Self-Play)

`vgc-lab` can generate datasets that directly tie open team sheets to battle outcomes:

- **Method**: Uses Showdown's internal `RandomPlayerAI` and the same engine as `js/random_selfplay.js`
- **Data stored**: For each battle:
  - Both sides' 6-Pokémon public teams (from the engine, via `Teams.unpack`)
  - The winner (`winner_side` / `winner_name`)
  - The path to the raw battle log
- **Generation**: The CLI command `dataset preview-outcome` runs complete battles and stores preview+outcome pairs
- **Dataset location**: `data/datasets/preview_outcome/preview_outcome.jsonl`
- **Usage example**:
  ```bash
  cd Pokai/vgc-lab
  python scripts/cli.py dataset preview-outcome --n 5
  ```
- **Data locations**:
  - Raw logs: `data/battles_raw/` (shared with other scripts)
  - Dataset: `data/datasets/preview_outcome/preview_outcome.jsonl`

You can then load `preview_outcome.jsonl` from Python / Jupyter for analysis or model training.

## Core Source Modules (Simplified Architecture)

All internal logic is now grouped into a small set of focused modules:

### `config.py`
Path configuration and constants.

- `PROJECT_ROOT`, `SHOWDOWN_ROOT`
- `DATA_ROOT`, `TEAMS_DIR`, `BATTLES_RAW_DIR`, `MODELS_DIR`
- `DEFAULT_FORMAT` (defaults to `gen9vgc2026regf`)
- `ensure_paths()` to create data directories on first use.

### `showdown.py`
Pokemon Showdown bridge + log parsing + random bots.

Includes all functionality previously split across `showdown_cli.py`, `random_bot.py`, `lineup.py`, and `team_preview.py`:

- CLI / process bridge:
  - `run_showdown_command`
  - `pack_team`
  - `validate_team`
  - `generate_random_team`
  - `simulate_battle`
  - `play_battle_with_random_bots`
  - `run_random_selfplay_json`
  - `run_random_selfplay`
- Random bot:
  - `choose_team_preview`
  - `choose_turn_action`
  - `choose_action`
- Team preview & lineups:
  - `PokemonPublicInfo`, `SideTeamPreview`, `TeamPreviewSnapshot`, `PreviewOutcomeRecord`
  - `extract_team_preview_requests`
  - `parse_team_preview_snapshot`
  - `extract_lineups_from_log`
  - `append_snapshot_to_dataset`
  - `append_preview_outcome_record`

### `catalog.py`
Set and team catalogs + team building utilities.

Merged from `set_catalog.py`, `team_pool.py`, `teams.py`, and `team_builder.py`:

- Set catalog:
  - `SetEntry`, `SetCatalog`
  - load from `data/catalog/sets_regf.yaml`
- Team pool:
  - `TeamDef`, `TeamPool`
  - load from `data/catalog/teams_regf*.yaml`
- Team import/export:
  - `import_text_to_packed`
  - `build_team_import_from_set_ids`
- Convenience helpers for evaluating catalog teams (wrapped again at the core level).

### `dataio.py`
All JSONL / file I/O for battles, matchups, and teams-vs-pool.

Merged from `battle_logger.py`, `team_matchup_data.py`, and `teams_vs_pool_data.py`:

- Battle logging:
  - `BattleResult`, `FullBattleRecord`
  - `save_raw_log`, `infer_winner_from_log`, `count_turns`
  - `save_battle_result`, `append_full_battle_record`
- Matchup data:
  - `TeamMatchupRecord`
  - `ensure_matchup_dir`, `append_matchup_record`, `iter_team_matchup_records`
- Teams-vs-pool data:
  - `ensure_teams_vs_pool_dir`
  - `append_result_to_dataset`
  - constants like `TEAMS_VS_POOL_JSONL_PATH`, etc.

### `features.py`
Team encoding and feature extraction for ML.

Merged from `team_encoding.py` and `team_features.py`:

- `SetIdVocab`
- `one_hot_team`
- `TeamsVsPoolRecord`
- `iter_teams_vs_pool_records`
- `team_bag_of_sets_feature`
- `build_default_vocab` / `build_vocab_from_default_catalog`

### `models.py`
All ML models (team value + matchup).

Merged from `team_value_model.py` and `team_matchup_model.py`:

- Team value:
  - `TeamValueSample`, `TeamValueTorchDataset`, `TeamValueModel`
  - `TrainingConfig`
  - `load_team_value_samples`
  - `train_team_value_model`
  - `save_team_value_checkpoint`, `load_team_value_checkpoint`
- Matchup:
  - `MatchupTrainingConfig`, `TeamMatchupTorchDataset`, `TeamMatchupModel`
  - `train_team_matchup_model`
  - `save_team_matchup_checkpoint`, `load_team_matchup_checkpoint`
  - `predict_matchup_win_prob`

### `analysis.py`
Evaluation, aggregation, meta analysis, and team search.

Merged from `eval.py`, `team_analysis.py`, `matchup_meta_analysis.py`, and `team_search.py`:

- Evaluation:
  - `TeamEvalSummary`, `MatchupEvalSummary`
  - `evaluate_team_vs_random`
  - `evaluate_team_vs_team`
- Aggregation:
  - `TeamAggregateStats`
  - `aggregate_teams`
  - `load_and_aggregate_teams_vs_pool`
- Meta / matchup:
  - `MetaTeam`, `MetaDistribution`
  - `build_uniform_meta`, `normalize_meta`
  - `compute_expected_win_vs_meta`
  - `rank_teams_vs_meta`
  - `predict_matchup_win_prob` (model wrapper)
- Search:
  - `PoolEvalSummary`, `CandidateTeamResult`, `ModelGuidedProposal`
  - `evaluate_team_against_pool`
  - `sample_random_team_set_ids`
  - `random_search_over_pool`
  - `propose_candidates_with_model`
  - `evaluate_top_model_candidates_against_pool`
  - `mutate_team_set_ids`

### `core.py`
High-level pipeline functions that orchestrate the modules above.

All dataset generation, training, search, evaluation, and analysis pipelines are exposed here as functions that can be called from Python code or via the CLI:

- Dataset generation: `generate_full_battle_dataset`, `generate_preview_outcome_dataset`, `generate_team_preview_dataset`, `generate_team_matchup_dataset`, `generate_teams_vs_pool_dataset`
- Training: `train_team_value_model`, `train_team_matchup_model`, `auto_team_value_iter`
- Search: `random_catalog_team_search`, `model_guided_team_search`
- Evaluation: `evaluate_catalog_team`, `evaluate_catalog_team_vs_pool`, `evaluate_main_team_vs_random`, `evaluate_matchup`
- Analysis: `analyze_teams_vs_pool`, `analyze_meta_matchups`, `suggest_counter_teams`
- Tools: `pack_team`, `validate_team`, `dump_teams_vs_pool_features`, `run_demo_battle`

### `cli.py`
Unified command-line interface that maps CLI commands to `core.py` functions.

## Future Extensions

This project will later be extended for RL/DL training environments, but currently focuses on:
1. Establishing clean battle simulation workflows
2. Collecting battle logs and structured data
3. Providing reusable Python APIs for team generation and validation
4. Building datasets for team preview decision-making

## Dependencies

- Python 3.10+
- Node.js 16+ and npm (for the underlying `pokemon-showdown` CLI)
- The `pokemon-showdown/` folder must be present as a sibling directory

## Quick Start

### CLI Usage Examples

All functionality is accessed through the unified CLI: `scripts/cli.py` (or `python -m vgc_lab.cli`).

**Generate datasets:**
```bash
# Generate full battle dataset
python scripts/cli.py dataset full-battles --n 10

# Generate teams vs pool dataset
python scripts/cli.py dataset teams-vs-pool --n-samples 100 --n-opponents 5 --n-per 5

# Generate team matchup dataset
python scripts/cli.py dataset team-matchups --max-pairs 200 --n-battles-per-pair 12
```

**Train models:**
```bash
# Train team value model
python scripts/cli.py train value-model --epochs 50 --device cuda

# Train matchup model
python scripts/cli.py train matchup-model --epochs 40 --device cuda
```

**Search and analyze:**
```bash
# Model-guided team search
python scripts/cli.py search model-guided --n-proposals 500 --top-k 20

# Analyze top teams
python scripts/cli.py analyze teams-vs-pool --top-k 20 --min-battles 10

# Analyze team matchups vs meta
python scripts/cli.py analyze meta-matchups my_main_team --top-k-best 10 --top-k-worst 10
```

**Evaluate teams:**
```bash
# Evaluate a catalog team vs random
python scripts/cli.py eval catalog-team set_id_1 set_id_2 set_id_3 set_id_4 set_id_5 set_id_6 --n 50

# Score a team with trained model
python scripts/cli.py eval score-catalog-team my_team_id --ckpt data/models/team_value_mlp.pt
```

### Programmatic Usage

For programmatic access, import directly from `vgc_lab`:

```python
import vgc_lab as v

# Generate dataset
v.generate_full_battle_dataset(n=10)

# Train model
v.train_team_value_model(epochs=50, device='cuda')

# Analyze teams
v.analyze_teams_vs_pool(top_k=20, min_battles=10)
```

### Detailed Documentation

For complete API reference, module documentation, and internal architecture details, see [REPOSITORY_STRUCTURE.md](REPOSITORY_STRUCTURE.md).

