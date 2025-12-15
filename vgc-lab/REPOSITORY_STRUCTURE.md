# vgc-lab Repository Structure

## Overview

`vgc-lab` is a Python wrapper for Pokemon Showdown CLI, focused on Gen 9 VGC Reg F battle simulation, data collection, and machine learning for team analysis. It provides utilities for team generation, battle simulation, dataset creation, model training, and team search optimization.

**Simplified Architecture**: After a major refactoring, all logic is consolidated into a small set of focused modules, with a single unified CLI entrypoint.

## Repository Structure

```
vgc-lab/
├── data/                           # Generated data and datasets
│   ├── battles_json/               # JSON battle results
│   ├── battles_raw/                # Raw battle log files (.log)
│   ├── catalog/                    # Team and set catalogs
│   │   ├── sets_regf.yaml          # Pokemon set catalog (sets definitions)
│   │   ├── teams_regf.yaml         # Team catalog (6-Pokemon teams)
│   │   └── teams_regf_auto.yaml    # Auto-generated top teams (from analyze teams-vs-pool)
│   ├── datasets/                   # Structured datasets (JSONL format)
│   │   ├── full_battles/
│   │   │   └── full_battles.jsonl  # Complete battle records
│   │   ├── preview_outcome/
│   │   │   └── preview_outcome.jsonl  # Team preview + battle outcome pairs
│   │   ├── team_matchups/
│   │   │   └── team_matchups.jsonl    # Pairwise team matchup results
│   │   ├── team_preview/
│   │   │   └── team_preview.jsonl     # Team preview snapshots
│   │   └── teams_vs_pool/
│   │       ├── teams_vs_pool.jsonl    # Team evaluations vs pool
│   │       ├── features.csv           # ML features extracted from JSONL
│   │       └── top_teams.csv          # Top teams CSV export
│   ├── meta/
│   │   └── example_meta.json       # Example meta distribution JSON
│   ├── models/                     # Trained model checkpoints
│   │   ├── team_matchup_mlp.pt     # Team matchup prediction model
│   │   ├── team_value_mlp.pt       # Team value (win rate) prediction model
│   │   └── team_value_mlp_iter*.pt # Iteration checkpoints from auto-value-iter
│   └── teams/                      # Team import files (.txt)
│       ├── main_team.txt
│       └── test_team_2.txt
├── js/                             # Node.js bridge scripts
│   ├── import_to_packed.js         # Convert team import to packed format
│   └── random_selfplay.js          # Random self-play battle runner
├── scripts/                        # CLI tools (single entrypoint)
│   └── cli.py                      # Convenience wrapper that imports vgc_lab.cli.main
├── src/
│   └── vgc_lab/                    # Main Python package
│       ├── __init__.py             # Public API exports
│       ├── config.py               # Path configuration and constants
│       ├── showdown.py             # Showdown bridge + log parsing + random bots
│       ├── catalog.py              # Set and team catalogs + team building
│       ├── dataio.py               # JSONL / file I/O for battles and matchups
│       ├── features.py             # Team encoding and feature extraction
│       ├── models.py               # ML models (team value + matchup)
│       ├── analysis.py             # Evaluation, aggregation, meta analysis, search
│       ├── core.py                 # High-level pipeline functions
│       └── cli.py                  # CLI argument parsing and dispatch
├── tests/                          # Test suite
│   └── test_cli_smoke.py           # Smoke tests for CLI
├── pyproject.toml                  # Project configuration
├── README.md                       # Project documentation
└── REPOSITORY_STRUCTURE.md         # This file
```

## Unified CLI

All commands are accessed through a single entrypoint. You can use any of these methods:

- `python scripts/cli.py <command>` - Convenience wrapper (imports `vgc_lab.cli.main`)
- `python -m vgc_lab.cli <command>` - Direct module execution
- `vgc-lab <command>` - Console script (after installation via `pip install -e .`)

**Note:** `scripts/cli.py` is a thin wrapper that imports and calls `vgc_lab.cli.main()`. The actual CLI implementation is in `src/vgc_lab/cli.py`, which routes all commands to functions in `vgc_lab.core`. These functions are also re-exported at the package level (`import vgc_lab as v`). For programmatic usage, import directly from `vgc_lab` rather than `vgc_lab.core`.

**Device Auto-Detection:** All functions that accept a `device` parameter will automatically use 'cuda' if CUDA is available and `device` is `None`, otherwise they default to 'cpu'. This behavior is implemented via the internal `_resolve_device()` helper in `core.py`.

### CLI Command Structure

```
vgc-lab [dataset|train|search|eval|analyze|tools|demo] <subcommand> [options]
```

### Dataset Generation Commands

#### `dataset full-battles`
Generate full battle dataset from random self-play battles.

**Options:**
- `--n <int>`: Number of full battles to generate (default: 10)
- `--format <str>`: Showdown format ID (default: gen9vgc2026regf)

**Function**: `vgc_lab.generate_full_battle_dataset(n, format_id)` (also available as `vgc_lab.core.generate_full_battle_dataset`)

#### `dataset preview-outcome`
Generate dataset linking team previews to battle outcomes.

**Options:**
- `--n <int>`: Number of preview+outcome samples (default: 10)
- `--sleep-ms <int>`: Sleep between battles in milliseconds (default: 50)
- `--format <str>`: Showdown format ID

**Function**: `vgc_lab.generate_preview_outcome_dataset(n, sleep_ms, format_id)` (also available as `vgc_lab.core.generate_preview_outcome_dataset`)

#### `dataset team-preview`
Generate team preview snapshots from simulated battles.

**Options:**
- `--n <int>`: Number of snapshots to generate (default: 100)
- `--format <str>`: Showdown format ID

**Function**: `vgc_lab.generate_team_preview_dataset(n, format_id)` (also available as `vgc_lab.core.generate_team_preview_dataset`)

#### `dataset team-matchups`
Generate pairwise team matchup dataset.

**Options:**
- `--max-pairs <int>`: Maximum number of pairs to evaluate (default: 200)
- `--n-battles-per-pair <int>`: Battles per A vs B evaluation (default: 12)
- `--dataset-path <str>`: Optional override for JSONL output path
- `--format <str>`: Showdown format ID
- `--pool-yaml <str>`: Team catalog path (defaults to teams_regf_auto.yaml if exists)

**Function**: `vgc_lab.generate_team_matchup_dataset(max_pairs, n_battles_per_pair, dataset_path, format_id, pool_yaml)` (also available as `vgc_lab.core.generate_team_matchup_dataset`)

#### `dataset teams-vs-pool`
Generate dataset of random teams evaluated against a team pool.

**Options:**
- `--n-samples <int>`: Number of random teams to evaluate (default: 50)
- `--team-size <int>`: Number of Pokémon per team (default: 6)
- `--n-opponents <int>`: Opponent teams sampled per candidate (default: 5)
- `--n-per <int>`: Battles per opponent (default: 5)
- `--include-catalog-teams`: Also evaluate all catalog teams
- `--format <str>`: Showdown format ID
- `--teams-yaml <str>`: Path to teams YAML catalog

**Function**: `vgc_lab.generate_teams_vs_pool_dataset(n_samples, team_size, n_opponents, n_battles_per_opponent, include_catalog_teams, format_id, teams_yaml)` (also available as `vgc_lab.core.generate_teams_vs_pool_dataset`)

### Model Training Commands

#### `train value-model`
Train a neural team value model on teams_vs_pool.jsonl.

**Options:**
- `--jsonl <str>`: Path to teams_vs_pool.jsonl dataset
- `--out <str>`: Path to save model checkpoint
- `--epochs <int>`: Training epochs (default: 50)
- `--batch-size <int>`: Batch size (default: 64)
- `--lr <float>`: Learning rate (default: 3e-4)
- `--min-battles <int>`: Minimum battles per sample (default: 3)
- `--no-include-random`: Exclude 'random' source (default: include)
- `--no-include-catalog`: Exclude 'catalog' source (default: include)
- `--device <str>`: Device ('cpu' or 'cuda'); default is 'cuda' if available, else 'cpu'`

**Function**: `vgc_lab.train_team_value_model(jsonl_path: Path | str | None = None, out_path: Path | str | None = None, epochs: int = 50, batch_size: int = 64, lr: float = 3e-4, min_battles: int = 3, include_random: bool = True, include_catalog: bool = True, device: str | None = None)` (also available as `vgc_lab.core.train_team_value_model`)

#### `train matchup-model`
Train a neural team matchup model on team_matchups.jsonl.

**Options:**
- `--jsonl <str>`: Path to team_matchups.jsonl dataset
- `--out <str>`: Path to save model checkpoint
- `--epochs <int>`: Training epochs (default: 40)
- `--batch-size <int>`: Batch size (default: 64)
- `--lr <float>`: Learning rate (default: 1e-3)
- `--min-battles <int>`: Minimum battles per matchup (default: 10)
- `--max-records <int>`: Optional cap on number of records
- `--device <str>`: Device ('cpu' or 'cuda'); default is 'cuda' if available, else 'cpu'`

**Function**: `vgc_lab.train_team_matchup_model(jsonl_path: Path | str | None = None, out_path: Path | str | None = None, epochs: int = 40, batch_size: int = 64, lr: float = 1e-3, min_battles: int = 10, max_records: int | None = None, device: str | None = None)` (also available as `vgc_lab.core.train_team_matchup_model`)

### Team Search Commands

#### `search random-over-pool`
Random search baseline over catalog-defined teams vs pool.

**Options:**
- `--n-candidates <int>`: Number of random candidates (default: 20)
- `--team-size <int>`: Number of Pokémon per team (default: 6)
- `--n-opponents <int>`: Opponent teams sampled (default: 5)
- `--n-per <int>`: Battles per opponent (default: 5)
- `--top-k <int>`: Number of top teams to print (default: 5)
- `--format <str>`: Showdown format ID
- `--teams-yaml <str>`: Teams YAML catalog path

**Function**: `vgc_lab.random_catalog_team_search(n_candidates, team_size, n_opponents, n_battles_per_opponent, top_k, format_id, teams_yaml)` (also available as `vgc_lab.core.random_catalog_team_search`)

#### `search model-guided`
Single iteration of model-guided team search vs pool.

**Options:**
- `--ckpt <str>`: Path to team value model checkpoint (default: team_value_mlp.pt)
- `--n-proposals <int>`: Candidate teams to propose (default: 500)
- `--top-k <int>`: Top candidates to evaluate (default: 20)
- `--n-opponents <int>`: Opponent teams sampled (default: 5)
- `--n-per <int>`: Battles vs each opponent (default: 5)
- `--seed <int>`: Random seed (default: 123)
- `--device <str>`: Device ('cpu' or 'cuda'); default is 'cuda' if available, else 'cpu'
- `--source <str>`: Dataset source tag (default: 'model_guided_v1')
- `--jsonl <str>`: Optional explicit path to teams_vs_pool.jsonl

**Function**: `vgc_lab.model_guided_team_search(ckpt_path: Path | str | None = None, n_proposals: int = 500, top_k: int = 20, n_opponents: int = 5, n_per_opponent: int = 5, seed: int = 123, device: str | None = None, source_tag: str = "model_guided_v1", jsonl_path: Path | str | None = None)` (also available as `vgc_lab.core.model_guided_team_search`)

#### `search auto-value-iter`
Multi-iteration training + model-guided search loop.

**Options:**
- `--n-iters <int>`: Number of outer iterations (default: 5)
- `--init-random-samples <int>`: Initial random candidates (default: 0)
- `--per-iter-random-samples <int>`: Random candidates per iteration (default: 0)
- `--per-iter-model-proposals <int>`: Model proposals per iteration (default: 500)
- `--per-iter-top-k <int>`: Top candidates to evaluate (default: 40)
- `--n-mutation-bases <int>`: Top candidates to mutate (default: 0, disabled)
- `--n-mutations-per-base <int>`: Mutations per base (default: 1)
- `--n-opponents <int>`: Opponent teams sampled (default: 5)
- `--n-per <int>`: Battles per opponent (default: 5)
- `--epochs-per-iter <int>`: Training epochs per iteration (default: 20)
- `--batch-size <int>`: Batch size (default: 64)
- `--lr <float>`: Learning rate (default: 3e-4)
- `--seed <int>`: Random seed (default: 123)
- `--device <str>`: Device ('cpu' or 'cuda')`

**Function**: `vgc_lab.auto_team_value_iter(n_iters, init_random_samples, per_iter_random_samples, per_iter_model_proposals, per_iter_top_k, n_mutation_bases, n_mutations_per_base, n_opponents, n_per_opponent, epochs_per_iter, batch_size, lr, seed, device)` (also available as `vgc_lab.core.auto_team_value_iter`)

### Evaluation Commands

#### `eval catalog-team`
Evaluate a catalog-defined team vs random opponents.

**Options:**
- `set_ids <list>`: List of 6 set IDs forming the team (positional args)
- `--n <int>`: Number of self-play battles (default: 50)
- `--format <str>`: Showdown format ID
- `--team-as <str>`: Team placement: "p1", "p2", or "both" (default: "both")
- `--save-logs`: Save logs to data/battles_raw/

**Function**: `vgc_lab.evaluate_catalog_team(set_ids, n, format_id, team_as, save_logs)` (also available as `vgc_lab.core.evaluate_catalog_team`)

#### `eval catalog-vs-pool`
Evaluate a catalog-defined team vs a catalog-defined pool.

**Options:**
- `set_ids <list>`: List of 6 set IDs (positional args)
- `--n-opponents <int>`: Opponent teams sampled (default: 5)
- `--n-per <int>`: Battles per opponent (default: 5)
- `--format <str>`: Showdown format ID
- `--teams-yaml <str>`: Path to teams YAML catalog

**Function**: `core.evaluate_catalog_team_vs_pool(set_ids, n_opponents, n_per_opponent, format_id, teams_yaml)`

#### `eval main-team-vs-random`
Evaluate a hardcoded main team against random opponents.

**Options:**
- `--n <int>`: Number of self-play battles (default: 50)
- `--format <str>`: Showdown format ID
- `--team-as <str>`: Team placement: "p1", "p2", or "both" (default: "p1")
- `--save-logs`: Save logs

**Function**: `vgc_lab.evaluate_main_team_vs_random(n, format_id, team_as, save_logs)` (also available as `vgc_lab.core.evaluate_main_team_vs_random`)

#### `eval matchup`
Evaluate matchup between two teams using RandomPlayerAI.

**Options:**
- `team_a <str>`: Path to Team A import text (positional)
- `team_b <str>`: Path to Team B import text (positional)
- `--n <int>`: Number of battles (default: 50)
- `--format <str>`: Showdown format ID

**Function**: `vgc_lab.evaluate_matchup(team_a_path: Path | str, team_b_path: Path | str, n: int = 50, format_id: str | None = None)` (also available as `vgc_lab.core.evaluate_matchup`)

#### `eval score-catalog-team`
Score a catalog team using the trained value model.

**Options:**
- `team_id <str>`: Team ID from teams_regf.yaml (positional)
- `--ckpt <str>`: Model checkpoint path (default: team_value_mlp.pt)
- `--device <str>`: Device ('cpu' or 'cuda'); default is 'cuda' if available, else 'cpu'`

**Function**: `vgc_lab.score_catalog_team_with_model(team_id: str, ckpt_path: Path | str | None = None, device: str | None = None)` (also available as `vgc_lab.core.score_catalog_team_with_model`)

#### `eval score-matchup`
Score a matchup between two catalog teams using the matchup model.

**Options:**
- `team_a_id <str>`: Team A ID from catalog (positional)
- `team_b_id <str>`: Team B ID from catalog (positional)
- `--ckpt <str>`: Matchup model checkpoint (default: team_matchup_mlp.pt)
- `--teams-yaml <str>`: Team catalog path
- `--device <str>`: Device ('cpu' or 'cuda'); default is 'cuda' if available, else 'cpu'`

**Function**: `vgc_lab.score_matchup_with_model(team_a_id: str, team_b_id: str, ckpt_path: Path | str | None = None, teams_yaml: Path | str | None = None, device: str | None = None)` (also available as `vgc_lab.core.score_matchup_with_model`)

### Analysis Commands

#### `analyze teams-vs-pool`
Analyze teams_vs_pool dataset and list/discover top teams.

**Options:**
- `--top-k <int>`: Number of top teams to display (default: 20)
- `--min-battles <int>`: Minimum total battles required (default: 20)
- `--min-records <int>`: Minimum JSONL records required (default: 1)
- `--source-prefix <str>`: Filter by source prefix (e.g., 'auto_iter_')
- `--jsonl <str>`: Path to teams_vs_pool.jsonl
- `--output-csv <str>`: Optional CSV export path
- `--output-teams-yaml <str>`: Optional YAML catalog export path

**Function**: `vgc_lab.analyze_teams_vs_pool(top_k, min_battles, min_records, source_prefix, jsonl_path, output_csv, output_teams_yaml)` (also available as `vgc_lab.core.analyze_teams_vs_pool`)

#### `analyze meta-matchups`
Analyze a team's matchup vs a meta using the matchup model.

**Options:**
- `team_id <str>`: Team ID to analyze (positional, e.g., 'my_main_team' or 'auto_top_0001')
- `--teams-yaml <str>`: Team catalog path (defaults to teams_regf_auto.yaml)
- `--ckpt <str>`: Path to trained matchup model (default: team_matchup_mlp.pt)
- `--top-k-best <int>`: Number of best matchups to show (default: 10)
- `--top-k-worst <int>`: Number of worst matchups to show (default: 10)
- `--device <str>`: Device ('cpu' or 'cuda'); default is 'cuda' if available, else 'cpu'`

**Function**: `vgc_lab.analyze_meta_matchups(team_id: str, teams_yaml: Path | str | None = None, ckpt_path: Path | str | None = None, top_k_best: int = 10, top_k_worst: int = 10, device: str | None = None)` (also available as `vgc_lab.core.analyze_meta_matchups`)

#### `analyze suggest-counters`
Suggest counter teams vs a meta using the matchup model.

**Options:**
- `--teams-yaml <str>`: Team catalog (defaults to teams_regf_auto.yaml)
- `--ckpt <str>`: Matchup model checkpoint (default: team_matchup_mlp.pt)
- `--meta-json <str>`: Optional JSON file describing meta distribution (if None, uses uniform meta)
- `--candidate-prefix <str>`: Filter candidates by prefix (e.g., 'auto_top_')
- `--top-k <int>`: Number of best counter candidates to show (default: 10)
- `--device <str>`: Device ('cpu' or 'cuda'); default is 'cuda' if available, else 'cpu'`

**Function**: `vgc_lab.suggest_counter_teams(teams_yaml: Path | str | None = None, ckpt_path: Path | str | None = None, meta_json: Path | str | None = None, candidate_prefix: str | None = None, top_k: int = 10, device: str | None = None)` (also available as `vgc_lab.core.suggest_counter_teams`)

### Utility Tools Commands

#### `tools pack-team`
Convert team from export format to packed format.

**Options:**
- `--file <str>`: Input file (or read from stdin if not provided)

**Function**: `vgc_lab.pack_team(file)` (also available as `vgc_lab.core.pack_team`)

#### `tools validate-team`
Validate a Pokemon Showdown team against a format.

**Options:**
- `--file <str>`: Input file (or read from stdin)
- `--format <str>`: Format ID to validate against (default: gen9vgc2026regf)

**Function**: `vgc_lab.validate_team(file, format_id)` (also available as `vgc_lab.core.validate_team`)

#### `tools dump-features`
Convert teams_vs_pool JSONL to CSV features for ML.

**Options:**
- `--input <str>`: Path to teams_vs_pool.jsonl (default: datasets/teams_vs_pool/teams_vs_pool.jsonl)
- `--output <str>`: Output CSV path (default: datasets/teams_vs_pool/features.csv)

**Function**: `vgc_lab.dump_teams_vs_pool_features(input_path, output_csv)` (also available as `vgc_lab.core.dump_teams_vs_pool_features`)

### Demo Commands

#### `demo battle`
Demo script to generate two random teams and simulate a battle.

**Options:** None

**Function**: `vgc_lab.run_demo_battle()` (also available as `vgc_lab.core.run_demo_battle`)

## Core Source Modules

### `config.py`
**Purpose**: Path configuration and directory management

**Constants:**
- `PROJECT_ROOT: Final[Path]`: Root directory (containing pyproject.toml)
- `SHOWDOWN_ROOT: Final[Path]`: Sibling directory containing pokemon-showdown
- `DEFAULT_FORMAT: Final[str]`: Default format ID ("gen9vgc2026regf")
- `DEFAULT_FORMAT_BO3: Final[str]`: BO3 format ID ("gen9vgc2026regfbo3")
- `DATA_ROOT: Final[Path]`: Project data directory (`PROJECT_ROOT / "data"`)
- `TEAMS_DIR: Final[Path]`: Team import files directory
- `BATTLES_RAW_DIR: Final[Path]`: Raw battle log files directory
- `BATTLES_JSON_DIR: Final[Path]`: JSON battle results directory
- `MODELS_DIR: Final[Path]`: Trained model checkpoints directory

**Functions:**
- `ensure_paths() -> None`: Create the data directories (teams, battles_raw, battles_json, models) if they don't exist

### `showdown.py`
**Purpose**: Pokemon Showdown bridge, log parsing, random bots, and team preview

**Constants:**
- `DATASET_DIR: Path`: Team preview dataset directory
- `DATASET_PATH: Path`: Team preview JSONL file path
- `PREVIEW_OUTCOME_DATASET_DIR: Path`: Preview outcome dataset directory
- `PREVIEW_OUTCOME_DATASET_PATH: Path`: Preview outcome JSONL file path

**Classes:**
- `PokemonPublicInfo(BaseModel)`: Public Pokemon info (name, species, item, ability, moves, tera)
- `SideTeamPreview(BaseModel)`: Side team preview (6 Pokemon)
- `TeamPreviewSnapshot(BaseModel)`: Full snapshot (both sides + metadata)
- `PreviewOutcomeRecord(BaseModel)`: Preview + outcome record

**Functions:**

*Random Bot:*
- `choose_team_preview(request: dict[str, Any], rng: Optional[random.Random] = None) -> str`: Choose random subset of Pokémon for team preview
- `choose_turn_action(request: dict[str, Any], rng: Optional[random.Random] = None) -> str`: Choose random actions for a single turn
- `choose_action(state: dict, choices: dict, rng: Optional[random.Random] = None) -> str`: General action choice wrapper

*Lineup Extraction:*
- `extract_lineups_from_log(log_text: str, p1_team: list, p2_team: list) -> dict`: Extract chosen/lead indices from battle log

*Team Preview:*
- `extract_team_preview_requests(log_text: str) -> dict[str, dict]`: Extract preview requests from log
- `parse_team_preview_snapshot(log_text: str, *, format_id: str | None = None, raw_log_path: Path | None = None) -> TeamPreviewSnapshot`: Parse snapshot from log (format_id defaults to DEFAULT_FORMAT if not provided)
- `append_snapshot_to_dataset(snapshot: TeamPreviewSnapshot) -> None`: Append to team_preview.jsonl
- `append_preview_outcome_record(record: PreviewOutcomeRecord) -> None`: Append to preview_outcome.jsonl
- `ensure_dataset_dir() -> None`: Create team preview dataset directory
- `ensure_preview_outcome_dataset_dir() -> None`: Create preview outcome dataset directory

*Showdown CLI:*
- `run_showdown_command(args: list[str], input_text: Optional[str] = None) -> str`: Run arbitrary Showdown command
- `pack_team(export_text: str) -> str`: Convert export format to packed
- `validate_team(packed_team: str, format_id: str = DEFAULT_FORMAT) -> str`: Validate team, return error or ""
- `generate_random_team(format_id: str = DEFAULT_FORMAT) -> str`: Generate random team (packed format)
- `simulate_battle(format_id: str, p1_packed: str, p2_packed: str, save_log: bool = False) -> tuple[str, str, str]`: Run non-interactive battle simulation
- `play_battle_with_random_bots(format_id: str, p1_packed: Optional[str] = None, p2_packed: Optional[str] = None, save_log: bool = False) -> tuple[str, str, str]`: Play battle with random bots (streaming)
- `run_random_selfplay_json(format_id: str, p1_packed_team: Optional[str] = None, p2_packed_team: Optional[str] = None, sleep_ms: int = 50) -> dict`: Run random self-play, return JSON
- `run_random_selfplay(format_id: str, p1_packed_team: Optional[str] = None, p2_packed_team: Optional[str] = None) -> tuple[str, str, int, str, str]`: Run random self-play, return (log, winner, turns, p1_name, p2_name)

### `catalog.py`
**Purpose**: Set and team catalogs, team pools, and team building utilities

**Constants:**
- `CATALOG_PATH: Path`: Default set catalog path (`data/catalog/sets_regf.yaml`)
- `TEAMS_CATALOG_PATH: Path`: Default team catalog path (`data/catalog/teams_regf.yaml`)

**Classes:**
- `SetEntry`: Single Pokemon set entry (id, format_id, import_text, description)
- `SetCatalog`: Catalog of Pokemon sets with methods:
  - `from_yaml(path: Path = CATALOG_PATH) -> SetCatalog`: Load from YAML
  - `get(set_id: str) -> SetEntry`: Get entry by ID
  - `ids() -> list[str]`: List all set IDs
  - `__len__() -> int`: Number of sets
  - `__contains__(set_id: str) -> bool`: Check if set ID exists
- `TeamDef`: Team definition (id, format_id, set_ids, description)
- `TeamPool`: Pool of teams with methods:
  - `from_yaml(path: Path = TEAMS_CATALOG_PATH) -> TeamPool`: Load from YAML
  - `get(team_id: str) -> TeamDef`: Get team by ID
  - `ids() -> list[str]`: List all team IDs
  - `all() -> list[TeamDef]`: List all TeamDefs
  - `sample(n: int, exclude_ids: Optional[set[str]] = None, rng: Optional[random.Random] = None) -> list[TeamDef]`: Sample teams
  - `__len__() -> int`: Number of teams
  - `__contains__(team_id: str) -> bool`: Check if team ID exists

**Functions:**
- `import_text_to_packed(import_text: str) -> str`: Convert team import text to packed format using Showdown CLI
- `build_team_import_from_set_ids(set_ids: list[str], catalog: SetCatalog) -> str`: Build import text from set IDs
- `evaluate_catalog_team_vs_random(set_ids: list[str], n: int = 50, format_id: str = DEFAULT_FORMAT, team_as: str = "both", save_logs: bool = False) -> TeamEvalSummary`: Evaluate catalog team vs random
- `evaluate_catalog_matchup(set_ids_a: list[str], set_ids_b: list[str], n: int = 50, format_id: str = DEFAULT_FORMAT) -> MatchupEvalSummary`: Evaluate matchup between catalog teams

### `dataio.py`
**Purpose**: JSONL/file I/O for battles, matchups, and teams-vs-pool datasets

**Constants:**
- `FULL_BATTLE_DATASET_DIR: Path`: Full battle dataset directory
- `FULL_BATTLE_DATASET_PATH: Path`: Full battle JSONL file path
- `MATCHUP_DATASET_DIR: Path`: Matchup dataset directory
- `MATCHUP_JSONL_PATH: Path`: Matchup JSONL file path
- `TEAMS_VS_POOL_DIR: Path`: Teams vs pool dataset directory
- `TEAMS_VS_POOL_JSONL_PATH: Path`: Teams vs pool JSONL file path

**Classes:**
- `BattleResult(BaseModel)`: Battle result model (format_id, p1_name, p2_name, winner, raw_log_path, turns, created_at)
- `FullBattleRecord(BaseModel)`: Full battle record (extends BattleResult with additional metadata)
- `TeamMatchupRecord`: Matchup record dataclass (team_a_id, team_b_id, team_a_set_ids, team_b_set_ids, format_id, n_battles, n_a_wins, n_b_wins, n_ties, avg_turns, created_at, meta)

**Functions:**

*Battle Logging:*
- `save_raw_log(format_id: str, log_text: str) -> Path`: Save raw battle log to data/battles_raw/
- `infer_winner_from_log(log_text: str, p1_name: str, p2_name: str) -> str`: Infer winner from log text ("p1", "p2", "tie", or "unknown")
- `count_turns(log_text: str) -> Optional[int]`: Count turns in battle log
- `save_battle_result(result: BattleResult) -> Path`: Save result JSON, return path
- `append_full_battle_record(record: FullBattleRecord) -> None`: Append to full_battles.jsonl
- `ensure_full_battle_dataset_dir() -> None`: Create full battle dataset directory

*Matchup Data:*
- `ensure_matchup_dir() -> None`: Create matchup dataset directory
- `to_json_dict(record: TeamMatchupRecord) -> Dict[str, object]`: Convert record to JSON dict
- `from_json_dict(d: Dict[str, object]) -> TeamMatchupRecord`: Parse record from JSON dict
- `append_matchup_record(record: TeamMatchupRecord, path: Optional[Path] = None) -> None`: Append to team_matchups.jsonl
- `iter_team_matchup_records(path: Path) -> Iterator[TeamMatchupRecord]`: Iterator over records

*Teams vs Pool Data:*
- `ensure_teams_vs_pool_dir() -> None`: Create teams vs pool dataset directory
- `append_result_to_dataset(result: PoolEvalSummary, source: str, team_id: Optional[str] = None, jsonl_path: Optional[Path] = None) -> None`: Append result to teams_vs_pool.jsonl

### `features.py`
**Purpose**: Team encoding and feature extraction for ML

**Constants:**
- `DATASETS_ROOT: Path`: Root directory for all datasets

**Classes:**
- `SetIdVocab`: Mapping between set IDs and integer indices with methods:
  - `from_catalog(catalog: SetCatalog) -> SetIdVocab`: Build vocabulary from catalog
  - `from_idx_to_id(idx_to_id: list[str]) -> SetIdVocab`: Build vocabulary from index list
  - `encode_ids(set_ids: Sequence[str]) -> list[int]`: Encode set IDs to indices
  - `decode_indices(indices: Sequence[int]) -> list[str]`: Decode indices back to set IDs
  - `__len__() -> int`: Vocabulary size
- `TeamsVsPoolRecord`: Record from teams_vs_pool dataset (team_set_ids, win_rate, n_battles_total, source, meta)

**Functions:**
- `one_hot_team(indices: Sequence[int], vocab_size: int) -> list[list[int]]`: Convert indices to list of one-hot vectors
- `build_vocab_from_default_catalog() -> SetIdVocab`: Load catalog and build vocabulary
- `iter_teams_vs_pool_records(path: Path) -> Iterator[TeamsVsPoolRecord]`: Iterator over teams_vs_pool JSONL records
- `team_bag_of_sets_feature(team_set_ids: list[str], vocab: SetIdVocab) -> list[int]`: Build bag-of-sets feature vector
- `build_default_vocab() -> SetIdVocab`: Convenience wrapper for build_vocab_from_default_catalog()

### `models.py`
**Purpose**: Neural models for team value and matchup prediction

**Classes:**

*Team Value Model:*
- `TeamValueSample`: Training sample (set_indices: list[int], win_rate: float, n_battles: int, source: str)
- `TeamValueTorchDataset(Dataset)`: PyTorch Dataset for team value training
- `TeamValueModel(nn.Module)`: MLP model (embedding -> MLP -> win_rate prediction)
- `TrainingConfig`: Training configuration dataclass (jsonl_path, out_path, epochs, batch_size, lr, min_battles, include_sources, device)

*Team Matchup Model:*
- `MatchupTrainingConfig`: Training configuration dataclass
- `TeamMatchupTorchDataset(Dataset)`: PyTorch Dataset for matchup training
- `TeamMatchupModel(nn.Module)`: MLP model (team_a + team_b -> P(team_a wins))

**Functions:**
- `load_team_value_samples(jsonl_path: Path, min_battles: int = 3, include_sources: Optional[list[str]] = None, rng: Optional[random.Random] = None) -> tuple[list[TeamValueSample], SetIdVocab]`: Load samples from JSONL
- `train_team_value_model(cfg: TrainingConfig, device_override: Optional[str] = None) -> tuple[TeamValueModel, SetIdVocab, dict]`: Train model, return (model, vocab, meta)
- `save_team_value_checkpoint(model: TeamValueModel, vocab: SetIdVocab, out_path: Path, meta: Optional[dict] = None) -> None`: Save model + vocab
- `load_team_value_checkpoint(path: Path, device: str = "cpu") -> tuple[TeamValueModel, SetIdVocab, dict]`: Load model + vocab
- `train_team_matchup_model(cfg: MatchupTrainingConfig, device_override: Optional[str] = None) -> tuple[TeamMatchupModel, SetIdVocab, dict]`: Train matchup model
- `save_team_matchup_checkpoint(model: TeamMatchupModel, vocab: SetIdVocab, out_path: Path, meta: Optional[dict] = None) -> None`: Save matchup model
- `load_team_matchup_checkpoint(path: Path, device: str = "cpu") -> tuple[TeamMatchupModel, SetIdVocab, dict]`: Load matchup model
- `predict_matchup_win_prob(model: TeamMatchupModel, team_a_set_ids: list[str], team_b_set_ids: list[str], vocab: SetIdVocab, device: str = "cpu") -> float`: Predict P(team_a wins vs team_b)

### `analysis.py`
**Purpose**: Evaluation, aggregation, meta analysis, and team search

**Classes:**
- `TeamEvalSummary`: Evaluation summary for team vs random (format_id, team_role, team_packed, n_battles, n_wins, n_losses, n_ties, avg_turns, created_at)
- `MatchupEvalSummary`: Evaluation summary for team vs team (format_id, team_a_packed, team_b_packed, n_battles, n_a_wins, n_b_wins, n_ties, avg_turns, created_at)
- `TeamAggregateStats`: Aggregate stats (team_set_ids, n_records, n_battles_total, n_wins, n_losses, n_ties, win_rate, avg_turns, source_counts)
- `MetaTeam`: Meta team entry (team_id: str, weight: float)
- `MetaDistribution`: Meta distribution (entries: list[MetaTeam])
- `PoolEvalSummary`: Evaluation summary vs pool (team_set_ids, format_id, n_opponents, n_battles_per_opponent, n_battles_total, n_wins, n_losses, n_ties, win_rate, avg_turns, opponent_counts)
- `CandidateTeamResult`: Candidate team with evaluation result (team_set_ids, eval_summary)
- `ModelGuidedProposal`: Model proposal with predicted win_rate (team_set_ids, predicted_win_rate)

**Functions:**

*Evaluation:*
- `evaluate_team_vs_random(team_packed: str, format_id: str = DEFAULT_FORMAT, n: int = 100, team_as: str = "p1", save_logs: bool = False) -> TeamEvalSummary`: Evaluate team vs random opponents
- `evaluate_team_vs_team(team_a_packed: str, team_b_packed: str, format_id: str = DEFAULT_FORMAT, n: int = 100, save_logs: bool = False) -> MatchupEvalSummary`: Evaluate two teams head-to-head

*Aggregation:*
- `aggregate_teams(records: Iterable[TeamsVsPoolRecord]) -> list[TeamAggregateStats]`: Aggregate records by unique team_set_ids
- `load_and_aggregate_teams_vs_pool(jsonl_path: Path) -> list[TeamAggregateStats]`: Load JSONL and aggregate

*Meta / Matchup:*
- `build_uniform_meta(team_pool: TeamPool, team_ids: Iterable[str]) -> MetaDistribution`: Build uniform meta distribution
- `normalize_meta(meta: MetaDistribution) -> MetaDistribution`: Normalize weights to sum to 1
- `compute_expected_win_vs_meta(team_id: str, meta: MetaDistribution, team_pool: TeamPool, matchup_model: TeamMatchupModel, vocab: SetIdVocab, device: str = "cpu") -> float`: Compute expected win rate vs meta
- `rank_teams_vs_meta(team_ids: Iterable[str], meta: MetaDistribution, team_pool: TeamPool, matchup_model: TeamMatchupModel, vocab: SetIdVocab, device: str = "cpu") -> list[tuple[str, float]]`: Rank teams by expected win rate vs meta

*Search:*
- `evaluate_team_against_pool(team_set_ids: list[str], pool: TeamPool, format_id: str = DEFAULT_FORMAT, n_opponents: int = 5, n_battles_per_opponent: int = 5, save_logs: bool = False) -> PoolEvalSummary`: Evaluate team vs pool
- `sample_random_team_set_ids(catalog: SetCatalog, team_size: int = 6, rng: Optional[random.Random] = None) -> list[str]`: Sample random team from catalog
- `random_search_over_pool(n_candidates: int, catalog: SetCatalog, pool: TeamPool, format_id: str = DEFAULT_FORMAT, team_size: int = 6, n_opponents: int = 5, n_battles_per_opponent: int = 5, top_k: int = 5) -> list[CandidateTeamResult]`: Random search, return list of results
- `propose_candidates_with_model(model: TeamValueModel, vocab: SetIdVocab, catalog: SetCatalog, n_proposals: int = 500, seed: int = 123, device: str = "cpu") -> list[ModelGuidedProposal]`: Use model to propose candidates
- `evaluate_top_model_candidates_against_pool(model: TeamValueModel, vocab: SetIdVocab, catalog: SetCatalog, pool: TeamPool, n_proposals: int = 500, top_k: int = 20, format_id: str = DEFAULT_FORMAT, n_opponents: int = 5, n_per_opponent: int = 5, seed: int = 123, device: str = "cpu", save_logs: bool = False) -> list[CandidateTeamResult]`: Propose + evaluate top-K candidates
- `mutate_team_set_ids(team_set_ids: list[str], catalog: SetCatalog, rng: Optional[random.Random] = None) -> list[str]`: Mutate team by replacing one set ID

### `core.py`
**Purpose**: High-level pipeline functions that orchestrate the modules above

**Constants:**
- `AUTO_POOL_PATH: Path`: Path to auto-generated team pool (`data/catalog/teams_regf_auto.yaml`)
- `DEFAULT_VALUE_CKPT: Path`: Default value model checkpoint path
- `DEFAULT_MATCHUP_CKPT: Path`: Default matchup model checkpoint path
- `DEFAULT_MATCHUP_JSONL: Path`: Default matchup JSONL path
- `DEFAULT_VALUE_JSONL: Path`: Default teams_vs_pool JSONL path
- `MY_MAIN_TEAM_IMPORT: str`: Hardcoded main team import text
- `CUDA_AVAILABLE: bool`: Whether CUDA is available (detected at import time)

**Internal Helpers:**
- `_resolve_device(device: str | None) -> str`: Resolve effective device for model operations. If `device` is `None` or empty, returns `'cuda'` if CUDA is available, otherwise `'cpu'`. Otherwise returns `device` as-is. All public functions that accept a `device` parameter use this helper for consistent auto-detection.

**Functions:**

*Dataset Generation:*
- `generate_full_battle_dataset(n: int = 10, format_id: str | None = None) -> None`: Generate full battle dataset
- `generate_preview_outcome_dataset(n: int = 10, sleep_ms: int = 50, format_id: str | None = None) -> None`: Generate preview+outcome dataset
- `generate_team_preview_dataset(n: int = 100, format_id: str | None = None) -> None`: Generate team preview dataset
- `generate_team_matchup_dataset(max_pairs: int = 200, n_battles_per_pair: int = 12, dataset_path: Path | str | None = None, format_id: str | None = None, pool_yaml: Path | str | None = None) -> None`: Generate pairwise team matchup dataset
- `generate_teams_vs_pool_dataset(n_samples: int = 50, team_size: int = 6, n_opponents: int = 5, n_battles_per_opponent: int = 5, include_catalog_teams: bool = False, format_id: str | None = None, teams_yaml: Path | str | None = None) -> None`: Generate teams vs pool dataset

*Training:*
- `train_team_value_model(jsonl_path: Path | str | None = None, out_path: Path | str | None = None, epochs: int = 50, batch_size: int = 64, lr: float = 3e-4, min_battles: int = 3, include_random: bool = True, include_catalog: bool = True, device: str | None = None) -> None`: Train team value model (device auto-detects: 'cuda' if available, else 'cpu')
- `train_team_matchup_model(jsonl_path: Path | str | None = None, out_path: Path | str | None = None, epochs: int = 40, batch_size: int = 64, lr: float = 1e-3, min_battles: int = 10, max_records: int | None = None, device: str | None = None) -> None`: Train team matchup model (device auto-detects: 'cuda' if available, else 'cpu')
- `auto_team_value_iter(n_iters: int = 5, init_random_samples: int = 0, per_iter_random_samples: int = 0, per_iter_model_proposals: int = 500, per_iter_top_k: int = 40, n_mutation_bases: int = 0, n_mutations_per_base: int = 1, n_opponents: int = 5, n_per_opponent: int = 5, epochs_per_iter: int = 20, batch_size: int = 64, lr: float = 3e-4, seed: int = 123, device: str | None = None) -> None`: Multi-iteration training + model-guided search loop (device auto-detects)

*Search:*
- `random_catalog_team_search(n_candidates: int = 20, team_size: int = 6, n_opponents: int = 5, n_battles_per_opponent: int = 5, top_k: int = 5, format_id: str | None = None, teams_yaml: Path | str | None = None) -> None`: Random search over catalog teams vs pool
- `model_guided_team_search(ckpt_path: Path | str | None = None, n_proposals: int = 500, top_k: int = 20, n_opponents: int = 5, n_per_opponent: int = 5, seed: int = 123, device: str | None = None, source_tag: str = "model_guided_v1", jsonl_path: Path | str | None = None) -> None`: Model-guided team search (device auto-detects)

*Evaluation:*
- `evaluate_catalog_team(set_ids: list[str], n: int = 50, format_id: str | None = None, team_as: str = "both", save_logs: bool = False) -> None`: Evaluate catalog team vs random opponents
- `evaluate_catalog_team_vs_pool(set_ids: list[str], n_opponents: int = 5, n_per_opponent: int = 5, format_id: str | None = None, teams_yaml: Path | str | None = None) -> None`: Evaluate catalog team vs pool
- `evaluate_main_team_vs_random(n: int = 50, format_id: str | None = None, team_as: str = "p1", save_logs: bool = False) -> None`: Evaluate main team vs random opponents
- `evaluate_matchup(team_a_path: Path | str, team_b_path: Path | str, n: int = 50, format_id: str | None = None) -> None`: Evaluate matchup between two teams

*Analysis:*
- `analyze_teams_vs_pool(top_k: int = 20, min_battles: int = 20, min_records: int = 1, source_prefix: str | None = None, jsonl_path: Path | str | None = None, output_csv: Path | str | None = None, output_teams_yaml: Path | str | None = None) -> None`: Analyze teams_vs_pool dataset and discover top teams
- `analyze_meta_matchups(team_id: str, teams_yaml: Path | str | None = None, ckpt_path: Path | str | None = None, top_k_best: int = 10, top_k_worst: int = 10, device: str | None = None) -> None`: Analyze team's matchup vs meta using matchup model (device auto-detects)
- `suggest_counter_teams(teams_yaml: Path | str | None = None, ckpt_path: Path | str | None = None, meta_json: Path | str | None = None, candidate_prefix: str | None = None, top_k: int = 10, device: str | None = None) -> None`: Suggest counter teams vs a meta (device auto-detects)

*Tools:*
- `dump_teams_vs_pool_features(input_path: Path | str | None = None, output_csv: Path | str | None = None) -> None`: Convert teams_vs_pool JSONL to CSV features
- `pack_team(file: str | None = None) -> None`: Convert team from export to packed format (CLI wrapper)
- `validate_team(file: str | None = None, format_id: str | None = None) -> None`: Validate a Pokemon Showdown team (CLI wrapper)
- `run_demo_battle() -> None`: Demo script to generate two random teams and simulate a battle

*Model Inference:*
- `score_catalog_team_with_model(team_id: str, ckpt_path: Path | str | None = None, device: str | None = None) -> None`: Score a catalog team using the value model (device auto-detects)
- `score_matchup_with_model(team_a_id: str, team_b_id: str, ckpt_path: Path | str | None = None, teams_yaml: Path | str | None = None, device: str | None = None) -> None`: Score a matchup using the matchup model (device auto-detects)

### `cli.py`
**Purpose**: Unified command-line interface that maps CLI commands to `vgc_lab.core` functions (which are also exported at package level)

**Location**: `src/vgc_lab/cli.py` (the actual implementation)

**Note**: `scripts/cli.py` is a convenience wrapper that simply imports and calls `vgc_lab.cli.main()`.

**Functions:**
- `main(argv: Optional[list[str]] = None) -> None`: Main CLI entrypoint that parses arguments and dispatches to appropriate core functions
- Various `_parse_*` helper functions that map parsed arguments to core function calls

### `__init__.py`
**Purpose**: Public API exports for the vgc-lab package

**Exports:**

*Functions (from `core`):*
- Dataset generation: `generate_full_battle_dataset`, `generate_preview_outcome_dataset`, `generate_team_preview_dataset`, `generate_team_matchup_dataset`, `generate_teams_vs_pool_dataset`
- Training: `train_team_value_model`, `train_team_matchup_model`, `auto_team_value_iter`
- Search: `random_catalog_team_search`, `model_guided_team_search`
- Evaluation: `evaluate_catalog_team`, `evaluate_catalog_team_vs_pool`, `evaluate_main_team_vs_random`, `evaluate_matchup`
- Analysis: `analyze_teams_vs_pool`, `analyze_meta_matchups`, `suggest_counter_teams`
- Tools: `pack_team`, `validate_team`, `dump_teams_vs_pool_features`, `run_demo_battle`
- Model inference: `score_catalog_team_with_model`, `score_matchup_with_model`

*Types (re-exported):*
- From `catalog`: `SetCatalog`, `TeamPool`
- From `models`: `TeamValueModel`, `TeamMatchupModel`
- From `analysis`: `TeamEvalSummary`, `MatchupEvalSummary`, `MetaDistribution`, `CandidateTeamResult`

**Constants:**
- `__version__`: Package version ("0.1.0")

## Key Data Files

### JSONL Datasets

#### `data/datasets/full_battles/full_battles.jsonl`
**Format**: One `FullBattleRecord` per line (JSON)

**Schema**:
- `format_id`: Format identifier
- `p1_name`, `p2_name`: Player names
- `winner`: "p1", "p2", "tie", or "unknown"
- `turns`: Number of turns
- `raw_log_path`: Path to raw battle log
- `created_at`: ISO timestamp

**Generated by**: `dataset full-battles`

#### `data/datasets/preview_outcome/preview_outcome.jsonl`
**Format**: One `PreviewOutcomeRecord` per line (JSON)

**Schema**:
- `format_id`: Format identifier
- `p1_name`, `p2_name`: Player names
- `p1_team_public`, `p2_team_public`: List of Pokemon public info (6 each)
- `p1_chosen_indices`, `p2_chosen_indices`: Indices of chosen Pokemon (4 each)
- `p1_lead_indices`, `p2_lead_indices`: Indices of lead Pokemon (2 each)
- `winner_side`, `winner_name`: Battle outcome
- `raw_log_path`: Path to raw battle log
- `created_at`: ISO timestamp
- `meta`: Additional metadata dict

**Generated by**: `dataset preview-outcome`

#### `data/datasets/team_matchups/team_matchups.jsonl`
**Format**: One `TeamMatchupRecord` per line (JSON)

**Schema**:
- `team_a_id`, `team_b_id`: Team identifiers
- `team_a_set_ids`, `team_b_set_ids`: Lists of 6 set IDs each
- `format_id`: Format identifier
- `n_battles`: Total battles
- `n_a_wins`, `n_b_wins`, `n_ties`: Win/loss/tie counts
- `avg_turns`: Average turns per battle
- `created_at`: ISO timestamp
- `meta`: Metadata (e.g., source, n_battles_per_pair)

**Generated by**: `dataset team-matchups`

**Used by**: `train matchup-model`

#### `data/datasets/teams_vs_pool/teams_vs_pool.jsonl`
**Format**: One record per line (JSON)

**Schema**:
- `team_set_ids`: List of 6 set IDs
- `format_id`: Format identifier
- `n_opponents`: Number of opponent teams sampled
- `n_battles_per_opponent`: Battles per opponent
- `n_battles_total`: Total battles
- `n_wins`, `n_losses`, `n_ties`: Win/loss/tie counts
- `win_rate`: Calculated win rate
- `avg_turns`: Average turns
- `opponent_counts`: Dict of opponent_id -> count
- `source`: Source tag (e.g., "random", "catalog", "model_guided_v1", "auto_iter_1_model")
- `created_at`: ISO timestamp
- `team_id` (optional): Team ID if from catalog

**Generated by**: 
- `dataset teams-vs-pool`
- `search model-guided`
- `search auto-value-iter`

**Used by**:
- `analyze teams-vs-pool` (generates `teams_regf_auto.yaml`)
- `train value-model`
- `tools dump-features`

### Catalog Files

#### `data/catalog/sets_regf.yaml`
**Format**: YAML list of set entries

**Schema** (per entry):
- `id`: Set identifier (unique)
- `format`: Format ID (default: gen9vgc2026regf)
- `import_text`: Pokemon Showdown import text
- `description` (optional): Description

**Used by**: All functions that build teams from catalog

#### `data/catalog/teams_regf.yaml`
**Format**: YAML list of team entries

**Schema** (per entry):
- `id`: Team identifier (unique)
- `format`: Format ID
- `description` (optional): Description
- `set_ids`: List of exactly 6 set IDs

**Used by**: All functions that use team pools

#### `data/catalog/teams_regf_auto.yaml`
**Format**: Same as `teams_regf.yaml`

**Generated by**: `analyze teams-vs-pool` with `--output-teams-yaml`

**Content**: Auto-discovered top teams from `teams_vs_pool.jsonl`

**Naming**: Teams have IDs like `auto_top_0001`, `auto_top_0002`, etc.

**Description**: Includes win_rate, battles, records in description

## Workflow Examples

### 1. Generate Initial Dataset and Train Value Model
```bash
# Generate random teams vs pool dataset
python scripts/cli.py dataset teams-vs-pool --n-samples 100 --n-opponents 5 --n-per 5

# Train value model
python scripts/cli.py train value-model --epochs 50 --device cuda

# Discover top teams
python scripts/cli.py analyze teams-vs-pool --top-k 20 --min-battles 10 --output-teams-yaml data/catalog/teams_regf_auto.yaml
```

### 2. Model-Guided Search Iteration
```bash
# Single iteration
python scripts/cli.py search model-guided --n-proposals 500 --top-k 20

# Multi-iteration training + search loop
python scripts/cli.py search auto-value-iter --n-iters 5 --per-iter-model-proposals 500 --per-iter-top-k 40
```

### 3. Generate Matchup Dataset and Train Matchup Model
```bash
# Generate pairwise matchups
python scripts/cli.py dataset team-matchups --max-pairs 200 --n-battles-per-pair 12

# Train matchup model
python scripts/cli.py train matchup-model --epochs 40 --device cuda

# Analyze team vs meta
python scripts/cli.py analyze meta-matchups my_main_team --top-k-best 10 --top-k-worst 10

# Suggest counter teams
python scripts/cli.py analyze suggest-counters --top-k 10
```

### 4. Generate Preview Datasets
```bash
# Team preview snapshots
python scripts/cli.py dataset team-preview --n 1000

# Preview + outcome pairs
python scripts/cli.py dataset preview-outcome --n 500
```

## Key Concepts

### Team Representation
- **Set ID**: Identifier for a single Pokemon set (from `sets_regf.yaml`)
- **Team**: List of 6 set IDs
- **Packed Format**: Showdown's internal team representation (base64-like string)
- **Import Format**: Human-readable Pokemon Showdown export format

### Evaluation Metrics
- **Win Rate**: n_wins / n_battles_total
- **Expected Win Rate**: Model-predicted win rate vs a meta distribution
- **Matchup Win Probability**: P(team_a wins vs team_b) from matchup model

### Source Tags
Used in `teams_vs_pool.jsonl` to track data origin:
- `random`: Random search
- `catalog`: Catalog teams
- `model_guided_v1`: Single model-guided search
- `auto_iter_{N}_model`: Model proposals from iteration N
- `auto_iter_{N}_mut`: Mutations from iteration N
- `auto_iter_{N}_random`: Random exploration in iteration N

### Model Types
1. **Team Value Model**: Predicts win_rate vs a team pool (single team input)
2. **Team Matchup Model**: Predicts P(team_a wins vs team_b) (pair of teams input)

Both use MLP architectures with set_id embeddings.
