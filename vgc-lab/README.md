# vgc-lab

Python wrapper and utilities for Pokemon Showdown CLI, focused on Gen 9 VGC Reg F battle simulation and data collection.

## Overview

`vgc-lab` is a helper project that wraps the sibling `pokemon-showdown/` folder (a fork of smogon/pokemon-showdown). It provides Python utilities to:

- Generate random teams for VGC formats
- Validate teams against format rules
- Run CLI simulated battles (AI vs AI) without requiring the web server
- Save battle logs and structured metadata under `vgc-lab/data/`

## Project Structure

- `src/vgc_lab/` - Main Python package
  - `core.py` ⭐ - **Single core module** containing all functionality (config, Showdown client, battle logging, datasets, team preview)
  - `__init__.py` - Package exports
- `scripts/` - CLI tools
  - `cli.py` ⭐ - **Unified Typer CLI** with all commands (`demo-battle`, `gen-full`, `gen-preview`, `pack-team`, `validate-team`)
- `js/` - Node.js bridge scripts
  - `random_selfplay.js` - Bridge to Showdown's BattleStream and RandomPlayerAI
- `scratch/` - Temporary experimental scripts (not part of core API)
- `data/` - Generated data (teams, battle logs, JSON results, datasets)

## Team Preview Dataset

`vgc-lab` can generate structured team preview datasets from simulated battles:

- **TeamPreviewSnapshot**: Represents the open team sheet state at the start of a Reg F battle, including both players' full teams (6 Pokémon each) with moves, items, abilities, and Tera types
- **Generation method**: Uses `generate_random_team` + `simulate_battle` to create battle logs, then parses team preview requests from the logs
- **Dataset location**: `data/datasets/team_preview/team_preview.jsonl` (JSON Lines format, one snapshot per line)
- **Usage example**:
  ```bash
  cd Pokai/vgc-lab
  python -m scripts.cli gen-preview 50
  ```

This dataset is intended as the input space for future agents making team-preview decisions (which 4 mons to bring, BO3 planning, etc.), but RL/ML implementation is out of scope for this phase.

## Full Battle Dataset (Random Self-Play)

`vgc-lab` can generate datasets of completed battles using Showdown's internal random AI:

- **Method**: Uses Showdown's `BattleStream` and `RandomPlayerAI` via a Node.js script (`js/random_selfplay.js`)
- **Generation**: The command `gen-full` runs complete battles and collects results
- **Output**: JSONL dataset with battle logs, winners, turn counts, and metadata
- **Usage example**:
  ```bash
  cd Pokai/vgc-lab
  python -m scripts.cli gen-full 100
  ```
- **Data locations**:
  - Raw logs: `data/battles_raw/`
  - Dataset: `data/datasets/full_battles/full_battles.jsonl`

The Node script `js/random_selfplay.js` serves as the bridge to Showdown's battle engine, running battles to completion and outputting structured JSON results.

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

1. **Install dependencies**:
   ```bash
   pip install -e .
   ```

2. **Run a demo battle**:
   ```bash
   python -m scripts.cli demo-battle
   ```

3. **Generate battle datasets**:
   ```bash
   # Generate 100 full battle records
   python -m scripts.cli gen-full 100

   # Generate 100 team preview snapshots
   python -m scripts.cli gen-preview 100
   ```

4. **Team utilities**:
   ```bash
   # Pack a team from export format
   python -m scripts.cli pack-team --file my_team.txt

   # Validate a team
   python -m scripts.cli validate-team --file my_team.txt
   ```

All generated data is stored under `data/`:
- Raw logs: `data/battles_raw/`
- Datasets: `data/datasets/full_battles/` and `data/datasets/team_preview/`

