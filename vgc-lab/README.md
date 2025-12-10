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
  - `config.py` - Paths and configuration
  - `showdown_cli.py` - Wrappers around `node pokemon-showdown` CLI commands
  - `battle_logger.py` - BattleResult model and persistence utilities
- `scripts/` - CLI tools and demo scripts
- `data/` - Generated data (teams, battle logs, JSON results)

## Future Extensions

This project will later be extended for RL/DL training environments, but currently focuses on:
1. Establishing clean battle simulation workflows
2. Collecting battle logs and structured data
3. Providing reusable Python APIs for team generation and validation

## Dependencies

- Python 3.10+
- Node.js 16+ and npm (for the underlying `pokemon-showdown` CLI)
- The `pokemon-showdown/` folder must be present as a sibling directory

## Quick Start

See the local server quickstart guide for setup instructions.

