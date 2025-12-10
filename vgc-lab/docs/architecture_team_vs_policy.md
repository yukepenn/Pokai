# Architecture: Team/Meta Layer vs Turn-Level Policy

## Overview

This document clarifies the two-layer architecture of our VGC learning system and the scope of this repository.

## Two Layers

### 1. Team/Meta Layer (This Repository)

**Scope:**
- Team space exploration (finding strong 6-Pokémon teams)
- Team value estimation (average win_rate vs a pool of opponents)
- Pairwise matchup prediction (P(A wins vs B) for any two teams)
- Meta analysis (identifying meta teams, counter relationships, expected win rates)
- Top-team catalogs and meta distributions

**What it learns:**
- From **full battle outcomes** in Showdown (who won/lost)
- Implicit patterns like "Urshifu + Chien-Pao tends to be strong"
- "Dondozo compositions do well vs certain team cores"
- Matchup relationships between team compositions

**What it does NOT do:**
- Explicit per-turn decision making
- Lead selection reasoning ("Seeing Urshifu, choose Flutter Mane as lead")
- In-battle tactical decisions (respecting Covert Cloak, switching patterns, Tera timing)
- BO3 game selection and adaptation strategies

**Key Models:**
- `TeamValueModel`: Predicts average win_rate vs a pool (supervised on `teams_vs_pool.jsonl`)
- `TeamMatchupModel`: Predicts P(A wins vs B) for any pair (supervised on `team_matchups.jsonl`)

**Key Datasets:**
- `teams_vs_pool.jsonl`: Team evaluations vs opponent pools
- `team_matchups.jsonl`: Pairwise matchup results
- `battle_trajectories.jsonl`: Turn-by-turn battle data (for future RL)

### 2. Turn-Level Policy / RL Layer (Future Work, Outside This Repo)

**Scope:**
- Given a specific battle state (open team sheet, current board, items, Tera types, weather, etc.)
- Decide moves, switches, Tera timing
- Choose which 4 Pokémon to bring in BO3 scenarios
- Adapt strategy based on revealed information

**Future Implementation:**
- Will likely use VGC-Bench or a Showdown-based environment
- Will leverage battle trajectories from this repo
- Will use team embeddings and matchup priors from this repo as features/priors

**Why Separate:**
- Different action spaces: team composition vs. turn-level moves
- Different training paradigms: supervised learning on outcomes vs. RL on trajectories
- Different evaluation metrics: win_rate vs. detailed battle performance

## Current Models: What They Implicitly Capture

Our current models learn from battle outcomes, so they **implicitly** encode some patterns:

- **Team synergies**: "Urshifu + Chien-Pao" appears frequently in winning teams
- **Matchup patterns**: "Dondozo teams tend to beat certain cores" emerges from matchup data
- **Meta relationships**: Popular teams and their counters are learned from pool evaluations

However, they are **not** per-turn policies. They cannot answer:
- "Given this specific board state, what move should I make?"
- "I see their Urshifu, should I lead with Flutter Mane or switch to Amoonguss?"
- "When should I use Tera in this specific game state?"

## This Repository's Goals

### 1. Make the Team/Meta Layer Powerful

- Better features for team representation
- Better self-play loops (auto-iteration of team search and evaluation)
- Stronger top-team catalogs and matchup estimates
- Continuous self-improvement of team value and matchup models

### 2. Prepare Clean Datasets + APIs for Future RL

- Battle trajectories: structured turn-by-turn data from Showdown logs
- Encoders: team embeddings, state encodings that RL agents can use
- Approximations: "bring 4" selection heuristics that can inform RL

### 3. Provide Long-Running Self-Play Orchestrator

- Continuously generate data (teams, matchups, battles)
- Retrain models periodically
- Search for better teams
- Update meta catalogs automatically

## Integration Path

When building turn-level RL in the future:

1. **Use team embeddings** from this repo as input features
2. **Use matchup priors** to inform value functions ("this team is strong vs this opponent")
3. **Use battle trajectories** as training data for RL
4. **Use 4-of-6 selection heuristics** as initial policy priors or action space constraints

The team/meta layer provides:
- **Strong priors**: Which teams are likely good
- **Matchup signals**: What to expect in specific matchups
- **Structured data**: Trajectories and encodings for RL training

## Summary

This repository focuses on **team-level** learning and meta analysis. It provides:
- Strong team compositions and meta understanding
- Datasets and encoders for future work
- Self-improving infrastructure for continuous discovery

Turn-level policy learning will be a separate layer that builds on top of these foundations.

