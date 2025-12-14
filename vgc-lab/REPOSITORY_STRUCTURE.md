# Repository Structure & API Reference

Complete overview of the vgc-lab repository structure, all files, classes, functions, and their purposes.

## Overview

**vgc-lab** is a reinforcement learning framework for Pokémon VGC (Video Game Championships) that provides:

1. **Battle Simulation**: Interface with Pokémon Showdown's battle engine via Node.js
2. **Data Collection**: Generate battle logs, trajectories, and team preview snapshots
3. **RL Training**: Three-layer RL approach:
   - **Layer 1 (Team-Building)**: Value iteration for team composition
   - **Layer 2 (Preview)**: Behavior cloning for team preview (bring-4) decisions
   - **Layer 3 (Battle)**: Behavior cloning and offline DQN for in-battle move selection
4. **Python-Driven Online Self-Play**: Real-time battles where Python policies (random/BC/DQN) make decisions
5. **Offline RL Support**: Convert battle trajectories to RL transitions for offline RL algorithms (DQN, etc.)

---

## Repository Layout

```
vgc-lab/
├── src/vgc_lab/          # Core library package
│   ├── __init__.py       # Public API exports
│   ├── core.py           # Core functionality (paths, battle client, data models)
│   ├── catalog.py        # Pokemon set catalog management
│   ├── datasets.py       # Dataset utilities and iterators
│   └── features.py       # Feature encoding utilities
├── projects/             # RL subprojects (3 layers)
│   ├── rl_battle/        # Layer 3: In-battle behavior cloning
│   ├── rl_preview/       # Layer 2: Team preview (bring-4) prediction
│   └── rl_team_build/    # Layer 1: Team-building value iteration
├── scripts/              # CLI scripts
│   └── cli.py            # Unified Typer CLI (all commands)
├── js/                   # JavaScript/Node.js bridge
│   ├── random_selfplay.js        # Node-driven random self-play
│   └── py_policy_selfplay.js     # Python-driven online self-play bridge
├── tests/                # Test files
├── data/                 # Data storage
│   ├── battles_raw/      # Raw battle logs (*.log)
│   ├── battles_json/     # Parsed battle JSON (*.json)
│   ├── catalog/          # Set/team YAML files
│   └── datasets/         # JSONL datasets
│       ├── full_battles/         # Full battle records
│       ├── trajectories/         # Step-by-step battle trajectories
│       ├── team_build/           # Team-building episodes
│       └── team_preview/         # Team preview snapshots
└── checkpoints/          # Model checkpoints
    ├── battle_bc.pt      # In-battle behavior cloning model
    ├── preview_bring4.pt # Team preview model
    └── team_value.pt     # Team value model
```

---

## Core Library: `src/vgc_lab/`

### `core.py` - Core Functionality

**Purpose**: Centralized module consolidating configuration, Showdown client, battle logging, and data models.

**Constants:**
- `PROJECT_ROOT: Path` - Root directory of the project
- `SHOWDOWN_ROOT: Path` - Path to pokemon-showdown directory
- `DEFAULT_FORMAT: str = "gen9vgc2026regf"` - Default battle format
- `DEFAULT_BO3_FORMAT: str = "gen9vgc2026regfbo3"` - Best-of-3 format
- `SanitizeReason: Literal[...]` - Type-safe sanitization reason values

**Classes:**

- **`Paths`** (dataclass) - Centralized path configuration
  - **Fields**: `data_root`, `battles_raw`, `battles_json`, `datasets_root`, `full_battles`, `team_preview`, `trajectories`, `team_build`
  - **Properties**: `full_battles_jsonl`, `team_preview_jsonl`, `trajectories_jsonl`, `team_build_jsonl`

- **`FullBattleRecord`** (dataclass) - Full battle record for completed battles
  - **Fields**: `format_id`, `p1_name`, `p2_name`, `winner_side`, `winner_name`, `turns`, `raw_log_path`, `created_at`, `meta`

- **`PokemonPublicInfo`** (dataclass) - Public info about a Pokemon (from team preview)
  - **Fields**: `species`, `item`, `ability`, `tera_type`, `moves`

- **`SideTeamPreview`** (dataclass) - One side's team preview
  - **Fields**: `pokemon`

- **`TeamPreviewSnapshot`** (Pydantic BaseModel) - Complete team preview state
  - **Fields**: `format_id`, `tier_name`, `p1_preview`, `p2_preview`, `raw_log_path`, `created_at`, `meta`

- **`BattleStep`** (Pydantic BaseModel) - Single request/response step for one side
  - **Fields**: `side`, `step_index`, `request_type`, `rqid`, `turn`, `request`, `choice`, `sanitize_reason`

- **`BattleTrajectory`** (Pydantic BaseModel) - Complete battle trajectory
  - **Fields**: `battle_id`, `format_id`, `p1_name`, `p2_name`, `winner_side`, `winner_name`, `turns`, `raw_log_path`, `log_path`, `p1_team_packed`, `p2_team_packed`, `steps_p1`, `steps_p2`, `created_at`, `reward_p1`, `reward_p2`, `meta`

- **`ShowdownClient`** - Encapsulates all Showdown / Node calls
  - **Methods**:
    - `run_showdown_command(args: List[str]) -> str` - Run a Showdown CLI command
    - `generate_random_team(format_id: str) -> str` - Generate a random team
    - `pack_team(export_text: str) -> str` - Convert team from export to packed format
    - `validate_team(team_text: str, format_id: str) -> None` - Validate a team
    - `simulate_battle(team1: str, team2: str, format_id: str) -> str` - Simulate a battle, returns log
    - `run_random_selfplay_json(format_id: str) -> Dict[str, Any]` - Run random self-play, returns JSON

- **`RandomAgent`** - Random decision-making agent
  - **Methods**:
    - `choose_team_preview(request: Dict[str, Any]) -> str` - Choose team preview
    - `choose_turn_action(request: Dict[str, Any]) -> str` - Choose turn action

- **`BattleStore`** - Handles saving logs, JSON, and JSONL datasets
  - **Methods**:
    - `save_raw_log(log_text: str, format_id: str) -> Path` - Save raw battle log
    - `infer_winner_from_log(log_text: str, p1_name: str, p2_name: str) -> str` - Extract winner from log
    - `count_turns(log_text: str) -> int` - Count turns in log
    - `append_full_battle_from_json(battle_json: Dict[str, Any]) -> FullBattleRecord` - Append battle record to JSONL
    - `append_preview_snapshot(snapshot: TeamPreviewSnapshot) -> None` - Append preview snapshot to JSONL
    - `append_battle_trajectory(traj: BattleTrajectory) -> None` - Append trajectory to JSONL

**Functions:**

- `get_paths() -> Paths` - Get the default Paths configuration
- `ensure_paths(paths: Optional[Paths]) -> None` - Create all necessary directories
- `extract_team_preview_requests(log_text: str) -> List[dict]` - Extract team preview requests from log
- `parse_team_preview_snapshot(...) -> Optional[TeamPreviewSnapshot]` - Parse team preview from log

---

### `catalog.py` - Pokemon Set Catalog Management

**Purpose**: Load and manage Pokemon sets from YAML files.

**Classes:**

- **`PokemonSetDef`** - Represents a single Pokemon set definition
  - **Fields**: `id`, `format`, `description`, `import_text`, `species`, `item`

**Functions:**

- `load_sets(catalog_path: Path) -> Dict[str, PokemonSetDef]` - Load sets from YAML file
- `sample_team_sets_random(sets: Dict[str, PokemonSetDef], team_size: int = 6) -> List[str]` - Sample random team
- `build_packed_team_from_set_ids(set_ids: List[str], sets: Dict[str, PokemonSetDef]) -> str` - Build packed team string
- `_normalize_species(raw: str) -> str` - Normalize species names

---

### `datasets.py` - Dataset Utilities

**Purpose**: Dataset utilities for loading and managing battle and team-building datasets.

**Classes:**

- **`TurnView`** (dataclass) - A turn-based view of a trajectory
  - **Fields**: `turn`, `p1_steps`, `p2_steps`

- **`TeamBuildStep`** (Pydantic BaseModel) - A single step in team-building
  - **Fields**: `step_index`, `available_set_ids`, `chosen_set_id`

- **`TeamBuildEpisode`** (Pydantic BaseModel) - A high-level episode for team-building
  - **Fields**: `episode_id`, `format_id`, `side`, `created_at`, `steps`, `final_reward`, `meta`

**Functions:**

- `iter_trajectories(paths: Optional[Paths]) -> Iterator[BattleTrajectory]` - Iterate over all BattleTrajectory rows
- `group_steps_by_turn(traj: BattleTrajectory) -> List[TurnView]` - Group steps by turn
- `build_turn_views(traj: BattleTrajectory) -> List[TurnView]` - Build turn views from trajectory
- `iter_team_build_episodes(paths: Optional[Paths]) -> Iterator[TeamBuildEpisode]` - Iterate over team build episodes
- `append_team_build_episode(episode: TeamBuildEpisode, paths: Optional[Paths]) -> None` - Append episode to JSONL
- `summarize_sanitize_reasons(dataset_root: Union[str, Path]) -> Dict[SanitizeReason, int]` - Summarize sanitize reasons distribution

---

### `features.py` - Feature Encoding Utilities

**Purpose**: Feature encoding utilities for RL/BC training.

**Functions:**

- `encode_state_from_request(request: Dict[str, Any]) -> Dict[str, Any]` - Convert Showdown request to feature dict
- `encode_step(step: BattleStep) -> Tuple[Dict[str, Any], str]` - Encode BattleStep into (state, action)
- `encode_trajectory_side(traj: BattleTrajectory, side: str) -> List[Tuple[Dict, str, float, bool]]` - Encode trajectory from one side
- `encode_trajectory_both_sides(traj: BattleTrajectory) -> List[Tuple[Dict, str, float, bool]]` - Encode trajectory from both sides
- `encode_trajectory(traj: BattleTrajectory) -> List[Tuple[Dict, str, float, bool]]` - Encode trajectory (alias for both_sides)
- `encode_team_from_set_ids(set_ids: List[str], sets: Dict[str, PokemonSetDef]) -> np.ndarray` - Encode team as vector
- `encode_team_build_episode(episode: TeamBuildEpisode, sets: Dict[str, PokemonSetDef]) -> Tuple[torch.Tensor, torch.Tensor]` - Encode episode for training

---

## RL Projects: `projects/`

### Layer 3: `rl_battle/` - In-Battle Behavior Cloning & Offline RL

**Purpose**: Train and use policies for in-battle move selection. Supports both Behavior Cloning (BC) and offline DQN training.

#### `dataset.py`

**Classes:**

- **`BattleStepDatasetConfig`** (Pydantic BaseModel) - Configuration for BattleStepDataset
  - **Fields**: `format_id`, `train_allowed_sanitize_reasons`

- **`BattleStepDataset`** (PyTorch Dataset) - Dataset for in-battle joint action choices (BC training)
  - **Methods**: `__init__`, `__len__`, `__getitem__`

**Functions:**

- `is_valid_battle_choice(choice: str) -> bool` - Check if choice string is valid
- `encode_battle_state_to_vec(state_dict: dict, vec_dim: int = 256) -> np.ndarray` - Encode state to vector
- `parse_move_choice(choice: str) -> Optional[int]` - Parse move choice string to action index

#### `rl_dataset.py`

**Purpose**: Offline RL transition dataset for converting battle trajectories into RL-style transitions.

**Classes:**

- **`BattleTransition`** (dataclass) - Single RL-style transition
  - **Fields**: `state`, `action_index`, `reward`, `next_state`, `done`, `battle_id`, `step_index`, `side`

- **`RlBattleDatasetConfig`** (dataclass) - Configuration for BattleTransitionDataset
  - **Fields**: `format_id`, `vec_dim`, `max_trajectories`, `min_turns`, `allowed_policy_ids`, `forbidden_policy_ids`

- **`BattleTransitionDataset`** (PyTorch Dataset) - Dataset for offline RL transitions
  - **Methods**: `__init__`, `__len__`, `__getitem__`

**Functions:**

- `trajectory_to_transitions(traj: BattleTrajectory, vec_dim: int = 256) -> List[BattleTransition]` - Convert BattleTrajectory to RL transitions

#### `train_bc.py`

**Classes:**

- **`BattleBCConfig`** (Pydantic BaseModel) - Configuration for battle BC training
- **`BattlePolicyBC`** (nn.Module) - Battle BC policy model (MLP)

**Functions:**

- `train_battle_bc(cfg: BattleBCConfig) -> Path` - Train battle BC model, returns checkpoint path

#### `train_dqn.py`

**Purpose**: Offline DQN training on battle trajectories.

**Classes:**

- **`BattleQNetwork`** (nn.Module) - Simple MLP Q-network for battle RL
  - **Input**: state vector of dimension `vec_dim`
  - **Output**: Q-values for each discrete action
  - **Architecture**: Linear(vec_dim, hidden_dim) → ReLU → Linear(hidden_dim, hidden_dim) → ReLU → Linear(hidden_dim, num_actions)

- **`BattleDqnConfig`** (dataclass) - Configuration for battle DQN training
  - **Fields**: `format_id`, `vec_dim`, `num_actions`, `gamma`, `batch_size`, `lr`, `epochs`, `steps_per_epoch`, `target_update_interval`, `max_trajectories`, `device`, `seed`, `ckpt_path`

**Functions:**

- `train_battle_dqn(cfg: BattleDqnConfig) -> Path` - Train offline DQN model, returns checkpoint path
- `_collate_transitions(batch: list[BattleTransition]) -> list[BattleTransition]` - Collate function for DataLoader

#### `eval_bc.py`

**Functions:**

- `evaluate_battle_bc(batch_size: int = 512, topk: int = 1) -> Tuple[float, float]` - Evaluate model accuracy

#### `policy.py`

**Purpose**: Policy wrappers for both BC and DQN models used in inference.

**Classes:**

- **`BattleBCPolicyConfig`** (dataclass) - Configuration for BattleBCPolicy
  - **Fields**: `format_id`, `device`

- **`BattleBCPolicy`** - Policy wrapper for battle BC model
  - **Methods**: `encode_request()`, `score_actions()`, `choose_action()`

- **`BattleDqnPolicyConfig`** (dataclass) - Configuration for BattleDqnPolicy
  - **Fields**: `format_id`, `ckpt_path`, `vec_dim`, `num_actions`, `device`

- **`BattleDqnPolicy`** - Policy wrapper for battle DQN model
  - **Methods**: `_encode_request()`, `score_actions()`, `choose_action_argmax()`, `choose_showdown_command()`

**Functions:**

- `load_battle_bc_checkpoint(cfg: BattleBCPolicyConfig) -> Tuple[BattlePolicyBC, Dict]` - Load trained BC model
- `load_battle_dqn_checkpoint(cfg: BattleDqnPolicyConfig) -> Tuple[BattleQNetwork, Dict]` - Load trained DQN model

#### `online_selfplay.py`

**Purpose**: Python-driven online self-play where Python policies make real-time decisions.

**Types:**

- **`PythonPolicyKind`** - `Literal["random", "bc", "dqn"]` - Python policy type identifier

**Classes:**

- **`OnlineSelfPlaySummary`** (TypedDict) - Summary structure returned by run_online_selfplay
  - **Fields**: `episodes`, `errors`, `p1_wins`, `p2_wins`, `draws`

- **`OnlineSelfPlayConfig`** (dataclass) - Configuration for online self-play
  - **Fields**: `num_episodes`, `format_id`, `p1_policy`, `p2_policy`, `p1_python_policy`, `p2_python_policy`, `seed`, `write_trajectories`, `strict_invalid_choice`, `debug`
  - **Policy Selection**: `p1_policy`/`p2_policy` can be `"node_random_v1"` or `"python_external_v1"`. When using `"python_external_v1"`, `p1_python_policy`/`p2_python_policy` specify which Python policy to use (`"random"`, `"bc"`, or `"dqn"`).

- **`PythonPolicyRouter`** - Routes requests to Python policies (RandomAgent / BattleBCPolicy / BattleDqnPolicy)
  - **Methods**: `__init__()`, `_choose_team_preview()`, `_choose_move_like()`, `choose_for_request()`

**Key Functions:**

- `_compute_required_slots(request: Dict[str, Any]) -> int` - Count required choice slots
- `_sanitize_choice_for_doubles_with_reasons(choice: str, request: Dict[str, Any]) -> Tuple[str, List[SanitizeReason]]` - Sanitize choice and return reasons
- `_sanitize_choice_for_doubles(choice: str, request: Dict[str, Any]) -> str` - Sanitize choice (without reasons)
- `_emergency_valid_choice(request: Dict[str, Any], req_type: str) -> str` - Generate emergency fallback choice
- `_random_preview_choice(request: Dict[str, Any]) -> str` - Generate random preview choice
- `_build_trajectory_from_result_and_steps(...) -> Optional[Dict[str, Any]]` - Build BattleTrajectory from Node result
- `_run_single_episode(...) -> Tuple[Optional[Dict], Optional[str]]` - Run single episode loop
- `run_online_selfplay(cfg: OnlineSelfPlayConfig) -> OnlineSelfPlaySummary` - Run multiple episodes, returns summary

**Protocol**: Node.js ↔ Python JSON communication over stdin/stdout. Node sends requests, Python responds with choices.

#### `eval_battle_policies.py`

**Purpose**: Evaluation harness for battle policies via online self-play.

**Classes:**

- **`BattleEvalConfig`** (dataclass) - Configuration for battle policy evaluation
  - **Fields**: `format_id`, `num_runs`, `episodes_per_run`, `p1_policy`, `p2_policy`, `p1_python_policy`, `p2_python_policy`, `seed`, `strict_invalid_choice`, `debug`

**Functions:**

- `run_battle_eval(cfg: BattleEvalConfig) -> Dict[str, Any]` - Run multiple online self-play evaluations and aggregate results
  - **Returns**: Dict with `total_episodes`, `total_errors`, `total_p1_wins`, `total_p2_wins`, `total_draws`, `p1_win_rate`, `runs`

---

### Layer 2: `rl_preview/` - Team Preview (Bring-4) Prediction

**Purpose**: Train and use policies for team preview decisions (which 4 mons to bring).

#### `dataset.py`

**Classes:**

- **`PreviewExample`** (dataclass) - Single preview example
- **`PreviewDataset`** (PyTorch Dataset) - Dataset for team preview

**Functions:**

- `parse_team_preview_choice(choice: str) -> Optional[Tuple[Tuple[int, int, int, int], int]]` - Parse preview choice

#### `train_preview.py`

**Classes:**

- **`PreviewModel`** (nn.Module) - Preview prediction model
- **`TrainPreviewConfig`** (dataclass) - Training configuration

**Functions:**

- `train_preview_model(config: TrainPreviewConfig) -> Path` - Train preview model

#### `eval.py`

**Classes:**

- **`EvalPreviewConfig`** (dataclass) - Evaluation configuration

**Functions:**

- `run_eval_preview(config: EvalPreviewConfig) -> Dict[str, float]` - Evaluate preview model

#### `policy.py`

**Classes:**

- **`SetIdMapping`** (dataclass) - Mapping between set IDs and indices
- **`PreviewPolicy`** - Policy wrapper for preview model

**Functions:**

- `_build_mapping_from_ckpt(...) -> SetIdMapping` - Build mapping from checkpoint

---

### Layer 1: `rl_team_build/` - Team-Building Value Iteration

**Purpose**: Train value models and run value iteration for team composition.

#### `dataset.py`

**Classes:**

- **`EncodedTeamEpisode`** (dataclass) - Encoded team-building episode
- **`TeamBuildDataset`** (PyTorch Dataset) - Dataset for team-building

**Functions:**

- `build_set_id_index(sets: Dict[str, PokemonSetDef]) -> Tuple[Dict[str, int], List[str]]` - Build index mapping
- `encode_episode_to_indices(...) -> EncodedTeamEpisode` - Encode episode

#### `train_value.py`

**Classes:**

- **`TeamValueModel`** (nn.Module) - Team value prediction model

**Functions:**

- `train_value_model(...) -> Path` - Train team value model

#### `eval.py`

**Classes:**

- **`EvalConfig`** (dataclass) - Evaluation configuration

**Functions:**

- `run_eval(config: EvalConfig) -> Dict[str, Any]` - Evaluate team value model

#### `policy.py`

**Classes:**

- **`SetIdIndexMapping`** (dataclass) - Mapping between set IDs and indices
- **`TeamValuePolicy`** - Policy wrapper for team value model

**Functions:**

- `load_team_value_checkpoint(...) -> TeamValuePolicy` - Load trained model
- `build_mapping_from_sets(...) -> SetIdIndexMapping` - Build mapping from sets

#### `loop.py`

**Classes:**

- **`ValueIterationConfig`** (dataclass) - Value iteration configuration

**Functions:**

- `run_value_iteration(config: ValueIterationConfig) -> None` - Run value iteration loop

#### `selfplay.py`

**Classes:**

- **`SelfPlayConfig`** (dataclass) - Self-play configuration

**Functions:**

- `_pick_team(...) -> List[str]` - Pick team using policy
- `_append_team_build_episode_for_side(...) -> None` - Append episode
- `run_selfplay(config: SelfPlayConfig) -> Dict[str, float]` - Run self-play loop

---

## CLI: `scripts/cli.py`

**Purpose**: Unified Typer-based CLI that replaces all individual scripts.

**All Commands:**

1. **`demo-battle`** - Run a single random selfplay battle
   - **Function**: `demo_battle(format_id)`
   - **Purpose**: Quick test of battle system, generates one battle log

2. **`gen-full`** - Generate full battle dataset
   - **Function**: `gen_full(n, format_id)`
   - **Purpose**: Generate `n` battles using random self-play, saves to `full_battles.jsonl`

3. **`gen-preview`** - Generate team preview snapshots
   - **Function**: `gen_preview(n, format_id)`
   - **Purpose**: Generate `n` team preview snapshots from battles, saves to `team_preview.jsonl`

4. **`pack-team`** - Convert team from export to packed format
   - **Function**: `pack_team(file)`
   - **Purpose**: Convert team text format

5. **`validate-team`** - Validate a team against format rules
   - **Function**: `validate_team(file, format_id)`
   - **Purpose**: Check if team is valid for format

6. **`clean-data`** - Clean generated data directories
   - **Function**: `clean_data(confirm)`
   - **Purpose**: Remove battle logs, JSON, and datasets (with confirmation)

7. **`preview-dataset-stats`** - Show team preview dataset statistics
   - **Function**: `preview_dataset_stats(format_id, include_default)`
   - **Purpose**: Display statistics about team preview dataset

8. **`train-battle-bc`** - Train battle BC model
   - **Function**: `train_battle_bc_cmd(...)`
   - **Purpose**: Train in-battle behavior cloning model

9. **`battle-bc-stats`** - Show battle BC dataset statistics
   - **Function**: `battle_bc_stats(...)`
   - **Purpose**: Display statistics about battle trajectory dataset

10. **`gen-battles-from-sets`** - Generate battles from catalog sets
    - **Function**: `gen_battles_from_sets(...)`
    - **Purpose**: Generate battles using teams built from catalog sets

11. **`value-iter`** - Run value iteration for team-building
    - **Function**: `value_iter(...)`
    - **Purpose**: Run value iteration loop for team value model

12. **`train-battle-dqn`** - Train offline DQN model
    - **Function**: `train_battle_dqn_cmd(...)`
    - **Purpose**: Train offline DQN on battle trajectories, saves checkpoint to `checkpoints/battle_dqn.pt`

13. **`battle-rl-dataset-stats`** - Show offline RL dataset statistics
    - **Function**: `battle_rl_dataset_stats(...)`
    - **Purpose**: Display statistics about BattleTransitionDataset (episodes, transitions, avg reward, done fraction)

14. **`online-selfplay`** - Run Python-driven online self-play
    - **Function**: `online_selfplay_cmd(...)`
    - **Purpose**: Run battles where Python policies make decisions
    - **Options**: `--p1-policy`, `--p2-policy`, `--p1-python-policy`, `--p2-python-policy` (random/bc/dqn)

15. **`battle-eval`** - Evaluate battle policies over multiple runs
    - **Function**: `battle_eval_cmd(...)`
    - **Purpose**: Run multiple evaluation runs and aggregate results (win rates, etc.)
    - **Options**: `--num-runs`, `--episodes-per-run`, policy configuration options

16. **`analyze-sanitizer`** - Analyze sanitizer reason statistics
    - **Function**: `analyze_sanitizer(dataset_root)`
    - **Purpose**: Show distribution of sanitize_reason values in trajectories

---

## JavaScript Bridge: `js/`

### `random_selfplay.js`

**Purpose**: Node.js script for running random self-play battles via Showdown's BattleStream.

**Usage**: Called via subprocess from Python's `ShowdownClient.run_random_selfplay_json()`

**Output**: JSON with battle results, log, and metadata

---

### `py_policy_selfplay.js`

**Purpose**: Node.js bridge for Python-driven online self-play.

**Key Components:**

- **`buildFallbackChoiceFromRequest(request)`** - Generate fallback choice when Python choice is invalid
  - Handles team preview, wait, move, and force-switch requests
  - Implements same invariants as Python sanitizer
  - Never returns "pass" unless truly unavoidable

- **`PythonBridgePlayerAI`** - PlayerAI wrapper for Python policies
  - **Methods**:
    - `receiveRequest(request)` - Handle request from Showdown
    - `receiveError(error)` - Handle invalid choice errors
    - `_handleRequestAsync(request)` - Async request handler

- **`LoggingRandomPlayerAI`** - Wrapper for random AI with logging

**Protocol**: 
- Node → Python: `{type: "request", side: "p1"|"p2", request_type: "preview"|"move"|"force-switch"|"wait", request: {...}}`
- Python → Node: `{type: "action", choice: "..."}`
- Node → Python: `{type: "result", ...}` (at battle end)

**Environment Variables:**
- `PY_POLICY_STRICT_INVALID_CHOICE=1` - Strict mode: invalid choices abort episode
- `PY_POLICY_DEBUG=1` - Enable debug logging

---

## Tests: `tests/`

### Test Files

1. **`test_choice_counting.py`** - Tests for choice counting logic
2. **`test_invalid_choice_regressions.py`** - Regression tests for invalid choice bugs
3. **`test_online_selfplay_sanitizer.py`** - Tests for sanitizer functions
4. **`test_sanitize_reason_filtering.py`** - Tests for sanitize reason filtering in datasets
5. **`test_sanitizer_mixed_force_switch.py`** - Tests for mixed force-switch scenarios
6. **`test_sanitizer_reasons.py`** - Tests for sanitize reason assignment
7. **`test_sanitizer_statistics.py`** - Tests for sanitize reason statistics
8. **`test_rl_battle_dataset.py`** - Tests for RL transition dataset (trajectory_to_transitions, BattleTransitionDataset)
9. **`test_train_battle_dqn.py`** - Tests for DQN training (BattleQNetwork, train_battle_dqn)
10. **`test_battle_dqn_policy.py`** - Tests for DQN policy (BattleDqnPolicy, load_battle_dqn_checkpoint)
11. **`test_online_selfplay_dqn_router.py`** - Tests for DQN integration in online self-play router
12. **`test_eval_battle_policies.py`** - Tests for battle policy evaluation harness (run_battle_eval)

---

## Data Storage: `data/`

### Directories

- **`battles_raw/`** - Raw battle logs (*.log files)
- **`battles_json/`** - Parsed battle JSON (*.json files)
- **`catalog/`** - Pokemon set and team YAML files
  - `sets_regf.yaml` - Pokemon set definitions
  - `teams_regf.yaml` - Predefined teams
- **`datasets/`** - JSONL datasets
  - `full_battles/full_battles.jsonl` - Full battle records
  - `trajectories/trajectories.jsonl` - Step-by-step battle trajectories
  - `team_build/episodes.jsonl` - Team-building episodes
  - `team_preview/team_preview.jsonl` - Team preview snapshots

---

## Model Checkpoints: `checkpoints/`

- **`battle_bc.pt`** - In-battle behavior cloning model (Layer 3)
- **`battle_dqn.pt`** - In-battle offline DQN model (Layer 3)
- **`preview_bring4.pt`** - Team preview model (Layer 2)
- **`team_value.pt`** - Team value model (Layer 1)

---

## Key Concepts

### SanitizeReason

Type-safe enum for tracking why a choice was sanitized:
- `"ok"` - No sanitization needed
- `"fixed_pass"` - "pass" was converted to a move
- `"fixed_disabled_move"` - Disabled move was replaced
- `"fixed_switch_to_move"` - Non-forced switch was converted to move
- `"fallback_switch"` - Fallback switch was used
- `"fallback_pass"` - Fallback "pass" was used (extremely rare)

### Request Types

- `"preview"` - Team preview request (choose 4 from 6)
- `"move"` - Normal move turn request
- `"force-switch"` - Force-switch request (Pokemon fainted)
- `"wait"` - Wait request (opponent's turn)

### Data Flow

1. **Battle Generation**: `gen-full` → generates battles → saves to `full_battles.jsonl`
2. **Trajectory Extraction**: `online-selfplay` → generates trajectories → saves to `trajectories.jsonl`
3. **BC Training**: `train-battle-bc` → loads trajectories → trains BC model → saves checkpoint to `battle_bc.pt`
4. **DQN Training**: `train-battle-dqn` → loads trajectories → converts to RL transitions → trains DQN model → saves checkpoint to `battle_dqn.pt`
5. **BC Evaluation**: `battle-bc-stats` → loads trajectories → computes statistics
6. **RL Dataset Stats**: `battle-rl-dataset-stats` → loads transitions → displays dataset statistics
7. **Policy Evaluation**: `battle-eval` → runs multiple online self-play episodes → aggregates win rates

---

## External Dependencies

- **pokemon-showdown/** - Upstream Showdown repository (battle engine)
  - Used via Node.js: `dist/sim/index.js`
  - Provides: `BattleStream`, `RandomPlayerAI`, `Teams`, etc.

---

## Development Workflow

### Basic Workflow (BC):
1. **Generate Data**: `python -m scripts.cli gen-full 100`
2. **Run Online Self-Play**: `python -m scripts.cli online-selfplay --num-episodes 30`
3. **Train BC Model**: `python -m scripts.cli train-battle-bc`
4. **Evaluate BC**: `python -m scripts.cli battle-bc-stats`

### DQN Workflow:
1. **Generate Trajectories**: `python -m scripts.cli online-selfplay --num-episodes 100`
2. **Train DQN**: `python -m scripts.cli train-battle-dqn --epochs 10 --steps-per-epoch 200`
3. **Check Dataset Stats**: `python -m scripts.cli battle-rl-dataset-stats`
4. **Evaluate Policies**: `python -m scripts.cli battle-eval --p1-python-policy dqn --p2-python-policy random --num-runs 5`

### Testing:
- **Run Tests**: `pytest`

---

## Notes

- All paths are managed via `Paths` dataclass in `core.py`
- All data models use Pydantic for validation
- JSONL format used for all datasets (one JSON object per line)
- Node.js bridge scripts communicate with Python via stdin/stdout JSON protocol
- Sanitizer ensures all choices are valid for Showdown's battle engine
- **RL Transitions**: `BattleTransition` represents the standard RL tuple `(state, action, reward, next_state, done)`
- **Policy Routing**: PythonPolicyRouter supports three policy types: `"random"` (RandomAgent), `"bc"` (BattleBCPolicy), and `"dqn"` (BattleDqnPolicy)
- **DQN Training**: Uses target network and MSE loss, supports configurable gamma, batch size, learning rate, and update intervals
