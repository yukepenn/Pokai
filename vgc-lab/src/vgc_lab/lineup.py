"""Lineup extraction from battle logs."""

from typing import Dict, List, Tuple


def extract_lineups_from_log(
    log_text: str,
    p1_team_public: List[Dict],
    p2_team_public: List[Dict],
) -> Tuple[List[int], List[int], List[int], List[int]]:
    """
    Parse a Showdown battle log and determine, for each side:

    - Which 4 Pokémon from the 6 in team_public were actually brought.
    - Which 2 were the initial leads.

    Args:
        log_text: Full battle log text
        p1_team_public: List of dicts with public team info for p1
        p2_team_public: List of dicts with public team info for p2

    Returns:
        Tuple of (p1_chosen_indices, p2_chosen_indices, p1_lead_indices, p2_lead_indices)
        All indices are 0-based into the corresponding team_public list.
    """
    # Build species -> index mappings
    p1_species_to_idx: Dict[str, int] = {}
    p2_species_to_idx: Dict[str, int] = {}

    for idx, pokemon in enumerate(p1_team_public):
        species = pokemon.get("species", "")
        if species:
            p1_species_to_idx[species] = idx

    for idx, pokemon in enumerate(p2_team_public):
        species = pokemon.get("species", "")
        if species:
            p2_species_to_idx[species] = idx

    # Track chosen and leads
    p1_chosen_set: set[int] = set()
    p2_chosen_set: set[int] = set()
    p1_lead_indices: List[int] = []
    p2_lead_indices: List[int] = []

    battle_started = False
    lines = log_text.splitlines()

    for line in lines:
        # Mark when battle starts
        if line.strip() == "|start":
            battle_started = True
            continue

        if not battle_started:
            continue

        # Parse switch/drag lines
        if line.startswith("|switch|") or line.startswith("|drag|"):
            parts = line.split("|")
            if len(parts) < 3:
                continue

            # Extract side (p1a, p1b, p2a, p2b, etc.)
            switch_part = parts[2]
            if not switch_part.startswith(("p1", "p2")):
                continue

            is_p1 = switch_part.startswith("p1")
            side_idx = 1 if is_p1 else 2

            # Extract species from the details part (after the next |)
            if len(parts) < 4:
                continue

            details = parts[3]
            # Format: "Species, L50, M" or "Species, L50" or "Necrozma-Dawn-Wings, L50"
            # Extract species name (everything before the first comma)
            species_name = details.split(",")[0].strip()

            # Map to index
            species_map = p1_species_to_idx if is_p1 else p2_species_to_idx
            chosen_set = p1_chosen_set if is_p1 else p2_chosen_set
            lead_list = p1_lead_indices if is_p1 else p2_lead_indices

            if species_name in species_map:
                idx = species_map[species_name]
                chosen_set.add(idx)

                # Track leads (first 2 switches per side after |start)
                # Only count |switch| lines, not |drag| lines for leads
                if line.startswith("|switch|") and len(lead_list) < 2:
                    lead_list.append(idx)
            else:
                # Species not found - log a warning (but don't crash)
                print(
                    f"[WARN] Could not map species '{species_name}' to team_public for {'p1' if is_p1 else 'p2'}"
                )

    # Convert sets to sorted lists for consistency
    p1_chosen_indices = sorted(list(p1_chosen_set))
    p2_chosen_indices = sorted(list(p2_chosen_set))

    # Limit to 4 (VGC format)
    if len(p1_chosen_indices) > 4:
        print(f"[WARN] p1 has more than 4 chosen Pokémon: {p1_chosen_indices}")
        p1_chosen_indices = p1_chosen_indices[:4]
    if len(p2_chosen_indices) > 4:
        print(f"[WARN] p2 has more than 4 chosen Pokémon: {p2_chosen_indices}")
        p2_chosen_indices = p2_chosen_indices[:4]

    return (
        p1_chosen_indices,
        p2_chosen_indices,
        p1_lead_indices,
        p2_lead_indices,
    )

