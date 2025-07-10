# shapiq_student/coalition_finder.py
from typing import Dict, Set, FrozenSet, Any, List, Tuple
from itertools import combinations

def compute_value(S: Set[int], explanation: Dict[str, Any]) -> float:
    value = explanation.get('bias', 0.0)
    
    # Add node weights
    for i in S:
        value += explanation['nodes'].get(i, 0.0)

    # Add edge weights (pairwise interactions)
    for i in S:
        for j in S:
            if i < j:
                value += explanation['edges'].get(frozenset({i, j}), 0.0)

    # Add hyperedge weights (higher-order interactions)
    for T, val in explanation['hyperedges'].items():
        if T.issubset(S):
            value += val

    return value

def greedy_coalition(explanation: Dict[str, Any], l: int, maximize: bool = True) -> Set[int]:
    N = list(explanation['nodes'].keys())
    selected: Set[int] = set()
    remaining = set(N)

    while len(selected) < l and remaining:
        best_feature = None
        best_value = float('-inf') if maximize else float('inf')

        for f in remaining:
            candidate = selected | {f}
            val = compute_value(candidate, explanation)

            if (maximize and val > best_value) or (not maximize and val < best_value):
                best_value = val
                best_feature = f

        if best_feature is None:
            break

        selected.add(best_feature)
        remaining.remove(best_feature)

    return selected

import heapq

def beam_search_coalition(
    explanation: dict, l: int, beam_width: int = 100, maximize: bool = True
) -> set:
    """Beam Search for best coalition of size l."""
    N = list(explanation['nodes'].keys())
    beam = [set()]  # Start with empty set

    for _ in range(l):
        candidates = []
        for coalition in beam:
            remaining = set(N) - coalition
            for feature in remaining:
                new_set = coalition | {feature}
                value = compute_value(new_set, explanation)
                score = -value if not maximize else value
                heapq.heappush(candidates, (score, new_set))

        top_candidates = heapq.nlargest(beam_width, candidates) if maximize else heapq.nsmallest(beam_width, candidates)
        beam = [s for _, s in top_candidates]

    best = max(beam, key=lambda s: compute_value(s, explanation)) if maximize else min(beam, key=lambda s: compute_value(s, explanation))
    return best

def subset_finding(interaction_values, max_size: int) -> list:
    """Return all subsets of features up to given size as list of tuples."""
    n_players = interaction_values.n_players  # <- Fix hier
    feature_indices = list(range(n_players))

    all_subsets = []
    for size in range(max_size + 1):
        all_subsets += list(combinations(feature_indices, size))

    return list(all_subsets)