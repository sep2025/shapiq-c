"""Module for finding coalitions using greedy, beam search, and recursive search algorithms."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from your_module import InteractionValues

import heapq
from itertools import combinations


def greedy_coalition(
    interaction_values: InteractionValues, coalition_size: int, *, maximize: bool = True
) -> set[int]:
    """Finds a coalition of the given size maximizing or minimizing the subset value."""
    N = list(range(interaction_values.n_players))
    selected: set[int] = set()
    remaining = set(N)

    while len(selected) < coalition_size and remaining:
        best_feature = None
        best_value = float("-inf") if maximize else float("inf")

        for f in remaining:
            candidate = selected | {f}
            val = interaction_values.get_subset(candidate).values.sum()

            if (maximize and val > best_value) or (not maximize and val < best_value):
                best_value = val
                best_feature = f

        if best_feature is None:
            break

        selected.add(best_feature)
        remaining.remove(best_feature)

    return selected


def beam_search_coalition(
    interaction_values: InteractionValues,
    coalition_size: int,
    *,
    beam_width: int = 100,
    maximize: bool = True,
) -> set[int]:
    """Beam Search for best coalition of coalition size."""
    N = list(range(interaction_values.n_players))
    beam = [set()]

    for _ in range(coalition_size):
        candidates = []
        for coalition in beam:
            remaining = set(N) - coalition
            for feature in remaining:
                new_set = coalition | {feature}
                value = interaction_values.get_subset(new_set).values.sum()
                score = -value if not maximize else value
                heapq.heappush(candidates, (score, new_set))

        top_candidates = (
            heapq.nlargest(beam_width, candidates)
            if maximize
            else heapq.nsmallest(beam_width, candidates)
        )
        beam = [s for _, s in top_candidates]

    best = (
        max(beam, key=lambda s: interaction_values.get_subset(s).values.sum())
        if maximize
        else min(beam, key=lambda s: interaction_values.get_subset(s).values.sum())
    )
    return best


def recursive_greedy_coalition_max(
    checked: set[frozenset], interaction_values: InteractionValues, max_length: int, path: list[int]
) -> tuple[float, set[int]]:
    """Recursive greedy search to find maximum value coalition."""
    values = []
    best = []
    best_value = float("-inf")
    for i in range(interaction_values.n_players):
        if i in path:
            values.append(float("-inf"))
        else:
            values.append(interaction_values.get_subset([*path, i]).values.sum())
    next_indices = [
        values.index(i)
        for i in heapq.nlargest(round(interaction_values.n_players / (len(path) + 2)), values)
    ]
    for i in next_indices:
        if frozenset([*path, i]) in checked:
            continue
        checked.add(frozenset([*path, i]))
        path.append(i)
        if len(path) == max_length - 1:
            max_index = -1
            max_value = float("-inf")
            for j in range(interaction_values.n_players):
                if j not in path:
                    path_value = interaction_values.get_subset([*path, j]).values.sum()
                    if path_value > max_value:
                        max_value = path_value
                        max_index = j
            path.append(max_index)
            return max_value, path.copy()
        value, subset = recursive_greedy_coalition_max(
            checked, interaction_values, max_length, path.copy()
        )
        if value > best_value:
            best_value = value
            best = subset
        path.pop()
    return best_value, best


def recursive_greedy_coalition_min(
    checked: set[frozenset], interaction_values: InteractionValues, max_length: int, path: list[int]
) -> tuple[float, set[int]]:
    """Recursive greedy search to find minimum value coalition."""
    values = []
    best = []
    best_value = float("inf")
    for i in range(interaction_values.n_players):
        if i in path:
            values.append(float("inf"))
        else:
            values.append(interaction_values.get_subset([*path, i]).values.sum())
    next_indices = [
        values.index(i)
        for i in heapq.nsmallest(round(interaction_values.n_players / (len(path) + 2)), values)
    ]
    for i in next_indices:
        if frozenset([*path, i]) in checked:
            continue
        checked.add(frozenset([*path, i]))
        path.append(i)
        if len(path) == max_length - 1:
            min_index = -1
            min_value = float("inf")
            for j in range(interaction_values.n_players):
                if j not in path:
                    path_value = interaction_values.get_subset([*path, j]).values.sum()
                    if path_value < min_value:
                        min_value = path_value
                        min_index = j
            path.append(min_index)
            return min_value, path.copy()
        value, subset = recursive_greedy_coalition_min(
            checked, interaction_values, max_length, path.copy()
        )
        if value < best_value:
            best_value = value
            best = subset
        path.pop()
    return best_value, best


def recursive_greedy_coalition(
    interaction_values: InteractionValues, max_length: int
) -> tuple[InteractionValues]:
    """Recursive greedy search for best coalition of size l."""
    min_subset = recursive_greedy_coalition_min(set(), interaction_values, max_length, [])[1]
    max_subset = recursive_greedy_coalition_max(set(), interaction_values, max_length, [])[1]
    return interaction_values.get_subset(min_subset), interaction_values.get_subset(max_subset)


def subset_finding(interaction_values: InteractionValues, max_size: int) -> list:
    """Return all subsets of features up to given size as list of tuples."""
    n_players = interaction_values.n_players  # <- Fix hier
    feature_indices = list(range(n_players))

    all_subsets = []
    for size in range(max_size + 1):
        all_subsets += list(combinations(feature_indices, size))

    return list(all_subsets)
