"""Module for finding the maximum and minimum coalitions of an InteractionValues object using brute force, greedy, beam search, and recursive search algorithms."""

from __future__ import annotations
import heapq
from itertools import combinations
from shapiq import InteractionValues

def subset_finding(*, interaction_values: InteractionValues, max_size: int) -> InteractionValues:
    """
    Find the maximum and minimum coalitions of size l using the best algorithms we found for max and min coalitions.
    Parameters:
        interaction_values: The Interaction_Values object from which the maximum and minimum coalition are to be determined
        max_size: The maximum length of the coalition to be found.
    Returns:
        The InteractionValues object of the maximum and minimum coalition.
    """
    return recursive_greedy_coalition(interaction_values, max_size)

def brute_force(
    Interaction_values: InteractionValues, max_size: int
) -> InteractionValues:
    """
    Brute force method for the minimum and maximum coalitions of size l. Used for having the correct output for testing.
    Parameters:
        Interaction_values: The Interaction_Values object from which the minimum and maximum coalition are to be determined.
        max_size: The maximum length of the coalition to be found.
    Returns:
        Tuple of the InteractionValues object of the minimum and maximum coalition.
    """
    minNum = float("inf")
    maxNum = float("-inf")
    minCol = maxCol = []
    indices = list(range(Interaction_values.n_players))
    for combo in combinations(indices, max_size):
        coalition = Interaction_values.get_subset(list(combo)).values.sum()
        if coalition<minNum: 
            minNum = coalition 
            minCol = combo
        if coalition>maxNum: 
            maxNum = coalition 
            maxCol = combo
    if maxCol == minCol:
        interaction_lookup = {tuple(maxCol): 0}
    else:
        interaction_lookup = {tuple(maxCol): 0, tuple(minCol): 1}
    interaction = InteractionValues(
        values=(maxNum, minNum),
        interaction_lookup=interaction_lookup,
        index=Interaction_values.index,
        max_order=Interaction_values.max_order,
        n_players=Interaction_values.n_players,
        min_order=Interaction_values.min_order,
        baseline_value=Interaction_values.baseline_value,
    )
    return interaction


def greedy_coalition_call(
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


def greedy_coalition(
    interaction_values: InteractionValues, coalition_size: int
) -> InteractionValues:
    """Returns an InteractionValues object for the maximizing and minimizing greedy coalitions."""
    greedy_max = greedy_coalition_call(interaction_values, coalition_size, maximize=True)
    greedy_min = greedy_coalition_call(interaction_values, coalition_size, maximize=False)
    max_value = interaction_values.get_subset(greedy_max).values.sum()
    min_value = interaction_values.get_subset(greedy_min).values.sum()
    interaction = InteractionValues(
        values=(max_value, min_value),
        interaction_lookup={tuple(greedy_max): 0, tuple(greedy_min): 1},
        index=interaction_values.index,
        max_order=interaction_values.max_order,
        n_players=interaction_values.n_players,
        min_order=interaction_values.min_order,
        baseline_value=interaction_values.baseline_value,
    )
    return interaction


def beam_search_coalition(
    interaction_values: InteractionValues,
    coalition_size: int,
    *,
    beam_width: int = 100,
) -> InteractionValues:
    """Beam Search for best (max and min) coalition of coalition size. Returns an InteractionValues object."""
    N = list(range(interaction_values.n_players))
    # Maximize
    beam_max = [set()]
    for _ in range(coalition_size):
        candidates = []
        for coalition in beam_max:
            remaining = set(N) - coalition
            for feature in remaining:
                new_set = coalition | {feature}
                value = interaction_values.get_subset(new_set).values.sum()
                heapq.heappush(candidates, (value, new_set))
        top_candidates = heapq.nlargest(beam_width, candidates)
        beam_max = [s for _, s in top_candidates]
    max_set = max(beam_max, key=lambda s: interaction_values.get_subset(s).values.sum())
    max_value = interaction_values.get_subset(max_set).values.sum()
    # Minimize
    beam_min = [set()]
    for _ in range(coalition_size):
        candidates = []
        for coalition in beam_min:
            remaining = set(N) - coalition
            for feature in remaining:
                new_set = coalition | {feature}
                value = interaction_values.get_subset(new_set).values.sum()
                heapq.heappush(candidates, (value, new_set))
        top_candidates = heapq.nsmallest(beam_width, candidates)
        beam_min = [s for _, s in top_candidates]
    min_set = min(beam_min, key=lambda s: interaction_values.get_subset(s).values.sum())
    min_value = interaction_values.get_subset(min_set).values.sum()
    interaction = InteractionValues(
        values=(max_value, min_value),
        interaction_lookup={tuple(max_set): 0, tuple(min_set): 1},
        index=interaction_values.index,
        max_order=interaction_values.max_order,
        n_players=interaction_values.n_players,
        min_order=interaction_values.min_order,
        baseline_value=interaction_values.baseline_value,
    )
    return interaction


def recursive_greedy_coalition_max(
    Interaction_values: InteractionValues, max_size: int, path: list[int], checked: set[frozenset]
) -> tuple[float, set[int]]:
    """
    Function to find the maximum value coalition using recursive greedy search.
    Parameters: 
        Interaction_values: The Interaction_Values object from which the maximum coalition is to be determined
        max_size: The maximum length of the coalition to be found
        path: The current path of the coalition being explored
        checked: The set of already checked coalitions to avoid duplicates
    Returns:
        A tuple containing the maximum value of any checked coalition and a list containing the indices of said coalition 
    """
    values = []
    best = []
    best_value = float("-inf")
    for i in range(Interaction_values.n_players):
        if i in path:
            values.append(float("-inf"))
        else:
            values.append(Interaction_values.get_subset([*path, i]).values.sum())
    next_indices = [
        values.index(i)
        for i in heapq.nlargest(round(Interaction_values.n_players / (len(path) + 2)), values)
    ]
    for i in next_indices:
        if frozenset([*path, i]) in checked:
            continue
        checked.add(frozenset([*path, i]))
        path.append(i)
        if len(path) == max_size - 1:
            max_index = -1
            max_value = float("-inf")
            for j in range(Interaction_values.n_players):
                if j not in path:
                    path_value = Interaction_values.get_subset([*path, j]).values.sum()
                    if path_value > max_value:
                        max_value = path_value
                        max_index = j
            path.append(max_index)
            return max_value, path.copy()
        value, subset = recursive_greedy_coalition_max(
            Interaction_values, max_size, path.copy(), checked
        )
        if value > best_value:
            best_value = value
            best = subset
        path.pop()
    return best_value, best


def recursive_greedy_coalition_min(
    Interaction_values: InteractionValues, max_size: int, path: list[int], checked: set[frozenset]
) -> tuple[float, set[int]]:
    """
    Function to find the minimum value coalition using recursive greedy search.
    Parameters: 
        Interaction_values: The Interaction_Values object from which the minimum coalition is to be determined
        max_size: The minimum length of the coalition to be found
        path: The current path of the coalition being explored
        checked: The set of already checked coalitions to avoid duplicates
    Returns:
        A tuple containing the minimum value of any checked coalition and a list containing the indices of said coalition 
    """
    print("running with interaction_values", Interaction_values)
    values = []
    best = []
    best_value = float("inf")
    for i in range(Interaction_values.n_players):
        if i in path:
            values.append(float("inf"))
        else:
            values.append(Interaction_values.get_subset([*path, i]).values.sum())
    next_indices = [
        values.index(i)
        for i in heapq.nsmallest(round(Interaction_values.n_players / (len(path) + 2)), values)
    ]
    for i in next_indices:
        if frozenset([*path, i]) in checked:
            continue
        checked.add(frozenset([*path, i]))
        path.append(i)
        if len(path) == max_size - 1:
            min_index = -1
            min_value = float("inf")
            for j in range(Interaction_values.n_players):
                if j not in path:
                    path_value = Interaction_values.get_subset([*path, j]).values.sum()
                    if path_value < min_value:
                        min_value = path_value
                        min_index = j
            path.append(min_index)
            return min_value, path.copy()
        value, subset = recursive_greedy_coalition_min(
            Interaction_values, max_size, path.copy(), checked
        )
        if value < best_value:
            best_value = value
            best = subset
        path.pop()
    return best_value, best


def recursive_greedy_coalition(
    Interaction_values: InteractionValues, max_size: int
) -> InteractionValues:
    """ 
    Function to return the maximum and minimum coalitions of size l using a recursive greedy approach.
    Parameters:
        Interaction_values: The Interaction_Values object from which the maximum and minimum coalition are to be determined
        max_size: The maximum length of the coalition to be found
    Returns:
        The InteractionValues object of the maximum and minimum coalition
    """
    min_subset = recursive_greedy_coalition_min(Interaction_values, max_size, [], set())
    max_subset = recursive_greedy_coalition_max(Interaction_values, max_size, [], set())
    interaction = InteractionValues(
        values=(max_subset[0],min_subset[0]), 
        interaction_lookup={tuple(max_subset[1]):0, tuple(min_subset[1]):1}, 
        index=Interaction_values.index,
        max_order=Interaction_values.max_order,
        n_players=Interaction_values.n_players,
        min_order=Interaction_values.min_order,
        baseline_value=Interaction_values.baseline_value,)
    return interaction

def recursive_greedy_min_coalition(
    Interaction_values: InteractionValues, max_size: int
) -> InteractionValues:
    """
    Function to return the minimum coalition of size l using a recursive greedy approach.
    Parameters:
        Interaction_values: The Interaction_Values object from which the minimum coalition is to be determined
        max_size: The maximum length of the coalition to be found
    Returns:
        The InteractionValues object of the minimum coalition
    """
    min_subset = recursive_greedy_coalition_min(Interaction_values, max_size, [], set())
    interaction_min = InteractionValues(
        values=(min_subset[0]), 
        interaction_lookup={tuple(min_subset[1]):0}, 
        index=Interaction_values.index,
        max_order=Interaction_values.max_order,
        n_players=Interaction_values.n_players,
        min_order=Interaction_values.min_order,
        baseline_value=Interaction_values.baseline_value,)
    return interaction_min

def recursive_greedy_max_coalition(
    Interaction_values: InteractionValues, max_size: int
) -> InteractionValues:
    """
    Function to return the maximum coalition of size l using a recursive greedy approach.
    Parameters:
        Interaction_values: The Interaction_Values object from which the maximum coalition is to be determined
        max_size: The maximum length of the coalition to be found
    Returns:
        The InteractionValues object of the maximum coalition
    """
    max_subset = recursive_greedy_coalition_max(Interaction_values, max_size, [], set())
    interaction_max = InteractionValues(
        values=(max_subset[0]), 
        interaction_lookup={tuple(max_subset[1]):0}, 
        index=Interaction_values.index,
        max_order=Interaction_values.max_order,
        n_players=Interaction_values.n_players,
        min_order=Interaction_values.min_order,
        baseline_value=Interaction_values.baseline_value,)
    return interaction_max


def get_subset(interaction_values: InteractionValues, max_size: int) -> list:
    """Return all subsets of features up to given size as list of tuples."""
    n_players = interaction_values.n_players  # <- Fix hier
    feature_indices = list(range(n_players))

    all_subsets = []
    for size in range(max_size + 1):
        all_subsets += list(combinations(feature_indices, size))

    return list(all_subsets)
