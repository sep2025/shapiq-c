"""Module for finding the maximum and minimum coalitions of an InteractionValues object using brute force, greedy, beam search, and recursive search algorithms."""

from __future__ import annotations

import heapq
from itertools import combinations

import numpy as np
from shapiq import InteractionValues


def check_edge_cases(Interaction_values: InteractionValues, max_size: int) -> None:
    """Check edge cases for coalition finding functions.

    Parameters:
        Interaction_values: The InteractionValues object to validate
        max_size: The maximum size of coalitions to find

    Raises:
        TypeError: If Interaction_values is not an InteractionValues object
        ValueError: If max_size is larger than n_players or less than 1
    """
    if not isinstance(Interaction_values, InteractionValues):
        msg = "Did not pass an InteractionValues object as Interaction_values."
        raise TypeError(msg)
    if max_size > Interaction_values.n_players:
        msg = "Max_size is larger than the amount of features in the InteractionValues object."
        raise ValueError(msg)
    if max_size < 1:
        msg = "Max_size 0 or smaller. No coalition can be computed."
        raise ValueError(msg)


def subset_finding(*, interaction_values: InteractionValues, max_size: int) -> InteractionValues:
    """Find the maximum and minimum coalitions of size l using the best algorithms we found for max and min coalitions.

    Parameters:
        interaction_values: The Interaction_Values object from which the maximum and minimum coalition are to be determined
        max_size: The maximum length of the coalition to be found.

    Returns:
        The InteractionValues object of the maximum and minimum coalition.
    """
    check_edge_cases(interaction_values, max_size)
    return recursive_greedy_coalition(interaction_values, max_size)


def brute_force(Interaction_values: InteractionValues, max_size: int) -> InteractionValues:
    """Brute force method for the minimum and maximum coalitions of size l. Used for having the correct output for testing.

    Parameters:
        Interaction_values: The Interaction_Values object from which the minimum and maximum coalition are to be determined.
        max_size: The maximum length of the coalition to be found.

    Returns:
        Tuple of the InteractionValues object of the minimum and maximum coalition.
    """
    check_edge_cases(Interaction_values, max_size)
    minNum = float("inf")
    maxNum = float("-inf")
    minCol: tuple[int, ...] = ()
    maxCol: tuple[int, ...] = ()
    indices = list(range(Interaction_values.n_players))
    for combo in combinations(indices, max_size):
        coalition = Interaction_values.get_subset(list(combo)).values.sum()
        if coalition < minNum:
            minNum = coalition
            minCol = combo
        if coalition > maxNum:
            maxNum = coalition
            maxCol = combo
    if maxCol == minCol:
        interaction_lookup = {tuple(maxCol): 0}
    else:
        interaction_lookup = {tuple(maxCol): 0, tuple(minCol): 1}
    interaction = InteractionValues(
        values=np.array([maxNum, minNum]),
        interaction_lookup=interaction_lookup,
        index=Interaction_values.index,
        max_order=Interaction_values.max_order,
        n_players=Interaction_values.n_players,
        min_order=Interaction_values.min_order,
        baseline_value=Interaction_values.baseline_value,
    )
    return interaction


def greedy_coalition_call(
    interaction_values: InteractionValues,
    coalition_size: int,
    *,
    maximize: bool,
) -> set[int]:
    """Find a coalition of a given size that either maximizes or minimizes the interaction value using a greedy algorithm.

    Parameters:
        interaction_values: The InteractionValues object from which the coalition is to be determined.
        coalition_size: The desired size of the coalition.
        maximize: If True, the algorithm maximizes the value. If False, it minimizes.

    Returns:
        A set of feature indices forming the selected coalition.

    Raises:
        ValueError: If coalition_size is smaller than 1 or larger than the number of players.
        TypeError: If interaction_values is not of type InteractionValues.
    """
    N = list(range(interaction_values.n_players))
    selected: set[int] = set()
    remaining = set(N)

    while len(selected) < coalition_size and remaining:
        best_feature = None
        best_value = float("-inf") if maximize else float("inf")

        for f in remaining:
            candidate = selected | {f}
            val = interaction_values.get_subset(list(candidate)).values.sum()

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
"""Find the maximum and minimum coalitions of a given size using a greedy approach.
This function is a wrapper for `greedy_coalition_call` and simplifies usage by requiring only two parameters.
    
    Parameters:
        interaction_values: The InteractionValues object from which the coalitions are to be determined.
        coalition_size: The desired size of the coalitions.

    Returns:
        An InteractionValues object containing the maximum and minimum coalition values.

    Raises:
        ValueError: If coalition_size is smaller than 1 or larger than the number of players.
        TypeError: If interaction_values is not of type InteractionValues.
    """
    check_edge_cases(interaction_values, coalition_size)
    # Maximize
    greedy_max = greedy_coalition_call(interaction_values, coalition_size, maximize=True)
    max_value = interaction_values.get_subset(list(greedy_max)).values.sum()

    # Minimize
    greedy_min = greedy_coalition_call(interaction_values, coalition_size, maximize=False)
    min_value = interaction_values.get_subset(list(greedy_min)).values.sum()

    interaction = InteractionValues(
        values=np.array([max_value, min_value]),
        interaction_lookup={tuple(greedy_max): 0, tuple(greedy_min): 1},
        index=interaction_values.index,
        max_order=interaction_values.max_order,
        n_players=interaction_values.n_players,
        min_order=interaction_values.min_order,
        baseline_value=interaction_values.baseline_value,
    )
    return interaction


def beam_search_coalition_call(
    interaction_values: InteractionValues,
    coalition_size: int,
    *,
    beam_width: int,
) -> InteractionValues:
    """Find the maximum and minimum coalitions of a given size using beam search.

    Parameters:
        interaction_values: The InteractionValues object from which the coalitions are to be determined.
        coalition_size: The desired size of the coalitions.
        beam_width: The number of top candidates to retain at each level of the search.

    Returns:
        An InteractionValues object containing the maximum and minimum coalition values.

    Raises:
        ValueError: If coalition_size is smaller than 1 or larger than the number of players.
        TypeError: If interaction_values is not of type InteractionValues.
    """
    check_edge_cases(interaction_values, coalition_size)
    N = list(range(interaction_values.n_players))
    # Maximize
    beam_max: list[set[int]] = [set()]
    for _ in range(coalition_size):
        max_candidates: list[tuple[float, set[int]]] = []
        for coalition in beam_max:
            remaining = set(N) - coalition
            for feature in remaining:
                new_set = coalition | {feature}
                value = interaction_values.get_subset(list(new_set)).values.sum()
                heapq.heappush(max_candidates, (value, new_set))
        top_candidates = heapq.nlargest(beam_width, max_candidates)
        beam_max = [s for _, s in top_candidates]
    max_set = max(beam_max, key=lambda s: interaction_values.get_subset(list(s)).values.sum())
    max_value = interaction_values.get_subset(list(max_set)).values.sum()
    # Minimize
    beam_min: list[set[int]] = [set()]
    for _ in range(coalition_size):
        min_candidates: list[tuple[float, set[int]]] = []
        for coalition in beam_min:
            remaining = set(N) - coalition
            for feature in remaining:
                new_set = coalition | {feature}
                value = interaction_values.get_subset(list(new_set)).values.sum()
                heapq.heappush(min_candidates, (value, new_set))
        top_candidates = heapq.nsmallest(beam_width, min_candidates)
        beam_min = [s for _, s in top_candidates]
    min_set = min(beam_min, key=lambda s: interaction_values.get_subset(list(s)).values.sum())
    min_value = interaction_values.get_subset(list(min_set)).values.sum()
    interaction = InteractionValues(
        values=np.array([max_value, min_value]),
        interaction_lookup={tuple(max_set): 0, tuple(min_set): 1},
        index=interaction_values.index,
        max_order=interaction_values.max_order,
        n_players=interaction_values.n_players,
        min_order=interaction_values.min_order,
        baseline_value=interaction_values.baseline_value,
    )
    return interaction


def beam_search_coalition(
    interaction_values: InteractionValues, coalition_size: int
) -> InteractionValues:
    """Find the maximum and minimum coalitions of a given size using beam search with a fixed beam width of 3. 
    This wrapper simplifies usage by requiring only two parameters and internally calls beam_search_coalition_call.

    Parameters:
        interaction_values: The InteractionValues object from which the coalitions are to be determined.
        coalition_size: The desired size of the coalitions.

    Returns:
        An InteractionValues object containing the maximum and minimum coalition values.

    Raises:
        ValueError: If coalition_size is smaller than 1 or larger than the number of players.
        TypeError: If interaction_values is not of type InteractionValues.
    """
    check_edge_cases(interaction_values, coalition_size)
    return beam_search_coalition_call(interaction_values, coalition_size, beam_width=3)


def recursive_greedy_coalition_max(
    Interaction_values: InteractionValues, max_size: int, path: list[int], checked: set[frozenset]
) -> tuple[float, set[int]]:
    """Function to find the maximum value coalition using recursive greedy search.

    Parameters:
        Interaction_values: The Interaction_Values object from which the maximum coalition is to be determined
        max_size: The maximum length of the coalition to be found
        path: The current path of the coalition being explored
        checked: The set of already checked coalitions to avoid duplicates

    Returns:
        A tuple containing the maximum value of any checked coalition and a list containing the indices of said coalition.
    """
    values: list[float] = []
    best: set[int] = set()
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
            return max_value, set(path.copy())
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
    """Function to find the minimum value coalition using recursive greedy search.

    Parameters:
        Interaction_values: The Interaction_Values object from which the minimum coalition is to be determined
        max_size: The minimum length of the coalition to be found
        path: The current path of the coalition being explored
        checked: The set of already checked coalitions to avoid duplicates

    Returns:
        A tuple containing the minimum value of any checked coalition and a list containing the indices of said coalition.
    """
    values: list[float] = []
    best: set[int] = set()
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
            return min_value, set(path.copy())
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
    """Function to return the maximum and minimum coalitions of size l using a recursive greedy approach.

    Parameters:
        Interaction_values: The Interaction_Values object from which the maximum and minimum coalition are to be determined
        max_size: The maximum length of the coalition to be found

    Returns:
        The InteractionValues object of the maximum and minimum coalition.
        
    Raises:
        ValueError: If coalition_size is smaller than 1 or larger than the number of players.
        TypeError: If interaction_values is not of type InteractionValues.
    """
    check_edge_cases(Interaction_values, max_size)
    min_subset = recursive_greedy_coalition_min(Interaction_values, max_size, [], set())
    max_subset = recursive_greedy_coalition_max(Interaction_values, max_size, [], set())
    interaction = InteractionValues(
        values=np.array([max_subset[0], min_subset[0]]),
        interaction_lookup={tuple(max_subset[1]): 0, tuple(min_subset[1]): 1},
        index=Interaction_values.index,
        max_order=Interaction_values.max_order,
        n_players=Interaction_values.n_players,
        min_order=Interaction_values.min_order,
        baseline_value=Interaction_values.baseline_value,
    )
    return interaction


def recursive_greedy_min_coalition(
    Interaction_values: InteractionValues, max_size: int
) -> InteractionValues:
    """Function to return the minimum coalition of size l using a recursive greedy approach.

    Parameters:
        Interaction_values: The Interaction_Values object from which the minimum coalition is to be determined
        max_size: The maximum length of the coalition to be found

    Returns:
        The InteractionValues object of the minimum coalition.
        
    Raises:
        ValueError: If coalition_size is smaller than 1 or larger than the number of players.
        TypeError: If interaction_values is not of type InteractionValues.
    """
    min_subset = recursive_greedy_coalition_min(Interaction_values, max_size, [], set())
    interaction_min = InteractionValues(
        values=np.array([min_subset[0]]),
        interaction_lookup={tuple(min_subset[1]): 0},
        index=Interaction_values.index,
        max_order=Interaction_values.max_order,
        n_players=Interaction_values.n_players,
        min_order=Interaction_values.min_order,
        baseline_value=Interaction_values.baseline_value,
    )
    return interaction_min


def recursive_greedy_max_coalition(
    Interaction_values: InteractionValues, max_size: int
) -> InteractionValues:
    """Function to return the maximum coalition of size l using a recursive greedy approach.

    Parameters:
        Interaction_values: The Interaction_Values object from which the maximum coalition is to be determined
        max_size: The maximum length of the coalition to be found

    Returns:
        The InteractionValues object of the maximum coalition.
        
    Raises:
        ValueError: If coalition_size is smaller than 1 or larger than the number of players.
        TypeError: If interaction_values is not of type InteractionValues.
    """
    max_subset = recursive_greedy_coalition_max(Interaction_values, max_size, [], set())
    interaction_max = InteractionValues(
        values=np.array([max_subset[0]]),
        interaction_lookup={tuple(max_subset[1]): 0},
        index=Interaction_values.index,
        max_order=Interaction_values.max_order,
        n_players=Interaction_values.n_players,
        min_order=Interaction_values.min_order,
        baseline_value=Interaction_values.baseline_value,
    )
    return interaction_max


def get_subset(interaction_values: InteractionValues, max_size: int) -> list:
    """Return all subsets of features up to given size as list of tuples."""
    n_players = interaction_values.n_players  # <- Fix hier
    feature_indices = list(range(n_players))

    all_subsets = []
    for size in range(max_size + 1):
        all_subsets += list(combinations(feature_indices, size))

    return list(all_subsets)
