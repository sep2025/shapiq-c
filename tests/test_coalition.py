from __future__ import annotations

import csv
import os
import random
import time

import pytest
from shapiq import ExactComputer
from shapiq.games.benchmark import SOUM

from shapiq_student.coalition_finding import beam_search_coalition, greedy_coalition, get_subset, brute_force



def data():
    """Generates interaction values for testing."""
    results = []
    for seed in [0, 1, 2]:
        random.seed(seed)
        game = SOUM(n=11, n_basis_games=100)
        computer = ExactComputer(n_players=game.n_players, game=game)
        results.append(computer(index="FSII", order=2))
    return results


@pytest.mark.parametrize("max_size", [5, 6, 7, "n_players"])
@pytest.mark.parametrize("Interaction_values", data())
def test_all_coalition_methods(max_size, Interaction_values):
    max_size = max_size if max_size != "n_players" else Interaction_values.n_players
    subset_runtime = brute_runtime = greedy_runtime = beam_runtime = 0

    # Subset finding
    start_subset = time.perf_counter()
    subset_output = get_subset(interaction_values=Interaction_values, max_size=max_size)
    end_subset = time.perf_counter()
    subset_runtime += end_subset - start_subset

    # Brute force
    start_brute = time.perf_counter()
    brute_result = brute_force(Interaction_values, max_size)
    end_brute = time.perf_counter()
    brute_runtime += end_brute - start_brute

    # Greedy
    start_greedy = time.perf_counter()
    greedy_result = greedy_coalition(Interaction_values, max_size)
    end_greedy = time.perf_counter()
    greedy_runtime += end_greedy - start_greedy
    greedy_max_value = greedy_result.values[0]
    greedy_min_value = greedy_result.values[1]
    keys = [k for k in greedy_result.interaction_lookup.keys() if len(k) > 0]
    if len(keys) == 1:
        only_key = keys[0]
        greedy_max_set = greedy_min_set = set(only_key)
    else:
        greedy_max_set = set(next(k for k,v in greedy_result.interaction_lookup.items() if v==0))
        greedy_min_set = set(next(k for k,v in greedy_result.interaction_lookup.items() if v==1))

    # Beam search
    start_beam = time.perf_counter()
    beam_result = beam_search_coalition(Interaction_values, max_size, beam_width=3)
    end_beam = time.perf_counter()
    beam_runtime += end_beam - start_beam
    beam_max_value = beam_result.values[0]
    beam_min_value = beam_result.values[1]
    keys = [k for k in beam_result.interaction_lookup.keys() if len(k) > 0]
    if len(keys) == 1:
        only_key = keys[0]
        beam_max_set = beam_min_set = set(only_key)
    else:
        beam_max_set = set(next(k for k,v in beam_result.interaction_lookup.items() if v==0))
        beam_min_set = set(next(k for k,v in beam_result.interaction_lookup.items() if v==1))

    # Brute force sets
    try:
        keys = [k for k in brute_result.interaction_lookup.keys() if len(k) > 0]
        if len(keys) == 1:
            only_key = keys[0]
            brute_max_set = brute_min_set = set(only_key)
        else:
            brute_max_set = set(next(k for k,v in brute_result.interaction_lookup.items() if v==0))
            brute_min_set = set(next(k for k,v in brute_result.interaction_lookup.items() if v==1))
    except StopIteration:
        print("DEBUG: interaction_lookup keys:", list(brute_result.interaction_lookup.keys()))
        raise RuntimeError("Could not find max or min coalition in brute_result.interaction_lookup")

    # Overlaps
    overlap_greedy_max = len(greedy_max_set.intersection(brute_max_set)) / max_size
    overlap_greedy_min = len(greedy_min_set.intersection(brute_min_set)) / max_size
    overlap_beam_max = len(beam_max_set.intersection(brute_max_set)) / max_size
    overlap_beam_min = len(beam_min_set.intersection(brute_min_set)) / max_size

    print(f"\n=== COMPARISON max_size={max_size} ===")
    print(f"Greedy overlap min: {overlap_greedy_min:.2%}")
    print(f"Greedy overlap max: {overlap_greedy_max:.2%}")
    print(f"Beam   overlap min: {overlap_beam_min:.2%}")
    print(f"Beam   overlap max: {overlap_beam_max:.2%}")
    print(f"Runtime Subset: {subset_runtime:.6f}")
    print(f"Runtime Brute : {brute_runtime:.6f}")
    print(f"Runtime Greedy: {greedy_runtime:.6f}")
    print(f"Runtime Beam  : {beam_runtime:.6f}")

    # Assertions
    assert greedy_max_value <= brute_result.values[0]  # Greedy should not overshoot
    assert greedy_min_value >= brute_result.values[1]  # Greedy should not undershoot
    assert beam_max_value <= brute_result.values[0]
    assert beam_min_value >= brute_result.values[1]


def write_results_to_csv(results, filename):  # makes csv to make comparison and analysis easier
    keys = results[0].keys()
    with open(filename, "w", newline="") as f:
        dict_writer = csv.DictWriter(f, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(results)


def run_all_tests_multiple_times(
    num_runs=100,
):  # runs all tests multiple times. This was supposed to compare the algos with bigger sample size
    all_results = []
    for run in range(num_runs):
        for max_size in [5, 6, 7, "n_players"]:
            for Interaction_values in data():
                current_max_size = (
                    max_size if max_size != "n_players" else Interaction_values.n_players
                )
                # Prepare explanation_dict
                single_values = Interaction_values.get_n_order_values(1)
                pairwise_values = Interaction_values.get_n_order_values(2)
                nodes = {i: single_values[i] for i in range(len(single_values))}
                edges = {}
                for i in range(len(pairwise_values)):
                    for j in range(i + 1, len(pairwise_values)):
                        edges[frozenset({i, j})] = pairwise_values[i][j]
                hyperedges = {}
                explanation_dict = {
                    "bias": Interaction_values.baseline_value,
                    "nodes": nodes,
                    "edges": edges,
                    "hyperedges": hyperedges,
                }

                # Brute force
                start = time.perf_counter()
                brute_result = brute_force(Interaction_values, current_max_size)
                t_brute = time.perf_counter() - start

                try:
                    if len(brute_result.interaction_lookup) == 1:
                        only_key = next(iter(brute_result.interaction_lookup.keys()))
                        brute_max_set = brute_min_set = set(only_key)
                    else:
                        brute_max_set = set(next(k for k,v in brute_result.interaction_lookup.items() if v==0))
                        brute_min_set = set(next(k for k,v in brute_result.interaction_lookup.items() if v==1))
                except StopIteration:
                    print("DEBUG: interaction_lookup keys:", list(brute_result.interaction_lookup.keys()))
                    raise RuntimeError("Could not find max or min coalition in brute_result.interaction_lookup")

                # Greedy
                start = time.perf_counter()
                greedy_result = greedy_coalition(Interaction_values, current_max_size)
                t_greedy = time.perf_counter() - start
                keys = [k for k in greedy_result.interaction_lookup.keys() if len(k) > 0]
                if len(keys) == 1:
                    only_key = keys[0]
                    greedy_max_set = greedy_min_set = set(only_key)
                else:
                    greedy_max_set = set(next(k for k,v in greedy_result.interaction_lookup.items() if v==0))
                    greedy_min_set = set(next(k for k,v in greedy_result.interaction_lookup.items() if v==1))
                # Beam
                start = time.perf_counter()
                beam_result = beam_search_coalition(
                    Interaction_values, current_max_size, beam_width=3
                )
                t_beam = time.perf_counter() - start
                keys = [k for k in beam_result.interaction_lookup.keys() if len(k) > 0]
                if len(keys) == 1:
                    only_key = keys[0]
                    beam_max_set = beam_min_set = set(only_key)
                else:
                    beam_max_set = set(next(k for k,v in beam_result.interaction_lookup.items() if v==0))
                    beam_min_set = set(next(k for k,v in beam_result.interaction_lookup.items() if v==1))

                # Scores
                def overlap(a, b):
                    return len(a.intersection(b)) / current_max_size

                record = {
                    "run": run + 1,
                    "max_size": current_max_size,
                    "greedy_overlap_min": round(overlap(greedy_min_set, brute_min_set), 2),
                    "greedy_overlap_max": round(overlap(greedy_max_set, brute_max_set), 2),
                    "beam_overlap_min": round(overlap(beam_min_set, brute_min_set), 2),
                    "beam_overlap_max": round(overlap(beam_max_set, brute_max_set), 2),
                    "runtime_brute": round(t_brute, 6),
                    "runtime_greedy": round(t_greedy, 6),
                    "runtime_beam": round(t_beam, 6),
                }
                all_results.append(record)
    write_results_to_csv(all_results, os.path.join(os.getcwd(), "coalition_comparison_results.csv"))


# Direkt am Ende einfügen
if __name__ == "__main__":
    run_all_tests_multiple_times(num_runs=100)
