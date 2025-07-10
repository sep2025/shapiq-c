import time
import pytest
from shapiq_student.subset_finding import brute_force
from shapiq_student.coalition_finder import greedy_coalition, compute_value, subset_finding
from shapiq import ExactComputer
from shapiq.games.benchmark import SOUM
import random
import csv
import os

def data():
    results = []
    for seed in [0, 1, 2]:
        random.seed(seed)
        game = SOUM(n=11, n_basis_games=100)
        computer = ExactComputer(n_players=game.n_players, game=game)
        results.append(computer(index="FSII", order=2))
    return results

@pytest.mark.parametrize("max_size", [5, 6, 7, "n_players"])
@pytest.mark.parametrize("Interaction_values", data())
def test_subset(max_size, Interaction_values):
    max_size = max_size if max_size != "n_players" else Interaction_values.n_players
    subset_runtime = brute_runtime = greedy_runtime = 0

    # Subset finding
    start_subset = time.perf_counter()
    subset_output = subset_finding(interaction_values=Interaction_values, max_size=max_size)
    end_subset = time.perf_counter()
    subset_runtime += end_subset - start_subset

    # Brute force
    start_brute = time.perf_counter()
    brute_min, brute_max = brute_force(Interaction_values, max_size)
    end_brute = time.perf_counter()
    brute_runtime += end_brute - start_brute

    # Greedy
    single_values = Interaction_values.get_n_order_values(1)
    pairwise_values = Interaction_values.get_n_order_values(2)
    
    # Create nodes dict: {feature_index: value}
    nodes = {i: single_values[i] for i in range(len(single_values))}
    
    # Create edges dict: {frozenset({i, j}): value}
    edges = {}
    for i in range(len(pairwise_values)):
        for j in range(i+1, len(pairwise_values)):
            edges[frozenset({i, j})] = pairwise_values[i][j]
    
    # For now, no higher-order interactions (order > 2)
    hyperedges = {}
    
    explanation_dict = {
        'bias': Interaction_values.baseline_value,
        'nodes': nodes,
        'edges': edges,
        'hyperedges': hyperedges
    }

    start_greedy = time.perf_counter()
    greedy_max = greedy_coalition(explanation_dict, max_size, maximize=True)
    greedy_min = greedy_coalition(explanation_dict, max_size, maximize=False)
    end_greedy = time.perf_counter()
    greedy_runtime += end_greedy - start_greedy

    greedy_max_value = compute_value(greedy_max, explanation_dict)
    greedy_min_value = compute_value(greedy_min, explanation_dict)
    brute_max_value = compute_value(set(brute_max), explanation_dict)
    brute_min_value = compute_value(set(brute_min), explanation_dict)
    
    overlap_max = len(greedy_max.intersection(set(brute_max))) / max_size
    overlap_min = len(greedy_min.intersection(set(brute_min))) / max_size

    print(f"Feature overlap for max: {overlap_max:.2%}")
    print(f"Feature overlap for min: {overlap_min:.2%}")

    print(f"\ntotal runtime Subset: {subset_runtime}",
          f"\ntotal runtime Brute: {brute_runtime}",
          f"\ntotal runtime Greedy: {greedy_runtime}")

    assert greedy_max_value <= brute_max_value  # Greedy should not overshoot
    assert greedy_min_value >= brute_min_value  # Greedy should not undershoot


# Test for beam_search_coalition
from shapiq_student.coalition_finder import beam_search_coalition

@pytest.mark.parametrize("max_size", [5, 6, 7, "n_players"])
@pytest.mark.parametrize("Interaction_values", data())
def test_beam_search(max_size, Interaction_values):
    max_size = max_size if max_size != "n_players" else Interaction_values.n_players
    beam_runtime = brute_runtime = 0

    single_values = Interaction_values.get_n_order_values(1)
    pairwise_values = Interaction_values.get_n_order_values(2)

    nodes = {i: single_values[i] for i in range(len(single_values))}
    edges = {}
    for i in range(len(pairwise_values)):
        for j in range(i + 1, len(pairwise_values)):
            edges[frozenset({i, j})] = pairwise_values[i][j]
    hyperedges = {}

    explanation_dict = {
        'bias': Interaction_values.baseline_value,
        'nodes': nodes,
        'edges': edges,
        'hyperedges': hyperedges
    }

    start_greedy = time.perf_counter()
    greedy_max = greedy_coalition(explanation_dict, max_size, maximize=True)
    greedy_min = greedy_coalition(explanation_dict, max_size, maximize=False)
    end_greedy = time.perf_counter()
    greedy_runtime = end_greedy - start_greedy

    start_beam = time.perf_counter()
    beam_max = beam_search_coalition(explanation_dict, max_size, beam_width=3, maximize=True)
    beam_min = beam_search_coalition(explanation_dict, max_size, beam_width=3, maximize=False)
    end_beam = time.perf_counter()
    beam_runtime += end_beam - start_beam

    # Brute Force
    start_brute = time.perf_counter()
    brute_min, brute_max = brute_force(Interaction_values, max_size)
    end_brute = time.perf_counter()
    brute_runtime += end_brute - start_brute

    beam_max_value = compute_value(beam_max, explanation_dict)
    beam_min_value = compute_value(beam_min, explanation_dict)
    brute_max_value = compute_value(set(brute_max), explanation_dict)
    brute_min_value = compute_value(set(brute_min), explanation_dict)

    overlap_max = len(beam_max.intersection(set(brute_max))) / max_size
    overlap_min = len(beam_min.intersection(set(brute_min))) / max_size

    print(f"\n=== COMPARISON max_size={max_size} ===")
    print(f"Greedy overlap min: {len(greedy_min.intersection(set(brute_min))) / max_size:.2%}")
    print(f"Greedy overlap max: {len(greedy_max.intersection(set(brute_max))) / max_size:.2%}")
    print(f"Beam   overlap min: {overlap_min:.2%}")
    print(f"Beam   overlap max: {overlap_max:.2%}")
    print(f"Runtime Brute : {brute_runtime:.6f}")
    print(f"Runtime Greedy: {greedy_runtime:.6f}")
    print(f"Runtime Beam  : {beam_runtime:.6f}")

    assert beam_max_value <= brute_max_value
    assert beam_min_value >= brute_min_value

def write_results_to_csv(results, filename):
    keys = results[0].keys()
    with open(filename, "w", newline="") as f:
        dict_writer = csv.DictWriter(f, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(results)

def run_all_tests_multiple_times(num_runs=100):
    all_results = []
    for run in range(num_runs):
        for max_size in [5, 6, 7, "n_players"]:
            for Interaction_values in data():
                current_max_size = max_size if max_size != "n_players" else Interaction_values.n_players
                # Prepare explanation_dict
                single_values = Interaction_values.get_n_order_values(1)
                pairwise_values = Interaction_values.get_n_order_values(2)
                nodes = {i: single_values[i] for i in range(len(single_values))}
                edges = {}
                for i in range(len(pairwise_values)):
                    for j in range(i+1, len(pairwise_values)):
                        edges[frozenset({i, j})] = pairwise_values[i][j]
                hyperedges = {}
                explanation_dict = {
                    'bias': Interaction_values.baseline_value,
                    'nodes': nodes,
                    'edges': edges,
                    'hyperedges': hyperedges
                }

                # Brute force
                start = time.perf_counter()
                brute_min, brute_max = brute_force(Interaction_values, current_max_size)
                t_brute = time.perf_counter() - start

                # Greedy
                start = time.perf_counter()
                greedy_min = greedy_coalition(explanation_dict, current_max_size, maximize=False)
                greedy_max = greedy_coalition(explanation_dict, current_max_size, maximize=True)
                t_greedy = time.perf_counter() - start

                # Beam
                start = time.perf_counter()
                beam_min = beam_search_coalition(explanation_dict, current_max_size, beam_width=3, maximize=False)
                beam_max = beam_search_coalition(explanation_dict, current_max_size, beam_width=3, maximize=True)
                t_beam = time.perf_counter() - start

                # Scores
                def overlap(a, b): return len(a.intersection(set(b))) / current_max_size
                record = {
                    "run": run + 1,
                    "max_size": current_max_size,
                    "greedy_overlap_min": round(overlap(greedy_min, brute_min), 2),
                    "greedy_overlap_max": round(overlap(greedy_max, brute_max), 2),
                    "beam_overlap_min": round(overlap(beam_min, brute_min), 2),
                    "beam_overlap_max": round(overlap(beam_max, brute_max), 2),
                    "runtime_brute": round(t_brute, 6),
                    "runtime_greedy": round(t_greedy, 6),
                    "runtime_beam": round(t_beam, 6),
                }
                all_results.append(record)
    write_results_to_csv(all_results, os.path.join(os.getcwd(), "coalition_comparison_results.csv"))

# Direkt am Ende einfügen
if __name__ == "__main__":
    run_all_tests_multiple_times()