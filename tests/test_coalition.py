"""Tests for coalition finding algorithms with benchmark games."""

from __future__ import annotations

import csv
from pathlib import Path
import random
import time

import pytest
from shapiq import ExactComputer, InteractionValues
from shapiq.games.benchmark import SOUM

from shapiq_student.coalition_finding import (
    beam_search_coalition,
    beam_search_coalition_call,
    brute_force,
    get_subset,
    greedy_coalition,
    greedy_coalition_call,
    recursive_greedy_coalition,
    subset_finding,
)

# Constants for testing
MIN_EXPECTED_VALUES = 2  # Minimum expected values (max and min)


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
    """Test all coalition methods (brute, greedy, beam) against each other."""
    max_size = max_size if max_size != "n_players" else Interaction_values.n_players
    brute_runtime = greedy_runtime = beam_runtime = 0

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

    # Beam search
    start_beam = time.perf_counter()
    beam_result = beam_search_coalition(Interaction_values, max_size)
    end_beam = time.perf_counter()
    beam_runtime += end_beam - start_beam

    # Assertions
    assert greedy_result.values[0] <= brute_result.values[0]  # Greedy should not overshoot
    assert greedy_result.values[1] >= brute_result.values[1]  # Greedy should not undershoot
    assert beam_result.values[0] <= brute_result.values[0]
    assert beam_result.values[1] >= brute_result.values[1]


def test_subset_finding():
    """Test subset_finding function."""
    for Interaction_values in data():
        for max_size in [3, 4, 5]:
            result = subset_finding(interaction_values=Interaction_values, max_size=max_size)
            assert isinstance(result, InteractionValues)
            assert len(result.values) >= MIN_EXPECTED_VALUES  # At least max and min values
            assert result.values[0] >= result.values[1]  # max >= min


def test_get_subset():
    """Test get_subset function."""
    for Interaction_values in data():
        for max_size in [2, 3, 4]:
            result = get_subset(Interaction_values, max_size)
            assert isinstance(result, list)
            assert all(isinstance(item, tuple) for item in result)
            # Check that all subsets have size <= max_size
            assert all(len(item) <= max_size for item in result)


def test_greedy_coalition_call():
    """Test greedy_coalition_call function."""
    for Interaction_values in data():
        for max_size in [3, 4, 5]:
            # Test maximize=True
            result_max = greedy_coalition_call(Interaction_values, max_size, maximize=True)
            assert isinstance(result_max, set)
            assert len(result_max) == max_size

            # Test maximize=False
            result_min = greedy_coalition_call(Interaction_values, max_size, maximize=False)
            assert isinstance(result_min, set)
            assert len(result_min) == max_size


def test_beam_search_coalition_call():
    """Test beam_search_coalition_call function."""
    for Interaction_values in data():
        for max_size in [3, 4, 5]:
            result = beam_search_coalition_call(Interaction_values, max_size, beam_width=3)
            assert isinstance(result, InteractionValues)
            assert len(result.values) >= MIN_EXPECTED_VALUES  # At least max and min values
            assert result.values[0] >= result.values[1]  # max >= min


def test_greedy_coalition():
    """Test greedy_coalition function."""
    for Interaction_values in data():
        for max_size in [3, 4, 5]:
            result = greedy_coalition(Interaction_values, max_size)
            assert isinstance(result, InteractionValues)
            assert len(result.values) >= MIN_EXPECTED_VALUES  # At least max and min values
            assert result.values[0] >= result.values[1]  # max >= min


def test_beam_search_coalition():
    """Test beam_search_coalition function."""
    for Interaction_values in data():
        for max_size in [3, 4, 5]:
            result = beam_search_coalition(Interaction_values, max_size)
            assert isinstance(result, InteractionValues)
            assert len(result.values) >= MIN_EXPECTED_VALUES  # At least max and min values
            assert result.values[0] >= result.values[1]  # max >= min


def test_recursive_greedy_coalition():
    """Test beam_search_coalition function."""
    for Interaction_values in data():
        for max_size in [3, 4, 5]:
            result = recursive_greedy_coalition(Interaction_values, max_size)
            assert isinstance(result, InteractionValues)
            assert len(result.values) >= MIN_EXPECTED_VALUES  # At least max and min values
            assert result.values[0] >= result.values[1]  # max >= min


def test_edge_cases():
    """Test edge cases and error conditions."""
    for Interaction_values in data():
        n_players = Interaction_values.n_players

        # Test with max_size = 1
        result = brute_force(Interaction_values, 1)
        assert isinstance(result, InteractionValues)
        assert len(result.values) >= MIN_EXPECTED_VALUES

        # Test with max_size = n_players
        result = brute_force(Interaction_values, n_players)
        assert isinstance(result, InteractionValues)
        assert len(result.values) >= MIN_EXPECTED_VALUES

        # Test greedy with max_size = 1
        result = greedy_coalition(Interaction_values, 1)
        assert isinstance(result, InteractionValues)
        assert len(result.values) >= MIN_EXPECTED_VALUES

        # Test beam search with max_size = 1
        result = beam_search_coalition(Interaction_values, 1)
        assert isinstance(result, InteractionValues)
        assert len(result.values) >= MIN_EXPECTED_VALUES

        # Test beam search with max_size = 1
        result = recursive_greedy_coalition(Interaction_values, 1)
        assert isinstance(result, InteractionValues)
        assert len(result.values) >= MIN_EXPECTED_VALUES


def test_error_handling():
    """Test error handling for invalid inputs."""
    for Interaction_values in data():
        n_players = Interaction_values.n_players

        # Test with max_size > n_players (should raise error or handle gracefully)
        try:
            result = brute_force(Interaction_values, n_players + 1)
            # If no error is raised, the function should handle it gracefully
            assert isinstance(result, InteractionValues)
        except (ValueError, IndexError):
            # Error is also acceptable
            pass

        # Test with max_size = 0 (should raise error or handle gracefully)
        try:
            result = brute_force(Interaction_values, 0)
            # If no error is raised, the function should handle it gracefully
            assert isinstance(result, InteractionValues)
        except (ValueError, IndexError):
            # Error is also acceptable
            pass


def write_results_to_csv(results, filename):
    """Write results to a CSV file."""
    keys = results[0].keys()
    with Path(filename).open("w", newline="") as f:
        dict_writer = csv.DictWriter(f, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(results)


def overlap(a, b, max_size):
    """Compute the overlap ratio between two sets."""
    return len(a.intersection(b)) / max_size


def run_all_tests_multiple_times(num_runs=100):
    """Run all coalition finding methods multiple times and log the results."""
    all_results = []
    for run in range(num_runs):
        for max_size in [5, 6, 7, "n_players"]:
            for Interaction_values in data():
                current_max_size = (
                    max_size if max_size != "n_players" else Interaction_values.n_players
                )
                # Brute force
                start = time.perf_counter()
                brute_result = brute_force(Interaction_values, current_max_size)
                t_brute = time.perf_counter() - start

                try:
                    keys = [k for k in brute_result.interaction_lookup if len(k) > 0]
                    if len(keys) == 1:
                        only_key = keys[0]
                        brute_max_set = brute_min_set = set(only_key)
                    else:
                        brute_max_set = set(
                            next(k for k, v in brute_result.interaction_lookup.items() if v == 0)
                        )
                        brute_min_set = set(
                            next(k for k, v in brute_result.interaction_lookup.items() if v == 1)
                        )
                except StopIteration as err:
                    debug_message = (
                        "Could not find max or min coalition in brute_result.interaction_lookup"
                    )
                    raise RuntimeError(debug_message) from err

                # Greedy
                start = time.perf_counter()
                greedy_result = greedy_coalition(Interaction_values, current_max_size)
                t_greedy = time.perf_counter() - start
                keys = [k for k in greedy_result.interaction_lookup if len(k) > 0]
                if len(keys) == 1:
                    only_key = keys[0]
                    greedy_max_set = greedy_min_set = set(only_key)
                else:
                    greedy_max_set = set(
                        next(k for k, v in greedy_result.interaction_lookup.items() if v == 0)
                    )
                    greedy_min_set = set(
                        next(k for k, v in greedy_result.interaction_lookup.items() if v == 1)
                    )

                # Beam
                start = time.perf_counter()
                beam_result = beam_search_coalition(Interaction_values, current_max_size)
                t_beam = time.perf_counter() - start
                keys = [k for k in beam_result.interaction_lookup if len(k) > 0]
                if len(keys) == 1:
                    only_key = keys[0]
                    beam_max_set = beam_min_set = set(only_key)
                else:
                    beam_max_set = set(
                        next(k for k, v in beam_result.interaction_lookup.items() if v == 0)
                    )
                    beam_min_set = set(
                        next(k for k, v in beam_result.interaction_lookup.items() if v == 1)
                    )

                record = {
                    "run": run + 1,
                    "max_size": current_max_size,
                    "greedy_overlap_min": round(
                        overlap(greedy_min_set, brute_min_set, current_max_size), 2
                    ),
                    "greedy_overlap_max": round(
                        overlap(greedy_max_set, brute_max_set, current_max_size), 2
                    ),
                    "beam_overlap_min": round(
                        overlap(beam_min_set, brute_min_set, current_max_size), 2
                    ),
                    "beam_overlap_max": round(
                        overlap(beam_max_set, brute_max_set, current_max_size), 2
                    ),
                    "runtime_brute": round(t_brute, 6),
                    "runtime_greedy": round(t_greedy, 6),
                    "runtime_beam": round(t_beam, 6),
                }
                all_results.append(record)
    write_results_to_csv(all_results, Path.cwd() / "coalition_comparison_results.csv")


# Direkt am Ende einfügen
if __name__ == "__main__":
    run_all_tests_multiple_times(num_runs=100)
