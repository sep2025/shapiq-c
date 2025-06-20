import time
import pytest
from shapiq_student import subset_finding
from shapiq import ExactComputer
from shapiq.games.benchmark import SOUM

#test, ob subset_finding function aus shapiq_student.subset_finding das korrekte subset liefert. mit "pytest --count x -s" im terminal runnen. x entspricht anzahl der tests
#-s damit runtime auf der konsole ausgegeben wird

def data():
    game = SOUM(n=10, n_basis_games=50)
    computer = ExactComputer(n_players=game.n_players, game=game)       
    return [computer(index="FSII", order=2),]
    
@pytest.mark.parametrize("max_size", [5])
@pytest.mark.parametrize("Interaction_values",data())
def test_subset(max_size,Interaction_values):
    subset_runtime = brute_runtime = 0
    start_subset = time.perf_counter()
    output = subset_finding.subset_finding(Interaction_values,max_size)
    end_subset = time.perf_counter()
    subset_runtime += end_subset - start_subset
    start_brute = time.perf_counter()
    correct = subset_finding.brute_force(Interaction_values,max_size)
    end_brute = time.perf_counter()
    brute_runtime += end_brute - start_brute
    print(f"\ntotal runtime Subset: {subset_runtime}",f"\ntotal runtime Brute: {brute_runtime}")
    assert output == correct
