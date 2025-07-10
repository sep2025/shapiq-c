from itertools import combinations
import numpy

#aktuell noch brute force für test setup, einfach mit zu testender function replacen
def subset_finding(interaction_values, max_size):
    minNum = float("inf")
    maxNum = float("-inf")
    minCol = maxCol = numpy.empty(max_size)
    indices = list(range(interaction_values.n_players))
    for combo in combinations(indices, max_size):
        coalition = interaction_values.get_subset(list(combo)).values.sum()
        if coalition<minNum: 
            minNum = coalition 
            minCol = combo
        if coalition>maxNum: 
            maxNum = coalition 
            maxCol = combo
    return [minCol, maxCol]

#brute force function für korrekte Lösung für Tests
def brute_force(InteractionValues,max_value):
    minNum = float("inf")
    maxNum = float("-inf")
    minCol = MaxCol = numpy.empty(max_value)
    indices = list(range(InteractionValues.n_players))
    for combo in combinations(indices, max_value):
        coalition = InteractionValues.get_subset(list(combo)).values.sum()
        if coalition<minNum: 
            minNum = coalition 
            minCol = combo
        if coalition>maxNum: 
            maxNum = coalition 
            MaxCol = combo
    return(minCol,MaxCol)
