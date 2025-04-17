#basic hill climbing search provided as base code for the DSA/ISE 5113 course
#author: Charles Nicholson
#date: 4/5/2019

#NOTE: You will need to change various parts of this code.  However, please keep the majority of the code intact (e.g., you may revise existing logic/functions and add new logic/functions, but don't completely rewrite the entire base code!)
#However, I would like all students to have the same problem instance, therefore please do not change anything relating to:
#   random number generation
#   number of items (should be 150)
#   random problem instance
#   weight limit of the knapsack

#------------------------------------------------------------------------------

#Student name: Ali Abdullah, Arman Radmanesh, and Lakshmi Sahasra Jangoan
#Date: 04-04-2025


#need some python libraries
import copy
from random import Random   #need this for the random number generation -- do not change
import numpy as np
import math


#to setup a random number generator, we will specify a "seed" value
#need this for the random number generation -- do not change
seed = 51132023
myPRNG = Random(seed)

#to get a random number between 0 and 1, use this:             myPRNG.random()
#to get a random number between lwrBnd and upprBnd, use this:  myPRNG.uniform(lwrBnd,upprBnd)
#to get a random integer between lwrBnd and upprBnd, use this: myPRNG.randint(lwrBnd,upprBnd)

#number of elements in a solution
n = 150

#create an "instance" for the knapsack problem
value = []
for i in range(0,n):
    value.append(round(myPRNG.triangular(5,1000,200),1))

weights = []
for i in range(0,n):
    weights.append(round(myPRNG.triangular(10,200,60),1))

#define max weight for the knapsack
maxWeight = 2500

#change anything you like below this line ------------------------------------

#monitor the number of solutions evaluated
solutionsChecked = 0

#function to repair a solution x by simply removing items from the knapsack until it is feasible
def repair(x):

    a=np.array(x)
    b=np.array(value)
    c=np.array(weights)

    totalValue = np.dot(a,b)     #compute the value of the knapsack selection
    totalWeight = np.dot(a,c)    #compute the weight value of the knapsack selection

    if totalWeight > maxWeight:
        print ("The solution is infeasible! Repairing...")
        # sort the items by value-to-weight ratio in descending order
        items = [(value[i], weights[i], i) for i in range(n)]
        items.sort(key=lambda item: item[0] / item[1], reverse=True)
        # remove items from the knapsack until it is feasible
        for i in range(n):
            if totalWeight > maxWeight:
                if x[i] == 1:
                    x[i] = 0
                    totalWeight -= weights[i]

    return x   #returns the repaired solution

#function to evaluate a solution x
def evaluate(x):

    a=np.array(x)
    b=np.array(value)
    c=np.array(weights)

    totalValue = np.dot(a,b)     #compute the value of the knapsack selection
    totalWeight = np.dot(a,c)    #compute the weight value of the knapsack selection

    if totalWeight > maxWeight:
        #print ("Oh no! The solution is infeasible!  What to do?  What to do?")   #you will probably want to change this...
        #return evaluate(repair(x))   #if the solution is infeasible, repair it and return the evaluation of the repaired solution
        totalValue = maxWeight-totalWeight   #if the solution is infeasible, return a negative value equal to weight overage and total weight

    return [totalValue, totalWeight]   #returns a list of both total value and total weight


#here is a simple function to create a neighborhood
#1-flip neighborhood of solution x
def neighborhood(x):

    return neighborhood1flip(x)
    #return neighborhood2flip(x)

#1-flip neighborhood of solution x
def neighborhood1flip(x):

    nbrhood = []

    for i in range(0,n):
        nbrhood.append(x[:])
        if nbrhood[i][i] == 1:
            nbrhood[i][i] = 0
        else:
            nbrhood[i][i] = 1

    return nbrhood

#2-flip neighborhood of solution x
def neighborhood2flip(x):

    nbrhood = []
    for i in range(n):
        for j in range(i + 1, n):       # ensures j > i to avoid duplicate pairs
            neighbor = x[:]             # copy the current solution
            neighbor[i] = 1 - neighbor[i]  # flip bit i
            neighbor[j] = 1 - neighbor[j]  # flip bit j
            nbrhood.append(neighbor)    # store the modified neighbor
    return nbrhood


#create the initial solution
def initial_solution():
    return initial_solution_random()


# create the initial solution by randomly selecting some percentage of items (10% by default)
def initial_solution_random(percentage=0.1):
    print("Creating initial solution...")
    x = [0] * 150   #i recommend creating the solution as a list
    # we want to create a random solution by selectin 10% of the items
    # and then repairing it if it is infeasible
    size = int(n * percentage)  # number of items to select

    totalWeight = 0
    for i in range(size):
        index = myPRNG.randint(0, n-1)  # select a random index
        if x[index] == 0:  # if the item is not already selected
            x[index] = 1  # select the item
            totalWeight += weights[index]  # add the weight of the item to the total weight
        else:
            i -= 1
            continue  # if the item is already selected, select another item
        if totalWeight >= maxWeight: # break if the weight limit is reached
            break
    print("Initial solution: ", x)
    print("Initial solution weight: ", evaluate(x)[1])

    return x   #return the solution

# Simulated Annealing
# This function will implement the simulated annealing algorithm
# to find a good solution to the knapsack problem.
def initial_temperature(x_initial):
    """
    Determines an appropriate initial temperature for simulated annealing.
    Samples random moves from the neighborhood and sets temperature based on average worsening move.
    """

    # Number of random samples to take
    num_samples = 100
    # List to store deltas of worsening moves
    worsening_deltas = []

    # Get current solution evaluation
    current_eval = evaluate(x_initial)[0]

    # Generate random neighbors and calculate deltas
    for _ in range(num_samples):
        # Select random index to flip
        idx = myPRNG.randint(0, n-1)
        neighbor = x_initial[:]
        neighbor[idx] = 1 - neighbor[idx]  # Flip the bit

        # Calculate delta (change in objective function)
        neighbor_eval = evaluate(neighbor)[0]
        delta = neighbor_eval - current_eval

        # If it's a worsening move, record the delta
        if delta < 0:
            worsening_deltas.append(abs(delta))

    # If no worsening moves found, return a default value
    if not worsening_deltas:
        return 100.0

    # Set temperature so that average worsening move is accepted with ~80% probability
    avg_worsening = sum(worsening_deltas) / len(worsening_deltas)
    initial_temp = -avg_worsening / math.log(0.8)

    return initial_temp

# Generated by Copilot
def cooling_schedule(current_temp, alpha=0.95):
    """
    Implements a simple geometric cooling schedule.

    Args:
        current_temp: The current temperature
        alpha: The cooling rate (between 0 and 1)

    Returns:
        The new temperature
    """
    return current_temp * alpha

# Generated by Copilot
def acceptance_probability(delta, temperature):
    """
    Calculates the probability of accepting a worse solution.

    Args:
        delta: The change in objective function value (new - current)
        temperature: The current temperature

    Returns:
        Probability of accepting the move [0-1]
    """
    # Always accept improving moves
    if delta >= 0:
        return 1.0

    # For worsening moves, calculate probability based on Metropolis criterion
    return math.exp(delta / temperature)


#varaible to record the number of solutions evaluated
solutionsChecked = 0

x_curr = initial_solution()  #x_curr will hold the current solution
x_best = x_curr[:]           #x_best will hold the best solution
f_curr = evaluate(x_curr)    #f_curr will hold the evaluation of the current soluton
f_best = f_curr[:]



#begin local search overall logic ----------------
done = 0

while done == 0:

    Neighborhood = neighborhood(x_curr)   #create a list of all neighbors in the neighborhood of x_curr

    for s in Neighborhood:                #evaluate every member in the neighborhood of x_curr
        solutionsChecked = solutionsChecked + 1
        if evaluate(s)[0] > f_best[0]:
            x_best = s[:]                 #find the best member and keep track of that solution
            f_best = evaluate(s)[:]       #and store its evaluation

    if f_best == f_curr:                  #if there were no improving solutions in the neighborhood
        done = 1
    else:

        x_curr = x_best[:]         #else: move to the neighbor solution and continue
        f_curr = f_best[:]         #evalute the current solution

        print ("\nTotal number of solutions checked: ", solutionsChecked)
        print ("Best value found so far: ", f_best)

print ("\nFinal number of solutions checked: ", solutionsChecked)
print ("Best value found: ", f_best[0])
print ("Weight is: ", f_best[1])
print ("Total number of items selected: ", np.sum(x_best))
print ("Best solution: ", x_best)
