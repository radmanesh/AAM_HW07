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

#Student name: Arman Radmanesh, Ali Abdullah
#Date: 04-16-2025


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
# this function is the the same as the on Dr Nicholson provided in the base code
# just more succinct
def neighborhood1flip(x):

    nbrhood = []
    for i in range(0,n):
        nbrhood.append(x[:])
        if nbrhood[i][i] == 1:
            nbrhood[i][i] = 0
        else:
            nbrhood[i][i] = 1
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

    # repair the solution if it is infeasible
    if evaluate(x)[1] > maxWeight:
        print("Repairing solution...")
        x = repair(x)
        print("Repaired solution: ", x)
        print("Repaired solution weight: ", evaluate(x)[1])
        print("Repaired solution value: ", evaluate(x)[0])

    print("Initial solution: ", x)
    print("Initial solution weight: ", evaluate(x)[1])
    print("Initial solution value: ", evaluate(x)[0])

    return x   #return the solution

# This function will implement the calculation of initial temperature for simulated annealing algorithm
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

# Cooling schedule function
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

# Cauchy Cooling schedule function
def cooling_schedule_cauchy(current_temp, initial_temp, k):
    """
    Implements a cooling schedule based on the Cauchy distribution.
    Args:
        current_temp: The current temperature
        initial_temp: The initial temperature
        k: The iteration number
    Returns:
        The new temperature
    """
    # Cauchy cooling schedule
    return initial_temp / (1 + k)  # Cauchy cooling

# Boltzmann Cooling schedule function
def cooling_schedule_boltzmann(current_temp, initial_temp, k):
    """
    Implements a cooling schedule based on the Boltzmann distribution.
    Args:
        current_temp: The current temperature
        initial_temp: The initial temperature
        k: The iteration number
    Returns:
        The new temperature
    """
    return initial_temp / math.log(1 + k)  # Boltzmann cooling

# Ali's Cooling schedule function
def cooling_schedule_Ali(current_temp, initial_temp):
    """
    Implements a cooling schedule based on the initial temperature.

    Args:
        current_temp: The current temperature
        initial_temp: The initial temperature

    Returns:
        The new temperature
    """
    #return initial_temp / (log(1 + current_temp)) - sqrt(current_temp)  # Ali's cooling schedule
    return current_temp * (1 - (current_temp / initial_temp))  # Exponential cooling

# Acceptance probability function
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
initial_solution_percentage = 0.2  # percentage of items to select for the initial solution
x_curr = initial_solution_random(initial_solution_percentage)  #x_curr will hold the current solution
x_best = x_curr[:]           #x_best will hold the best solution
f_curr = evaluate(x_curr)    #f_curr will hold the evaluation of the current soluton
f_best = f_curr[:]
print("Initial solution: ", x_curr)
print("Initial solution weight: ", f_curr[1])
print("Initial solution value: ", f_curr[0])

current_temperature = initial_temperature(x_curr)  #set the current temperature
initial_temp = current_temperature # store the initial temperature
print("Initial temperature: ", current_temperature)

# Simulated Annealing parameters
cooling_rate = 0.95         # Cooling rate
min_temp = 0.01              # Minimum temperature (stopping criterion)
max_iterations = 1000       # Maximum number of iterations
k = 0               # Iteration counter
Mk = 50                     # number of neighbors to check in each temperature

#begin local search overall logic ----------------
done = 0

while done == 0:

    m = 0
    print("iteration: ", k)
    print("current temperature: ", current_temperature)
    print("Best value so far: ", f_best[0])
    print("Mk: ", int(10+5*initial_temp/current_temperature))
    Neighborhood = neighborhood(x_curr)   #create a list of all neighbors in the neighborhood of x_curr

    # while m < int(10+1000/current_temperature):
    while m < Mk:
        randIndex = myPRNG.randint(0, len(Neighborhood)-1)  #select a random index from the neighborhood
        s = Neighborhood[randIndex][:]
        if evaluate(s)[0] > f_curr[0]:  #if the randomly selected member is better than the current solution
            x_curr = s[:]                 #move to the randomly selected member
            f_curr = evaluate(s)[:]       #evaluate the new current solution
            # print("Found a better solution than current, moving: ", f_curr[0])
            if f_curr[0] > f_best[0]:
                # print("Found a better solution: ", s)
                x_best = x_curr[:]             #update the best solution
                f_best = evaluate(x_curr)[:]
        else:     # now we try to use a deteriorative solution
            delta = evaluate(s)[0] - f_curr[0]
            epsilon = myPRNG.uniform(0, 1)
            # if the solution is worse than the current solution, we will accept it with a certain probability
            if epsilon < acceptance_probability(delta, current_temperature):
                x_curr = s[:]
                f_curr = evaluate(s)[:]
        m += 1
        solutionsChecked += 1

    k += 1
    # current_temperature = cooling_schedule_cauchy(current_temperature, initial_temp, k)  #cool the system
    current_temperature = cooling_schedule_boltzmann(current_temperature, initial_temp, k)  #cool the system
    if current_temperature < min_temp or k > max_iterations:
        done = 1

print("\nInitial temperature: ", initial_temp)
print ("\nFinal number of solutions checked: ", solutionsChecked)
print ("Best value found: ", f_best[0])
print ("Weight is: ", f_best[1])
print ("Total number of items selected: ", np.sum(x_best))
print ("Best solution: ", x_best)
