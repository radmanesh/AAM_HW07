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
# plotting the results
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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

# Simulated Annealing Algorithm
# Main function
def simulated_annealing(initial_solution_percentage=0.2, cooling_rate=0.95, min_temp=0.01, max_iterations=1000, Mk=10, cooling_schedule_type='geometric'):
    """
    Main function to run the simulated annealing algorithm.
    """

    print("Simulated Annealing Algorithm with configs : ", initial_solution_percentage, cooling_rate, min_temp, max_iterations, Mk)
    # Initialize variables
    #varaible to record the number of solutions evaluated
    solutionsChecked = 0
    iterations = 0

    # Create initial solution
    x_curr = initial_solution_random(initial_solution_percentage)  #x_curr will hold the current solution
    x_best = x_curr[:]           #x_best will hold the best solution
    f_curr = evaluate(x_curr)    #f_curr will hold the evaluation of the current soluton
    f_best = f_curr[:]


    current_temperature = initial_temperature(x_curr)  #set the current temperature
    init_temperature = current_temperature #store the initial temperature
    print("Initial temperature: ", current_temperature)
    # Simulated Annealing parameters
    print("Cooling rate: ", cooling_rate)
    print("Minimum temperature: ", min_temp)
    print("Maximum iterations: ", max_iterations)
    print("Mk: ", Mk)

    #begin local search overall logic ----------------
    done = 0

    while done == 0:

        m = 0
        # print("iteration: %d , temp: %d , bestVal: %d", iterations, current_temperature, f_best[0])
        Neighborhood = neighborhood(x_curr)   #create a list of all neighbors in the neighborhood of x_curr

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
        iterations += 1
        match cooling_schedule_type:
            case 'geometric':
                current_temperature = cooling_schedule(current_temperature, cooling_rate)
            case 'cauchy':
                current_temperature = cooling_schedule_cauchy(current_temperature, init_temperature, iterations)
            case 'boltzmann':
                current_temperature = cooling_schedule_boltzmann(current_temperature, init_temperature, iterations)
            case 'Ali':
                current_temperature = cooling_schedule_Ali(current_temperature, init_temperature)
        if current_temperature < min_temp or iterations > max_iterations:
            done = 1

    print ("\nFinal number of solutions checked: ", solutionsChecked)
    print ("Best value found: ", f_best[0])
    print ("Weight is: ", f_best[1])
    print ("Total number of items selected: ", np.sum(x_best))
    print ("Best solution: ", x_best)

    return [x_best, f_best, iterations, solutionsChecked, init_temperature]

# run the simulated annealing algorithm with different parameters
# init_solution_range = list(range(1, 30, 1))  # creates [0.01, 0.02, ..., 0.29]
init_solution_range = list(range(5, 30, 10))  # creates [0.01, 0.02, ..., 0.29]

# Different min temperatures from 0.01 to 1
min_temps = [0.01, 0.1, 0.5]

# Different max iterations from 100 to 1000
max_iterations = list(range(100, 1100, 200))  # creates [100, 300, 500, 700, 900]

# Different Mk values from 10 to 50
Mk_values = list(range(5, 105, 5))  # creates [10, 20, 30, 40, 50]

# Cooling rates
cooling_rates = [0.95, 0.99, 0.999]  # list(range(0.9, 1.1, 0.01))  # creates [0.9, 0.91, ..., 1.09]

#create a DataFrame to store the results
results_df = pd.DataFrame(columns=["Initial Solution Percentage", "Cooling Rate", "Min Temp", "Max Iterations", "Mk", "Best Value", "Weight", "Total Items Selected"])
# Run the simulated annealing algorithm with different parameters
for cooling_type in ['geometric', 'cauchy', 'boltzmann']:
    for init_sol_perc in list(range(1, 30, 1)):
        # Run simulated annealing to get results
        results = simulated_annealing(
            initial_solution_percentage=init_sol_perc/100,
            cooling_rate=0.95,
            min_temp=min_temps[0],
            max_iterations=1000,
            Mk=50,
            cooling_schedule_type=cooling_type
        )

        # Create a single-row DataFrame with the new results
        new_row = pd.DataFrame({
            "Initial Solution Percentage": [init_sol_perc],  # Use list to create proper Series
            "Cooling Rate": [0.95],  # Each value needs to be in a list
            "Min Temp": [min_temps[0]],
            "Max Iterations": [1000],
            "Mk": [50],
            "Best Value": [results[1][0]],
            "Weight": [results[1][1]],
            "Total Items Selected": [np.sum(results[0])],
            "Cooling Type": [cooling_type]
        })

        # Use pd.concat to append the new row
        results_df = pd.concat([results_df, new_row], ignore_index=True)

# Plotting the results
# Plot initial solution percentage vs best value for different cooling types for geometric cooling
# filter the DataFrame for geometric cooling
geometric_df = results_df[results_df["Cooling Type"] == "geometric"]
boltzmann_df = results_df[results_df["Cooling Type"] == "boltzmann"]
cauchy_df = results_df[results_df["Cooling Type"] == "cauchy"]
plt.figure(figsize=(12, 6))
sns.lineplot(data=geometric_df, x="Initial Solution Percentage", y="Best Value",  style="Cooling Type", markers=True, dashes=False)
sns.lineplot(data=boltzmann_df, x="Initial Solution Percentage", y="Best Value",  style="Cooling Type", markers=True, dashes=False)
sns.lineplot(data=cauchy_df, x="Initial Solution Percentage", y="Best Value",  style="Cooling Type", markers=True, dashes=False)
plt.title("Best Value vs Initial Solution Percentage")
plt.xlabel("Initial Solution Percentage")
plt.ylabel("Best Value")
plt.legend(title="Cooling Type")
plt.grid()
plt.show()

for cooling_rate in range(50, 101, 2):
    # Run simulated annealing to get results
    results = simulated_annealing(
        initial_solution_percentage=20/100,
        cooling_rate=cooling_rate/100,
        min_temp=min_temps[0],
        max_iterations=max_iterations[0],
        Mk=50,
        cooling_schedule_type='geometric'
    )

    # Create a single-row DataFrame with the new results
    new_row = pd.DataFrame({
        "Initial Solution Percentage": [0.2],  # Use list to create proper Series
        "Cooling Rate": [cooling_rate/100],  # Each value needs to be in a list
        "Min Temp": [min_temps[0]],
        "Max Iterations": [max_iterations[0]],
        "Mk": [50],
        "Best Value": [results[1][0]],
        "Weight": [results[1][1]],
        "Total Items Selected": [np.sum(results[0])],
        "Cooling Type": ['geometric']
    })

    # Use pd.concat to append the new row
    results_df = pd.concat([results_df, new_row], ignore_index=True)

# Plotting the results
plt.figure(figsize=(12, 6))
sns.lineplot(data=results_df, x="Cooling Rate", y="Best Value", style="Mk", markers=True, dashes=False)
plt.title("Best Value vs Cooling Rate")
plt.xlabel("Cooling Rate")
plt.ylabel("Best Value")
plt.legend(title="Initial Solution Percentage and Mk")
plt.grid()
plt.show()

for cooling_type in ['geometric', 'cauchy', 'boltzmann']:
  for mk in Mk_values:
      # Run simulated annealing to get results
      results = simulated_annealing(
          initial_solution_percentage=10/100,
          cooling_rate=0.95,
          min_temp=min_temps[0],
          max_iterations=1000,
          Mk=mk,
          cooling_schedule_type=cooling_type
      )

      # Create a single-row DataFrame with the new results
      new_row = pd.DataFrame({
          "Initial Solution Percentage": [0.1],  # Use list to create proper Series
          "Cooling Rate": [0.95],  # Each value needs to be in a list
          "Min Temp": [min_temps[0]],
          "Max Iterations": [1000],
          "Mk": [mk],
          "Best Value": [results[1][0]],
          "Weight": [results[1][1]],
          "Total Items Selected": [np.sum(results[0])],
          "Cooling Type": [cooling_type]
      })

      # Use pd.concat to append the new row
      results_df = pd.concat([results_df, new_row], ignore_index=True)

geometric_df = results_df[results_df["Cooling Type"] == "geometric"]
boltzmann_df = results_df[results_df["Cooling Type"] == "boltzmann"]
cauchy_df = results_df[results_df["Cooling Type"] == "cauchy"]
plt.figure(figsize=(12, 6))
sns.lineplot(data=geometric_df, x="Mk", y="Best Value",  style="Cooling Type", markers=True, dashes=False)
sns.lineplot(data=boltzmann_df, x="Mk", y="Best Value",  style="Cooling Type", markers=True, dashes=False)
sns.lineplot(data=cauchy_df, x="Mk", y="Best Value",  style="Cooling Type", markers=True, dashes=False)
plt.title("Best Value vs Mk")
plt.xlabel("Mk")
plt.ylabel("Best Value")
plt.legend(title="Cool Schedule Type")
plt.grid()
plt.show()

print(simulated_annealing(0.1,0.95,0.01,1000,50,'boltzmann'))

exit()

#-----------------------------------------------------------
# Set the style of seaborn
sns.set_theme(style="whitegrid")
# Create a DataFrame to store the results
results_df = pd.DataFrame(columns=["Initial Solution Percentage", "Cooling Rate", "Min Temp", "Max Iterations", "Mk", "Best Value", "Weight", "Total Items Selected"])
# Append the results to the DataFrame
for cooling_type in ['geometric', 'cauchy', 'boltzmann']:
  for init_sol_perc in init_solution_range:
      for min_temp in min_temps:
          for max_iter in max_iterations:
              for mk in Mk_values:
                for cooling_rate in cooling_rates:
                    results = simulated_annealing(
                        initial_solution_percentage=init_sol_perc/100,
                        cooling_rate=cooling_rate,
                        min_temp=min_temp,
                        max_iterations=max_iter,
                        Mk=mk,
                        cooling_schedule_type=cooling_type
                    )
                    # Append the results to the DataFrame
                    new_row = pd.DataFrame({
                        "Initial Solution Percentage": [init_sol_perc],
                        "Cooling Rate": [cooling_rate],
                        "Min Temp": [min_temp],
                        "Max Iterations": [max_iter],
                        "Mk": [mk],
                        "Best Value": [results[1][0]],
                        "Weight": [results[1][1]],
                        "Total Items Selected": [np.sum(results[0])],
                        "Cooling Type": [cooling_type]
                    })
                    results_df = pd.concat([results_df, new_row], ignore_index=True)
# Plotting the results
plt.figure(figsize=(12, 6))
sns.lineplot(data=results_df, x="Initial Solution Percentage", y="Best Value", hue="Cooling Rate", style="Mk", markers=True, dashes=False)
plt.title("Best Value vs Initial Solution Percentage")
plt.xlabel("Initial Solution Percentage")
plt.ylabel("Best Value")
plt.legend(title="Cooling Rate and Mk")
plt.grid()
plt.show()
# Plotting the results
plt.figure(figsize=(12, 6))
sns.lineplot(data=results_df, x="Cooling Rate", y="Best Value", hue="Initial Solution Percentage", style="Mk", markers=True, dashes=False)
plt.title("Best Value vs Cooling Rate")
plt.xlabel("Cooling Rate")
plt.ylabel("Best Value")
plt.legend(title="Initial Solution Percentage and Mk")
plt.grid()
plt.show()
# Plotting the results
plt.figure(figsize=(12, 6))
sns.lineplot(data=results_df, x="Min Temp", y="Best Value", hue="Initial Solution Percentage", style="Mk", markers=True, dashes=False)
plt.title("Best Value vs Min Temp")
plt.xlabel("Min Temp")
plt.ylabel("Best Value")


plt.legend(title="Initial Solution Percentage and Mk")
plt.grid()

plt.show()
# Plotting the results


plt.figure(figsize=(12, 6))
sns.lineplot(data=results_df, x="Max Iterations", y="Best Value", hue="Initial Solution Percentage", style="Mk", markers=True, dashes=False)
plt.title("Best Value vs Max Iterations")
plt.xlabel("Max Iterations")
plt.ylabel("Best Value")
plt.legend(title="Initial Solution Percentage and Mk")
plt.grid()
plt.show()
# Plotting the results
plt.figure(figsize=(12, 6))
sns.lineplot(data=results_df, x="Mk", y="Best Value", hue="Initial Solution Percentage", style="Cooling Rate", markers=True, dashes=False)
plt.title("Best Value vs Mk")
plt.xlabel("Mk")
plt.ylabel("Best Value")
plt.legend(title="Initial Solution Percentage and Cooling Rate")
plt.grid()
plt.show()

# plotting different cooling types
plt.figure(figsize=(12, 6))
sns.lineplot(data=results_df, x="Cooling Type", y="Best Value", hue="Initial Solution Percentage", style="Mk", markers=True, dashes=False)
plt.title("Best Value vs Cooling Type")
plt.xlabel("Cooling Type")
plt.ylabel("Best Value")
plt.legend(title="Initial Solution Percentage and Mk")
plt.grid()
plt.show()


# Save the results to a CSV file
results_df.to_csv("simulated_annealing_results.csv", index=False)


