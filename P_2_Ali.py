#basic genetic algorithm Python code provided as base code for the DSA/ISE 5113 course
#author: Charles Nicholson
#date: 4/5/2019

#NOTE: You will need to change various parts of this code.  However, please keep the majority of the code intact (e.g., you may revise existing logic/functions and add new logic/functions, but don't completely rewrite the entire base code!)  
#However, I would like all students to have the same problem instance, therefore please do not change anything relating to:
#   random number generation
#   number of items (should be 150)
#   random problem instance
#   weight limit of the knapsack

#------------------------------------------------------------------------------

#Student name:
#Date: 


#need some python libraries
import copy
import math
from random import Random
import numpy as np


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
    #value.append(round(myPRNG.expovariate(1/500)+1,1))
    value.append(round(myPRNG.triangular(150,2000,500),1))
    
weights = []
for i in range(0,n):
    weights.append(round(myPRNG.triangular(8,300,95),1))
    
#define max weight for the knapsack
maxWeight = 2500


#change anything you like below this line, but keep the gist of the program ------------------------------------

# First, I import the libraries I need
from random import choices, randint, randrange, random
from typing import List, Optional, Callable, Tuple
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt

#monitor the number of solutions evaluated, and define some parmaters
solutionsChecked: int = 0
populationSize: int = 100
Generations: int = 100
crossOverRate: float = 0.8
mutationRate: float = 0.05
mutatuinRate_cancer: float = 0.3
eliteSolutions: int = 10
prop_one: float = 0.2
penalty_factor: float = 2.0
cancer_prop: float = 0.2

# Type aliases
Genome = NDArray[np.int8] # A genome is a 1D array of integers (0 or 1)
Population = NDArray[np.int8] # A population is a 2D array of genomes


# First: I create the greedy solution, this is the solution that I will use as a benchmark, I want to at least get this solution with my GA.

# Greedy initial solution
def initial_solution_greedy() -> NDArray[np.int8]:
    ratios: NDArray[np.float64] = np.array(value) / np.array(weights)
    sorted_indices: NDArray[np.int64] = np.argsort(-ratios)
    x: NDArray[np.int8] = np.zeros(n, dtype=np.int8)
    total_weight: float = 0.0
    
    for idx in sorted_indices:
        if total_weight + weights[idx] <= maxWeight:
            x[idx] = 1
            total_weight += weights[idx]
    return x


#create an continuous valued chromosome 

def createChromosome(n: int, prop_one: float = 0.5) -> Genome:
    return np.random.choice([0, 1], size=n, p=[1-prop_one, prop_one]).astype(np.int8)

#function to compute the weight of chromosome x
def calcWeight(x: Genome) -> float:
    return np.dot(x, weights)


#function to determine how many items have been selected in a particular chromosome x
def itemsSelected(x: Genome) -> int:
    return np.sum(x)  #returns total number of items selected 


#function to evaluate a solution x

def evaluate(x: Genome, penalty_factor: float = 2.0) -> float:
    total_weight: float = calcWeight(x) #compute the weight of the knapsack selection
    total_value: float = np.dot(x, value)  #compute the value of the knapsack selection
    if total_weight > maxWeight:
        excess: float = total_weight - maxWeight
        # Penalize solutions that are overweight
        # More overweight = more negative fitness
        return -(excess * penalty_factor)
    return total_value #return the value of the knapsack selection

#create initial population by calling the "createChromosome" function
def initializePopulation() -> list[tuple[Genome, float]]:
    # Initialize matrices
    population: Population = np.zeros((populationSize, n), dtype=np.int8)
    populationFitness: NDArray[np.float64] = np.zeros(populationSize)
    
    # Generate and evaluate population
    for i in range(populationSize):
        population[i] = createChromosome(n, prop_one)
        populationFitness[i] = evaluate(population[i])
    
    # Sort by fitness (descending)
    sorted_indices: NDArray[np.int64] = np.argsort(-populationFitness)
    
    # Return list of tuples (chromosome, fitness)
    return list(zip(population[sorted_indices], populationFitness[sorted_indices]))



#implement a crossover



def crossover(parent1: Genome, parent2: Genome):
    # Initializing offspring as copies of parents
    offspring1: Genome = parent1.copy()
    offspring2: Genome = parent2.copy()
    # Performing crossover with probability crossOverRate
    if np.random.random() < crossOverRate:
        # Selecting random crossover point
        crossover_point: int = np.random.randint(1, len(parent1)-1)
        # Performing crossover
        offspring1[crossover_point:] = parent2[crossover_point:]
        offspring2[crossover_point:] = parent1[crossover_point:]

    return offspring1, offspring2  #two offspring are returned 





#performs tournament selection; k chromosomes are selected (with repeats allowed) and the best advances to the mating pool
#function returns the mating pool with size equal to the initial population
def tournamentSelection(population: list[tuple[Genome, float]], 
                       k: int) -> Population:
    
    # Initialize mating pool as numpy array
    matingPool: Population = np.zeros((populationSize, n), dtype=np.int8)
    
    # Get all chromosomes and fitness values as separate arrays
    chromosomes: Population = np.array([p[0] for p in population])
    fitness: NDArray[np.float64] = np.array([p[1] for p in population])
    
    for i in range(populationSize):
        # Select k random indices for tournament
        tournament_indices: NDArray[np.int64] = np.random.randint(0, populationSize, k)
        
        # Get tournament fitness values
        tournament_fitness: NDArray[np.float64] = fitness[tournament_indices]
        
        # Select winner
        winner_idx: int = tournament_indices[np.argmax(tournament_fitness)]
        
        # Add winner to mating pool
        matingPool[i] = chromosomes[winner_idx]
    
    return matingPool





def rouletteWheel(population: list[tuple[Genome, float]]) -> Population:
    # Initialize mating pool
    matingPool: Population = np.zeros((populationSize, n), dtype=np.int8)
    
    # Extract chromosomes and fitness values
    chromosomes: Population = np.array([p[0] for p in population])
    fitness: NDArray[np.float64] = np.array([p[1] for p in population])
    
    # Calculate selection probabilities
    total_fitness: float = np.sum(fitness)
    probabilities: NDArray[np.float64] = fitness / total_fitness
    cumulative_probs: NDArray[np.float64] = np.cumsum(probabilities)
    
    # Select parents
    for i in range(populationSize):
        # Generate random number
        r: float = np.random.random()
        # Find first probability greater than r
        selected_idx: int = np.searchsorted(cumulative_probs, r)
        # Add selected chromosome to mating pool
        matingPool[i] = chromosomes[selected_idx]
    
    return matingPool
    

#function to mutate solutions
def mutate(x: Genome) -> Genome:
    # Create mutation mask using random numbers
    mutation_mask: NDArray[np.bool_] = np.random.random(len(x)) < mutationRate
    
    # Create mutated chromosome (XOR with mutation mask)
    mutated: Genome = x ^ mutation_mask.astype(np.int8)
    
    return mutated
            
    

#breeding -- uses the "mating pool" and calls "crossover" function    
def breeding(matingPool: Population) -> list[tuple[Genome, float]]:
    # Initialize numpy arrays for children
    children: Population = np.zeros((populationSize, n), dtype=np.int8)
    childrenFitness: NDArray[np.float64] = np.zeros(populationSize)
    
    # Breed pairs of parents
    for i in range(0, populationSize-1, 2):
        # Crossover
        children[i], children[i+1] = crossover(matingPool[i], matingPool[i+1])
        
        # Mutate
        children[i] = mutate(children[i])
        children[i+1] = mutate(children[i+1])
        
        # Evaluate
        childrenFitness[i] = evaluate(children[i])
        childrenFitness[i+1] = evaluate(children[i+1])
    
    # Sort by fitness
    sorted_indices: NDArray[np.int64] = np.argsort(-childrenFitness)
    
    # Return sorted list of tuples
    return list(zip(children[sorted_indices], childrenFitness[sorted_indices]))


#insertion step
def insert(pop: list[tuple[Genome, float]], 
          kids: list[tuple[Genome, float]]) -> list[tuple[Genome, float]]:
    """Combine population and offspring using elitism"""
    
    # Keep best eliteSolutions from current population
    new_population: list[tuple[Genome, float]] = pop[:eliteSolutions]
    
    # Add best offspring for remaining spots
    new_population.extend(kids[:populationSize - eliteSolutions])
    
    # Sort by fitness (already sorted, but ensuring order after combine)
    return sorted(new_population, key=lambda x: x[1], reverse=True)
    
    
    
def summaryFitness(pop: list[tuple[Genome, float]]) -> tuple[float, float, float, float]:
    """Calculate population fitness statistics"""
    fitness_values: NDArray[np.float64] = np.array([p[1] for p in pop])
    return (np.max(fitness_values), 
            np.mean(fitness_values), 
            np.min(fitness_values),
            np.std(fitness_values))

def bestSolutionInPopulation(pop: list[tuple[Genome, float]]) -> None:
    """Print details of best solution in population"""
    best_chromosome: Genome = pop[0][0]
    print(f"Best solution: {best_chromosome}")
    print(f"Items selected: {itemsSelected(best_chromosome)}")
    print(f"Value: {pop[0][1]:.2f}")
    print(f"Weight: {calcWeight(best_chromosome):.2f}")





def create_performance_plot(best_values: list[float], normal_gens: int) -> None:
    """Create performance plot with cancer phase marker"""
    plt.figure(figsize=(10, 6))
    plt.plot(best_values, 'b-', label='Best Fitness')
    plt.axvline(x=normal_gens, color='r', linestyle='--', label='Cancer Phase Start')
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness')
    plt.title('GA Performance with Cancer Phase')
    plt.legend()
    plt.grid(True)
    plt.show()

def main() -> tuple[list[float], int]:
    """Main GA loop with cancer phase"""
    population: list[tuple[Genome, float]] = initializePopulation()
    best_values: list[float] = []
    cancerProp: float = 0.2
    
    # Normal evolution phase
    for generation in range(Generations):
        mating_pool: Population = tournamentSelection(population, 10)
        offspring: list[tuple[Genome, float]] = breeding(mating_pool)
        population = insert(population, offspring)
        
        maxVal, meanVal, minVal, stdVal = summaryFitness(population)
        best_values.append(maxVal)
        print(f"Generation {generation}: Max={maxVal:.2f}, Mean={meanVal:.2f}")
    
    # Cancer mutation phase
    cancer_generations: int = int(Generations * cancerProp)
    
    for generation in range(cancer_generations):
        mating_pool: Population = tournamentSelection(population, 10)
        offspring: list[tuple[Genome, float]] = breeding(mating_pool)
        
        # Apply cancer mutations
        for i in range(len(offspring)):
            if np.random.random() < mutationRate_cancer:
                offspring[i] = (mutate(offspring[i][0]), evaluate(offspring[i][0]))
        
        population = insert(population, offspring)
        maxVal, _, _, _ = summaryFitness(population)
        best_values.append(maxVal)
        print(f"Cancer Generation {generation}: Max={maxVal:.2f}")
    
    print("\nFinal Solution:")
    bestSolutionInPopulation(population)
    
    return best_values, Generations

if __name__ == "__main__":
    best_vals, gens = main()
    create_performance_plot(best_vals, gens)

