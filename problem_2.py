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

#Student name: Arman Radmanesh, Ali Abdullah
#Date: 04-19-2025


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

#monitor the number of solutions evaluated
solutionsChecked = 0


populationSize = 150 #size of GA population
Generations = 100   #number of GA generations

crossOverRate = 0.8  #currently not used in the implementation; neeeds to be used.
mutationRate = 0.05  #currently not used in the implementation; neeeds to be used.
eliteSolutions = 10  #currently not used in the implementation; neeed to use some type of elitism


#create an continuous valued chromosome
def createChromosome(d):
    #this code as-is expects chromosomes to be stored as a list, e.g., x = []
    #write code to generate chromosomes, most likely want this to be randomly generated

    x = [0]*d   #i recommend creating the solution as a list
    percentage = 0.15
    totalWeight = 0
    size = int(d*percentage)  #size of the chromosome to be randomly generated
    for i in range(size):
        idx = myPRNG.randint(0,d-1)  #randomly select an index in the chromosome
        if x[idx] == 0:  #if the index is not already selected, then select it
            x[idx] = 1
            totalWeight = totalWeight + weights[idx]  #add the weight of the item to the total weight
        else:  #if the index is already selected, then randomly select another index
            i = i - 1
            continue
        if totalWeight > maxWeight: #if the total weight exceeds the max weight, then randomly select another index
            break
    return x


#create initial population by calling the "createChromosome" function many times and adding each to a list of chromosomes (a.k.a., the "population")
def initializePopulation(): #n is size of population; d is dimensions of chromosome

    population = []
    populationFitness = []

    for i in range(populationSize):
        population.append(createChromosome(n))
        populationFitness.append(evaluate(population[i]))

    tempZip = zip(population, populationFitness)
    popVals = sorted(tempZip, key=lambda tempZip: tempZip[1], reverse = True)

    #the return object is a reversed sorted list of tuples:
    #the first element of the tuple is the chromosome; the second element is the fitness value
    #for example:  popVals[0] is represents the best individual in the population
    #popVals[0] for a 2D problem might be  ([-70.2, 426.1], 483.3)  -- chromosome is the list [-70.2, 426.1] and the fitness is 483.3

    return popVals

#implement a crossover
def crossover(x1,x2):

    #with some probability (i.e., crossoverRate) perform breeding via crossover,
    #i.e. two parents (x1 and x2) should produce two offsrping (offspring1 and offspring2)
    # --- the first part of offspring1 comes from x1, and the second part of offspring1 comes from x2
    # --- the first part of offspring2 comes from x2, and the second part of offspring2 comes from x1

    #if no breeding occurs, then offspring1 and offspring2 can simply be copies of x1 and x2, respectively

    p = myPRNG.random()  #random number between 0 and 1
    if p < crossOverRate:  #if the random number is less than the crossover rate, then perform crossover
        idx = myPRNG.randint(0,n-1)  #randomly select an index in the chromosome
        offspring1 = x1[:idx] + x2[idx:]
        offspring2 = x2[:idx] + x1[idx:]
    else:  #if the random number is greater than the crossover rate, then do not perform crossover
        offspring1 = x1
        offspring2 = x2

    return offspring1, offspring2  #two offspring are returned


#function to compute the weight of chromosome x
def calcWeight(x):

    a=np.array(x)
    c=np.array(weights)

    totalWeight = np.dot(a,c)    #compute the weight value of the knapsack selection

    return totalWeight   #returns total weight


#function to determine how many items have been selected in a particular chromosome x
def itemsSelected(x):

    a=np.array(x)
    return np.sum(a)   #returns total number of items selected



#function to evaluate a solution x
def evaluate(x):

    a=np.array(x)
    b=np.array(value)
    c=np.array(weights)
    totalWeight = np.dot(a,c)    #compute the weight value of the knapsack selection

    totalValue = np.dot(a,b)     #compute the value of the knapsack selection

    #you will VERY LIKELY need to add some penalties or sometype of modification of the totalvalue to compute the chromosome fitness
    #for instance, you may include penalties if the knapsack weight exceeds the maximum allowed weight
    if( totalWeight > maxWeight):
        totalValue = maxWeight - totalWeight
    fitness  = totalValue

    return fitness   #returns the chromosome fitness




#performs tournament selection; k chromosomes are selected (with repeats allowed) and the best advances to the mating pool
#function returns the mating pool with size equal to the initial population
def tournamentSelection(pop,k):

    #randomly select k chromosomes; the best joins the mating pool
    matingPool = []

    while len(matingPool)<populationSize:

        ids = [myPRNG.randint(0,populationSize-1) for i in range(k)]
        competingIndividuals = [pop[i][1] for i in ids]
        bestID=ids[competingIndividuals.index(max(competingIndividuals))]
        matingPool.append(pop[bestID][0])

    return matingPool


def rouletteWheel(pop):


    #create sometype of rouletteWheel selection -- can be based on fitness function or fitness rank
    #(remember the population is always ordered from most fit to least fit, so pop[0] is the fittest chromosome in the population, and pop[populationSize-1] is the least fit!
    # Generated by Copilot
    # Function to implement rouletteWheel selection based on fitness values
    matingPool = []
    # Calculate total fitness of the population
    totalFitness = sum(ind[1] for ind in pop)

    # Handle case where all solutions have negative fitness
    if totalFitness <= 0:
      # Use rank-based selection instead
      ranks = list(range(1, populationSize + 1))
      ranks.reverse()  # Higher ranks for better solutions
      totalRank = sum(ranks)

      # Select individuals based on rank probability
      while len(matingPool) < populationSize:
        # Generate random number between 0 and total rank
        pick = myPRNG.uniform(0, totalRank)
        current = 0
        for i in range(populationSize):
          current += ranks[i]
          if current > pick:
            matingPool.append(pop[i][0])
            break
    else:
      # Traditional roulette wheel based on fitness
      while len(matingPool) < populationSize:
        # Generate random number between 0 and total fitness
        pick = myPRNG.uniform(0, totalFitness)
        current = 0
        for i in range(populationSize):
          current += pop[i][1]
          if current > pick:
            matingPool.append(pop[i][0])
            break

    return matingPool


#function to mutate solutions
def mutate(x):

    #create some mutation logic  -- make sure to incorporate "mutationRate" somewhere and dont' do TOO much mutation
    if myPRNG.random() < mutationRate:
        idx = myPRNG.randint(0,n-1)
        if x[idx] == 0:
            x[idx] = 1
        else:
            x[idx] = 0
    return x




#breeding -- uses the "mating pool" and calls "crossover" function
def breeding(matingPool):
    #the parents will be the first two individuals, then next two, then next two and so on

    children = []
    childrenFitness = []
    for i in range(0,populationSize-1,2):
        child1,child2=crossover(matingPool[i],matingPool[i+1])

        child1=mutate(child1)
        child2=mutate(child2)

        children.append(child1)
        children.append(child2)

        childrenFitness.append(evaluate(child1))
        childrenFitness.append(evaluate(child2))

    tempZip = zip(children, childrenFitness)
    popVals = sorted(tempZip, key=lambda tempZip: tempZip[1], reverse = True)

    #the return object is a sorted list of tuples:
    #the first element of the tuple is the chromosome; the second element is the fitness value
    #for example:  popVals[0] is represents the best individual in the population
    #popVals[0] for a 2D problem might be  ([-70.2, 426.1], 483.3)  -- chromosome is the list [-70.2, 426.1] and the fitness is 483.3

    return popVals

kBestToKeep = 10  #number of best solutions to keep from the previous generation at most
#insertion step
def insert(pop,kids):

    #this is not a good solution here... essentially this is replacing the previous generation with the offspring and not implementing any type of elitism
    #at the VERY LEAST evaluate the best solution from "pop" to make sure you are not losing a very good chromosome from last generation
    #maybe want to keep the top 5? 10? solutions from pop -- it's up to you.

    # Check the k best solutions from the previous generation and keep them in the new generation if they are better than the new ones
    for i in range(kBestToKeep):
        if pop[i][1] > kids[i][1]:
            kids[i] = pop[i]

    return kids



#perform a simple summary on the population: returns the best chromosome fitness, the average population fitness, and the variance of the population fitness
def summaryFitness(pop):
    a=np.array(list(zip(*pop))[1])
    return np.max(a), np.mean(a), np.min(a), np.std(a)


#the best solution should always be the first element...
def bestSolutionInPopulation(pop):
    print ("Best solution: ", pop[0][0])
    print ("Items selected: ", itemsSelected(pop[0][0]))
    print ("Value: ", pop[0][1])
    print ("Weight: ", calcWeight(pop[0][0]))



def main():
    #GA main code
    Population = initializePopulation()


    #optional: you can output results to a file -- i've commented out all of the file out put for now
    #f = open('out.txt', 'w')  #---uncomment this line to create a file for saving output


    for j in range(Generations):

        mates=tournamentSelection(Population,10)  #<--need to replace this with roulette wheel selection, e.g.:  mates=rouletteWheel(Population)
        Offspring = breeding(mates)
        Population = insert(Population, Offspring)

        #end of GA main code

        maxVal, meanVal, minVal, stdVal=summaryFitness(Population)          #check out the population at each generation
        print("Iteration: ", j, summaryFitness(Population))                 #print to screen; turn this off for faster results

        #f.write(str(minVal) + " " + str(meanVal) + " " + str(varVal) + "\n")  #---uncomment this line to write to  file

    #f.close()   #---uncomment this line to close the file for saving output

    print (summaryFitness(Population))
    bestSolutionInPopulation(Population)


if __name__ == "__main__":
    main()



