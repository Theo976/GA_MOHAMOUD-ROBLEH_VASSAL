# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 11:24:15 2022

@author: agademer & tdrumond

Template for exercise 1
(genetic algorithm module specification)
"""
import mastermind as mm
import random

MATCH = mm.MastermindMatch(secret_size=4)


class Individual:
    """Represents an Individual for a genetic algorithm"""

    def __init__(self, chromosome: list, fitness: float):
        """Initializes an Individual for a genetic algorithm 

        Args:
            chromosome (list[]): a list representing the individual's chromosome
            fitness (float): the individual's fitness (the higher, the better the fitness)
        """
        self.chromosome = chromosome
        self.fitness = fitness

    def __lt__(self, other):
        """Implementation of the less_than comparator operator"""
        return self.fitness < other.fitness

    def __repr__(self):
        """Representation of the object for print calls"""
        return f'Indiv({self.fitness:.1f},{self.chromosome})'


class GASolver:
    def __init__(self, selection_rate=0.5, mutation_rate=0.1):
        """Initializes an instance of a ga_solver for a given GAProblem

        Args:
            selection_rate (float, optional): Selection rate between 0 and 1.0. Defaults to 0.5.
            mutation_rate (float, optional): mutation_rate between 0 and 1.0. Defaults to 0.1.
        """
        self._selection_rate = selection_rate
        self._mutation_rate = mutation_rate
        self._population = []

    def reset_population(self, pop_size=50):
        """ Initialize the population with pop_size random Individuals """
        
        # Loop to create several random individuals and append them to the population list
        for _ in range (pop_size):
            chromosome = MATCH.generate_random_guess()
            fitness =  MATCH.rate_guess(chromosome)
            new_individual = Individual(chromosome, fitness)
            self._population.append(new_individual)


    def evolve_for_one_generation(self):
        """ Apply the process for one generation : 
            -	Sort the population (Descending order)
            -	Selection: Remove x% of population (less adapted)
            -   Reproduction: Recreate the same quantity by crossing the 
                surviving ones 
            -	Mutation: For each new Individual, mutate with probability 
                mutation_rate i.e., mutate it if a random value is below   
                mutation_rate
        """
        self._population.sort(reverse=True) 
        keep_count = int(len(self._population) * self._selection_rate)     #Keep the top x% of individuals with the highest fitness
        survivors = self._population[:keep_count]   
        while len(self._population) < len(survivors):   # Generate new individuals through crossover of two parents
            parent1, parent2 = random.sample(survivors, 2)
            crossover_point = random.randrange(0, len(parent1.chromosome))   # Take a random crossing point
            new_chromosome = parent1.chromosome[0:crossover_point] + parent2.chromosome[crossover_point:]   # Create a new chromosome by combining parent chromosomes
            if random.random() < self._mutation_rate:   # Mutate the new chromosome with a probability of mutation_rate
                mutation_index = random.randrange(0, len(new_chromosome))
                valid_colors = mm._colors()
                new_gene = random.choice(valid_colors)
                new_chromosome = new_chromosome[0:mutation_index] + [new_gene] + new_chromosome[mutation_index+1:]
            new_fitness = MATCH.rate_guess(new_chromosome)   # Evaluate the fitness of the new chromosome
            self._population.append(Individual(new_chromosome, new_fitness))

 
    def show_generation_summary(self):
        """ Print some debug information on the current state of the population """
        for individual in self._population:
            print(individual)
        


    def get_best_individual(self):
        """ Return the best Individual of the population """
        return max(self._population, key=lambda indiv: indiv.fitness)

    def evolve_until(self, max_nb_of_generations=500, threshold_fitness=None):
        """ Launch the evolve_for_one_generation function until one of the two condition is achieved : 
            - Max nb of generation is achieved
            - The fitness of the best Individual is greater than or equal to
              threshold_fitness
        """
        for _ in range(max_nb_of_generations):  # Iterate for a maximum number of generations
            self.evolve_for_one_generation()
            if threshold_fitness and self.get_best_individual().fitness >= threshold_fitness:
                break


solver = GASolver()
solver.reset_population()
solver.evolve_until(threshold_fitness=MATCH.max_score())

best = solver.get_best_individual()     # Get the best individual after evolution
print(f"Best guess {best.chromosome}")
print(f"Problem solved? {MATCH.is_correct(best.chromosome)}")
