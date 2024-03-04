# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 2022

@author: tdrumond & agademer

Template file for your Exercise 3 submission 
(generic genetic algorithm module)
"""
import random 

class Individual:
    """Represents an Individual for a genetic algorithm"""

    def __init__(self, chromosome: list, fitness: float):
        """Initializes an Individual for a genetic algorithm

        Args:
            chromosome (list[]): a list representing the individual's
            chromosome
            fitness (float): the individual's fitness (the higher the value,
            the better the fitness)
        """
        self.chromosome = chromosome
        self.fitness = fitness

    def __lt__(self, other):
        """Implementation of the less_than comparator operator"""
        return self.fitness < other.fitness

    def __repr__(self):
        """Representation of the object for print calls"""
        return f'Indiv({self.fitness:.1f},{self.chromosome})'


class GAProblem:
    """Defines a Genetic algorithm problem to be solved by ga_solver"""  
    def initialize_population(self, pop_size: int) -> list:
        """Initializes and returns a population of Individuals."""
        raise NotImplementedError

    def evaluate_fitness(self, individual: Individual) -> float:
        """Evaluates and returns the fitness of an Individual."""
        raise NotImplementedError

    def select_parents(self, population: list) -> tuple:
        """Selects and returns two parents from the population."""
        raise NotImplementedError

    def crossover(self, parent1: Individual, parent2: Individual) -> Individual:
        """Performs crossover on parents to produce and return offspring."""
        raise NotImplementedError

    def mutate(self, individual: Individual, mutation_rate: float) -> Individual:
        """Mutates the given Individual based on the mutation rate."""
        raise NotImplementedError

    def termination_condition_met(self, best_individual: Individual, generations: int) -> bool:
        """Checks if the termination condition is met."""
        raise NotImplementedError

class GASolver:
    def __init__(self, problem: GAProblem, selection_rate=0.5, mutation_rate=0.1):
        """Initializes an instance of a ga_solver for a given GAProblem

        Args:
            problem (GAProblem): GAProblem to be solved by this ga_solver
            selection_rate (float, optional): Selection rate between 0 and 1.0. Defaults to 0.5.
            mutation_rate (float, optional): mutation_rate between 0 and 1.0. Defaults to 0.1.
        """
        self._problem = problem
        self._selection_rate = selection_rate
        self._mutation_rate = mutation_rate
        self._population = []

    def reset_population(self, pop_size=50):
        """ Initialize the population with pop_size random Individuals """ 
        self._population = self._problem.initialize_population(pop_size)

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
          
        # Fitness assessment for each individual
        for individual in self._population:
            individual.fitness = self._problem.evaluate_fitness(individual)
    
        # Sorting the population in descending order of fitness
        self._population.sort(reverse=True)
        
        # Survivor selection based on selection rate
        survivors_count = int(len(self._population) * self._selection_rate)
        self._population = self._population[:survivors_count]

        # Ensure the population is not too small for parent selection
        if len(self._population) < 2:
            print("La population est trop petite, ajout d'individus aléatoires.")
            while len(self._population) < 2:
                self._population.append(self._problem.create_random_individual())

        # Reproduction (crossover and mutation) to fill the new generation
        new_population = self._population[:]
        while len(new_population) < len(self._population) * 2:
            parent1, parent2 = self._problem.select_parents(self._population)
            child = self._problem.crossover(parent1, parent2)
            child = self._problem.mutate(child, self._mutation_rate)
            child.fitness = self._problem.evaluate_fitness(child)
            new_population.append(child)
        
        
        # Updating the population with the new individuals generated
        self._population = new_population[:len(self._population)]




    def show_generation_summary(self):
        """ Print some debug information on the current state of the population """
        for individual in self._population:
            print(individual)


    def get_best_individual(self):
        """ Return the best Individual of the population """
        if not self._population:  
            print("The population is empty. Impossible to determine the best individual")
            return None
        return max(self._population, key=lambda indiv: indiv.fitness) 
        

    def evolve_until(self, max_nb_of_generations=500, threshold_fitness=None):
        """ Launch the evolve_for_one_generation function until one of the two condition is achieved : 
            - Max nb of generation is achieved
            - The fitness of the best Individual is greater than or equal to
              threshold_fitness
        """  
        for generation in range(max_nb_of_generations):
            self.evolve_for_one_generation()
            best_individual = self.get_best_individual()
            print(f"Generation {generation}, Best Fitness {best_individual.fitness}")
            if self._problem.termination_condition_met(best_individual, generation):
                break