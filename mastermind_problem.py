# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 2022

@author: tdrumond & agademer

Template file for your Exercise 3 submission 
(GA solving Mastermind example)
"""
from ga_solver import GAProblem
from ga_solver import Individual
import mastermind as mm
import random 

class MastermindProblem(GAProblem):
    """Implementation of GAProblem for the mastermind problem""" 
    def __init__(self, match):
        self.match = match

    def create_random_individual(self):
        """Crée et retourne un nouvel individu avec un chromosome généré aléatoirement."""
        chromosome = [random.choice(mm.get_possible_colors()) for _ in range(self.match.secret_size())]
        # Créer un nouvel individu avec ce chromosome. Initialement, on peut mettre la fitness à 0 ou à une valeur par défaut.
        new_individual = Individual(chromosome, 0)  # Initialiser la fitness à 0 ou à une autre valeur par défaut
        
        # Calculer la fitness de l'individu et mettre à jour sa valeur de fitness
        new_individual.fitness = self.evaluate_fitness(new_individual)
        
        return new_individual

    def initialize_population(self, pop_size: int):
        population = []
        for _ in range(pop_size):
            # Generation of a chromosome with random colors
            chromosome = [random.choice(mm.get_possible_colors()) for _ in range(self.match.secret_size())]
            individual = Individual(chromosome, 0)  # Initial fitness is set to 0
            population.append(individual)
        return population

    def evaluate_fitness(self, individual: Individual):
        score = self.match.rate_guess(individual.chromosome)
        return score

    def select_parents(self, population: list):
        if len(population) < 2:
            raise Exception("The population is not large enough for parent selection.")
        return random.sample(population, 2)

    def crossover(self, parent1: Individual, parent2: Individual):
        # Single-point crossover
        point = random.randint(1, len(parent1.chromosome) - 2)
        child_chromosome = parent1.chromosome[:point] + parent2.chromosome[point:]
        return Individual(child_chromosome, 0)

    def mutate(self, individual: Individual, mutation_rate: float):
        if random.random() < mutation_rate:
            index = random.randrange(len(individual.chromosome))
            individual.chromosome[index] = random.choice(mm.get_possible_colors())
        return individual

    def termination_condition_met(self, best_individual: Individual, generations: int):
        # Terminate if the correct solution is found or a maximum number of generations is reached
        return self.match.is_correct(best_individual.chromosome)

if __name__ == '__main__':
    from ga_solver import GASolver

    match = mm.MastermindMatch(secret_size=6)
    problem = MastermindProblem(match)
    solver = GASolver(problem)

    solver.reset_population()
    solver.evolve_until()

    best_individual = solver.get_best_individual()
    best_guess_colors = best_individual.chromosome 
    if best_individual:
        print(f"Best guess: {best_guess_colors} Fitness: {best_individual.fitness}")
        print(f"Problem solved? {match.is_correct(best_individual.chromosome)}")
    else:
        print("No better individual has been found.")
