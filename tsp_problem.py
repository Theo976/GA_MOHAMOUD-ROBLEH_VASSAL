# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 2022

@author: tdrumond & agademer

Template file for your Exercise 3 submission 
(GA solving TSP example)
"""
from ga_solver import GAProblem
import cities

from ga_solver import GAProblem, Individual
import cities
import random
city_dict = cities.load_cities("/Users/theo/Documents/EPF/4A /Professional programming /genetic tp/genetic_part3/cities.txt")

class TSProblem(GAProblem):
    def __init__(self, city_dict):
        self.city_dict = city_dict

    def initialize_population(self, pop_size):
        population = []
        city_names = list(self.city_dict.keys())
        for _ in range(pop_size):
            random.shuffle(city_names)
            chromosome = city_names[:]
            individual = Individual(chromosome, 0)  # Initial fitness set to 0
            population.append(individual)
        return population

    def evaluate_fitness(self, individual):
        # Fitness is the negative of the road length (since we want to minimize it)
        road_length = cities.road_length(self.city_dict, individual.chromosome)
        return -road_length

    def select_parents(self, population):
        # Simple random selection
        return random.sample(population, 2)

    def crossover(self, parent1, parent2):
        # Implement a simple crossover strategy
        crossover_point = random.randint(1, len(parent1.chromosome) - 2)
        child_chromosome = parent1.chromosome[:crossover_point] + parent2.chromosome[crossover_point:]
        return Individual(child_chromosome, 0)  # Initial fitness set to 0

    def mutate(self, individual, mutation_rate):
        if random.random() < mutation_rate:
            # Swap two cities in the chromosome for mutation
            i, j = random.sample(range(len(individual.chromosome)), 2)
            individual.chromosome[i], individual.chromosome[j] = individual.chromosome[j], individual.chromosome[i]
        return individual

    def termination_condition_met(self, best_individual, generations):
        # Terminate after a fixed number of generations or if a satisfactory solution is found
        return generations >= 500 or best_individual.fitness > -300  # Example condition



if __name__ == '__main__':
    city_dict = cities.load_cities("cities.txt")
    tsp_problem = TSProblem(city_dict)
    solver = GASolver(tsp_problem, selection_rate=0.5, mutation_rate=0.1)
    
    solver.reset_population(pop_size=100)
    solver.evolve_until(max_nb_of_generations=500, threshold_fitness=-300)
    
    best_individual = solver.get_best_individual()
    print(f"Best path: {best_individual.chromosome} with fitness: {best_individual.fitness}")
    cities.draw_cities(city_dict, best_individual.chromosome)

