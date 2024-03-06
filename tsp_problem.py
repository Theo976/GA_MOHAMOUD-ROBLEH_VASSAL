# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 2022

@author: tdrumond & agademer

Template file for your Exercise 3 submission 
(GA solving TSP example)
"""
from ga_solver import GAProblem, GASolver
import cities
import random


class TSProblem(GAProblem):
    """Implementation of GAProblem for the traveling salesperson problem"""

    def create_chromosome(self):
        """
        Creation of a chromosome, composed of 4 random colors
        takes no argument
        returns a chromosome
        """
        default = cities.default_road(city_dict) #list the cities
        chromosome = random.sample(default, len(default)) #random list with  the cities
        return chromosome #return a chromosome
    
    def create_fitness(self, chromosome):
        """
        Creation of a fitness, using the function 'rate_guess' imported from mastermind.py
        takes a chromosome as an argument
        returns a fitness 
        """
        fitness = - cities.road_length(city_dict, chromosome) #calculation of the fitness of the chromosome generated above
        return fitness #return a fitness value


    def reproduction(self, population, parent_a_index, parent_b_index):
        """
        Reproduction step :
        takes a population and two integers as arguments
        returns a chromosome
        
        Steps are : 
            1. Take the first half of the road from the first parent.
            2. Then, take the second half from the second parent, skipping any cities already
            present in the child.
            3. If the chromosome is not long enough after step 2, complete it with any
            remaining cities that are not yet present in the child’s chromosome.
        """
        
        x_point = int(len(population[0].chromosome)//2) #half the length of the population
        #x_point = random.randrange(0, len(population[0].chromosome)) 
        new_chrom = population[parent_a_index].chromosome[0:x_point] #the new chromosome is composed of the first half of the parent A's chromosome
        for city in population[parent_b_index].chromosome[x_point:] : #and the genes of the second half of the parent B's chromosome
            if city not in new_chrom : #that are not in the new chromosome
                new_chrom.append(city) 
            
        if len(new_chrom) != 2*x_point : #if not all the possible cities are in the new chromosome
            possible_cities = cities.default_road(city_dict) #list of all the possible cities 
            for city in possible_cities : #browse to the list "possible_cities"
                if city not in new_chrom : #if a city is not already in the new chromosome :
                    new_chrom.append(city) #it is added to the new chromosome
        return new_chrom #return a chromosome
        
        
    def mutation(self, population, Index):
        """
        Mutation step :
        takes a population and an integer as arguments
        returns a chromosome
        
        Steps are :
             - Takes two random integers as the positions of the genes
             - Swap the genes at these positions
        """
        pos1 = random.randrange(0,len(population[0].chromosome)) #random position that will be switched with the other one
        pos2 = random.randrange(0,len(population[0].chromosome)) #random position that will be switched with the other one
                
        #switch the selected positions
        change = population[Index].chromosome[pos1]
        population[Index].chromosome[pos1] = population[Index].chromosome[pos2]
        population[Index].chromosome[pos2] = change
        return population[Index].chromosome
 


if __name__ == '__main__':

    from ga_solver import GASolver

    city_dict = cities.load_cities("/Users/leo/Desktop/test tsp/cities.txt")
    problem = TSProblem()
    solver = GASolver(problem)
    solver.reset_population()
    solver.evolve_until()

    print("best road = ", (solver.get_best_individual()))
    cities.draw_cities(city_dict, (solver.get_best_individual()).chromosome)

