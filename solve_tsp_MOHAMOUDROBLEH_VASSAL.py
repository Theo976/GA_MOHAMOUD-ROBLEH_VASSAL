# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 11:24:15 2022

@author: agademer & tdrumond

Template for exercise 1
(genetic algorithm module specification)
"""
import matplotlib.pyplot as plt
import random
from typing import List, Dict, Tuple, Optional
from collections.abc import Iterable, Mapping

from cities import default_road, road_length

Coordinates = Tuple[int, int]


def load_cities(filename) -> Dict[str,Coordinates]:
    """ load the cities list from a text file """
    with open(filename) as file:
        nbCities = int(file.readline())
        cities = {}
        for _ in range(nbCities):
            city_name, x, y = file.readline().split(";")
            cities[city_name]=(int(x), int(y))
        return cities


def default_road(cities:Dict) -> List:
    """ Default road: all the cities in the order of the text file """
    return list(cities.keys())


def draw_cities(cities:Dict, road=Optional[Iterable[str]]):
    """ Plot the cities and the trajectory """
    x_cords, y_coords = tuple(zip(*cities.values()))
    plt.figure()
    plt.scatter(x_cords, y_coords, color="red")
    if road is not None:
        road_coordinates = [cities[c] for c in road]
        x_cords, y_coords = list(zip(*road_coordinates))
        plt.plot(x_cords, y_coords)
        for city_name in road:
            plt.annotate(
                city_name, 
                cities[city_name],
                xytext=(4, -1), 
                textcoords='offset points')
    plt.gca().set_aspect('equal')
    plt.show()


def distance(city1:Coordinates, city2:Coordinates) -> float:
    """ Euclidian distance between two cities """
    return ((city1[0] - city2[0])**2 + (city1[1] - city2[1])**2)**0.5


def road_length(cities:Dict[str, Coordinates], road:Iterable[str]) -> float:
    """ Calculate the length of the road """
    road_coords = [cities[c] for c in road]
    total = 0
    for i in range(len(road_coords)-1):
        total += distance(road_coords[i], road_coords[i+1])
    total += distance(road_coords[-1], road_coords[0])
    return total

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

    def reset_population(self, city_dict: Dict[str, Coordinates], pop_size: int) -> List[Individual]:
        """
    Initialize a population of individuals (chromosomes) with random permutations of city positions.

    Parameters:
    - city_dict: Dictionary containing city coordinates.
    - pop_size: Size of the population.

    Returns:
    - _population: List of Individuals in the population.
    """
        self._population = []

        # Loop to create a valid chromosome using the default_road function
        for _ in range (pop_size):
            chromosome = default_road(city_dict)
            random.shuffle(chromosome)           # Shuffle the chromosome to generate different permutation
            fitness =  -road_length(city_dict, chromosome)      # Evaluate the fitness of the chromosome
            new_individual = Individual(chromosome, fitness)     # Create a new Individual and append it to the population
            self._population.append(new_individual)
        return self._population


    def evolve_for_one_generation(self,city_dict: Dict[str, Coordinates]):
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

             # Reproduction (Crossover)
            crossover_point = random.randrange(0, len(parent1.chromosome))   # Take a random crossing point
            new_chromosome = (parent1.chromosome[:crossover_point] +
                              [city for city in parent2.chromosome if city not in parent1.chromosome[crossover_point:]])   # Create a new chromosome by combining the first half of parent1 and the remaining cities from parent2
            remaining_cities = [city for city in parent1.chromosome if city not in new_chromosome]       # Add any remaining cities not present in the child's chromosome
            new_chromosome += remaining_cities 
            
             # Mutation
            if random.random() < self._mutation_rate:   # Swap two random chromosome positions to introduce variability
                mutation_index =random.sample(range(len(new_chromosome)), 2)
                new_chromosome[mutation_index[0]], new_chromosome[mutation_index[1]] = \
                    new_chromosome[mutation_index[1]], new_chromosome[mutation_index[0]]
            new_fitness = -road_length(city_dict, new_chromosome)   # Evaluate the fitness of the new chromosome
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



if __name__ == '__main__':
    # Load the cities from the file
    city_dict = load_cities("cities.txt")

    # Print information about shuffled road
    print(city_dict)
    road = default_road(city_dict)
    random.shuffle(road)
    print(road)
    draw_cities(city_dict, road)
    print(road_length(city_dict, road))

    # Initialize the solver and reset the population
    solver = GASolver()
    population_size = 50  # You can adjust the population size as needed
    solver.reset_population(city_dict, population_size)

    # Evolve the population until a stopping criterion is met
    max_generations = 500  # You can adjust the maximum number of generations
    threshold_fitness = None  # You can set a threshold fitness if needed
    solver.evolve_until(max_nb_of_generations=max_generations, threshold_fitness=threshold_fitness)

    # Get the best individual from the final population
    best = solver.get_best_individual()

    # Plot the best path found
    draw_cities(city_dict, best.chromosome)
    plt.title("Best Path Found")
    plt.show()