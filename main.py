import random
import numpy as np 
import json
import math
from client import get_errors

# USE new_gen.txt FOR FURTHER GENERATIONS(USED THE BEST RESULT TO RUN THE GENERATION 15 TIMES)...run some more times to get closer to good answer
# Taking initial population from new_gen array(Comment it if not using population from previous generation)
with open('new_gen.txt','r') as read_file:
	gen_arr = json.load(read_file)
 
with open('valid.txt','r') as read_file:
	valid_arr = json.load(read_file)
 
with open('train.txt','r') as read_file:
	train_arr = json.load(read_file)
 
output = []

POPULATION_SIZE = 20
num_of_features = 11
num_of_generations = 3
delta = 0.08105   #see the mutation fxn for this
valid_factor = 1.2
dist_index = 2
elite_per = 20  #Percentage of elite
mating_pop = 10
mutation_prob = 8		#The numerator(denominator = 11) which u need to choose for mutation to happen
genNum = 17

population = [None for i in range(POPULATION_SIZE)]

# Getting the array from overfit.txt file but in string format
with open('overfit.txt','r') as read_file:
	overfit_arr = json.load(read_file)


def send_request(array):
    res = get_errors('7NieYKa7us9QRVnQzZDsGuWsqv6OAPYocq1TAIB2PevyRYSsjW',array)
    # res = [random.randint(1,100000000),random.randint(1,10000000000)]
    return res

    
def create_gnome(): 
	''' 
	create chromosome or string of genes 
	'''
	new_individual = [0.0, -2.6897862657227743e-13, -2.031229818614153e-13, 8.635785179729647e-12, -1.363883264010215e-10, -2.9154968893446633e-16, 8.963204972580152e-16, 2.44490779067968e-05, -1.5581181578555832e-06, -1.4113473844375399e-08, 6.963381927074255e-10]

	for index in range(num_of_features):
		vary = 0
		prob = random.randint(0, 10)
		if prob < mutation_prob:
			if (7<= index) and (index <= 9):
				vary = 1 + random.uniform(-0.005, 0.005)
			else:
				vary = random.uniform(0, 1.5)
			rem = overfit_arr[index]*vary

			if abs(rem) < 10:
				new_individual[index] = rem
			elif abs(new_individual[index]) >= 10:
				new_individual[index] = random.uniform(-0.01,0.01)
	return new_individual
 

class Individual(): 
	''' 
	Class representing individual in population 
	'''
	def __init__(self, training_error, validation_error): 
		self.chromosome = []
		self.training_error = training_error
		self.validation_error = validation_error
		self.fitness = self.cal_fitness() 

	def set_chromosome(self, chromosome):
		for element in chromosome:
			self.chromosome.append(element)

	def mate(self, parent2):
            child1 = np.empty(11)
            child2 = np.empty(11)

            u = random.random() 
            if (u < 0.5):
                beta = (2 * u)**((dist_index + 1)**-1)
            else:
                beta = ((2*(1-u))**-1)**((dist_index + 1)**-1)


            parent1 = np.array(self.chromosome)
            parent2 = np.array(parent2.chromosome)
    
            child1 = 0.5*((1 + beta) * parent1 + (1 - beta) * parent2)
            child2 = 0.5*((1 - beta) * parent1 + (1 + beta) * parent2)

            # Mutation
            for i in range(num_of_features):
                    prob = random.randint(0, 10)
                    if prob < mutation_prob:
                        vary = 1 + random.uniform(-delta, delta)
                        rem = child1[i]*vary 
                        if abs(rem) <= 10:
                            child1[i] = rem
        
                    prob = random.randint(0, 10)
                    if prob < mutation_prob:
                        vary = 1 + random.uniform(-delta, delta)
                        rem = child2[i]*vary 
                        if abs(rem) <= 10:
                            child2[i] = rem			

            temp1 = [0.0]*num_of_features
            temp2 = [0.0]*num_of_features
            for i in range(num_of_features):
                temp1[i] = child1[i]
                temp2[i] = child2[i]
    
            # create new Individual(offspring) using 
            # generated chromosome for offspring 
            res1 = send_request(temp1)
            res2 = send_request(temp2)
    
            obj1 = Individual(res1[0],res1[1])
            obj2 = Individual(res2[0],res2[1])
            
            obj1.set_chromosome(temp1)
            obj2.set_chromosome(temp2)
            return obj1, obj2 


	def cal_fitness(self): 
			''' 
			Calculate fittness score, it is the sum  of the ratio of validation error and ratio(>1) b/w training error and validation error
			'''
			fitness = abs(valid_factor*self.validation_error + self.training_error)
			return fitness 

# Driver code 
def main(): 
	# output.append("This is the initial population\n")
	global population
	#current generation 
	generation = 1
	# Storing the chromosomes in output array and finally dumping them in output.txt file
	new_gen = []
 
	store_train_error = []
	store_valid_error = []
 
	# create initial population 
	for i in range(POPULATION_SIZE): 
            # gnome = create_gnome() 
            gnome = gen_arr[i]
            
            response = []
            response.append(train_arr[i])
            response.append(valid_arr[i]) 
            # response = send_request(gnome)
            temp = Individual(response[0],response[1])
            temp.set_chromosome(gnome)
            population[i] = temp
            
    
	for gen in range(num_of_generations): 
		# sort the population in increasing order of fitness score 
		population = sorted(population, key = lambda x:x.fitness) 
  
		# Otherwise generate new offsprings for new generation 
		new_generation = [] 

		# Perform Elitism, that mean 20% of fittest population 
		# goes to the next generation 
		s = int((elite_per * POPULATION_SIZE)/100) 
		new_generation.extend(population[:s]) 

		# From 50% of fittest population, Individuals 
		# will mate to produce offspring
		# this is s is representing the remaining 80% population we are gonna to create
		s = int(((100 - elite_per) * POPULATION_SIZE)/100) 
		for _ in range(s//2): 
			# We can change the choices of parents too here...like 4/3...
			parent1 = random.choice(population[:mating_pop]) 
			parent2 = random.choice(population[:mating_pop]) 
			child1, child2 = parent1.mate(parent2) 
			new_generation.append(child1)
			new_generation.append(child2)

		population = new_generation 
		generation += 1
  
		# sort the population in increasing order of fitness score 
		population = sorted(population, key = lambda x:x.fitness) 
  
		# Only appending the fitttest individual in population
  
  
		for i in range(POPULATION_SIZE):
			# Appending the last population in the new_gen array to be used in next run
			if gen == num_of_generations - 1:
				new_gen.append(population[i].chromosome)
				store_train_error.append(population[i].training_error)
				store_valid_error.append(population[i].validation_error)
    
			print(population[i].chromosome)
			print('ind {}     :%%$#@&*               '.format(i),math.log(population[i].training_error, 10),'    ',population[i].training_error,'                 @#%#$%#$^$:   ',math.log(population[i].validation_error, 10),'     ',population[i].validation_error)
		print('<--end of generation-->\n\n')
    
	with open('output_new.txt', 'a') as write_file:
		json.dump(output, write_file, indent = 1)
  
	with open('new_gen.txt', 'w') as w_file:
		json.dump(new_gen, w_file, indent = 1)
  
	with open('train.txt', 'w') as w_file:
		json.dump(store_train_error, w_file, indent = 1)
  
	with open('valid.txt', 'w') as w_file:
		json.dump(store_valid_error, w_file, indent = 1)		

if __name__ == '__main__': 
	main() 
