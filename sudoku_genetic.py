import random
import time
from z3 import *
from tqdm import tqdm
from math import inf
from copy import deepcopy   

matrix = ["020100070", "000302000", "001080040", "900010007", "008060050", "000000000", "080036009", "005070006", "002000003"]
# matrix = ["8 0 6 0 0 0 1 0 7", "0 0 0 6 0 2 0 0 0", "0 5 3 0 0 4 8 0 6", "7 0 4 8 0 0 6 3 0", "0 0 0 0 0 0 0 9 0", "1 0 0 5 0 0 4 0 0", "0 0 1 2 0 0 7 0 9", "2 0 0 0 9 6 0 0 0", "0 7 0 0 1 0 0 8 0"]
matrix = [[int(char) for char in list(s) if char != " "] for s in matrix]

def make_genes(initial=None):
    initial_copy=deepcopy(initial)
    if initial_copy is None:
        initial_copy = list(range(1, 10))    
        random.shuffle(initial_copy)
        return initial_copy
    assert(len(initial_copy)==9)
    random_spots_to_fill=[]
    nums_seen=[0]*9
    for i in range(9):
        if initial_copy[i] == 0:
            random_spots_to_fill.append(i)
        else:
            nums_seen[initial_copy[i]-1] = 1
    random_spots_to_fill2 = [i for i in range(9) if nums_seen[i]==0]
    random.shuffle(random_spots_to_fill)
    for i in range(len(random_spots_to_fill)):
        initial_copy[random_spots_to_fill[i]] = random_spots_to_fill2[i]+1
    return initial_copy

def make_chromosomes(initial_chromosome=None):
    if initial_chromosome is None:
        return [make_genes() for _ in range(9)]
    assert(len(initial_chromosome)==9)
    return [make_genes(initial_gene) for initial_gene in initial_chromosome]

def make_population(size=100, initial_chromosome=None):
    if initial_chromosome is None:
        initial_chromosome = [[0] * 9 for _ in range(9)]
    population=[]
    for i in range(size):
        population.append(make_chromosomes(initial_chromosome))
    return population

def get_fit(chromosome):
    fit = 0
    for col in range(9):
        visited = [0] * 9
        for row in range(9):
            visited[chromosome[row][col]-1] += 1
        for i in visited:
            if i > 1:
                fit -= abs(i-1)
    for row in range(9):
        visited = [0] * 9
        for col in range(9):
            visited[chromosome[row][col]-1] += 1
        for i in visited:
            if i > 1:
                fit -= abs(i-1)
    for block_top in range(0,9,3):
        for block_left in range(0,9,3):
            visited = [0] * 9
            for i in range(3):
                for j in range(3):
                    visited[chromosome[block_top+i][block_left+j]-1] += 1
            for i in visited:
                if i > 1:
                    fit -= abs(i-1)
    return fit

def simulate_swap_genes(gene1, gene2):
    vars=[Bool(f"var_{i}") for i in range(9)]
    truly_variable=[vars[i] for i in range(9) if gene1[i]!=gene2[i]]
    if not truly_variable:
        return gene1, gene2
    s=Solver()
    for i in range(9):
        if gene1[i]==gene2[i]:
            continue
        ind=gene2.index(gene1[i])
        s.add(Implies(vars[i], vars[ind]))
        s.add(Implies(vars[ind], vars[i]))
    all_models=[]
    s.add(Or(truly_variable))
    s.add(Not(And(truly_variable)))
    if s.check()==unsat:
        if random.randint(0,1)==0:
            return gene2, gene1
        return gene1, gene2
    while s.check()==sat:
        m=s.model()
        all_models.append(m)
        s.add(Or([var!=m.evaluate(var) for var in truly_variable]))
    m=random.choice(all_models)
    return [gene1[i] if m.evaluate(vars[i])==True else gene2[i] for i in range(9)], [gene2[i] if m.evaluate(vars[i])==True else gene1[i] for i in range(9)]

def simulate_crossover(chromosome1, chromosome2):
    child1 = []
    child2 = []
    for i in range(9):
        # child1_gene, child2_gene = simulate_swap_genes(chromosome1[i], chromosome2[i])
        if random.random() < 0.5:
            child1_gene=chromosome1[i]
            child2_gene=chromosome2[i]
        else:
            child1_gene=chromosome2[i]
            child2_gene=chromosome1[i]
        child1.append(child1_gene)
        child2.append(child2_gene)
    return child1, child2

def simulate_mutation(chromosome):
    global matrix
    for i in range(9):
        if random.random() < 0.1:
            choice1=make_genes(matrix[i])
            # # truly_random has indices of row i where matrix[i][j] == 0
            truly_random=[j for j in range(9) if matrix[i][j] == 0]
            if len(truly_random) < 2:
                continue
            values=[chromosome[i][j] for j in truly_random]
            values=sorted(values)
            random.shuffle(truly_random)
            # for k in range(len(truly_random)):
            #     chromosome[i][truly_random[k]]=values[k]
            chromosome[i]=deepcopy(choice1)
        # if random.random() < 0.1:
        #     chromosome[i]=make_genes(matrix[i])
        # if random.random() < 0.1:
        #     truly_random=[j for j in range(9) if chromosome[i][j] not in matrix[i]]
        #     if len(truly_random) < 2:
        #         continue
        #     truly_random_copy=[j for j in truly_random]
        #     random.shuffle(truly_random_copy)
        #     chromosome_i_copy=[j for j in chromosome[i]]
        #     for k in range(len(truly_random)):
        #         chromosome[i][truly_random[k]]=chromosome_i_copy[truly_random_copy[k]]
        #     # for _ in range(50):
        #     #     a, b = random.sample(truly_random, 2)  
        #     #     temp=chromosome[i][a]
        #     #     chromosome[i][a]=chromosome[i][b]
        #     #     chromosome[i][b]=temp
    return chromosome

# def get_mating_pool(population):
#     fitnesses = [get_fit(chromosome) for chromosome in population]
#     fitnesses = [-1/fit  if fit != 0 else inf for fit in fitnesses]  
#     pool = random.choices(population, weights=fitnesses, k=len(fitnesses))
#     return pool

def get_mating_pool(population):
    fitness_list = []
    pool = []
    for chromosome in population:
        fitness = get_fit(chromosome)
        fitness_list.append((fitness, chromosome))
    fitness_list.sort()
    weight = list(range(1, len(fitness_list) + 1))
    for _ in range(len(population)):
        ch = random.choices(fitness_list, weight)[0]
        pool.append(ch[1])
    return pool

def get_offsprings(population):
    new_pool = []
    i = 0
    while i < len(population):
        ch1 = population[i]
        ch2 = population[(i + 1) % len(population)]
        x = random.random()
        if x < 0.95:
            ch1, ch2 = simulate_crossover(ch1, ch2)
        new_pool.append(simulate_mutation(ch1))
        new_pool.append(simulate_mutation(ch2))
        i += 2
    return new_pool

# initial_chromosome = make_chromosomes(matrix)
# second_chromosome = make_chromosomes(matrix)
# new_chromosome1, new_chromosome2 = simulate_crossover(initial_chromosome, second_chromosome)
# print(new_chromosome1[0])
# simulate_mutation(new_chromosome1)
# print(new_chromosome1[0])

POPULATION = 500

REPETITION = 1000

m=0
def genetic_algorithm():
    global m
    population = make_population(POPULATION, matrix)
    for _ in tqdm(range(REPETITION)):
        mating_pool = get_mating_pool(population)
        random.shuffle(mating_pool)
        population = get_offsprings(mating_pool)
        fit = [get_fit(c) for c in population]
        m = max(fit)
        print(m)
        if m == 0:
            return population
    return population

# population=make_population(100, matrix)
# new_pop=get_mating_pool(population)
# print(population[0][0])
# print(new_pop[0][0])

pop=genetic_algorithm()
print(m)
for row in pop[0]:
    print(row)