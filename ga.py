import numpy as np
from copy import deepcopy
from configs import NUM_VMS, NUM_TASKS, MAX_DURATION

# GA parameters
POP_SIZE = 100
GENES = NUM_VMS
GENERATIONS = 350
CROSSOVER_RATE = 0.5
MUTATION_RATE = 0.23


def fitness(chromosome, tasks, virtual_machines):
    total_cost = 0
    for vm in virtual_machines:
        vm['used'] = 0

    for idx, gene in enumerate(chromosome):
        task = tasks[idx]
        vm = virtual_machines[gene]

        if vm['used'] + vm['speed'] * task['deadline'] <= MAX_DURATION and vm['speed'] * task['deadline'] >= task[
            'cost']:
            vm['used'] += vm['speed'] * task['deadline']
            total_cost += task['cost'] * vm['cost']
    return total_cost


# Crossover
def crossover(parent1, parent2):
    child1 = parent1.copy()
    child2 = parent2.copy()
    for i in range(len(parent1)):
        if np.random.random() < CROSSOVER_RATE:
            child1[i], child2[i] = child2[i], child1[i]
    return child1, child2


# Mutation
def mutate(chromosome):
    for i in range(len(chromosome)):
        if np.random.random() < MUTATION_RATE:
            chromosome[i] = np.random.randint(0, GENES)
    return chromosome


# Genetic Algorithm
def genetic_algorithm(tasks_, virtual_machines_):
    tasks = sorted(deepcopy(tasks_), key=lambda x: (x['deadline'], -x['cost']), reverse=True)
    virtual_machines = sorted(deepcopy(virtual_machines_), key=lambda x: (x['speed'], -x['cost']), reverse=True)

    probabilities = np.array([1 / (index + 1) for index in range(len(virtual_machines))])
    probabilities /= probabilities.sum()

    population = [np.random.choice(range(len(virtual_machines)), size=len(tasks), p=probabilities) for _ in
                  range(POP_SIZE)]
    # population = [np.random.randint(0, GENES, size=len(tasks)) for _ in range(POP_SIZE)]
    for generation in range(GENERATIONS):
        population = sorted(population, key=lambda x: fitness(x, tasks, virtual_machines))

        # Keep the top 10% of the population
        new_population = population[:int(0.2 * POP_SIZE)]

        while len(new_population) < POP_SIZE:
            parent1 = population[np.random.randint(0, int(0.5 * POP_SIZE))]
            parent2 = population[np.random.randint(0, int(0.5 * POP_SIZE))]
            child1, child2 = crossover(parent1, parent2)
            new_population.extend([mutate(child1), mutate(child2)])

        population = new_population

    best_solution = population[0]

    return [{'task_id': tasks[idx]['task_id'], 'vm_id': virtual_machines[vm_id]['vm_id']} for idx, vm_id in
            enumerate(best_solution)]
    # return best_solution, tasks, virtual_machines
