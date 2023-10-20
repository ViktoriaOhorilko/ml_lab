import numpy as np

from configs import NUM_VMS

# GA parameters
POP_SIZE = 50
GENES = NUM_VMS
GENERATIONS = 100
CROSSOVER_RATE = 0.7
MUTATION_RATE = 0.2


def fitness(chromosome, tasks, virtual_machines):
    total_cost = 0
    for idx, gene in enumerate(chromosome):
        task = tasks[idx]
        vm = virtual_machines[gene]
        if vm['speed'] * task['deadline'] >= task['cost']:
            total_cost += task['cost'] * vm['cost']
        else:
            total_cost += float('inf')  # penalty for not meeting the deadline
    return -total_cost


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
def genetic_algorithm(tasks, virtual_machines):
    population = [np.random.randint(0, GENES, size=len(tasks)) for _ in range(POP_SIZE)]
    for generation in range(GENERATIONS):
        population = sorted(population, key=lambda x: fitness(x, tasks, virtual_machines))

        # Keep the top 10% of the population
        new_population = population[:int(0.1 * POP_SIZE)]

        while len(new_population) < POP_SIZE:
            parent1 = population[np.random.randint(0, int(0.5 * POP_SIZE))]
            parent2 = population[np.random.randint(0, int(0.5 * POP_SIZE))]
            child1, child2 = crossover(parent1, parent2)
            new_population.extend([mutate(child1), mutate(child2)])

        population = new_population

    best_solution = population[0]
    return best_solution


def get_unscheduled_tasks(best_solution, tasks, virtual_machines):
    unscheduled_tasks_ga = []
    for idx, vm_id in enumerate(best_solution):
        task = tasks[idx]
        vm = virtual_machines[vm_id]
        if vm['speed'] * task['deadline'] < task['cost']:
            unscheduled_tasks_ga.append({'task_id': idx})
    return unscheduled_tasks_ga
