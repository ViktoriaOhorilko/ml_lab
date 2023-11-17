import numpy as np
from configs import NUM_VMS, MAX_DURATION
from joblib import Parallel, delayed

# GA parameters
POP_SIZE = 100
GENES = NUM_VMS
GENERATIONS = 350
CROSSOVER_RATE = 0.5
MUTATION_RATE = 0.23
ELITE_SIZE = 10


def greedy_search(tasks, virtual_machines, probabilities):
    population = [0 for _ in range(len(tasks))]
    for index, task in enumerate(tasks):
        viable_vms = [vm for vm in virtual_machines if vm['speed'] * task['deadline'] >= task['cost']]

        if viable_vms:
            vm = min(viable_vms, key=lambda x: (-x['speed'], x['cost']))
            population[index] = vm['vm_id']
        else:
            population[index] = np.random.choice(range(len(virtual_machines)), size=1, p=probabilities)[0]

    return population


def fitness(chromosome, tasks, virtual_machines):
    total_cost = 0
    unscheduled_tasks = 0
    for vm in virtual_machines:
        vm['used'] = 0

    for idx, gene in enumerate(chromosome):
        task = tasks[idx]
        vm = virtual_machines[gene]
        if vm['used'] + vm['speed'] * task['deadline'] <= MAX_DURATION and vm['speed'] * task['deadline'] >= task[
            'cost']:
            vm['used'] += vm['speed'] * task['deadline']
            total_cost += task['cost'] * vm['cost']
        else:
            unscheduled_tasks += 1
    return [unscheduled_tasks, total_cost / max((len(tasks) - unscheduled_tasks), 1)]


def evaluate_population(population, tasks, virtual_machines):
    fitness_results = Parallel(n_jobs=-1)(
        delayed(fitness)(chromosome, tasks, virtual_machines) for chromosome in population)
    return fitness_results


def crossover(parent1, parent2):
    mask = np.random.rand(len(parent1)) < CROSSOVER_RATE
    child1 = np.where(mask, parent2, parent1)
    child2 = np.where(mask, parent1, parent2)
    return child1.tolist(), child2.tolist()


def mutate(chromosome):
    mutation_mask = np.random.rand(len(chromosome)) < MUTATION_RATE
    mutation_values = np.random.randint(0, GENES, len(chromosome))
    new_chromosome = np.where(mutation_mask, mutation_values, chromosome)
    return new_chromosome.tolist()


def generate_offspring(parents, population_size):
    children = Parallel(n_jobs=-1)(delayed(crossover)(parents[i], parents[i + 1])
                                   for i in range(0, population_size, 2))
    children = [child for pair in children for child in pair]
    return children


def optimised_genetic_algorithm(tasks, virtual_machines):
    tasks = sorted(tasks, key=lambda x: (x['deadline'], -x['cost']), reverse=True)
    virtual_machines = sorted(virtual_machines, key=lambda x: (x['speed'], -x['cost']), reverse=True)

    probabilities = np.array([1 / (index + 1) for index in range(len(virtual_machines))])
    probabilities /= probabilities.sum()
    greedy_search_pop = greedy_search(tasks, virtual_machines, probabilities)
    population = [greedy_search_pop for _ in range(int(POP_SIZE * 0.2))] + [
        np.random.choice(range(len(virtual_machines)), size=len(tasks), p=probabilities) for _ in
        range(POP_SIZE - int(POP_SIZE * 0.2))]
    for generation in range(GENERATIONS):
        fitness_values = evaluate_population(population, tasks, virtual_machines)
        population = [x for _, x in sorted(zip(fitness_values, population), key=lambda pair: pair[0])]

        # Елітизм: збереження кращих особин
        new_population = population[:ELITE_SIZE]

        while len(new_population) < POP_SIZE:
            parents = [population[np.random.randint(0, int(0.5 * POP_SIZE))] for _ in range(POP_SIZE)]
            children = generate_offspring(parents, POP_SIZE - len(new_population))
            mutated_children = [mutate(child) for child in children]
            new_population.extend(mutated_children)

        population = new_population

    best_solution = population[0]
    return [{'task_id': tasks[idx]['task_id'], 'vm_id': virtual_machines[vm_id]['vm_id']} for idx, vm_id in
            enumerate(best_solution)]
