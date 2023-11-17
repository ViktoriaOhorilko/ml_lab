import random
import time
import sys

from configs import NUM_TASKS, NUM_VMS, MAX_DURATION
from copy import deepcopy
from gs import greedy_search
from ga import genetic_algorithm
from aga import particle_swarm_optimization
from new_a import optimised_genetic_algorithm

random.seed(0)
def generate_data(num_tasks=10, num_vms=5):
    tasks = [{'task_id': i, 'deadline': random.randint(1, 10), 'cost': random.randint(1, 40)} for i in
             range(num_tasks)]
    virtual_machines = [{'vm_id': i, 'speed': random.uniform(0.5, 10.0), 'cost': random.randint(1, 20), 'used': 0} for i in
                        range(num_vms)]
    return tasks, virtual_machines


def evaluation(tasks, virtual_machines, run_algorithm):
    start_time = time.time()

    scheduled_tasks = run_algorithm(deepcopy(tasks), deepcopy(virtual_machines))

    end_time = time.time()
    execution_time = end_time - start_time

    memory_usage = sys.getsizeof(scheduled_tasks) + sys.getsizeof(tasks) + sys.getsizeof(virtual_machines)

    total_cost = 0
    completed_tasks = 0
    max_completion_time = 0


    for idx, sol in enumerate(scheduled_tasks):
        task = tasks[sol["task_id"]]
        vm = virtual_machines[sol["vm_id"]]
        if vm['used'] + vm['speed'] * task['deadline'] <= MAX_DURATION and vm['speed'] * task['deadline'] >= task['cost']:
            total_cost += task['cost'] * vm['cost']
            completed_tasks += 1
            max_completion_time = max(max_completion_time, task['cost'] / vm['speed'])
            vm['used'] += vm['speed'] * task['deadline']

    return {
        'execution_time': execution_time,
        'memory_usage': memory_usage,
        'total_cost': total_cost,
        'completed_tasks': completed_tasks,
        'max_completion_time': max_completion_time
    }



# Generate sample data
tasks, virtual_machines = generate_data(num_tasks=NUM_TASKS, num_vms=NUM_VMS)
# print("Generated tasks:")
# for task in tasks:
#     print(task)
# print("Generated VMs:")
# for vm in virtual_machines:
#     print(vm)

# Evaluate the greedy search algorithm
if __name__ == '__main__':

    evaluation_metrics = evaluation(tasks, virtual_machines, greedy_search)
    print(evaluation_metrics)

    # Evaluate the  non-adaptive genetic algorithm
    evaluation_metrics = evaluation(tasks, virtual_machines, genetic_algorithm)
    print(evaluation_metrics)

    # Evaluate the  adaptive genetic algorithm
    evaluation_metrics = evaluation(tasks, virtual_machines, particle_swarm_optimization)
    print(evaluation_metrics)

    # Evaluate the  new genetic algorithm
    evaluation_metrics = evaluation(tasks, virtual_machines, optimised_genetic_algorithm)
    print(evaluation_metrics)