import random
import time
import sys

from configs import NUM_TASKS, NUM_VMS

from gs import greedy_search
from ga import genetic_algorithm, get_unscheduled_tasks
from aga import particle_swarm_optimization


def generate_data(num_tasks=10, num_vms=5):
    tasks = [{'task_id': i, 'deadline': random.randint(10, 50), 'cost': random.randint(1, 10)} for i in
             range(num_tasks)]
    virtual_machines = [{'vm_id': i, 'speed': random.uniform(0.5, 2.0), 'cost': random.randint(1, 5)} for i in
                        range(num_vms)]
    return tasks, virtual_machines


def evaluation(tasks, virtual_machines, run_algorithm):
    start_time = time.time()

    scheduled_tasks, unscheduled_tasks = run_algorithm(tasks, virtual_machines)

    end_time = time.time()
    execution_time = end_time - start_time

    memory_usage = sys.getsizeof(scheduled_tasks) + sys.getsizeof(unscheduled_tasks)

    total_cost = 0
    completed_tasks = 0
    max_completion_time = 0

    for idx, sol in enumerate(scheduled_tasks):
        task = tasks[sol["task_id"]]
        vm = virtual_machines[sol["vm_id"]]
        if vm['speed'] * task['deadline'] >= task['cost']:
            total_cost += task['cost'] * vm['cost']
            completed_tasks += 1
            max_completion_time = max(max_completion_time, task['cost'] / vm['speed'])

    print("Scheduled Tasks")
    for s_t in scheduled_tasks:
        print(s_t)
    print("Unscheduled Tasks")
    for uns_t in unscheduled_tasks:
        print(uns_t)

    return {
        'execution_time': execution_time,
        'memory_usage': memory_usage,
        'scheduled_tasks': len(scheduled_tasks),
        'unscheduled_tasks': len(unscheduled_tasks),
        'total_cost': total_cost,
        'completed_tasks': completed_tasks,
        'max_completion_time': max_completion_time
    }


def test_greedy_search(*args, **kwargs):
    return greedy_search(*args, **kwargs)


def test_ga(*args, **kwargs):
    best_solution = genetic_algorithm(*args, **kwargs)
    scheduled_tasks_ga = [{'task_id': idx, 'vm_id': vm_id} for idx, vm_id in enumerate(best_solution)]
    unscheduled_tasks_ga = get_unscheduled_tasks(best_solution, tasks, virtual_machines)
    return scheduled_tasks_ga, unscheduled_tasks_ga


def test_optimized_aga(*args, **kwargs):
    best_solution = particle_swarm_optimization(*args, **kwargs)
    scheduled_tasks_ga = [{'task_id': idx, 'vm_id': vm_id} for idx, vm_id in enumerate(best_solution)]
    unscheduled_tasks_ga = get_unscheduled_tasks(best_solution, tasks, virtual_machines)
    return scheduled_tasks_ga, unscheduled_tasks_ga


# Generate sample data
tasks, virtual_machines = generate_data(num_tasks=NUM_TASKS, num_vms=NUM_VMS)
print("Generated tasks:")
for task in tasks:
    print(task)
print("Generated VMs:")
for vm in virtual_machines:
    print(vm)

# Evaluate the greedy search algorithm
evaluation_metrics = evaluation(tasks, virtual_machines, test_greedy_search)
print(evaluation_metrics)

# Evaluate the  non-adaptive genetic algorithm
evaluation_metrics = evaluation(tasks, virtual_machines, test_ga)
print(evaluation_metrics)

# Evaluate the  adaptive genetic algorithm
evaluation_metrics = evaluation(tasks, virtual_machines, test_optimized_aga)
print(evaluation_metrics)
