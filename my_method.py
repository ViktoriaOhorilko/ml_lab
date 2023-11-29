import numpy as np
import random
from configs import MAX_DURATION


def simulated_annealing(tasks, virtual_machines, max_iterations=1000):
    tasks = sorted(tasks, key=lambda x: (x['deadline'], -x['cost']))
    virtual_machines = sorted(virtual_machines, key=lambda x: (x['speed'], -x['cost']), reverse=True)

    probabilities = np.array([1 / (index + 1) for index in range(len(virtual_machines))])
    probabilities /= probabilities.sum()

    initial_state = np.random.choice(range(len(virtual_machines)), size=len(tasks), p=probabilities)

    current_state = initial_state
    current_cost, current_unscheduled = evaluate(current_state, tasks, virtual_machines)
    best_state = current_state
    best_cost = current_cost
    best_unscheduled = current_unscheduled
    temp = 1

    for iteration in range(max_iterations):
        temp *= 0.9
        neighbor_state = generate_neighbor(current_state, tasks, virtual_machines, probabilities, temp, False)
        neighbor_cost, neighbor_unscheduled = evaluate(neighbor_state, tasks, virtual_machines)

        neighbor_state2 = generate_neighbor(current_state, tasks, virtual_machines, probabilities, temp, True)
        neighbor_cost2, neighbor_unscheduled2 = evaluate(neighbor_state2, tasks, virtual_machines)

        if neighbor_unscheduled > neighbor_unscheduled2 or (
                neighbor_unscheduled == neighbor_unscheduled2 and neighbor_cost > neighbor_cost2):
            neighbor_state, neighbor_cost, neighbor_unscheduled = neighbor_state2, neighbor_cost2, neighbor_unscheduled2

        if current_unscheduled > neighbor_unscheduled or (
                current_unscheduled == neighbor_unscheduled and current_cost > neighbor_cost):
            current_state, current_cost, current_unscheduled = neighbor_state, neighbor_cost, neighbor_unscheduled

        if neighbor_unscheduled < best_unscheduled or (
                neighbor_unscheduled == best_unscheduled and neighbor_cost < best_cost):
            best_state = neighbor_state
            best_cost = neighbor_cost
            best_unscheduled = neighbor_unscheduled

    return [{'task_id': tasks[idx]['task_id'], 'vm_id': virtual_machines[vm_id]['vm_id']} for idx, vm_id in
            enumerate(best_state)]


def generate_neighbor(state, tasks, virtual_machines, probabilities, temp, change_scheduled=False):
    new_state = state.copy()
    unscheduled_tasks = evaluate(state, tasks, virtual_machines, return_unscheduled_list=True)

    for idx in range(len(tasks)):
        if change_scheduled:
            if should_reassign_task(idx, unscheduled_tasks, temp):
                new_state[idx] = np.random.choice(range(len(virtual_machines)), size=1, p=probabilities)[0]

        else:
            if idx in unscheduled_tasks:
                new_state[idx] = np.random.choice(range(len(virtual_machines)), size=1, p=probabilities)[0]
    return new_state


def should_reassign_task(task_idx, unscheduled_tasks, temp):
    if task_idx in unscheduled_tasks:
        return random.random() < temp  # Вищий шанс для невиконаних задач
    return random.random() < 0.5 * temp  # Нижчий шанс для інших задач


def evaluate(state, tasks, virtual_machines, return_unscheduled_list=False):
    total_cost = 0
    unscheduled_tasks = 0
    unscheduled_list = []

    for vm in virtual_machines:
        vm['used'] = 0
    for idx, gene in enumerate(state):
        task = tasks[idx]
        vm = virtual_machines[gene]
        if vm['used'] + vm['speed'] * task['deadline'] <= MAX_DURATION and vm['speed'] * task['deadline'] >= task[
            'cost']:
            vm['used'] += vm['speed'] * task['deadline']
            total_cost += task['cost'] * vm['cost']
        else:
            unscheduled_tasks += 1
            unscheduled_list.append(idx)

    if return_unscheduled_list:
        return unscheduled_list
    return total_cost, unscheduled_tasks
