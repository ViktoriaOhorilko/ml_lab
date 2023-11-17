from copy import deepcopy

import numpy as np
from configs import NUM_TASKS, MAX_DURATION

# PSO parameters
PARTICLE_COUNT = 100
MAX_ITERATIONS = 200
W = 0.5  # inertia weight
C1 = 1.5  # personal best weight
C2 = 1.5  # global best weight


def pso_fitness(position, tasks, virtual_machines):
    total_cost = 0
    for vm in virtual_machines:
        vm['used'] = 0

    for idx, vm_id in enumerate(position):
        task = tasks[idx]
        vm = virtual_machines[vm_id]
        if vm['used'] + vm['speed'] * task['deadline'] <= MAX_DURATION and vm['speed'] * task['deadline'] >= task[
            'cost']:
            vm['used'] += vm['speed'] * task['deadline']
            total_cost += task['cost'] * vm['cost']
        else:
            total_cost += float('inf')  # penalty for not meeting the deadline
    return -total_cost


class Particle:
    def __init__(self, tasks_count, vms_count):
        self.position = np.random.randint(0, vms_count, size=tasks_count)
        self.velocity = np.random.uniform(-1, 1, size=tasks_count)
        self.best_position = np.copy(self.position)
        self.best_score = float('inf')

    def update_velocity(self, global_best_position):
        inertia = W * self.velocity
        personal_attraction = C1 * np.random.random() * (self.best_position - self.position)
        global_attraction = C2 * np.random.random() * (global_best_position - self.position)
        self.velocity = inertia + personal_attraction + global_attraction

    def update_position(self, vms_count):
        self.position = np.mod(self.position + self.velocity.astype(int), vms_count)


def particle_swarm_optimization(tasks_, virtual_machines_):
    tasks = sorted(deepcopy(tasks_), key=lambda x: (x['deadline'], -x['cost']), reverse=True)
    virtual_machines = sorted(deepcopy(virtual_machines_), key=lambda x: (x['speed'], -x['cost']), reverse=True)

    particles = [Particle(len(tasks), len(virtual_machines)) for _ in range(PARTICLE_COUNT)]
    global_best_position = np.random.randint(0, len(virtual_machines), size=len(tasks))
    global_best_score = float('inf')

    for _ in range(MAX_ITERATIONS):
        for particle in particles:
            score = pso_fitness(particle.position, tasks, virtual_machines)
            if score < particle.best_score:
                particle.best_score = score
                particle.best_position = particle.position

            if score < global_best_score:
                global_best_score = score
                global_best_position = particle.position

        for particle in particles:
            particle.update_velocity(global_best_position)
            particle.update_position(len(virtual_machines))

    return [{'task_id': tasks[idx]['task_id'], 'vm_id': virtual_machines[vm_id]['vm_id']} for idx, vm_id in
            enumerate(global_best_position)]

