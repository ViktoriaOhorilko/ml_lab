def greedy_search(tasks, virtual_machines):
    scheduled_tasks = []
    unscheduled_tasks = []

    tasks_sorted = sorted(tasks, key=lambda x: x['deadline'], reverse=True)

    for task in tasks_sorted:
        viable_vms = [vm for vm in virtual_machines if vm['speed'] * task['deadline'] >= task['cost']]

        if viable_vms:
            vm = max(viable_vms, key=lambda x: x['speed'])
            scheduled_tasks.append({'task_id': task['task_id'], 'vm_id': vm['vm_id']})
        else:
            unscheduled_tasks.append(task['task_id'])

    return scheduled_tasks
