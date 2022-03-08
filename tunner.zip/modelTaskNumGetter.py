from tvm import auto_scheduler

def get_model_tasks(mod, params, target):

    print("Extract tasks...")
    tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target)

    return tasks, task_weights