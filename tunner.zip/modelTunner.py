from tvm import auto_scheduler

# Set some tune options
def run_tuning_x86(tasks, task_weights, num_measure_trails, log_file, runner):
    print("Begin tuning...")
    tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
    tune_options = auto_scheduler.TuningOptions(
        num_measure_trials=num_measure_trails,
        runner = runner,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)]
    )
    tuner.tune(tune_options)

def resume_tune_x86(tasks, task_weights, num_measure_trails, former_log_file, log_file):
    print("Resume tune from %s to %s" % (former_log_file, log_file))
    tuner = auto_scheduler.TaskScheduler(tasks, task_weights, load_log_file=former_log_file)
    tune_options = auto_scheduler.TuningOptions(
        num_measure_trials=num_measure_trails,
        runner = runner,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)]
    )
    tuner.tune(tune_options)