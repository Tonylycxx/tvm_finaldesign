import tvm

import benchmarkModelGetter
import modelEvaluater
import modelTaskNumGetter
import modelTunner
import modelCompiler


network = "bert-base-uncased"
batch_size = 1
layout = "NHWC"
# target = tvm.target.Target("llvm -mcpu=core-avx2")
    target = tvm.target.Target("cuda")
trails_cnt = 100
dtype = "float32"
base_log_file = "%s-%s-B%d-%s.json" % (network, layout, batch_size, target.kind.name)
log_file = "%s-%s-B%d-%s-%d.json" % (network, layout, batch_size, target.kind.name, trails_cnt * 10)

mod = benchmarkModelGetter.get_benchmark_model(
    network,
    batch_size,
    layout,
    dtype,
)

tasks, task_weights = modelTaskNumGetter.get_model_tasks(mod, params, target)
# print("Get %d tasks from model %s" % (len(tasks), network))

# modelTunner.run_tuning_x86(tasks, task_weights, trails_cnt * len(tasks), log_file)
# modelTunner.resume_tune_x86(tasks, task_weights, trails_cnt * len(tasks), base_log_file, base_log_file)

# module = modelCompiler.run_compile(mod, params, target)
# module = modelCompiler.run_compile(mod, params, target, log_file)
# modelEvaluater.evaluate_model_inference_time(module, target, input_shape, dtype, 100)