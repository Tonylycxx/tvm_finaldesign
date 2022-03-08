import os
import sys
import torch, onnx
import tvm
from tvm import relay
import tvm.relay.testing
import transformers

import modelTaskNumGetter, modelTunner

def get_benchmark_model(name, batch_size, layout="NHWC", dtype="float32"):

    if layout == "NHWC":
        image_shape = (224, 224, 3)
    elif layout == "NCHW":
        image_shape = (3, 224, 224)
    else:
        raise ValueError("Invalid layout: " + layout)

    model_path_root = "/home/tonylyc/.tvm_test_data/onnx/"

    input_shape = (batch_size,) + image_shape
    output_shape = (batch_size, 1000)

    if name.startswith("resnet-"):
        n_layer = int(name.split("-")[1])
        mod, params = relay.testing.resnet.get_workload(
            num_layers=n_layer,
            batch_size=batch_size,
            layout=layout,
            dtype=dtype,
            image_shape=image_shape
        )
    elif name.startswith("resnet3d-"):
        n_layer = int(name.split("-")[1])
        mod, params = relay.testing.resnet.get_workload(
            num_layers=n_layer,
            batch_size=batch_size,
            layout=layout,
            dtype=dtype,
            image_shape=image_shape,
        )
    elif name == "mobilenet":
        mod, params = relay.testing.mobilenet.get_workload(
            batch_size=batch_size, layout=layout, dtype=dtype, image_shape=image_shape
        )
    elif name == "inception_v3":
        input_shape = (batch_size, 3, 299, 299) if layout == "NCHW" else (batch_size, 299, 299, 3)
        mod, params = relay.testing.inception_v3.get_workload(
            batch_size=batch_size,
            dtype=dtype
        )
    elif name.startswith("vgg-"):
        n_layer = int(name.split("-")[1])
        mod, params = relay.testing.vgg.get_workload(
            num_layers=n_layer,
            batch_size=batch_size,
            dtype=dtype,
            image_shape=image_shape,
        )
    elif name == "squeezenet_v1.1":
        assert layout == "NCHW", "squeezenet_v1.1 only supports NCHW layout"
        mod, params = relay.testing.squeezenet.get_workload(
            version="1.1",
            batch_size=batch_size,
            dtype=dtype,
            image_shape=image_shape,
        )
    elif name == "dcgan":
        input_shape = (batch_size, 100)
        mod, params = relay.testing.dcgan.get_workload(
            batch_size=batch_size,
            dtype=dtype,
            layout=layout
        )
    elif name.startswith("densenet-"):
        densenet_size = int(name.split("-")[1])
        mod, params = relay.testing.densenet.get_workload(
            densenet_size=densenet_size,
            batch_size=batch_size,
            dtype=dtype
        )
    elif name == "bert":
        input_shape = (batch_size, 128)
        input_ids = torch.randint(30000, input_shape)
        model_class = transformers.BertModel
        model = model_class.from_pretrained("bert-base-uncased", return_dict=False)
        scripted_model = torch.jit.trace(model, input_ids, strict=False)
        shape_list = [('input_ids', input_shape)]
        mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
    elif name.endswith("onnx"):
        assert layout == "NCHW", "onnx model only supports NCHW layout"
        model_path = model_path_root + name
        onnx_model = onnx.load(model_path)
        input_name = "data_0"
        input_shape = [1, 3, 224, 224]
        shape_dict = {input_name: input_shape}
        mod, params = relay.frontend.from_onnx(onnx_model, shape=shape_dict)
    else:
        raise ValueError("Network not found.")

    return mod, params, input_shape

def auto_tune(model_name, round_start, round_end):

    # Init basic variables
    base_cnt = 100
    batch_size = 1
    dtype = "float32"
    layout = "NCHW"
    # target = tvm.target.Target("llvm -mcpu=core-avx2")
    target = tvm.target.Target("cuda")
    cpu_runner = auto_scheduler.LocalRunner(repeat=5, enable_cpu_cache_flush=True)

    # Init logs dir
    cur_path = os.path.abspath(os.path.curdir)
    target_dir_path = cur_path + '/' + model_name
    try:
        if not os.path.isdir(target_dir_path):
            os.makedirs(target_dir_path)
    except:
        print("Make log dir failed, target dir is already exist!")
        sys.exit()
    
    # Init exec log file and init tune needed variables
    exec_log_file = target_dir_path + "/log.txt"
    with open(exec_log_file, 'a') as exec_log:
        exec_log.write("Prepare for tune process:\n\n")

        base_tune_log_file = "%s-%s-B%d-%s.json" % (model_name, layout, batch_size, target.kind.name)
        base_tune_log_file = target_dir_path + '/' + base_tune_log_file
        exec_log.write("Dump tune log to %s\n\n" % base_tune_log_file)

        mod, params, input_shape = get_benchmark_model(
            model_name,
            batch_size,
            layout,
            dtype,
        )
        exec_log.write("Get model successfully\n\n")

        tasks, task_weights = modelTaskNumGetter.get_model_tasks(mod, params, target)
        exec_log.write("Extract %d tasks from model %s\n\n" % (len(tasks), model_name))

        # Do auto-tune process
        for tune_round in range(round_start, round_end):
            
            trail_cnt = base_cnt * tune_round
            per_tune_trails = base_cnt * len(tasks)
            exec_log.write("Round %d tune" % tune_round)

            if tune_round == 1:
                modelTunner.run_tuning_x86(tasks, task_weights, per_tune_trails, base_tune_log_file, cpu_runner)
            else:
                modelTunner.resume_tune_x86(tasks, task_weights, per_tune_trails, base_tune_log_file, base_tune_log_file, cpu_runner)

            log_file = "%s-%s-B%d-%s-%d.json" % (model_name, layout, batch_size, target.kind.name, trail_cnt)
            log_file = target_dir_path + '/' + log_file

            with open(base_tune_log_file, 'r') as base_log:
                with open(log_file, 'w') as tune_log:
                    s = base_log.readline()
                    while len(s) > 0:
                        tune_log.write(s)
                        s = base_log.readline()
            exec_log.write("Write base log %s to tune log %s" % (base_tune_log_file, log_file))
                    

            # try:
            #     tune_log = open(log_file, 'w')
            #     base_log = open(base_tune_log_file, 'r')

            #     s = base_log.readline()
            #     while len(s) > 0:
            #         tune_log.write(s)
            #         s = base_log.readline()
            #     exec_log.write("Write base log %s to tune log %s" % (base_tune_log_file, log_file))

            #     tune_log.close()
            #     base_log.close()
            # except:
            #     print("Make tune log failed, exit")
            #     sys.exit()

        exec_log.write("Auto tune process executed successfully\n\n")

# auto_tune("googlenet.onnx", 10, 11)


