from operator import mod
import re
import tvm
import onnx
import netron
import numpy as np
import tvm.relay as relay

import modelDownloader
import modelTaskNumGetter

mobilenetv2_url = "https://s3.amazonaws.com/onnx-model-zoo/mobilenet/mobilenetv2-1.0/mobilenetv2-1.0.onnx"
model_path_root = "/home/tonylyc/.tvm_test_data/onnx/"
model_name = "mobilenetv2.onnx"

def test():

    # model_path = modelDownloader.download_model(mobilenetv2_url, model_name, "onnx")
    model_path = model_path_root + model_name
    onnx_model = onnx.load(model_path)
    # target = tvm.target.Target("llvm -mcpu=core-avx2")
    target = tvm.target.Target("cuda")
    netron.start(model_path)
    return
    input_name = "data_0"
    input_shape = [1, 3, 224, 224]
    shape_dict = {input_name: input_shape}
    mod, params = relay.frontend.from_onnx(onnx_model, shape=shape_dict)

    tasks, task_weights = modelTaskNumGetter.get_model_tasks(mod, params, target)
    print("Get %d tasks from model %s" % (len(tasks), model_name))

test()
