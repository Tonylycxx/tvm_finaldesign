from tvm import nd, device
import numpy as np

def evaluate_model_inference_time(module, target, input_shape, dtype, repeat_time):

    # Create graph executor
    dev = device(str(target), 0)
    data_tvm = nd.array((np.random.uniform(size=input_shape)).astype(dtype))
    module.set_input("data", data_tvm)

    # Do the evaluation
    print("Evaluate inference time cost...")
    print(module.benchmark(dev, repeat=repeat_time, min_repeat_ms=500))