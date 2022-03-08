import torch, onnx
import transformers
from transformers import BertModel
from tvm import relay
import tvm.relay.testing

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
    elif name == "bert-base-uncased":
        input_shape = (batch_size, 128)
        input_ids = torch.randint(30000, input_shape)
        model_class = transformers.BertModel
        model = model_class.from_pretrained(name, return_dict=False)
        scripted_model = torch.jit.trace(model, input_ids, strict=False)
        shape_list = [('input_ids', input_shape)]
        mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
    elif name.endswith("onnx"):
        assert layout == "NCHW", "onnx model only supports NCHW layout"
        model_path = model_path_root + name
        onnx_model = onnx.load(model_path)
        mod, params = relay.frontend.from_onnx(onnx_model)
    else:
        raise ValueError("Network not found.")

    return mod, params, input_shape