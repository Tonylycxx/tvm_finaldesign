import torch
import transformers
import tvm
from tvm import relay

import modelTaskNumGetter

model_name = 'bert-base-uncased'
batch_size = 1

model_class = transformers.BertModel
model = model_class.from_pretrained(model_name, return_dict=False)
model.eval()

input_shape = [batch_size, 128]
input_name = 'input_ids'
target = tvm.target.Target("llvm -mcpu=core-avx2")
A = torch.randint(30000, input_shape)

scripted_model = torch.jit.trace(model, [A], strict=False)
shape_list = [('input_ids', input_shape)]
mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)

tasks, task_weights = modelTaskNumGetter.get_model_tasks(mod, params, target)
print("Get %d tasks from model %s" % (len(tasks), "BERT"))