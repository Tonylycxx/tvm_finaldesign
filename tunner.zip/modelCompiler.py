import tvm
from tvm import relay, auto_scheduler
from tvm.contrib import graph_executor

def run_compile(mod, params, target, log_file=None):

    if log_file is None:
        #Compile model without tune history
        print("Compile untuned model")
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(mod, target=target, params=params)
    else:
        # Compile model with history best
        print("Compile model with auto-scheduler history best by %s" % log_file)
        with auto_scheduler.ApplyHistoryBest(log_file):
            with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
                lib = relay.build(mod, target=target, params=params)

    dev = tvm.device(str(target), 0)
    module = graph_executor.GraphModule(lib["default"](dev))
    return module

