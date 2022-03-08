import tvm
from tvm.contrib.download import download_testdata

def download_model(model_url, model_name, model_frame):
    model_path = download_testdata(
        model_url,
        model_name,
        model_frame
    )
    return model_path