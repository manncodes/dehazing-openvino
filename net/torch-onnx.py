import argparse
import torch
from torch.cuda import device_count
import torch.nn as nn
from models import *

device = "cuda" if torch.cuda.is_available() else "cpu"

# program to convert torch models to onnx
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="trained_models/its_train_ffa_3_9.pk",
        help="path to the onnx model",
    )
    opt = parser.parse_args()

    dummy_input = torch.randn(1, 3, 240, 240).to(
        device=device
    )  # input to the first layer of the model
    model_name = opt.model_path.split("/")[-1].split(".")[0]

    model = FFA(gps=3, blocks=9).to(device)
    checkpoint = torch.load(opt.model_path)
    model.load_state_dict(checkpoint["model"])
    # converting to onnx
    torch.onnx.export(
        model,
        dummy_input,
        "trained_models/onnx/" + model_name + ".onnx",
        opset_version=11,
        do_constant_folding=False,
    )
