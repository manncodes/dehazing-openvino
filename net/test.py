import time
import os, argparse
import numpy as np
from PIL import Image
from models import *
import torch
import torch.nn as nn
import torchvision.transforms as tfs
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


torch.cuda.empty_cache()

abs = os.getcwd() + "/"


def tensorShow(tensors, titles=["haze"]):
    fig = plt.figure()
    for tensor, tit, i in zip(tensors, titles, range(len(tensors))):
        img = make_grid(tensor)
        npimg = img.numpy()
        ax = fig.add_subplot(221 + i)
        ax.imshow(np.transpose(npimg, (1, 2, 0)))
        ax.set_title(tit)
    plt.show()


parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="its", help="its or ots")
parser.add_argument("--test_imgs", type=str, default="test", help="Test imgs folder")
opt = parser.parse_args()
dataset = opt.task
gps = 3
blocks = 9
img_dir = abs + opt.test_imgs + "/"
output_dir = abs + f"pred_FFA_{dataset}/"
print("pred_dir:", output_dir)
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
model_dir = abs + f"trained_models/{dataset}_train_ffa_{gps}_{blocks}.pk"
# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
ckp = torch.load(model_dir, map_location=device)
net = FFA(gps=gps, blocks=blocks).to(device=device)
net.load_state_dict(ckp["model"])
net.eval()

times = []
for im in os.listdir(img_dir)[0:1]:
    start_time = time.time()
    haze = Image.open(img_dir + im)
    haze1 = tfs.Compose([tfs.ToTensor()])(haze)[None, ::].to(device=device)
    print(f"{im} | {haze1.shape}")
    compose_time = time.time()
    with torch.no_grad():
        pred = net(haze1)
    ts = torch.squeeze(pred.clamp(0, 1).cpu())
    pred_time = time.time()
    # tensorShow([haze_no, pred.clamp(0, 1).cpu()], ["haze", "pred"])
    vutils.save_image(ts, output_dir + im.split(".")[0] + "_FFA.png")
    print(
        f"compose time:{compose_time - start_time} | inference time : {pred_time - compose_time} | total time : {pred_time - compose_time}"
    )
    times.append(pred_time - compose_time)

print("average time:", np.mean(times), " s")
