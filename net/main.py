import torch, os, sys, torchvision, argparse
import torchvision.transforms as tfs
from metrics import psnr, ssim
from models import *
import time, math
import numpy as np
from torch.backends import cudnn
from torch import optim
import torch, warnings
from torch import nn
from tqdm import tqdm
import wandb


# from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils

warnings.filterwarnings("ignore")
from option import opt, model_name, log_dir
from data_utils import *
from torchvision.models import vgg16

print("log_dir :", log_dir)
print("model_name:", model_name)

models_ = {
    "ffa": FFA(gps=opt.gps, blocks=opt.blocks),
    "InvoFFA": InvoFFA(gps=opt.gps, blocks=opt.blocks),
}
loaders_ = {
    "its_train": ITS_train_loader,
    "its_test": ITS_test_loader,
    # 'ots_train':OTS_train_loader,
    # 'ots_test':OTS_test_loader
}
start_time = time.time()
T = opt.steps


def lr_schedule_cosdecay(t, T, init_lr=opt.lr):
    return 0.5 * (1 + math.cos(t * math.pi / T)) * init_lr


def train(net, loader_train, optim, criterion):  # sourcery no-metrics
    losses = []
    start_step = 0
    max_ssim = 0
    max_psnr = 0
    ssims = []
    psnrs = []
    if opt.resume and os.path.exists(opt.model_dir):
        print(f"resume from {opt.model_dir}")
        ckp = torch.load(opt.model_dir)
        losses = ckp["losses"]
        net.load_state_dict(ckp["model"])
        start_step = ckp["step"]
        max_ssim = ckp["max_ssim"]
        max_psnr = ckp["max_psnr"]
        psnrs = ckp["psnrs"]
        ssims = ckp["ssims"]
        print(f"start_step:{start_step} start training ---")
    else:
        print("train from scratch *** ")

    curr_psnr = 0.0

    t = tqdm(range(start_step + 1, opt.steps + 1), leave=True)
    for step in t:
        net.train()
        lr = opt.lr
        if not opt.no_lr_sche:
            lr = lr_schedule_cosdecay(step, T)
            for param_group in optim.param_groups:
                param_group["lr"] = lr
        x, y = next(iter(loader_train))
        x = x.to(opt.device)
        y = y.to(opt.device)
        # print(f"X shape:{x.shape} | Y shape: {y.shape}")
        out = net(x)
        loss = criterion[0](out, y)
        if opt.perloss:
            loss2 = criterion[1](out, y)
            loss = loss + 0.04 * loss2

        loss.backward()

        optim.step()
        optim.zero_grad()
        losses.append(loss.item())
        t.set_description(f"loss:{loss.item():.4} | lr:{lr:.4} | psnr:{curr_psnr:.2}")

        # with SummaryWriter(log_dir=log_dir, comment=log_dir) as writer:
        #     writer.add_scalar("data/loss", loss, step)

        if step % opt.eval_step == 0:
            with torch.no_grad():
                ssim_eval, psnr_eval = test(net, x, y, max_psnr, max_ssim, step)

            curr_psnr = psnr_eval
            # print(f"\nstep :{step} |ssim:{ssim_eval:.4f}| psnr:{psnr_eval:.4f}")

            # with SummaryWriter(logdir=log_dir, comment=log_dir) as writer:
            #     writer.add_scalar("data/ssim", ssim_eval, step)
            #     writer.add_scalar("data/psnr", psnr_eval, step)
            #     writer.add_scalars(
            #         "group", {"ssim": ssim_eval, "psnr": psnr_eval, "loss": loss}, step
            #     )
            ssims.append(ssim_eval)
            psnrs.append(psnr_eval)
            if ssim_eval > max_ssim and psnr_eval > max_psnr:
                max_ssim = max(max_ssim, ssim_eval)
                max_psnr = max(max_psnr, psnr_eval)
                torch.save(
                    {
                        "step": step,
                        "max_psnr": max_psnr,
                        "max_ssim": max_ssim,
                        "ssims": ssims,
                        "psnrs": psnrs,
                        "losses": losses,
                        "model": net.state_dict(),
                    },
                    opt.model_dir,
                )
                print(
                    f"\n model saved at step :{step}| max_psnr:{max_psnr:.4f}|max_ssim:{max_ssim:.4f}"
                )

    print(f"Running avg of psnr : {np.mean(psnrs):.4f}")
    np.save(f"./numpy_files/{model_name}_{opt.steps}_losses.npy", losses)
    np.save(f"./numpy_files/{model_name}_{opt.steps}_ssims.npy", ssims)
    np.save(f"./numpy_files/{model_name}_{opt.steps}_psnrs.npy", psnrs)


def test(net, inputs, targets, max_psnr, max_ssim, step):
    net.eval()
    ssims = []
    psnrs = []
    # s=True

    inputs = inputs.to(opt.device)
    targets = targets.to(opt.device)
    with torch.no_grad():
        pred = net(inputs)
    # print(pred)

    vutils.save_image(inputs.cpu(), "input.png")
    vutils.save_image(targets.cpu(), "target.png")
    vutils.save_image(pred.cpu(), "pred.png")
    ssim1 = ssim(pred, targets).item()
    psnr1 = psnr(pred, targets)
    ssims.append(ssim1)
    psnrs.append(psnr1)
    if psnr1 > max_psnr:
        ts = vutils.make_grid(
            [
                torch.squeeze(inputs.cpu()),
                torch.squeeze(targets.cpu()),
                torch.squeeze(pred.clamp(0, 1).cpu()),
            ]
        )
        vutils.save_image(ts, f"samples/{model_name}/{step}_{psnr1:.4}_{ssim1:.4}.png")
        s = False
    return np.mean(ssims), np.mean(psnrs)


if __name__ == "__main__":
    BATCH_SIZE = opt.bs
    print(f"{BATCH_SIZE = }")

    pwd = os.getcwd()
    print(pwd)
    path = "../data"  # path to your 'data' folder

    loader_train = DataLoader(
        dataset=RESIDE_Dataset(path + "/RESIDE/ITS", train=True, size=crop_size),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    net = models_[opt.net]
    net = net.to(opt.device)
    if opt.device == "cuda":
        cudnn.benchmark = True
    criterion = []
    criterion.append(nn.L1Loss().to(opt.device))
    if opt.perloss:
        vgg_model = vgg16(pretrained=True).features[:16]
        vgg_model = vgg_model.to(opt.device)
        for param in vgg_model.parameters():
            param.requires_grad = False
        criterion.append(PerLoss(vgg_model).to(opt.device))
    optimizer = optim.Adam(
        params=filter(lambda x: x.requires_grad, net.parameters()),
        lr=opt.lr,
        betas=(0.9, 0.999),
        eps=1e-08,
    )
    optimizer.zero_grad()
    print("---------------------------------")
    train(net, loader_train, optimizer, criterion)
