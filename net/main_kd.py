import torch, os, sys, torchvision, argparse
import torchvision.transforms as tfs
from metrics import psnr, ssim
from models import *
import time, math
import numpy as np
from torch.autograd import Variable
from torch.backends import cudnn
from torch import optim
import torch, warnings
from torch import nn
import torchvision.utils as vutils
from tqdm import tqdm
from main_kd import train

warnings.filterwarnings("ignore")
from option import opt, model_name, log_dir
from data_utils import *
from torchvision.models import vgg16

################################################################################
KNOWLEDGE_DISTILATION = False  # False means model training is enabled


################################################################################


print("log_dir :", log_dir)
print("model_name:", model_name)


models_ = {
    "ffa": FFA(gps=opt.gps, blocks=opt.blocks),
}
loaders_ = {
    "its_train": ITS_train_loader,
    "its_test": ITS_test_loader,
    # "ots_train": OTS_train_loader,
    # "ots_test": OTS_test_loader,
}
start_time = time.time()
T = opt.steps


def lr_schedule_cosdecay(t, T, init_lr=opt.lr):
    lr = 0.5 * (1 + math.cos(t * math.pi / T)) * init_lr
    return lr


class sampleKDLoss(nn.Module):
    """
    L : Final Loss
    L_r : Reconstruction Loss (L1)
    L_p : perceptual Loss (user defined)
    L_rm : Representation Mimicking Loss (L1)

    L = L_r + lambda_P*L_p + lambda_rm*L_rm
    """

    def __init__(self):
        super(sampleKDLoss, self).__init__()

    def forward(self, student_output, teacher_output, ground_truth, param):
        # sourcery skip: inline-immediately-returned-variable
        L_r = nn.L1Loss(student_output, ground_truth)
        L_p = criterion[1](student_output, ground_truth)
        L_rm = nn.L1Loss(student_output, teacher_output)

        loss = L_r + params["lamdba_p"] * L_p + params["lamdba_rm"] * L_rm
        return loss


def train_kd(
    student, teacher, loader_train, loader_test, optim, loss_func_kd, criterion, params
):
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
        student.load_state_dict(ckp["model"])
        start_step = ckp["step"]
        max_ssim = ckp["max_ssim"]
        max_psnr = ckp["max_psnr"]
        psnrs = ckp["psnrs"]
        ssims = ckp["ssims"]
        print(f"start_step:{start_step} start training ---")
    else:
        print("train from scratch *** ")

    # set student to train mode, and teacher to eval.
    student.train()
    teacher.eval()

    for step in tqdm(range(start_step + 1, opt.steps + 1)):
        lr = opt.lr
        if not opt.no_lr_sche:
            lr = lr_schedule_cosdecay(step, T)
            for param_group in optim.param_groups:
                param_group["lr"] = lr
        x, y = next(iter(loader_train))
        x = x.to(opt.device)
        y = y.to(opt.device)

        # Convert to torch Variables
        x, y = Variable(x), Variable(y)

        ##! Here starts the main portion of knowledge distillation
        ## compute model output, fetch teacher output, and compute KD loss

        # compute student output
        out = student(x)

        # compute teacher output
        with torch.no_grad():
            out_teacher = teacher(x)
            out_teacher.cuda()

        # compute KD loss
        loss = loss _func_kd(out, out_teacher, y, criterion, params)

        # clear previous gradients, compute gradients of all variables wrt loss
        optim.zero_grad()
        loss.backward()

        # performs updates using calculated gradients
        optimizer.step()

        # KD loss logging
        losses.append(loss.item())
        print(
            f"\rtrain loss : {loss.item():.5f}| step :{step}/{opt.steps}|lr :{lr :.7f} |time_used :{(time.time()-start_time)/60 :.1f}",
            end="",
            flush=True,
        )

        if step % opt.eval_step == 0:
            with torch.no_grad():
                ssim_eval, psnr_eval = test(
                    student, loader_test, max_psnr, max_ssim, step
                )

            print(f"\nstep :{step} |ssim:{ssim_eval:.4f}| psnr:{psnr_eval:.4f}")

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
                        "model": student.state_dict(),
                    },
                    opt.model_dir,
                )
                print(
                    f"\n model saved at step :{step}| max_psnr:{max_psnr:.4f}|max_ssim:{max_ssim:.4f}"
                )

    np.save(f"./numpy_files/{model_name}_{opt.steps}_losses.npy", losses)
    np.save(f"./numpy_files/{model_name}_{opt.steps}_ssims.npy", ssims)
    np.save(f"./numpy_files/{model_name}_{opt.steps}_psnrs.npy", psnrs)


def test(net, loader_test, max_psnr, max_ssim, step):
    net.eval()
    torch.cuda.empty_cache()
    ssims = []
    psnrs = []
    # s=True
    for i, (inputs, targets) in enumerate(loader_test):
        inputs = inputs.to(opt.device)
        targets = targets.to(opt.device)
        pred = net(inputs)
        # # print(pred)
        # tfs.ToPILImage()(torch.squeeze(targets.cpu())).save('111.png')
        # vutils.save_image(targets.cpu(),'target.png')
        # vutils.save_image(pred.cpu(),'pred.png')
        ssim1 = ssim(pred, targets).item()
        psnr1 = psnr(pred, targets)
        ssims.append(ssim1)
        psnrs.append(psnr1)
        # if (psnr1>max_psnr or ssim1 > max_ssim) and s :
        # 		ts=vutils.make_grid([torch.squeeze(inputs.cpu()),torch.squeeze(targets.cpu()),torch.squeeze(pred.clamp(0,1).cpu())])
        # 		vutils.save_image(ts,f'samples/{model_name}/{step}_{psnr1:.4}_{ssim1:.4}.png')
        # 		s=False
    return np.mean(ssims), np.mean(psnrs)


if __name__ == "__main__":
    loader_train = loaders_[opt.trainset]
    loader_test = loaders_[opt.testset]

    # load student
    student = models_[opt.net]
    student = student.to(opt.device)

    # load teacher
    # teacher = torch.load()

    if opt.device == "cuda":
        cudnn.benchmark = True
    criterion = [nn.L1Loss().to(opt.device)]
    #! Initialize KD loss here
    loss_func_kd = sampleKDLoss

    vgg_model = vgg16(pretrained=True).features[:16]
    vgg_model = vgg_model.to(opt.device)
    for param in vgg_model.parameters():
        param.requires_grad = False
    criterion.append(PerLoss(vgg_model).to(opt.device))
    optimizer = optim.Adam(
        params=filter(lambda x: x.requires_grad, student.parameters()),
        lr=opt.lr,
        betas=(0.9, 0.999),
        eps=1e-08,
    )
    optimizer.zero_grad()
    params = {"lambda_p": 0.04, "lamdda_rm": 0.1}

    if KNOWLEDGE_DISTILATION == True:
        train_kd(
            student,
            teacher,
            loader_train,
            loader_test,
            optimizer,
            loss_func_kd,
            criterion,
            params,
        )
    else:
        train(
            student,
            loader_train,
            loader_test,
            optimizer,
            loss_func_kd,
            criterion,
            params,
        )
