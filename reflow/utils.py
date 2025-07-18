import os
join = os.path.join
import torch

import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import save_image, make_grid
import subprocess
import shutil
import logging
import random
import cv2 
import math
import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw






def seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


def restore_checkpoint(ckpt_dir, state, device):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir, exist_ok=True)
        logging.warning(f"No checkpoint found at {ckpt_dir}. "
                        f"Returned the same state as input")
        return state
    else:
        loaded_state = torch.load(ckpt_dir, map_location=device)
        state['optimizer'].load_state_dict(loaded_state['optimizer'])
        state['model'].load_state_dict(loaded_state['model'], strict=False)
        state['ema'].load_state_dict(loaded_state['ema'])
        state['step'] = loaded_state['step']
        return state


def save_checkpoint(ckpt_dir, state):
    saved_state = {
        'optimizer': state['optimizer'].state_dict(),
        'model': state['model'].state_dict(),
        'ema': state['ema'].state_dict(),
        'step': state['step']
    }
    torch.save(saved_state, ckpt_dir)


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def save_image_batch(batch, img_size, sample_path, log_name="examples"):
        sample_grid = make_grid(batch, nrow=int(np.ceil(np.sqrt(batch.shape[0]))), padding=img_size // 16)
        save_image(sample_grid, join(sample_path, log_name))


def update_curve(values, label, x_label, work_path, run_id):
    fig, ax = plt.subplots()
    ax.plot(values, label=label)
    ax.set_xlabel(x_label)
    ax.set_ylabel(label)
    ax.legend()
    plt.savefig(f'{work_path}/{label}_curve_{run_id}.png', dpi=600)
    plt.close()


def get_file_list():
    return [
        b.decode()
        for b in set(
            subprocess.check_output(
                'git ls-files -- ":!:load/*"', shell=True
            ).splitlines()
        )
        | set(  # hard code, TODO: use config to exclude folders or files
            subprocess.check_output(
                "git ls-files --others --exclude-standard", shell=True
            ).splitlines()
        )
    ]


def save_code_snapshot(model_path):
    os.makedirs(model_path, exist_ok=True)
    for f in get_file_list():
        if not os.path.exists(f) or os.path.isdir(f):
            continue
        os.makedirs(os.path.join(model_path, os.path.dirname(f)), exist_ok=True)
        shutil.copyfile(f, os.path.join(model_path, f))

def DarkChannel(im,sz):
    r,g,b = cv2.split(im)
    dc = cv2.min(cv2.min(r,g),b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(sz,sz))
    dark = cv2.erode(dc,kernel)
    return dark

def AtmLight(im,dark):
    [h,w] = im.shape[:2]
    imsz = h*w
    numpx = int(max(math.floor(imsz/1000),1))
    darkvec = dark.reshape(imsz)
    imvec = im.reshape(imsz,3)

    indices = darkvec.argsort()
    indices = indices[imsz-numpx::]

    atmsum = np.zeros([1,3])
    for ind in range(1,numpx):
       atmsum = atmsum + imvec[indices[ind]]

    A = atmsum / numpx
    return A

def TransmissionEstimate(im,A,sz):
    omega = 0.95
    im3 = np.empty(im.shape,im.dtype)

    for ind in range(0,3):
        im3[:,:,ind] = im[:,:,ind]/A[0,ind]

    transmission = 1 - omega*DarkChannel(im3,sz)
    return transmission

def Guidedfilter(im,p,r,eps):
    mean_I = cv2.boxFilter(im,cv2.CV_64F,(r,r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F,(r,r))
    mean_Ip = cv2.boxFilter(im*p,cv2.CV_64F,(r,r))
    cov_Ip = mean_Ip - mean_I*mean_p

    mean_II = cv2.boxFilter(im*im,cv2.CV_64F,(r,r))
    var_I   = mean_II - mean_I*mean_I

    a = cov_Ip/(var_I + eps)
    b = mean_p - a*mean_I

    mean_a = cv2.boxFilter(a,cv2.CV_64F,(r,r))
    mean_b = cv2.boxFilter(b,cv2.CV_64F,(r,r))

    q = mean_a*im + mean_b
    return q

def TransmissionRefine(im,et):
    if im.max() > 100: 
        gray = cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)
        gray = np.float64(gray)/255
    else: 
        gray = cv2.cvtColor((im*255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        gray = np.float64(gray)/255
    r = 100
    eps = 0.0001
    t = Guidedfilter(gray,et,r,eps)

    return t

def get_dcp_t(hazy, return_A=False, A1=False): 
    dark = DarkChannel(hazy, 15)
    A = AtmLight(hazy, dark)
    
    if A1: 
        te = TransmissionEstimate(hazy, np.array([[1.0, 1.0, 1.0]]), 15)
    else: 
        te = TransmissionEstimate(hazy, A, 15)
    t = TransmissionRefine(hazy, te)
    if return_A: 
        return A[0], t
    else:
        return t

def AutoContrast(img, _):
    return PIL.ImageOps.autocontrast(img)

def Blur(img, _):
    kernel_size = int(random.random() * 4.95)
    kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size
    return img.filter(PIL.ImageFilter.GaussianBlur(kernel_size))

def Brightness(img, v):
    assert v >= 0.0
    return PIL.ImageEnhance.Brightness(img).enhance(v)


def Color(img, v):
    assert v >= 0.0
    return PIL.ImageEnhance.Color(img).enhance(v)


def Contrast(img, v):
    assert v >= 0.0
    return PIL.ImageEnhance.Contrast(img).enhance(v)


def Equalize(img, _):
    return PIL.ImageOps.equalize(img)


def Posterize(img, v):  # [4, 8]
    v = int(v)
    v = max(1, v)
    return PIL.ImageOps.posterize(img, v)

def Sharpness(img, v):  # [0.1,1.9]
    assert v >= 0.0
    return PIL.ImageEnhance.Sharpness(img).enhance(v)



def augment_list_no_geometric():
    l = [
        (Contrast, 0.05, 0.95),
        (Brightness, 0.05, 0.95),
        # (Color, 0.05, 0.95),
        (Contrast, 0.25, 0.95),
        # (Equalize, 0, 1),
        (Posterize, 4, 8),
        (Sharpness, 0.05, 0.95),
        (Blur, 0, 1),
    ]
    return l


class RandAugment:
    def __init__(self, n, m, exclude_color_aug=False):
        self.n = n
        self.m = m  # [0, 30] in fixmatch, deprecated.

        self.augment_list = augment_list_no_geometric()

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)
        for op, min_val, max_val in ops:
            val = min_val + float(max_val - min_val) * random.random()
            img = op(img, val)
        return img