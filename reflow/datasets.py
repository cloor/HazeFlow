import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision
from PIL import Image
import os
import logging
import torchvision.transforms.functional as F
import torchvision.transforms as transforms 
import cv2 
import random
from scipy.linalg import orth
from typing import Any
from reflow.utils import get_dcp_t, RandAugment
from reflow.transform import augment, paired_random_crop, quadra_random_crop, random_crop_pair, triple_random_crop

def add_Gaussian_noise(img, noise_level1=2, noise_level2=25):
    noise_level = random.randint(noise_level1, noise_level2)
    rnum = np.random.rand()
    if rnum > 0.6:   # add color Gaussian noise
        img += np.random.normal(0, noise_level/255.0, img.shape).astype(np.float32)
    elif rnum < 0.4: # add grayscale Gaussian noise
        img += np.random.normal(0, noise_level/255.0, (*img.shape[:2], 1)).astype(np.float32)
    else:            # add  noise
        L = noise_level2/255.
        D = np.diag(np.random.rand(3))
        U = orth(np.random.rand(3,3))
        conv = np.dot(np.dot(np.transpose(U), D), U)
        img += np.random.multivariate_normal([0,0,0], np.abs(L**2*conv), img.shape[:2]).astype(np.float32)
    img = np.clip(img, 0.0, 1.0)
    return img

def random_resize(img, scale_factor=1.):
    return cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

def uint2single(img):
    return np.float32(img/255.)

def single2uint(img):
    return np.uint8((img.clip(0, 1)*255.).round())

def add_JPEG_noise(img):
    quality_factor = random.randint(30, 95)
    img = cv2.cvtColor(single2uint(img), cv2.COLOR_RGB2BGR)
    result, encimg = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality_factor])
    img = cv2.imdecode(encimg, 1)
    img = cv2.cvtColor(uint2single(img), cv2.COLOR_BGR2RGB)
    return img

def to_tensor(img): 
    if img.shape[2] == 3 : 
        img = img[:,:,::-1].copy()
    return torch.from_numpy(img).permute(2,0,1).float()
    
def img2tensor(imgs, bgr2rgb=True, float32=True):
    """Numpy array to tensor.

    Args:
        imgs (list[ndarray] | ndarray): Input images.
        bgr2rgb (bool): Whether to change bgr to rgb.
        float32 (bool): Whether to change to float32.

    Returns:
        list[tensor] | tensor: Tensor images. If returned results only have
            one element, just return tensor.
    """

    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            if img.dtype == 'float64':
                img = img.astype('float32')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)
    
join = os.path.join

extensions = ['.jpg', '.jpeg', '.JPEG', '.png', '.bmp']

def get_data_scaler(config):
  """Data normalizer. Assume data are always in [0, 1]."""
  if config.data.centered:
    # Rescale to [-1, 1]
    return lambda x: x * 2. - 1.
  else:
    return lambda x: x


def get_data_inverse_scaler(config):
  """Inverse data normalizer."""
  if config.data.centered:
    # Rescale [-1, 1] to [0, 1]
    return lambda x: (x + 1.) / 2.
  else:
    return lambda x: x


def get_dataset(config, phase="train"):
    if config.data.dataset=='mcbm':
        dataset = MCBM(config.data.custom_data_root)   
    elif config.data.dataset=='reflow':
        dataset = REFLOW(config.data.custom_data_root)      
    elif config.data.dataset=='distill': 
        dataset = DISTILL(data_root='datasets/URHI/')
    elif config.data.dataset=='reals':
        dataset = Reals(data_root='./datasets/')
        
    else:
        # TODO: add other datasets
        raise NotImplementedError
    dataloader = torch.utils.data.DataLoader(dataset,
                                                batch_size=config.training.batch_size,
                                                shuffle=True,
                                                num_workers=4,
                                                pin_memory=True)
    return dataloader

def get_test_dataloader(config=None, phase='test', dataset_name='example'): 
    # dataset = ImageTranslationDataset(config.data.custom_data_root, phase='test')
    if dataset_name.lower()=='urhi': 
        dataset = URHI('datasets/URHI/', phase='test')
    else:
        dataset = Example(config.data.test_data_root)
    dataloader = torch.utils.data.DataLoader(dataset, 
                                             batch_size = 1,
                                             shuffle = False,
                                             num_workers = 0)
    return dataloader

def get_rtts_dataloader(config, phase='test', depth = False): 
    dataset = RTTS(config.data.test_data_root, depth = depth)
    dataloader = torch.utils.data.DataLoader(dataset, 
                                             batch_size=1, 
                                             shuffle=False,
                                             num_workers=0, 
                                             )
    return dataloader

def get_real_dataloader(config, **kwargs: Any,): 
    dataset = Reals(config.data.test_data_root)
    dataloader = torch.utils.data.DataLoader(dataset, 
                                             batch_size=1, 
                                             shuffle=False,
                                             num_workers=0, 
                                             )
    return dataloader
    
class Example(Dataset): 
    def __init__(self, data_root, Fattal=None): 
        self.data_root = data_root
        
        self.imgs = os.listdir(self.data_root)
        
        self.img_list = [os.path.join(self.data_root, x) for x in self.imgs]

            
    def __len__(self): 
        return len(self.imgs)
    
    def __getitem__(self, index):
        name = self.img_list[index].split('/')[-1]
        img = cv2.imread(self.img_list[index]).astype(np.float32) / 255.0
        dcp = get_dcp_t(img[:,:,::-1], return_A=False, A1=True)[:,:,None]
        
       
        img = img2tensor(img, bgr2rgb=True, float32=True)
        dcp = to_tensor(dcp)

        
        return {'hazy':img, 
                'dcp':dcp, 
                'name':name}
    
class MCBM(Dataset): 
    def __init__(self, data_root, crop_size=256,nh=False):
        self.root_dir = data_root
        self.crop_size = crop_size
        self.nh = nh # MCBM
        self.img_names = [x for x in sorted(os.listdir(os.path.join(self.root_dir, 'rgb_500'))) if x.endswith('jpg')]
        self.depth_names = [x for x in sorted(os.listdir(os.path.join(self.root_dir, 'da_depth_500'))) if x.endswith('npy')]
        self.nh_names = [x for x in os.listdir(os.path.join(self.root_dir, 'MCBM')) if x.endswith('png')]
        
        self.img_list = [os.path.join(self.root_dir, 'rgb_500', x) for x in self.img_names]
        self.depth_list = [os.path.join(self.root_dir, 'da_depth_500', x) for x in self.depth_names]
        self.nh_list = [os.path.join(self.root_dir, 'MCBM', x) for x in self.nh_names]
        
        self.A_range = [0.25, 1.8]
        self.beta_range = [0.2, 2.8]
        self.color_p = 1.0
        self.color_range = [-0.025, 0.025]
                    
    def __getitem__(self, index):
        img_gt = cv2.imread(self.img_list[index])[:,:,::-1] / 255.0
        depth = np.array(np.load(self.depth_list[index]))
        depth = -(depth - depth.max())
        depth = (depth - depth.min()) / (depth.max() - depth.min())
        rand_idx = random.randint(0, 999)
        nh = cv2.imread(self.nh_list[rand_idx], 0) / 255.0
        nh = (nh-nh.min())/(nh.max() - nh.min())

        beta = np.random.rand(1) * (self.beta_range[1] - self.beta_range[0]) + self.beta_range[0]

        ## Non Honmogeneous haze (MCBM)
        if self.nh:
            beta = nh * (np.random.rand()+1)/2 + np.ones(nh.shape) * beta

        A = np.random.rand(1) * (self.A_range[1] - self.A_range[0]) + self.A_range[0]
        if np.random.rand(1) < self.color_p:
            A_random = np.random.rand(3) * (self.color_range[1] - self.color_range[0]) + self.color_range[0]
            A = A + A_random

        img_lq = img_gt.copy()
        # adjust luminance
        if np.random.rand(1) < 0.5:
            img_lq = np.power(img_lq, np.random.rand(1) * 1.5 + 1.5)
        # add gaussian noise
        if np.random.rand(1) < 0.5:
            img_lq = add_Gaussian_noise(img_lq)
        
        input_gt_size = np.min(img_gt.shape[:2])
        input_lq_size = np.min(img_lq.shape[:2])
        scale = input_gt_size // input_lq_size
        gt_size = self.crop_size   
                                
        # random resize
        if input_gt_size > gt_size:
            input_gt_random_size = random.randint(gt_size, input_gt_size)
            input_gt_random_size = input_gt_random_size - input_gt_random_size % scale # make sure divisible by scale 
            resize_factor = input_gt_random_size / input_gt_size
        else:
            resize_factor = (gt_size+1) / input_gt_size
        img_gt = random_resize(img_gt, resize_factor)
        depth = random_resize(depth,resize_factor)
        img_lq = random_resize(img_lq, resize_factor)
        
        
        input_gt_size = np.min(img_gt.shape[:2])
        input_lq_size = np.min(img_lq.shape[:2])
        scale = input_gt_size // input_lq_size
        gt_size = self.crop_size

        # random crop
        img_gt,img_lq, depth = triple_random_crop(img_gt,img_lq, depth, gt_size, input_gt_size // input_lq_size)
        # flip, rotation
        img_gt,img_lq, depth = augment([img_gt,img_lq, depth ], True, False)
        
        t = np.exp( - depth * 2.0 * beta)
        t = t[:, :, np.newaxis]
        # add haze
        img_lq = img_lq * t + A * (1 - t)

        # add JPEG noise
        if np.random.rand(1) < 0.5:
            img_lq = add_JPEG_noise(img_lq)
        
        dcp_t = get_dcp_t(img_lq[:,:,::-1], return_A=False, A1=True)[:,:,None]
        
        gt, hazy = img2tensor([img_gt, img_lq], True, True)

        dcp_t = to_tensor(dcp_t)
        t = to_tensor(t)
        A = torch.ones((3,1,1)).float()

        
        
        
        return {'hazy': hazy, 
                'gt': gt, 
                'A':A, 
                't':dcp_t,
                'gt_t':t,
                }
    
    def __len__(self): 
        return len(self.img_names)
    
class RTTS(Dataset): 
    def __init__(self, data_root, depth=False): 
        self.depth = depth
        
        self.hazy_root = os.path.join(data_root, 'JPEGImages')
        # self.dcp_root = os.path.join(data_root, 'JPEGImages_dcp')
        if depth: 
            self.depth_root = os.path.join(data_root, 'JPEGImages_depth')
            
        
        
        self.imgs = sorted(os.listdir(self.hazy_root))
        # self.dcps = sorted(os.listdir(self.dcp_root)) 
        if depth: 
            self.depths = sorted(os.listdir(self.depth_root))
        
        self.img_list = [os.path.join(self.hazy_root, x) for x in self.imgs]
        # self.dcp_list = [os.path.join(self.dcp_root, x) for x in self.dcps]
        if depth: 
            self.depth_list = [os.path.join(self.depth_root, x) for x in self.depths]
        
        self.mean = [0.5, 0.5, 0.5]
        self.std = [0.5, 0.5, 0.5]        
    
    def __len__(self): 
        return len(self.imgs)
    
    def __getitem__(self, index): 
        name = self.img_list[index].split('/')[-1]
        
        img = cv2.imread(self.img_list[index]).astype(np.float32) / 255.0
        # dcp = cv2.imread(self.dcp_list[index], 0).astype(np.float32) / 255.0
        dcp = get_dcp_t(img[:,:,::-1], return_A=False, A1=True)[:,:,None]
        img = img2tensor(img)
        dcp = to_tensor(dcp)
        # dcp = to_tensor(dcp[:,:,None])
        
        return {'hazy': img, 
            'dcp': dcp, 
            'name': name}
        
class URHI(Dataset): 
    def __init__(self, data_root): 
        self.hazy_root = os.path.join(data_root, 'hazy')
        # self.dcp_root = os.path.join(data_root, 'dcp')
        
        
        self.imgs = sorted(os.listdir(self.hazy_root))
        # self.dcps = sorted(os.listdir(self.dcp_root)) 
        
        self.img_list = [os.path.join(self.hazy_root, x) for x in self.imgs]
        # self.dcp_list = [os.path.join(self.dcp_root, x) for x in self.dcps]
                
        self.crop_size = 256       

    def __len__(self): 
        return len(self.imgs)
    
    def __getitem__(self, index): 
        
        img_lq = cv2.imread(self.img_list[index]).astype(np.float32) / 255.0
        name = self.img_list[index].split('/')[-1]
        dcp_t = get_dcp_t(img_lq[:,:,::-1], return_A=False, A1=True)[:,:,None]
        img_lq = img2tensor(img_lq, True, True)
        dcp_t = to_tensor(dcp_t)
        return {'hazy':img_lq, 
                'dcp':dcp_t,
                'name':name}
                   
class DISTILL(Dataset): 
    def __init__(self, data_root): 
        self.hazy_root = os.path.join(data_root, 'hazy')
        
        self.imgs = sorted(os.listdir(self.hazy_root))
        
        self.img_list = [os.path.join(self.hazy_root, x) for x in self.imgs]
                
        self.crop_size = 256     

        self.strong_aug = torchvision.transforms.Compose([
            # TO PIL
            torchvision.transforms.ToPILImage(),
            RandAugment(2, 10),
            torchvision.transforms.ToTensor()
        ])
    def __len__(self): 
        return len(self.imgs)
    
    def __getitem__(self, index): 
        
        img_gt = cv2.imread(self.img_list[index]).astype(np.float32) / 255.0
        img_gt = img2tensor(img_gt, True, True)
        
        img_lq = self.strong_aug(img_gt)
        
        
        
        # Random crop
        if img_gt.shape[1] < self.crop_size or img_gt.shape[2] < self.crop_size:
            img_gt = F.resize(img_gt, (self.crop_size, self.crop_size), antialias=True)
            img_lq = F.resize(img_lq, (self.crop_size, self.crop_size), antialias=True)
    
        i, j, h, w = torchvision.transforms.RandomCrop.get_params(
            img_gt, output_size=(self.crop_size, self.crop_size))
        img_gt = F.crop(img_gt, i, j, h, w)
        img_lq = F.crop(img_lq, i, j, h, w)
        
        # random flip 
        if random.random() > 0.5:
            img_gt = F.hflip(img_gt)
            img_lq = F.hflip(img_lq)
        
        gt_t = to_tensor(get_dcp_t(img_gt.permute(1,2,0).numpy(), A1=True)[:,:,None])
        lq_t = to_tensor(get_dcp_t(img_lq.permute(1,2,0).numpy(), A1=True)[:,:,None])
        
        A = torch.ones((3,1,1)).float()
        
        return {'hazy': img_lq,
                'gt': img_gt, 
                't': lq_t, 
                'gt_t': gt_t,
                'A':A }

class REFLOW(Dataset): 
    def __init__(self, data_root, crop_size=256):
        self.root_dir = data_root
        self.crop_size = crop_size

        self.hazy_names = [x for x in sorted(os.listdir(os.path.join(self.root_dir, 'hazy'))) if x.endswith('.png') or x.endswith('.jpg') or x.endswith('.JPG')]

        self.hazy_list = [os.path.join(self.root_dir, 'hazy', x) for x in self.hazy_names]
        
        self.A_range = [0.25, 1.8]
        self.beta_range = [0.2, 2.8]
        self.color_p = 1.0
        self.color_range = [-0.025, 0.025]
                    
    def __getitem__(self, index):

        img_gt = cv2.imread(self.hazy_list[index]).astype(np.float32) / 255.0
                

        input_gt_size = np.min(img_gt.shape[:2])
        input_lq_size = np.min(img_gt.shape[:2])
        scale = input_gt_size // input_lq_size
        gt_size = self.crop_size   
                                
        # random resize
        if input_gt_size > gt_size:
            input_gt_random_size = random.randint(gt_size, input_gt_size)
            input_gt_random_size = input_gt_random_size - input_gt_random_size % scale # make sure divisible by scale 
            resize_factor = input_gt_random_size / input_gt_size
        else:
            resize_factor = (gt_size+1) / input_gt_size
        img_gt = random_resize(img_gt, resize_factor)
        
        # random crop
        img_gt, _= paired_random_crop(img_gt, img_gt, gt_size, input_gt_size // input_lq_size)
        # flip, rotation
        img_gt, _ = augment([img_gt, img_gt], True, False)
        
        # dcp_t = get_dcp_t(img_lq[:,:,::-1], return_A=False, A1=True)[:,:,None]
        t = get_dcp_t(img_gt[:,:,::-1], return_A=False, A1=True)[:,:,None]
        
        img_gt = img2tensor(img_gt, True, True)
        t = to_tensor(t)
        A = torch.ones((3,1,1)).float()
        
        return {'hazy': img_gt, 
                # 'gt': gt, 
                'A':A, 
                't':t,
                # 'gt_t':gt_t
                }
    
    def __len__(self): 
        return len(self.hazy_names)

class Reals(Dataset): 
    def __init__(self, data_root): 
        self.dataset_list = ['Fattal','O_haze','I_haze','NH_haze','Dense_haze']
        self.img_list = []
        for dataset_name in self.dataset_list:
            
            self.data_root = os.path.join(data_root, dataset_name,'hazy')
        
            self.imgs = os.listdir(self.data_root)
        
            self.imgs = [os.path.join(self.data_root, x) for x in self.imgs]
            self.img_list.extend(self.imgs)

            
    def __len__(self): 
        return len(self.img_list)
    
    def __getitem__(self, index):
        name = os.path.join(self.img_list[index].split('/')[-3],self.img_list[index].split('/')[-1])
        img = cv2.imread(self.img_list[index]).astype(np.float32) / 255.0
        dcp = get_dcp_t(img[:,:,::-1], return_A=False, A1=True)[:,:,None]
        

        img = img2tensor(img, bgr2rgb=True, float32=True)
        dcp = to_tensor(dcp)

        
        return {'hazy':img, 
                'dcp':dcp, 
                'name':name}