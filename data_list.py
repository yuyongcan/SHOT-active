import torch
from torchvision import transforms,utils
import numpy as np
import random
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import os.path
import cv2
import torchvision
from randaugment import RandAugmentMC
from utils import get_embedding

def make_dataset(image_list, labels=None, source=False):
    if source:
        if len(image_list[0].split()) > 2:
          images = [(val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
        else:
          images = [(val.split()[0], int(val.split()[1])) for val in image_list]
        return images
    image_list = open(image_list).readlines()
    if labels:
        len_ = len(image_list)
        images = [(image_list[i].strip(), labels[i, :]) for i in range(len_)]
    else:
        if len(image_list[0].split()) > 2:
            images = [(val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
        else:
            images = [(val.split()[0], int(val.split()[1])) for val in image_list]
    return images

def rgb_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def l_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')

class ImageList(Dataset):
    def __init__(self, image_path, labels=None, transform=None, target_transform=None, mode='RGB', input_list = False):
        if input_list:
            imgs = image_path
        else:
            imgs = make_dataset(image_path, labels, source =True)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)

class ImageList_idx(Dataset):
    def __init__(self, image_path, labels=None, transform=None, target_transform=None, mode='RGB', input_list = False):
        if input_list:
            imgs = image_path
        else:
            imgs = make_dataset(image_path, labels)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        
        if self.transform is not None:
            if type(self.transform).__name__ == "list":
                img = [t(img) for t in self.transform]
            else:
                img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return len(self.imgs)

def image_train(resize_size=256, crop_size=224, alexnet=False):
  if not alexnet:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
  else:
    normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
  return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

def image_test(resize_size=256, crop_size=224, alexnet=False):
  if not alexnet:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
  else:
    normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
  return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])

def data_load(args): 
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    
    dsets["target"] = ImageList_idx(args.t_dset_path, transform=image_train())
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=False, pin_memory=True)
    dsets["test"] = ImageList_idx(args.test_dset_path, transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs*3, shuffle=False, num_workers=args.worker, drop_last=False, pin_memory=True)

    return dset_loaders

def data_load_active(args, l_list, u_list):
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    strong_aug = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        RandAugmentMC(n=2, m=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
    ])
    
    dsets["l"] = ImageList_idx(l_list, transform=[image_train(), strong_aug], input_list=True)
    dsets["u"] = ImageList_idx(u_list, transform=[image_train(), strong_aug], input_list=True)
    dsets["test"] = ImageList_idx(l_list + u_list, transform=image_test(), input_list=True)
    dsets["u_test"] = ImageList_idx(u_list, transform=image_test(), input_list=True)

    dset_loaders["l"] = DataLoader(dsets["l"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=True, pin_memory=True)
    dset_loaders["u"] = DataLoader(dsets["u"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=True, pin_memory=True)
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=False, pin_memory=True)
    dset_loaders["u_test"] = DataLoader(dsets["u_test"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=False, pin_memory=True)


    return dset_loaders

def data_load_Q(args, list=None, shuff=False, drop=False, bs=None, aug=False):
    strong_aug = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        RandAugmentMC(n=2, m=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
    ])
    if aug:
        trans = [image_test(), strong_aug]
    else:
        trans = image_test()
    target_set = ImageList_idx(list, transform=trans, input_list=True)

    dset_loaders = {}
    dset_loaders["Q"] = DataLoader(target_set, batch_size=128, shuffle=shuff, num_workers=args.worker, drop_last=False)

    return dset_loaders

def data_load_aug(args, l_list, u_list):
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    
    dsets["l"] = ImageList_idx(l_list, transform=image_train() , input_list=True)
    dsets["u"] = ImageList_idx(u_list, transform=[image_train(), image_train()], input_list=True)
    dsets["test"] = ImageList_idx(l_list + u_list, transform=image_test(), input_list=True)
    dsets["u_test"] = ImageList_idx(u_list, transform=image_test(), input_list=True)

    dset_loaders["l"] = DataLoader(dsets["l"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=True, pin_memory=True)
    dset_loaders["u"] = DataLoader(dsets["u"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=True, pin_memory=True)
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=False, pin_memory=True)
    dset_loaders["u_test"] = DataLoader(dsets["u_test"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=False, pin_memory=True)


    return dset_loaders


@torch.no_grad()
def data_split(args, sp_list, sp_idx, netF, netB, netC):
    netF.eval()
    netB.eval()
    netC.eval()
    split_loaders = data_load_Q(args, sp_list)['Q']
    NUMS = args.easynum  # easy samples of each class

    if(args.skip_split):
        filename_e = './data/{}/easy_{}{}_{}_list.txt'.format(args.dset, args.s, args.t, NUMS) 
        filename_h = './data/{}/hard_{}{}_{}_list.txt'.format(args.dset, args.s, args.t, NUMS)
        easy_path = utils.make_dataset('', filename_e)
        hard_path = utils.make_dataset('', filename_h)
        print('load txt from ' + filename_e + ' and ' + filename_h )             
        args.out_file.write('load txt from ' + filename_e + 'and' + filename_h  + '\n')             
        args.out_file.flush()
    else:
        easy_path, hard_path, easy_idx, hard_idx = [], [], [], []

        # base_network.eval()
        """ the full (path, label) list """
        img = sp_list
        img_idx = sp_idx

        # with torch.no_grad():
        """ extract the prototypes """
        for name, param in netC.named_parameters():
            if('fc.weight' in name):
                prototype = param

        _, features_bank = get_embedding(args, netF, netB, netC, split_loaders)
        # features_bank = F.normalize(features_bank)
        # prototype = F.normalize(prototype)
        dists = prototype.mm(features_bank.t())  # 31*len

        sort_idxs = torch.argsort(dists, dim=1, descending=True)
        fault = 0.
        # count = 0

        aux_idx = []
        for i in range(args.class_num):
            ## check if the repeated index in the list
            s_idx = 0
            for _ in range(NUMS):
                idx = sort_idxs[i, s_idx]

                while idx in aux_idx:
                    s_idx += 1
                    idx = sort_idxs[i, s_idx]

                assert idx not in aux_idx

                if not img[idx][1] == i:
                    fault += 1
                    if args.test4: # 不选错的
                        s_idx += 1
                        continue

                aux_idx.append(idx)
                easy_path.append((img[idx][0], i))
                easy_idx.append(sp_idx[idx])

                s_idx += 1

        for id in range(len(img)):
            if id not in aux_idx:
                hard_path.append(img[id])
                hard_idx.append(sp_idx[id])

        """mindist: a distance matrix which store the minimum cosine distance between the prototypes and features """
        # print(len(sp_list), len(easy_path), len(hard_path))
        # print(min(sp_idx), min(easy_idx), min(hard_idx), max(sp_idx), max(easy_idx), max(hard_idx),)
        acc = 1 - fault / (args.class_num*NUMS)

        print('Splited data list label Acc:{}'.format(acc))
        args.out_file.write('Splited data list label Acc:{}'.format(acc) + '\n')
        args.out_file.flush()
        
        # exit(0)
        # filename_e = './data/{}/easy_{}{}_{}_list.txt'.format(args.dset, args.s, args.t, NUMS)
        # filename_h = './data/{}/hard_{}{}_{}_list.txt'.format(args.dset, args.s, args.t, NUMS)
        # list2txt(easy_path, filename_e)
        # list2txt(hard_path, filename_h)
        # print('Splited data list saved in ' + filename_e + ' and ' + filename_h )
        # args.out_file.write('Splited data list saved in ' + filename_e + 'and' + filename_h  + '\n')
        # args.out_file.flush()
        # exit(0)

    return  easy_path, easy_idx, hard_path, hard_idx
