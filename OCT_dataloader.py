# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 15:58:40 2021

@author: hudew
"""

from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import torch
import numpy as np
import os
import sys
import glob
import cv2
# %%
from scipy.ndimage import zoom
from transforms import AffineTransformation, GrayscaleAugmentation, ZoomTransformation, FlipTransformation, FlipZTransformation


class TopconDataset(Dataset):

    def __init__(self, non_glaucoma_dir, glaucoma_dir, data_size, att_type, dataset_type='train', augment=False, weighting='all', seed=100, aug_type='fgz', ONH_only=False):
        super().__init__()

        self.dataset_type = dataset_type
        self.data_size = data_size
        # must use 128 slices
        assert self.data_size[0] == 128
        self.augment = augment

        self.weighting = weighting
        self.aug_type = aug_type
        self.att_type = att_type

        self.non_glaucoma_dir = glob.glob(
            non_glaucoma_dir + '/**/*.npy', recursive=True)
        self.glaucoma_dir = glob.glob(
            glaucoma_dir + '/**/*.npy', recursive=True)

        # randomly select samples from non_glaucoma list
        self.seed = seed
        gen = np.random.default_rng(seed=self.seed)
        gen.shuffle(self.non_glaucoma_dir)
        gen.shuffle(self.glaucoma_dir)

        if self.weighting == 'same':
            self.non_glaucoma_dir = self.non_glaucoma_dir[:len(
                self.glaucoma_dir)]
        elif self.weighting == '4x':
            self.non_glaucoma_dir = self.non_glaucoma_dir[:len(
                self.glaucoma_dir)*4]
        elif self.weighting == '2x':
            self.non_glaucoma_dir = self.non_glaucoma_dir[:len(
                self.glaucoma_dir)*2]

        self.vol_dir = self.non_glaucoma_dir + self.glaucoma_dir
        gen.shuffle(self.vol_dir)

        self.labels = []

        if self.dataset_type == 'test':
            self.vol_dir = self.vol_dir[int(
                0.8*len(self.vol_dir)):]
        elif self.dataset_type == 'val':
            self.vol_dir = self.vol_dir[int(
                0.65*len(self.vol_dir)):int(0.8*len(self.vol_dir))]
        else:
            self.vol_dir = self.vol_dir[:int(
                0.65*len(self.vol_dir))]

        self.data = {}
        count = 0
        num_glaucoma = 0
        num_non_glaucoma = 0
        for i in range(len(self.vol_dir)):
            vol = np.load(self.vol_dir[i])

            if (self.data_size[1] != vol.shape[1] and self.data_size[2] != vol.shape[2]) or ONH_only:

                new_vol = np.zeros(
                    [128, self.data_size[1], self.data_size[2]], dtype=np.float32)
                for slice in range(len(vol)):
                    resized = cv2.resize(
                        vol[slice], (self.data_size[2], self.data_size[1]))
                    new_vol[slice] = resized

                vol = new_vol
                if ONH_only:
                    vol = vol[:, :, 0:vol.shape[2]//2]

            if 'Non-Glaucomas' in self.vol_dir[i]:
                num_non_glaucoma += 1
                self.labels.append(0)
            else:
                num_glaucoma += 1
                self.labels.append(1)

            self.data[count] = (self.vol_dir[i], vol)
            count += 1

        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        filename, x = self.data[idx]

        # transform if training and enabled
        if self.augment and self.dataset_type == 'train':
            x = self.transform(x)

        x = x[None, :, :]

        if self.att_type == 'TimeSformer':
            # channels has to be second
            x = np.transpose(x, axes=[1, 0, 2, 3])

        x = torch.tensor(x).type(torch.FloatTensor)

        # get label

        if 'Non-Glaucomas' in filename:
            gt = torch.tensor([1., 0.])
        else:
            gt = torch.tensor([0., 1.])

        return x, gt

    def ImageRescale(self, im, I_range):
        im_range = im.max() - im.min()
        target_range = I_range[1] - I_range[0]

        if im_range == 0:
            target = np.zeros(im.shape, dtype=np.float32)
        else:
            target = I_range[0] + target_range/im_range * (im - im.min())
        return np.float32(target)

    def transform(self, x):
        rescale = np.max(x) > 1
        if rescale:
            x = x/255

        if 'a' in self.aug_type:
            affine = AffineTransformation(random=True, rotation_step=90)
            x = affine.apply(x)
        if 'g' in self.aug_type:
            grayscale = GrayscaleAugmentation(random=True)
            x = grayscale.apply(x)
        if 'z' in self.aug_type:
            zoom = ZoomTransformation(random=True, range=(1, 1.25))
            x = zoom.apply(x)
        if 'f' in self.aug_type:
            flip = FlipTransformation(random=True)
            x = flip.apply(x)
        if 's' in self.aug_type:
            zflip = FlipZTransformation(random=True)
            x = zflip.apply(x)

        if rescale:
            x = (x*255).astype(int)

        return x


def load_topcon_data(*, batch_size, data_size, att_type, non_glaucoma_dir="/path/to/your/non_glaucoma/data",
                     glaucoma_dir="/path/to/your/glaucoma/data",
                     dataset_type='train', shuffle=False, augment=False, weighting='all', seed=100, aug_type='fgz', ONH_only=False):
    dataset = TopconDataset(non_glaucoma_dir=non_glaucoma_dir, glaucoma_dir=glaucoma_dir, dataset_type=dataset_type,
                            augment=augment, weighting=weighting, data_size=data_size, seed=seed, aug_type=aug_type, att_type=att_type, ONH_only=ONH_only)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader


def load_zeiss_data(*, batch_size, data_size, att_type, data_dir='/path/to/your/zeiss/data',
                    dataset_type='train', shuffle=False, augment=False, seed=113, aug_type='fgz'):
    dataset = ZeissEqualDataset(data_dir=data_dir, data_size=data_size, dataset_type=dataset_type,
                                augment=augment, seed=seed, aug_type=aug_type, att_type=att_type)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader


def load_weighted_topcon_data(*, batch_size, data_size, att_type, non_glaucoma_dir="/path/to/your/non_glaucoma/data",
                              glaucoma_dir="/path/to/your/glaucoma/data",
                              dataset_type='train', shuffle=False, augment=False, weighting='all', seed=100, aug_type='fgz', ONH_only=False):
    """
    Load Topcon data with weighted sampling for class balancing.
    This is an alias for load_topcon_data with weighting parameter.
    """
    return load_topcon_data(
        batch_size=batch_size, 
        data_size=data_size, 
        att_type=att_type, 
        non_glaucoma_dir=non_glaucoma_dir,
        glaucoma_dir=glaucoma_dir,
        dataset_type=dataset_type, 
        shuffle=shuffle, 
        augment=augment, 
        weighting=weighting, 
        seed=seed, 
        aug_type=aug_type, 
        ONH_only=ONH_only
    )


class ZeissEqualDataset(Dataset):
    def __init__(self, data_dir, data_size, att_type, dataset_type='train', augment=False, seed=113, aug_type='fgz'):
        super().__init__()

        self.dataset_type = dataset_type
        self.augment = augment
        self.data_size = data_size

        self.aug_type = aug_type
        self.att_type = att_type

        self.non_glaucoma_dir = glob.glob(
            data_dir + '/**/*Normal*.npy', recursive=True)
        self.glaucoma_dir = glob.glob(
            data_dir + '/**/*POAG*.npy', recursive=True)

        # randomly select samples from non_glaucoma list
        self.seed = seed
        gen = np.random.default_rng(seed=self.seed)
        gen.shuffle(self.non_glaucoma_dir)
        gen.shuffle(self.glaucoma_dir)

        # equally weight
        self.glaucoma_dir = self.glaucoma_dir[:len(self.non_glaucoma_dir)]

        self.vol_dir = self.non_glaucoma_dir + self.glaucoma_dir
        gen.shuffle(self.vol_dir)

        # split filenames based on patient (can be left and right eye so we want to keep the same patient together)
        self.patients = {}
        for filename in self.vol_dir:
            shortened = filename[filename.rfind('/')+1:]
            shortened = shortened[shortened.find('-')+1:]
            patient_id = shortened[:shortened.find('-')]

            # print(filename, patient_id)

            # check if in dictionary already
            if patient_id in self.patients:
                self.patients[patient_id].append(filename)
            else:
                self.patients[patient_id] = [filename]

        self.patient_ids = list(self.patients.keys())

        # randomly select samples from non_glaucoma list
        self.seed = seed
        gen = np.random.default_rng(seed=self.seed)
        gen.shuffle(self.patient_ids)

        # split filenames for test/val/train
        def add_files(patients, patient_ids):
            # add each file for each corresponding patient
            vols = []
            for id in patient_ids:
                vols.extend(patients[id])
            return vols

        if self.dataset_type == 'test':
            self.patient_ids = self.patient_ids[int(
                0.8*len(self.patient_ids)):]
            self.vol_dir = add_files(self.patients, self.patient_ids)
        elif self.dataset_type == 'val':
            self.patient_ids = self.patient_ids[int(
                0.65*len(self.patient_ids)):int(0.8*len(self.patient_ids))]
            self.vol_dir = add_files(self.patients, self.patient_ids)
        else:
            self.patient_ids = self.patient_ids[:int(
                0.65*len(self.patient_ids))]
            self.vol_dir = add_files(self.patients, self.patient_ids)

        self.data = {}
        self.labels = []
        count = 0
        num_glaucoma = 0
        num_non_glaucoma = 0
        for i in range(len(self.vol_dir)):
            # print(f'using file: {self.vol_dir[i]}')
            vol = np.load(self.vol_dir[i])

            if self.data_size[0] != vol.shape[0] or self.data_size[1] != vol.shape[1] or self.data_size[2] != vol.shape[2]:
                if i == 1:
                    print(
                        f'Experiment will modify shape: {vol.shape}')
                # resize uniform
                f1 = self.data_size[0]/vol.shape[0]
                f2 = self.data_size[1]/vol.shape[1]
                f3 = self.data_size[2]/vol.shape[2]
                vol = zoom(vol, (f1, f2, f3), order=1)
                if i == 1:
                    print(
                        f'Experiment will be using RESIZED shape: {vol.shape}')
            elif i == 1:
                print(
                    f'Experiment will be using shape: {vol.shape}')
            # transpose
            # vol = np.transpose(vol, (1, 0, 2))

            if 'Normal' in self.vol_dir[i]:
                num_non_glaucoma += 1
                self.labels.append(0)
            else:
                num_glaucoma += 1
                self.labels.append(1)

            self.data[i] = (self.vol_dir[i], vol)
            count += 1
        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        filename, x = self.data[idx]

        # transform if training and enabled
        if self.augment and self.dataset_type == 'train':
            x = self.transform(x)

        x = x[None, :, :]
        if self.att_type == 'TimeSformer':
            # channels has to be second
            x = np.transpose(x, axes=[1, 0, 2, 3])

        x = torch.tensor(x).type(torch.FloatTensor)

        # get label

        if 'Normal' in filename:
            gt = torch.tensor([1., 0.])
        else:
            gt = torch.tensor([0., 1.])

        return x, gt

    def ImageRescale(self, im, I_range):
        im_range = im.max() - im.min()
        target_range = I_range[1] - I_range[0]

        if im_range == 0:
            target = np.zeros(im.shape, dtype=np.float32)
        else:
            target = I_range[0] + target_range/im_range * (im - im.min())
        return np.float32(target)

    def transform(self, x):
        rescale = np.max(x) > 1
        if rescale:
            x = x/255

        if 'a' in self.aug_type:
            affine = AffineTransformation(random=True, rotation_step=90)
            x = affine.apply(x)
        if 'g' in self.aug_type:
            grayscale = GrayscaleAugmentation(random=True)
            x = grayscale.apply(x)
        if 'z' in self.aug_type:
            zoom = ZoomTransformation(random=True, range=(1, 1.25))
            x = zoom.apply(x)
        if 'f' in self.aug_type:
            flip = FlipTransformation(random=True)
            x = flip.apply(x)
        if 's' in self.aug_type:
            zflip = FlipZTransformation(random=True)
            x = zflip.apply(x)

        if rescale:
            x = (x*255).astype(int)

        return x
