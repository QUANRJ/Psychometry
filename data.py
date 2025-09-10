import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import utils
import kornia
from kornia.augmentation.container import AugmentationSequential


def build_image_augmentor():
    return AugmentationSequential(
        kornia.augmentation.RandomResizedCrop((224, 224), (0.8, 1.0), p=0.3),
        kornia.augmentation.Resize((224, 224)),
        kornia.augmentation.RandomBrightness((0.8, 1.2), clip_output=True, p=0.2),
        kornia.augmentation.RandomContrast((0.8, 1.2), clip_output=True, p=0.2),
        kornia.augmentation.RandomGamma((0.8, 1.2), (1.0, 1.3), p=0.2),
        kornia.augmentation.RandomSaturation((0.8, 1.2), p=0.2),
        kornia.augmentation.RandomHue((-0.1, 0.1), p=0.2),
        kornia.augmentation.RandomSharpness((0.8, 1.2), p=0.2),
        kornia.augmentation.RandomGrayscale(p=0.2),
        data_keys=["input"],
    )

def build_large_image_augmentor():
    return AugmentationSequential(
        kornia.augmentation.RandomBrightness((0.8, 1.2), clip_output=True, p=0.2),
        kornia.augmentation.RandomContrast((0.8, 1.2), clip_output=True, p=0.2),
        kornia.augmentation.RandomGamma((0.8, 1.2), (1.0, 1.3), p=0.2),
        kornia.augmentation.RandomSaturation((0.8, 1.2), p=0.2),
        kornia.augmentation.RandomHue((-0.1, 0.1), p=0.2),
        kornia.augmentation.RandomSharpness((0.8, 1.2), p=0.2),
        kornia.augmentation.RandomGrayscale(p=0.2),
        kornia.augmentation.RandomSolarize(p=0.2),
        kornia.augmentation.RandomGaussianBlur((7, 7), (0.1, 2.0), p=0.1),
        kornia.augmentation.RandomResizedCrop((512, 512), scale=(0.5, 1.0)),
        data_keys=["input"],
    )

img_augment = build_image_augmentor()
img_augment_ll = build_large_image_augmentor()

class BrainDataset(Dataset):
    def __init__(self, data_dir, file_types=None, pool_size=8192, pool_mode="max", sample_count=None):
        self.data_dir = data_dir
        self.file_types = file_types if file_types else []
        self.pool_size = pool_size
        self.pool_mode = pool_mode
        self.entries = self._gather_entries()
        self.keys = sorted(self.entries.keys())
        self.sample_count = sample_count
        if sample_count is not None:
            if sample_count > len(self.keys):
                pass
            elif sample_count > 0:
                self.keys = self.keys[:sample_count]
            elif sample_count < 0:
                self.keys = self.keys[sample_count:]
            elif sample_count == 0:
                raise ValueError("sample_count must be non-zero!")
        else:
            self.sample_count = len(self.keys)

    def _gather_entries(self):
        all_files = os.listdir(self.data_dir)
        mapping = {}
        for fname in all_files:
            fpath = os.path.join(self.data_dir, fname)
            base, ext = fname.split(".", 1)
            if ext in self.file_types:
                if base not in mapping:
                    mapping[base] = {"subj": fpath}
                mapping[base][ext] = fpath
        return mapping

    def _read_image(self, path):
        img = Image.open(path).convert('RGB')
        arr = np.asarray(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr.transpose(2, 0, 1))
        return tensor

    def _read_npy(self, path):
        arr = np.load(path)
        return torch.from_numpy(arr)

    def _process_voxel(self, vox):
        if self.pool_size is not None:
            vox = pool_voxels(vox, self.pool_size, self.pool_mode)
        return vox

    def _subject_id(self, subj_path):
        return int(subj_path.split("/")[-2].replace("subj", ""))

    def _process_brain3d(self, arr):
        return arr

    def __len__(self):
        return self.sample_count

    def __getitem__(self, index):
        index = index % len(self.keys)
        key = self.keys[index]
        entry = self.entries[key]
        result = []
        for ext in self.file_types:
            if ext == "jpg":
                result.append(self._read_image(entry[ext]))
            elif ext == "nsdgeneral.npy":
                vox = self._read_npy(entry[ext])
                result.append(vox)
            elif ext == "coco73k.npy":
                result.append(self._read_npy(entry[ext]))
            elif ext == "subj":
                result.append(self._subject_id(entry[ext]))
            elif ext == "wholebrain_3d.npy":
                arr = self._read_npy(entry[ext])
                result.append(self._process_brain3d(arr))
        return result

def pool_voxels(voxel_tensor, pool_size, pool_mode):
    voxel_tensor = voxel_tensor.float()
    if pool_mode == 'avg':
        return nn.AdaptiveAvgPool1d(pool_size)(voxel_tensor)
    elif pool_mode == 'max':
        return nn.AdaptiveMaxPool1d(pool_size)(voxel_tensor)
    elif pool_mode == 'resize':
        temp = voxel_tensor.unsqueeze(1)
        temp = F.interpolate(temp, size=pool_size, mode='linear', align_corners=False)
        return temp.squeeze(1)
    return voxel_tensor

def get_dataloader(
    root_dir,
    batch_size,
    num_workers=1,
    seed=42,
    is_shuffle=True,
    extensions=['nsdgeneral.npy', 'jpg', 'coco73k.npy', 'subj'],
    pool_type=None,
    pool_num=None,
    length=None,
):
    utils.seed_everything(seed)
    dataset = BrainDataset(
        data_dir=root_dir,
        file_types=extensions,
        pool_size=pool_num,
        pool_mode=pool_type,
        sample_count=length
    )
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=is_shuffle)

def get_dls(subject, data_path, batch_size, val_batch_size, num_workers, pool_type, pool_num, length, seed):
    train_dir = f"{data_path}/webdataset_avg_split/train/subj0{subject}"
    val_dir = f"{data_path}/webdataset_avg_split/val/subj0{subject}"
    exts = ['nsdgeneral.npy', 'jpg', 'coco73k.npy', 'subj']
    train_loader = get_dataloader(
        train_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
        extensions=exts,
        pool_type=pool_type,
        pool_num=pool_num,
        is_shuffle=True,
        length=length,
    )
    val_loader = get_dataloader(
        val_dir,
        batch_size=val_batch_size,
        num_workers=num_workers,
        seed=seed,
        extensions=exts,
        pool_type=pool_type,
        pool_num=pool_num,
        is_shuffle=False,
    )
    print(train_dir, "\n", val_dir)
    print("number of train data:", len(train_loader.dataset))
    print("batch_size", batch_size)
    print("number of val data:", len(val_loader.dataset))
    print("val_batch_size", val_batch_size)
    return train_loader, val_loader
