from glob import glob
from PIL import Image
from typing import *

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import albumentations as A


class ObstacleDetectionDataset(Dataset):
    """
    Args:
        path: a dataset folder directory
        subset: train, valid or test set
    """
    def __init__(
        self,
        path: str,
        subset: str='train',
        crop_size: Optional[Tuple[int]]=None,
        transforms_: Optional[bool]=None,
    ):
        assert subset in ('train', 'valid', 'test'), \
            f'{subset} does not exists'
        self.subset = subset
        
        self.image_files = sorted(glob(path+'Bbox_*/Bbox_*/*.jpg') + \
            glob(path+'Bbox_*/Bbox_*/*.png'))
        self.label_files = sorted(glob(path+'Bbox_*/Label_*/*.txt'))
        
        assert len(self.image_files) == len(self.label_files), \
            f'The number of images {len(self.image_files)} and labels {len(self.label_files)} does not match'
        
        self.transforms_ = A.Compose([
            A.RandomResizedCrop(height=crop_size[0], width=crop_size[1]),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=40, p=0.5),
            A.CLAHE(p=0.01),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.01),
            A.Blur(p=0.01),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])) if transforms_ is not None else None
        
        self.totensor = transforms.ToTensor()
       
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        images = Image.open(self.image_files[idx])
        label_idx = self.label_files[idx]
        labels = self._load_labels_(label_idx)
        if self.transforms_ is not None:
            augmented = self.transfroms_(image=images, bboxes=labels[:,1:], class_labels=labels[:,0])
            images, labels = augmented['image'], torch.Tensor([[c, *b] for c, b in zip(augmented['class_labels'], augmented['bboxes'])])
        labels_out = torch.zeros((len(labels), 6))
        labels_out[:, 1] = labels[:, 0]
        labels_out[:, 2:] = labels[:, 1:]
        
        return self.totensor(images), labels_out
        
    def _load_labels_(self, label_dir):
        with open(label_dir, 'r') as f:
            label_list = []
            for anno in f.readlines():
                sub_label_list = []
                for i, sub_anno in enumerate(anno.rstrip().split(' ')):
                    sub_label_list.append(int(sub_anno)) if i == 0 else \
                        sub_label_list.append(float(sub_anno))
                label_list.append(sub_label_list)

        return torch.Tensor(label_list)

    @staticmethod
    def collate_fn(batch):
        imagee, labels = zip(*batch)
        for i, label in enumerate(labels):
            label[:, 0] = i
        return torch.stack(images, 0), torch.cat(labels, 0)

