import torch
import os
from data_handler.dataset_factory import GenericDataset
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torchvision.datasets.utils import download_file_from_google_drive, check_integrity, verify_str_arg
import torch
import torchvision
import pandas as pd
from functools import partial
from os.path import join
from torchvision import transforms

class CelebA(GenericDataset):
    
    file_list = [
        # File ID                         MD5 Hash                            Filename
        ("0B7EVK8r0v71pZjFTYXZWM3FlRnM", "00d2c5bc6d35e252742224ab0c1e8fcb", "img_align_celeba.zip"),
        # ("0B7EVK8r0v71pbWNEUjJKdDQ3dGc", "b6cd7e93bc7a96c2dc33f819aa3ac651", "img_align_celeba_png.7z"),
        # ("0B7EVK8r0v71peklHb0pGdDl6R28", "b6cd7e93bc7a96c2dc33f819aa3ac651", "img_celeba.7z"),
        ("0B7EVK8r0v71pblRyaVFSWGxPY0U", "75e246fa4810816ffd6ee81facbd244c", "list_attr_celeba.txt"),
        ("1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS", "32bd1bd63d3c78cd57e08160ec5ed1e2", "identity_CelebA.txt"),
        ("0B7EVK8r0v71pbThiMVRxWXZ4dU0", "00566efa6fedff7a56946cd1c10f1c16", "list_bbox_celeba.txt"),
        ("0B7EVK8r0v71pd0FJY3Blby1HUTQ", "cc24ecafdb5b50baae59b03474781f8c", "list_landmarks_align_celeba.txt"),
        # ("0B7EVK8r0v71pTzJIdlJWdHczRlU", "063ee6ddb681f96bc9ca28c6febb9d1a", "list_landmarks_celeba.txt"),
        ("0B7EVK8r0v71pY0NSMzRuSXJEVkk", "d32c9cbf5e040fd4025c592c306e6668", "list_eval_partition.txt"),
    ]
    # mean, std = get_mean_std('celeba')
    # train_transform = transforms.Compose(
    #         [transforms.Resize((256,256)),
    #          transforms.RandomCrop(224),
    #          transforms.RandomHorizontalFlip(),
    #          transforms.ToTensor(),
    #          transforms.Normalize(mean=mean, std=std)]
    #     )
    # test_transform = transforms.Compose(
    #         [transforms.Resize((224, 224)),
    #          transforms.ToTensor(),
    #          transforms.Normalize(mean=mean, std=std)] 
    #     )
    
    name = 'celeba'    
    def __init__(self, transform=None, processor=None, **kwargs):
    # transform=torchvision.transforms.ToTensor(), embedding_model=None, binarize_age=True):
        GenericDataset.__init__(self, **kwargs)
        self.dataset_path = os.path.join(self.root, 'celeba')
        self.processor = processor
        self.transform = transform

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')
        # SELECT the features
        # self.sensitive_attr = 'Male'
        split_map = {
            "train": 0,
            "valid": 1,
            "test": 2,
            "all": None,
        }
        split = split_map[verify_str_arg(self.split.lower(), "split",
                                         ("train", "valid", "test", "all" ))]
        fn = partial(join, self.dataset_path)
        splits = pd.read_csv(fn("list_eval_partition.txt"), sep='\s+', header=None, index_col=0)
        attr = pd.read_csv(fn("list_attr_celeba.txt"), sep='\s+', header=1)
        
        mask = slice(None) if split is None else (splits[1] == split)
        
        self.filename = splits[mask].index.values
        self.attr = torch.as_tensor(attr[mask].values)
        self.attr = (self.attr + 1) // 2  # map from {-1, 1} to {0, 1}
        self.attr_names = list(attr.columns)
        
        # self.target_idx = self.attr_names.index(self.target_attr)


    def __len__(self):
        return len(self.filename)

    def __getitem__(self, idx):
        img_name = self.filename[idx]
        path = os.path.join(self.dataset_path, "img_align_celeba", img_name)
        image = Image.open(path)
        if self.transform is not None:
            image = self.transform(image)
        if self.processor is not None:
            image = self.processor(images=image, return_tensors="pt")
            image = image['pixel_values'][0]
        
        return image, self.attr[idx]

    def _check_integrity(self):
        for (_, md5, filename) in self.file_list:
            fpath = os.path.join(self.dataset_path, filename)
            _, ext = os.path.splitext(filename)
            # Allow original archive to be deleted (zip and 7z)
            # Only need the extracted images
            if ext not in [".zip", ".7z"] and not check_integrity(fpath, md5):
                return False

        # Should check a hash of the images
        return os.path.isdir(os.path.join(self.dataset_path, "img_align_celeba"))
