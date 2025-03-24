import torch
from data_handler.dataset_factory import GenericDataset
import os
from PIL import Image
from datasets import load_from_disk
import numpy as np
    
class StableBiasProfession(GenericDataset):

    def __init__(self, transform=None, processor=None, **kwargs):
        GenericDataset.__init__(self, **kwargs)

        self.transform = transform
        self.processor = processor

        if self.args.target_profession not in ['firefighter', 'CEO', 'musician']:
            raise ValueError(f"Profession {self.args.target_profession} not in dataset")
        
        self.datapath = f'/n/holyscratch01/calmon_lab/Lab/datasets/{self.args.target_model}/{self.args.target_profession}'
        self.filenames = os.listdir(self.datapath)
        self.filenames = [f for f in self.filenames if f.endswith('.png')]
        self.filenames = np.array(self.filenames)
        filenames_id = [int(f.split('.')[0]) for f in self.filenames]
        filenames_id = np.argsort(filenames_id)
        self.filenames = self.filenames[filenames_id]
        # self.filenames = self.filenames[:15000]
        print(self.filenames)

        self.profession_set = [self.args.target_profession]
        print(len(self.filenames))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        imagepath = os.path.join(self.datapath, filename)
        image = Image.open(imagepath).convert("RGB")
        
        if self.transform is not None:
            image = self.transform(image)
        if self.processor is not None:
            image = self.processor(images=image, return_tensors="pt")
            image = image['pixel_values'][0]
        return image, 0
