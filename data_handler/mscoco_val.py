import torch
from data_handler.dataset_factory import GenericDataset
import os
from PIL import Image
import numpy as np
import json
import pickle
class MSCoco(GenericDataset):
    infofile = 'mscoco_info.pkl'
    dataset_path = './datasets/mscoco/val2014'
    def __init__(self, transform=None, processor=None, **kwargs):
        GenericDataset.__init__(self, **kwargs)
        if self.dataset_path is None:
            raise ValueError(f"Dataset path is not provided")
        
        if not os.path.isdir(self.dataset_path):
            raise ValueError(f"Dataset path is not valid")
        
        self.transform = transform
        self.processor = processor

        self.filenames = os.listdir(self.dataset_path)
        self.filenames = [f for f in self.filenames if f.endswith('.png') or f.endswith('.jpg')]
        self.filenames = np.array(self.filenames)
        filenames_id = [f.split('.')[0] for f in self.filenames]
        filenames_id = np.argsort(filenames_id)
        self.filenames = self.filenames[filenames_id]
        # print(self.filenames)

        # self.concept_set = [self.args.target_concept]
        print('The number of samples : ', len(self.filenames))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = self.dataset_path + '/' + self.filenames[idx]
        # img_path = './datasets/mscoco/' + "train2017/" + str(self.img_ids[idx]).zfill(12) + ".jpg"
        image = Image.open(img_path).convert('RGB')
        
        if self.face_detect:
            left, top, right, bottom = self.bbox_dic[idx]
            image = image.crop((left, top, right, bottom))

        if self.transform is not None: # clip transform
            image1 = self.transform(image)
            
        if self.processor is not None: # vae transform
            image2 = self.processor(image)
            # image = image['pixel_values'][0]

        return image1, 0, idx
