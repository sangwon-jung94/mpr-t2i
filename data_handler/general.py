import torch
from data_handler.dataset_factory import GenericDataset
import os
from PIL import Image, ImageOps
import numpy as np
import torchvision
from torchvision import transforms
class General(GenericDataset):
    def __init__(self, transform=None, processor=None, **kwargs):
        GenericDataset.__init__(self, **kwargs)
        self.dataset_path = self.args.dataset_path

        if self.dataset_path is None:
            raise ValueError(f"Dataset path is not provided")
        
        if not os.path.isdir(self.dataset_path):
            raise ValueError(f"Dataset path is not valid")
        
        # self.check_path_validation()
        self.transform = transform
        self.processor = processor

        # sort the filenames
        self.filenames = os.listdir(self.dataset_path)
        self.filenames = [f for f in self.filenames if f.endswith('.png') or f.endswith('.jpg')]
        self.filenames = np.array(self.filenames)
        if 'openimages' not in self.dataset_path:
            filenames_id = [int(f.split('.')[0]) for f in self.filenames]
            filenames_id = np.argsort(filenames_id)
        else:
            filenames_id = [f.split('.')[0] for f in self.filenames]
            filenames_id = np.argsort(filenames_id)
        self.filenames = self.filenames[filenames_id]
        print('The number of generated samples : ', len(self.filenames))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        imagepath = os.path.join(self.dataset_path, filename)
        image_ori = Image.open(imagepath)

        if self.face_detect:
            left, top, right, bottom, pad_left, pad_top, pad_right, pad_bottom = self.bbox_dic[idx]
            
            image_ori = image_ori.crop((left, top, right, bottom))
            
            if pad_left>0 or pad_top>0 or pad_right>0 or pad_bottom>0:
                image_ori = transforms.ToTensor()(image_ori)
                image_ori = transforms.Pad([pad_left,pad_top,pad_right,pad_bottom], fill=0)(image_ori)
                image_ori = transforms.ToPILImage()(image_ori)

        if self.transform is not None:
           image = self.transform(image_ori)

        if self.processor is not None:
            image = self.processor(images=image_ori, return_tensors="pt")
            image = image['pixel_values'][0]
        return image, 0, idx
