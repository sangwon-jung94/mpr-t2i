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

        # self.query_dataset = self.args.query_dataset
        # self.target_model = self.args.target_model
        # self.target_concept = self.args.target_concept

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
        # print(self.filenames)

        # self.concept_set = [self.args.target_concept]
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

    # def check_path_validation(self):
    #     folders = self.dataset_path.split('/')
    #     target_model = folders[-2]
    #     trainer = folders[1]
    #     concept = folders[-1].split('_')[0]
    #     print(self.dataset_path)
    #     if not os.path.isdir(self.dataset_path):
    #         raise ValueError(f"Dataset path is not valid")

    #     if self.args.target_concept != concept:
    #         raise ValueError(f"The concept ({concept}) in the data path and the target concept ({self.args.target_concept}) are not matching")

    #     if self.args.target_model != target_model:
    #         raise ValueError(f"The model ({target_model}) in the data path and the target model ({self.args.target_model}) are not matching")
        
    #     if self.args.trainer != trainer:
    #         raise ValueError(f"The trainer ({trainer}) in the data path and the trainer ({self.args.trainer}) are not matching")

    #     if self.args.p_ver == 'v2':
    #         if 'adj' not in self.dataset_path:
    #             raise ValueError(f"The data path should contain 'adj' for the p_ver v2")

    #     args_group_name = "".join([_g[0] for _g in self.args.trainer_group])        
    #     if self.args.trainer != 'scratch':
    #         group_name = folders[2]
    #         if args_group_name != group_name:
    #             raise ValueError(f"The group ({group_name}) in the data path and the trainer-group ({args_group_name}) are not matching")
        
