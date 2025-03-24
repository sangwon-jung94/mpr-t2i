import torch
from data_handler.dataset_factory import GenericDataset
import os
from PIL import Image
import numpy as np
import json
import pickle
class MSCoco(GenericDataset):
    infofile = 'mscoco_info.pkl'
    dataset_path = './datasets/mscoco/'
    def __init__(self, transform=None, processor=None, **kwargs):
        GenericDataset.__init__(self, **kwargs)
        if self.dataset_path is None:
            raise ValueError(f"Dataset path is not provided")
        
        if not os.path.isdir(self.dataset_path):
            raise ValueError(f"Dataset path is not valid")
        
        self.transform = transform
        self.processor = processor

        self.img_ids = self._data_dict()

        with open(self.dataset_path + self.infofile, 'rb') as f:
            self.info = pickle.load(f)
            self.bbox_dic = self.info['bbox']

        for key in self.bbox_dic.keys():
            if key not in self.img_ids:
                raise ValueError(f"Image id {key} not found in MSCOCO dataset")
        self.img_ids = list(self.bbox_dic.keys())
        print('The number of samples : ', len(self.img_ids))

        self.weights = self._make_sampling_prob()
        
    def _make_sampling_prob(self): 
        # print(np.unique([i for i in self.info['gender'].values()]))
        # print(np.unique([i for i in self.info['age'].values()]))
        # print(np.unique([i for i in self.info['race'].values()]))
        self.group_count = np.zeros((2,2,7))
        for id in self.img_ids:
            g = self.info['gender'][id]
            a = self.info['age'][id]
            r = self.info['race'][id]
            self.group_count[g][a][r] += 1
        print(self.group_count.sum())
        mask = self.group_count==0
        self.group_count[mask] = 1
        
        c = 1. / self.group_count
        probs = c / np.sum(c)
        probs_sample = []
        for id in self.img_ids:
            g = self.info['gender'][id]
            a = self.info['age'][id]
            r = self.info['race'][id]
            probs_sample.append(probs[g][a][r])
        return probs_sample
        
        
    def _data_dict(self):
        f = open(self.dataset_path + 'annotations/captions_train2017.json')
        data = json.load(f)
        f.close()

        img_ids = []

        # img_id_set = {}
        for x in data['annotations']:
            img_id = x["image_id"]
            img_ids.append(img_id)
            # if img_id in img_id_set:
            #     img_id_set[img_id].append(x["caption"])
            # else:
            #     img_id_set[img_id] = [x["caption"]]
        img_ids = sorted(img_ids)
        img_ids = list(set(img_ids))
        return img_ids

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_path = self.dataset_path + "train2017/" + str(self.img_ids[idx]).zfill(12) + ".jpg"
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

        return image1, image2, idx
