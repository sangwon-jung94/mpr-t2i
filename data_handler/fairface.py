import torch
import os
from data_handler.dataset_factory import GenericDataset
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import torch
import torchvision
import pandas as pd
    
class FairFace(GenericDataset):
    
    def __init__(self, transform=None, processor=None, **kwargs):
    # transform=torchvision.transforms.ToTensor(), embedding_model=None, binarize_age=True):
        GenericDataset.__init__(self, **kwargs)
        self.dataset_path = os.path.join(self.dataset_root, 'fairface')
        
        self.processor = processor
        self.transform = transform

        self.image_paths = []
        self.labels = []

        self.labeltags = [
            "gender",
            "age",
        ]

        if self.split=='train':
            df = pd.read_csv(self.dataset_path + "/fairface_label_train.csv")
        else:
            df = pd.read_csv(self.dataset_path + "/fairface_label_val.csv")

        self.race_to_idx = {}
        races = ['East Asian', 'Indian', 'Black', 'White', 'Middle Eastern', 'Latino_Hispanic', 'Southeast Asian']
        # races = ['White', 'Black', 'Latino_Hispanic', 'Asian']
                        
        # for i, race in enumerate(df.race.unique()):
        for i, race in enumerate(races):
            self.labeltags.append(race)
            self.race_to_idx[race] = i

        self.gender_to_idx = {
            'Male': 0,
            'Female': 1
        }

        self.age_to_idx = {
            '0-2': 0,
            '3-9': 1,
            '10-19': 2,
            '20-29': 3,
            '30-39': 4,
            '40-49': 5,
            '50-59': 6,
            '60-69': 7,
            'more than 70': 8
        }

        one_hot = torch.nn.functional.one_hot(torch.tensor([self.race_to_idx[race] for race in df.race])).numpy()
        if self.args.race_reduce:
            one_hot[:,0] = one_hot[:,0] + one_hot[:,6] 
            one_hot[:,3] = one_hot[:,3] + one_hot[:,4] + one_hot[:,5] 
            one_hot = one_hot[:,[0,1,2,3]]
        gender_idx = [self.gender_to_idx[gen] for gen in df.gender]
        age_idx = [self.age_to_idx[age] for age in df.age]

        # if self.args.binarize_age:
        age_idx = [int(ag>4) for ag in age_idx]
        ## labels is [gender_binary, age_categorical, race_one_hot]

        self.labels = []
        for i in range(len(gender_idx)):
            self.labels.append([gender_idx[i], age_idx[i]] + list(one_hot[i]))

        self.labels = torch.tensor(self.labels)
        self.img_paths = df.file.to_list()
        
        if (hasattr(self.args, 'mpr_group') and 'race2' in self.args.mpr_group) or \
            (hasattr(self.args, 'trainer_group') and 'race2' in self.args.trainer_group):
        
            labels = torch.zeros((self.labels.shape[0], 6))
            labels[:,:2] = self.labels[:,:2]
            labels[:,2] = self.labels[:,5] # white
            labels[:,3] = self.labels[:,4] # black
            labels[:,4] = self.labels[:,7] # white
            labels[:,5] = self.labels[:,2] + self.labels[:,3] + self.labels[:,6] + self.labels[:,8] # white
            self.labels = labels
            
            # reduce Asian images
            indices = torch.nonzero(self.labels[:, 5] == 1).squeeze()
            indices = indices[torch.randperm(indices.size(0))]
            num_indices_to_keep = indices.size(0) // 4
            selected_indices = indices[:num_indices_to_keep]         
            remained_indices = torch.nonzero(self.labels[:, 5] != 1).squeeze()
            total_indices = torch.cat((selected_indices, remained_indices))
            total_indices = total_indices[torch.argsort(total_indices)]
            self.labels = self.labels[total_indices]
            print('the number of total images:', len(self.img_paths))
            print('the number of remained images:', len(self.labels))
            self.img_paths = [self.img_paths[i] for i in total_indices]
        # construct_path = lambda x: os.path.join(self.dataset_path, x)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # if self.embedding_model is not None:
        #     path = self.img_paths[idx]
        #     image_id = path.split(".")[0]
        #     embeddingpath = os.path.join(self.dataset_path, self.embedding_model, image_id+".pt")
        #     return torch.load(embeddingpath), self.labels[idx]
        
        path = os.path.join(self.dataset_path, self.img_paths[idx])
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        if self.processor is not None:
            image = self.processor(images=image, return_tensors="pt")
            image = image['pixel_values'][0]
        
        return image, self.labels[idx], idx
