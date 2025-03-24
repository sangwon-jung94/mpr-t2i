import importlib
import torch.utils.data as data
from collections import defaultdict
import os
import pickle
dataset_dict = {'stable_bias_i' : ['data_handler.stable_bias','StableBiasIdentity'],
                'stable_bias_p' : ['data_handler.stable_bias','StableBiasProfession'],
                'stable_bias_large' : ['data_handler.stable_bias_large','StableBiasProfession'],
                'fairface' : ['data_handler.fairface','FairFace'],
                'general' : ['data_handler.general','General'],
                'celeba' : ['data_handler.celeba','CelebA'],
                'openface': ['data_handler.openface','OpenFace'],
                'custom': ['data_handler.custom_dataset','CustomDataset'],
                'mscoco': ['data_handler.mscoco','MSCoco'],
                'mscoco_val': ['data_handler.mscoco_val','MSCoco'],
                # 'fairdiffusion' : ['data_handler.general','General'],
                # 'fairdiffusion_gender' : ['data_handler.general','General']
               }

class DatasetFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_dataset(args, name, transform, processor, split='test'):
        
        if name not in dataset_dict.keys():
            raise Exception('Not allowed dataset')
        
        module = importlib.import_module(dataset_dict[name][0])
        class_ = getattr(module, dataset_dict[name][1])
        
        return class_(args=args, transform=transform, processor=processor, split=split)

class GenericDataset(data.Dataset):
    def __init__(self, args, split):
        self.args = args
        self.split = split
        self.dataset_root = f'/n/holylabs/LABS/calmon_lab/Lab/datasets/'
        # self.face_detect = True if hasattr(self.args, 'face_detect') and self.args.face_detect else False
        self.face_detect = False
        
        # if self.face_detect:
            # self.bbox_dic = self._load_bbox_dic()

    def _load_bbox_dic(self):
        # Make sure this function should not used in mscoco
        path = os.path.join(self.dataset_path,'bbox_dic.pkl')
        if os.path.exists(path):
            with open(path, 'rb') as f:
                bbox_dic = pickle.load(f)
        else:
            raise ValueError(f"bbox_dic.pkl does not exist")

        return bbox_dic
    
    def turn_on_detect(self):
        self.face_detect = True
        if not hasattr(self, 'bbox_dic'):
            self.bbox_dic = self._load_bbox_dic()
    
    def turn_off_detect(self):
        self.face_detect = False


