import webdataset as wds
from PIL import Image
import io
import matplotlib.pyplot as plt
import os
import json

from warnings import filterwarnings


# os.environ["CUDA_VISIBLE_DEVICES"] = "0"    # choose GPU if you are on a multi GPU server
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import tqdm

from os.path import join
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import json

import clip


from PIL import Image, ImageFile


#####  This script will predict the aesthetic score for this image file:

img_path = "test.jpg"


class AestheticScorer(torch.nn.Module):
    def __init__(self, path_to_params="improved-aesthetic-predictor/sac+logos+ava1-l14-linearMSE.pth"):
        super().__init__()
        self.model = MLP(768)  # CLIP embedding dim is 768 for CLIP ViT L 14
        self.model.load_state_dict(torch.load(path_to_params))

        self.clipmodel, self.preprocess = clip.load("ViT-L/14", device='cuda')
        self.models = torch.nn.ModuleList([self.clipmodel, self.model])
        
    def forward(self, image):
        image_features = self.models[0].encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        image_features = image_features.to(torch.float32)
        prediction = self.models[1](image_features)
        return prediction

    def get_preprocess(self):
        return self.preprocess


# if you changed the MLP architecture during training, change it also here:
class MLP(torch.nn.Module):
    def __init__(self, input_size, xcol='emb', ycol='avg_rating'):
        super().__init__()
        self.input_size = input_size
        self.xcol = xcol
        self.ycol = ycol
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            #nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, 16),
            #nn.ReLU(),

            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
            x = batch[self.xcol]
            y = batch[self.ycol].reshape(-1, 1)
            x_hat = self.layers(x)
            loss = F.mse_loss(x_hat, y)
            return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = F.mse_loss(x_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

def normalized(a, axis=-1, order=2):
    import numpy as np  # pylint: disable=import-outside-toplevel

    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='representational-generation')
        
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n-workers', type=int, default=1)
    parser.add_argument('--dataset', type=str, default='general')
    parser.add_argument('--dataset-path', type=str, default=None, help='it is only used when query-dataset is general')

    parser.add_argument('--train', default=False, action='store_true', help='train the model')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--vision-encoder', type=str, default='CLIP',choices = ['BLIP', 'CLIP', 'PATHS'])
    parser.add_argument('--filter_w_aesthetic', default=False, action='store_true', help='additional filtering with aesthetic scoree')
    args = parser.parse_args()

    model = MLP(768)  # CLIP embedding dim is 768 for CLIP ViT L 14

    s = torch.load("improved-aesthetic-predictor/sac+logos+ava1-l14-linearMSE.pth")   # load the model you trained previously or the model available in this repo

    model.load_state_dict(s)
    loader = data_handler.DataloaderFactory.get_dataloader(dataname=args.dataset, args=args)

    model.to("cuda")
    model.eval()
    
    model2, preprocess_for_filter = clip.load("ViT-L/14", device='cuda')

    # filtered_ids = []
    # unfiltered_ids = []
    score_dic = {}
    loader.dataset.processor = None
    loader.dataset.transform = preprocess_for_filter
    with torch.no_grad():
        for image, label, idxs in loader:
            image_features = model2.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            image_features.to(torch.float32)
            prediction = model(image_features)
            print(prediction)
            for idx, score in zip(idxs, prediction):
                score_dic[idx.item()] = score.item()

    with open(os.path.join(args.dataset_path,'score_dic.pkl'), 'wb') as f:
        pickle.dump(bbox_dic, f)
    #print mean of the scores
    print(np.mean(list(score_dic.values())))
    print(np.std(list(score_dic.values())))
    print(np.min(list(score_dic.values())))
    print(np.max(list(score_dic.values())))
        # for id, bbox in bbox_dic.items():
            # f.write(f"{id}: {bbox}\n")
    
    # percentage of filtered images
