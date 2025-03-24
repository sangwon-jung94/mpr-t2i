import face_alignment
from skimage import io
from glob import glob # type: ignore
from tqdm import tqdm
import os
import numpy as np
from PIL import Image
import json
import argparse
from torchvision import transforms
import torch

class FanCrop:
    def __init__(self):
        self.device = 'cuda'
        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device=self.device)
        
    def tensor_to_pil(self, tensor):
        """Convert torch.tensor image to PIL.Image"""
        transform = transforms.ToPILImage()
        return transform(tensor)

    def crop_images(self, images):
        cropped_images = []
        valid_indices = []
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device="cuda")

        # images = glob(f'{args.image_folder}/*/*.png')

        for idx, image in enumerate(tqdm(images)):
            if isinstance(image, torch.Tensor):
                im = self.tensor_to_pil(image)
                input = image.numpy().transpose(1, 2, 0)  # Convert to HWC format
            else:
                im = Image.open(image)
                input = io.imread(image)

            preds = fa.get_landmarks(input)
            if preds is None:
                continue
            
            for i, pred in enumerate(preds):
                cropped_image = im.crop((pred[:, 0].min(), pred[:, 1].min(), pred[:, 0].max(), pred[:, 1].max()))
                cropped_images.append((cropped_image, pred))
                valid_indices.append(idx)


        return valid_indices, cropped_images
