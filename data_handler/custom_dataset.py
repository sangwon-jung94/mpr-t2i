import os
from data_handler.dataset_factory import GenericDataset
from PIL import Image
import glob
from torchvision import transforms
class CustomDataset(GenericDataset):
    name = 'openface'
    def __init__(self, transform=None, processor=None, **kwargs):
        GenericDataset.__init__(self, **kwargs)        
        # self.dataset_path = os.path.join('datasets/custom_mpr_datasets', "_".join(self.args.target_concept.split(" ")))
        self.dataset_path = os.path.join('datasets/custom_mpr_datasets', "_".join(self.args.target_concept.split(" ")))

        self.filepaths = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            self.filepaths += glob.glob(os.path.join(self.dataset_path, ext))
        print('# of images in the dataset:', len(self.filepaths))

        self.processor = processor
        self.transform = transform


    def __getitem__(self, idx):
        image = Image.open(self.filepaths[idx]).convert('RGB')
        if self.face_detect:
            left, top, right, bottom, pad_left, pad_top, pad_right, pad_bottom = self.bbox_dic[idx]
            # left, top, right, bottom = self.bbox_dic[idx]
            
            image = image.crop((left, top, right, bottom))
            
            if pad_left>0 or pad_top>0 or pad_right>0 or pad_bottom>0:
                image = transforms.ToTensor()(image)
                image = transforms.Pad([pad_left,pad_top,pad_right,pad_bottom], fill=0)(image)
                image = transforms.ToPILImage()(image)

        if self.transform is not None:
            image = self.transform(image)
        if self.processor is not None:
            image = self.processor(image, return_tensors="pt")
        return image, 0, idx

    def __len__(self):
        return len(self.filepaths)

    def _check_integrity(self):
        # I dont know how to check the integrity of Huggingface datasets so I'm leaving this
        return True