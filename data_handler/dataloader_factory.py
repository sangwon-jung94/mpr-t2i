
from data_handler.dataset_factory import DatasetFactory

import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
import random 

# from transformers import BlipProcessor, BlipModel, BlipForConditionalGeneration, BlipForQuestionAnswering

class DataloaderFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_dataloader(dataname, args):
        # Make a transform function
        processor = None
        transform = None
        test_transform = None
        if args.vision_encoder == 'BLIP':
            from transformers import BlipProcessor
            processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        
        elif args.vision_encoder == 'CLIP':
            import clip
            _, transform = clip.load("ViT-B/32", device= 'cpu')
            # print(transform)
            # mean = [0.48145466, 0.4578275, 0.40821073]
            # std = [0.26862954, 0.26130258, 0.27577711]
            # transform = transforms.Compose(
            #                  [
            #                     transforms.Resize((224,224)),
            #                     transforms.CenterCrop(224),
            #         transforms.Normalize(mean=mean, std=std)]
            # )

        else:
            # For CelebA
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]

            transform = transforms.Compose(
                    [transforms.Resize((256,256)),
                    transforms.RandomCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std)]
                )
            test_transform = transforms.Compose(
                    [transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std)] 
                )
            
        if args.trainer == 'rag':
            processor = transforms.Compose(
                [
                    transforms.Resize((256, 256)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ]
            )
            import clip
            _, transform = clip.load("ViT-L/14", device= 'cpu')


        def _init_fn(worker_id):
            np.random.seed(int(args.seed))

        if dataname != 'mscoco': # mscoco doesn't have the test dataset
            test_dataset = DatasetFactory.get_dataset(args, dataname, transform, processor, split='test')
            test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers, worker_init_fn=_init_fn, pin_memory=True)
        else:
            test_dataloader = None
            
        # If training, use the training dataset
        if args.train:
            train_dataset = DatasetFactory.get_dataset(args, dataname, transform, processor, split='train')            
            if args.bal_sampling:
                if not hasattr(train_dataset, 'weights'):
                    raise ValueError(f"Dataset does not have weights for balanced sampling")
                from torch.utils.data.sampler import WeightedRandomSampler
                weights = train_dataset.weights
                sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
            else:
                sampler = None
            train_dataloader = DataLoader(train_dataset, batch_size=args.train_GPU_batch_size,  worker_init_fn=_init_fn, pin_memory=True, sampler=sampler)
            # val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.n_workers, worker_init_fn=_init_fn, pin_memory=True, drop_last=True)
        
            # return train_dataloader, val_dataloader, test_dataloader
            return train_dataloader, test_dataloader
        
        return test_dataloader
    

# if __name__ == "__main__":

#     import argparse

#     parser = argparse.ArgumentParser(description='representational-generation')

#     parser.add_argument('--seed', type=int, default=0)
#     parser.add_argument('--n-workers', type=int, default=1)
#     parser.add_argument('--dataset', type=str, default='general')
#     parser.add_argument('--dataset-path', type=str, default=None, help='it is only used when query-dataset is general')

#     parser.add_argument('--train', default=False, action='store_true', help='train the model')
#     parser.add_argument('--batch-size', type=int, default=256)
#     parser.add_argument('--vision-encoder', type=str, default='CLIP',choices = ['BLIP', 'CLIP', 'PATHS'])

#     args = parser.parse_args()
    
#     loader = DataloaderFactory.get_dataloader('celeba', args)
