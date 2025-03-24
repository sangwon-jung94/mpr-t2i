import torch
from data_handler.dataset_factory import GenericDataset
from PIL import Image
from datasets import load_from_disk

# https://github.com/huggingface/transformers/blob/main/src/transformers/models/blip/image_processing_blip.py
class StableBiasIdentity(GenericDataset):
    race_set = ['African-American',
                'American_Indian',
                'Black',
                'Caucasian',
                'East_Asian',
                'First_Nations',
                'Hispanic',
                'Indigenous_American',
                'Latino',
                'Latinx',
                'Multiracial',
                'Native_American',
                'Pacific_Islander',
                'South_Asian',
                'Southeast_Asian',
                'White',
                'no_ethnicity_specified']
    gender_set = ['man', 'woman', 'non-binary', 'no_gender_specified']

    def __init__(self, transform=None, processor=None, **kwargs):
        # self.dataset = dataset
        GenericDataset.__init__(self, **kwargs)
        # path = '/n/holylabs/LABS/calmon_lab/Lab/datasets/stable_bias'
        self.datapath = '/n/holyscratch01/calmon_lab/Lab/datasets/stable_bias/identities'
        # self.dataset = load_dataset('tti-bias/identities', split='train', cache_dir=path)
        self.dataset = load_from_disk(self.datapath)

        # Original dataset contains generated samples from three diffreent models
        # self.dataset = self.dataset.filter(lambda x: x['model'] == self.args.target_model) 
        model_list = self.dataset['model']
        idx = [i for i, x in enumerate(model_list) if x == self.args.target_model]
        self.dataset = self.dataset.select(idx)
        
        self.processor = processor
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        # image = Image.open(data['image']).convert("RGB")
        image = data['image']
        
        if self.transform is not None:
            image = self.transform(image)
        if self.processor is not None:
            image = self.processor(images=image, return_tensors="pt")
            image = image['pixel_values'][0]
        race = self.race_set.index(data["ethnicity"])
        gender = self.gender_set.index(data["gender"])
        return image, race, gender
    
class StableBiasProfession(GenericDataset):

    def __init__(self, transform=None, processor=None, **kwargs):
        GenericDataset.__init__(self, **kwargs)
        # path = '/n/holylabs/LABS/calmon_lab/Lab/datasets/stable_bias'
        self.datapath = '/n/holyscratch01/calmon_lab/Lab/datasets/stable_bias/professions'
        self.dataset = load_from_disk(self.datapath)
        # self.dataset = self.dataset.select(range(1000))
        # self.dataset = self.dataset.filter(lambda x: x['model'] == self.args.target_model) 
        model_list = self.dataset['model']
        idx = [i for i, x in enumerate(model_list) if x == self.args.target_model]
        self.dataset = self.dataset.select(idx)

        self.profession_set = list(set(self.dataset['profession']))
        self.profession_set.sort()

        # Original dataset contains generated samples from three diffreent models
        # if self.args.target_profession != 'all':
        #     if self.args.target_profession not in self.profession_set:
        #         raise ValueError(f"Profession {self.args.target_profession} not in dataset")
        #     # self.dataset = self.dataset.filter(lambda x: x['profession'] == self.args.target_profession)
        #     profession_list = self.dataset['profession']
        #     idx = [i for i, x in enumerate(profession_list) if x == self.args.target_profession]
        #     self.dataset = self.dataset.select(idx)
        #     self.profession_set = [self.args.target_profession]

        self.processor = processor
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        # image = Image.open(data['image']).convert("RGB")
        image = data['image']
        
        if self.transform is not None:
            image = self.transform(image)
        if self.processor is not None:
            image = self.processor(images=image, return_tensors="pt")
            image = image['pixel_values'][0]
        profession = self.profession_set.index(data["profession"])
        return image, profession

