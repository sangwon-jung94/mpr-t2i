
import torch
import numpy as np 
import os
from transformers import BlipProcessor
import pickle
from tqdm import tqdm
import torch
import torch.nn as nn
import clip 
from transformers import Blip2Processor, AutoProcessor, AutoTokenizer, Blip2ForConditionalGeneration
import torch

class BLIPPredictor:
    def __init__(self):
        cache_dir='/n/holylabs/LABS/calmon_lab/Lab/diffusion_models'
        self.vis_processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b", cache_dir=cache_dir)    
        self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b", cache_dir=cache_dir)
        self.tokenizer = AutoTokenizer.from_pretrained("Salesforce/blip2-opt-2.7b", cache_dir=cache_dir)
        self.model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16, cache_dir=cache_dir)
        self.model.to('cuda').eval()
    
    def forward(self,images):
        images = images['pixel_values'][0].to('cuda')
        # images = images.to(torch.float16)
        generated_ans = []
        # for image in tqdm(images):
            # image = image.unsqueeze(0)
        prompt = "Question: What objects are in the image? Answer:"
        inputs = self.tokenizer([prompt]*images.shape[0], padding=True, return_tensors="pt").to('cuda')
        inputs = inputs.to(torch.float16)
        # print(inputs['input_ids'].shape)
        generated_ids = self.model.generate(pixel_values=images, **inputs)#, max_new_tokens=10)
        # print(generated_ids.shape)
        # generated_ids = generated_ids.to(torch.float32)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        generated_text = [t.strip().lower() for t in generated_text]
        print(generated_text)
        for text in generated_text:
            if 'wheelchair'  in text:
                generated_ans.append(1)
            else:
                generated_ans.append(0)
        return torch.tensor(generated_ans)
        
class CLIPExtractor:
    def __init__(self, encoder):
        self.encoder = encoder

    def extract(self, images, query=True):
        if torch.cuda.is_available():
            images = images.cuda()
        with torch.no_grad():
            outputs = self.encoder.encode_image(images)
        return outputs

def identity_embedding(args, encoder, dataloader, groups, query=True):
    dataset_name = args.refer_dataset if not query else 'query dataset'
    path = dataloader.dataset.dataset_path
    
    feature_dic = {}
    
    # need to consider different versions of dataloader depending on the groups
    for ver in ['normal', 'face_detect']:

        filename = f'{args.vision_encoder}_{ver}_feature.pkl'
        filepath = os.path.join(path,filename)

        save_flag = False
        features = []
        if os.path.exists(filepath):
            with open(os.path.join(path,filename), 'rb') as f:
                feature_dic[ver] =  pickle.load(f)
            if feature_dic[ver].shape[0] == len(dataloader.dataset):
                print(f'embedding vectors of {dataset_name} are successfully loaded in {path}')
                continue

        save_flag = True

        if ver == 'face_detect':
            dataloader.dataset.turn_on_detect()

        encoder.eval()
        encoder = encoder.cuda() if torch.cuda.is_available() else encoder
        # encoder = encoder.to(torch.float16)

        feature_extractor = None

        if args.vision_encoder == 'CLIP':
            feature_extractor =  CLIPExtractor(encoder)
        else:
            raise ValueError(f'vision encoder {args.vision_encoder} is not supported')
    
        for batch in tqdm(dataloader):
            image, label, idxs =  batch

            if torch.cuda.is_available():
                image = image.cuda()
            feature = feature_extractor.extract(image)
            features.append(feature.cpu())
        
        features = torch.cat(features).numpy()
        feature_dic[ver] = features

        if ver == 'face_detect':
            dataloader.dataset.turn_off_detect()
        
        if save_flag:
            with open(filepath, 'wb') as f:
                pickle.dump(features, f)

    # group estimation
    estimated_groups = []
    for group in groups:
        if group in ['gender', 'race', 'race2', 'age','emotion']:
            feature = feature_dic['face_detect']
        elif group in ['background', 'house']:
            feature = feature_dic['normal']
        elif group in ['wheelchair']:            
            feature = None
        else:
            raise ValueError(f'group {group} is not supported')

        estimated_group = group_estimation(feature,group, args.vision_encoder, onehot=args.mpr_onehot, loader=dataloader, encoder=encoder, query=query)
        estimated_groups.append(estimated_group)
    estimated_groups = np.concatenate(estimated_groups, axis=1)
    return estimated_groups, feature_dic
            
def group_estimation(features, group='gender', vision_encoder_name='CLIP', onehot=False, loader=None, encoder=None, query=True):
    path = './mpr_stuffs/'
    if group in ['gender', 'age','race', 'race2']:
        clf_path = os.path.join(path,f'clf_{group}.pkl')
        with open(clf_path, 'rb') as f:
            clf = pickle.load(f)
            estimated_group = clf.predict_proba(features)
            if onehot:
                one_hot_indices = np.argmax(estimated_group, axis=1)
                estimated_group = np.eye(estimated_group.shape[1])[one_hot_indices]
                
    # if you find that BLIP doesn't work well, please double check the face_detect version of dataloader
    elif group == 'wheelchair':
        model = BLIPPredictor()
        with torch.no_grad():
            item_presence = []
            transform = loader.dataset.transform
            loader.dataset.transform = model.vis_processor
            for data in loader:
                images, labels, idxs = data
                result = model.forward(images)
                item_presence.append(result)
            loader.dataset.transform = transform
            item_presence = torch.cat(item_presence)
            item_presence = torch.stack((1-item_presence, item_presence), dim=-1)
        estimated_group = item_presence
        # print(f'wheelchair 1/0: {torch.sum(item_presence[:,1])}/{item_presence.shape[0]-torch.sum(item_presence[:,0])}')
    
    elif group == 'emotion':
        estimated_group = emotion_estimate(features, encoder)
        if onehot:
            one_hot_indices = np.argmax(estimated_group, axis=1)
            estimated_group = np.eye(estimated_group.shape[1])[one_hot_indices]
            
    estimated_group = estimated_group * 2 - 1
    
    return estimated_group

def emotion_estimate(features, encoder):

    adjectives = ['ambitious','assertive','confident','decisive','determined','intelligent','outspoken','self-confident','stubborn','unreasonable','committed','supportive','sensitive','emotional','gendtle','honest','modest','compassionate','considerate','pleasant']

    tmp = []
    texts = []
    for adjective in adjectives:
        text = f'A photo of a {adjective} person'
        texts.append(text)
        # print(text)
        # tmp.append(text)
        text_inputs = clip.tokenize(text).to('cuda')
        with torch.no_grad():
            text_embedding = encoder.encode_text(text_inputs)
        text_embedding = text_embedding / torch.norm(text_embedding, dim=-1, keepdim=True)
        tmp.append(text_embedding)
    text_embedding = torch.cat(tmp)


    n_iter = features.shape[0]//256  
    n_iter = n_iter + 1 if features.shape[0]%256 != 0 else n_iter
    tmp_probs = []
    for i in range(n_iter):
        if i == n_iter-1:
            image = features[i*256:]
        else:
            image = features[i*256:(i+1)*256]
        image = torch.tensor(image).to('cuda')
        image = image / torch.norm(image, dim=-1, keepdim=True)
        tmp_probs.append(image @ text_embedding.T)
    tmp_probs = torch.cat(tmp_probs)
    male_probs = tmp_probs[:,:11].mean(dim=1)
    female_probs = tmp_probs[:,11:].mean(dim=1)
    probs = torch.stack([male_probs, female_probs], dim=1)
    probs = torch.softmax(probs, dim=1)
    return probs.cpu().numpy()
