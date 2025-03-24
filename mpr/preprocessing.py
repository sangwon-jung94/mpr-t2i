
import torch
import numpy as np 
import os
from transformers import BlipProcessor
import pickle
from tqdm import tqdm
import torch
import torch.nn as nn
import clip 
from transformers import Blip2Processor, Blip2Model, AutoProcessor, AutoTokenizer, Blip2ForConditionalGeneration
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
    dataset_name = args.refer_dataset if not query else args.query_dataset
    path = dataloader.dataset.dataset_path
    
    feature_dic = {}
    
    for ver in ['normal', 'face_detect']:
    # for ver in ['normal']:#, 'face_detect']:
        # make feature vectors

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

        if args.vision_encoder == 'BLIP':
            feature_extractor = BlipExtractor(encoder, args)
        
        elif args.vision_encoder == 'CLIP':
            feature_extractor =  CLIPExtractor(encoder)
    
        for batch in tqdm(dataloader):
            image, label, idxs =  batch

            if torch.cuda.is_available():
                image = image.cuda()
            # image = image.to(torch.float16)
            # with torch.autocast("cuda"):
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
        if group in ['gender', 'race', 'race2', 'age', 'face', 'skintone', 'emotion','paths', 'eyeglasses']:
            feature = feature_dic['face_detect']
            # feature = feature_dic['normal']
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
    path = '/n/holylabs/LABS/calmon_lab/Lab/datasets/mpr_stuffs/'
    if group in ['gender', 'age','race', 'race2']:
        with open(os.path.join(path,'clfs',f'fairface_{vision_encoder_name}_clf_{group}.pkl'), 'rb') as f:
            clf = pickle.load(f)
            # clf.best_estimator_.coef_ = clf.best_estimator_.coef_.astype(np.float16)
            # clf.best_estimator_.intercept_ = clf.best_estimator_.intercept_.astype(np.float16)
            estimated_group = clf.predict_proba(features)
            if onehot:
                # if estimated_group.shape[-1] == 1:
                    # estimated_group = estimated_group>0.5
                # else:
                one_hot_indices = np.argmax(estimated_group, axis=1)
                estimated_group = np.eye(estimated_group.shape[1])[one_hot_indices]

    elif group == 'face':
        with open(os.path.join(path,'clfs',f'celeba_{vision_encoder_name}_clf_{group}.pkl'), 'rb') as f:                
            clf = pickle.load(f)
        estimated_group_list = []
        attrs = ['Bald', 'Bangs', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Double_Chin', 'Eyeglasses',
            'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'Mustache', 'No_Beard', 'Sideburns', 'Smiling', 'Wearing_Hat']
        for attr in attrs:
            estimated_group = clf[attr].predict_proba(features)
            if onehot:
                one_hot_indices = np.argmax(estimated_group, axis=1)
                estimated_group = np.eye(estimated_group.shape[1])[one_hot_indices]
                # print statistics
                print(f'{attr} 1/0: {np.sum(one_hot_indices)}/{len(one_hot_indices)-np.sum(one_hot_indices)}')
            estimated_group_list.append(estimated_group)
        estimated_group = np.concatenate(estimated_group_list, axis=1)

    # if you find that BLIP doesn't work well, please double check the face_detect version of dataloader
    elif group == 'wheelchair':
        model = BLIPPredictor()
        with torch.no_grad():
            item_presence = []
            transform = loader.dataset.transform
            loader.dataset.transform = model.vis_processor
            # tmp_loader = DataLoader(loader.dataset, batch_size=1)
            for data in loader:
                images, labels, idxs = data
                # images['pixel_values'] = images['pixel_values']
                result = model.forward(images)
                item_presence.append(result)
            loader.dataset.transform = transform
            item_presence = torch.cat(item_presence)
            item_presence = torch.stack((1-item_presence, item_presence), dim=-1)
        print(item_presence[:,0])
        estimated_group = item_presence
        print(f'wheelchair 1/0: {torch.sum(item_presence[:,1])}/{item_presence.shape[0]-torch.sum(item_presence[:,0])}')
    
    elif group == 'paths':
        estimated_group = path_estimate(features, encoder)
        if onehot:
            one_hot_indices = np.argmax(estimated_group, axis=1)
            estimated_group = np.eye(estimated_group.shape[1])[one_hot_indices]
        print(estimated_group.shape)
        print(estimated_group.sum(axis=0))

    elif group == 'emotion':
        estimated_group = emotion_estimate(features, encoder)
        if onehot:
            one_hot_indices = np.argmax(estimated_group, axis=1)
            estimated_group = np.eye(estimated_group.shape[1])[one_hot_indices]
        print(estimated_group.shape)
        print(estimated_group.sum(axis=0))

    elif group == 'eyeglasses':
        estimated_group = eyeglasses_estimate(features, encoder, query)
        if onehot:
            one_hot_indices = np.argmax(estimated_group, axis=1)
            estimated_group = np.eye(estimated_group.shape[1])[one_hot_indices]
        print(estimated_group.shape)
        print(estimated_group.sum(axis=0))

    # elif g == 'skintone':
    #     from skintone_esti import FanCrop
    #     face_detector = FanCrop()
    #     for image, label in loader:
    #         valid_idxs, images = face_detector.crop_images(image)

            
    estimated_group = estimated_group * 2 - 1
    
    # normalization
    # outputs = outputs / np.linalg.norm(outputs, axis=-1, keepdims=True)    
    # for i in range(outputs.shape[0]):
        # outputs[i, :] = outputs[i, :] / np.linalg.norm(outputs[i, :])

    return estimated_group


def path_estimate(features, encoder):

    genders = ['man', 'woman', 'non-binary']    
    ethnicities = ['African-American', 'American Indian', 'Asian', 'Black', 'Caucasian', 'East Asian', 'First Nations', 'Hispanic', 'Indigenous','American', 'Latino','Latinx', 'Native American', 'Multiracial', 'Pacific Islander', 'South Asian', 'Southeast Asian', 'White']
    adjectives = ['ambitious','assertive','confident','decisive','determined','intelligent','outspoken','self-confident','stubborn','unreasonable','committed','supportive','sensitive','emotional','gendtle','honest','modest','compassionate','considerate','pleasant']

    total_embeddings = []
    texts = []
    for adjective in adjectives:
        for gender in genders:
            tmp = []
            for ethnicity in ethnicities:
                text = f'A photo of a {adjective} {gender} {ethnicity} person'
                texts.append(text)
                # print(text)
                # tmp.append(text)
                text_inputs = clip.tokenize(text).to('cuda')
                with torch.no_grad():
                    text_embedding = encoder.encode_text(text_inputs)
                text_embedding = text_embedding / torch.norm(text_embedding, dim=-1, keepdim=True)
                tmp.append(text_embedding)
            text_embedding = torch.cat(tmp)


            # Generate text embeddings
            # with torch.no_grad():
                # text_embedding = encoder.encode_text(text_inputs)
            # print(text_embedding.shape)
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
            # print(tmp_probs.shape)
            total_embeddings.append(tmp_probs)
    total_embeddings = torch.cat(total_embeddings, dim=1)
    idxs = total_embeddings.argmax(dim=1)
    # if features.shape[0] > 5002:
        # for idx in idxs:
            # print(texts[idx])
    # print(total_embeddings.shape)
    return total_embeddings.cpu().numpy()

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

def eyeglasses_estimate(features, encoder, query=True):
    tmp = []
    prompts = ['A photo of a person wearing eye glasses', 'A photo of a person not wearing eye glasses']
    if query:
        for prompt in prompts:
            text = prompt
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
        probs = torch.softmax(tmp_probs, dim=1)
    else:
        n_samples = features.shape[0]
        # give True with the probability of 0.5 for each sample
        probs = torch.rand(n_samples, 1)
        print(probs.mean())
        probs = torch.cat([1-probs, probs], dim=1)
    return probs.cpu().numpy()

# old version
def _blip_extraction(encoder, dataloader, args, query=True):
    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
#     raw_text=["What word best describes the person's appearance?"]
#     text = processor(text=raw_text, return_tensors='pt')
#     input_ids=text['input_ids']
#     attention_mask=text['attention_mask']
    
#     raw_dummy_label = ['0']
#     dummy_label = processor(text=raw_dummy_label, return_tensors='pt')['input_ids']

#     outputs = []
#     professions = []
        
#     # professions = []
#     for batch in dataloader:
#         with torch.no_grad():
#             image = batch[0]
#             batch_size = image.shape[0]
            
#             _input_ids = input_ids.repeat(batch_size, 1)
#             _attention_mask = attention_mask.repeat(batch_size, 1)
#             _dummy_label = dummy_label.repeat(batch_size, 1)

#             if torch.cuda.is_available():
#                 image = image.cuda()
#                 _input_ids = _input_ids.cuda()
#                 _attention_mask = _attention_mask.cuda()
#                 _dummy_label = _dummy_label.cuda()
            
#             output = encoder.forward(pixel_values=image, input_ids=_input_ids, attention_mask=_attention_mask, labels=_dummy_label, return_dict=True)
#             outputs.append(output['image_embeds'])
            
#             if query:
#                 professions.append(batch[1])

#     outputs = torch.cat(outputs)
#     outputs = outputs.mean(axis=1)
#     return outputs if not query else (outputs, torch.cat(professions))

