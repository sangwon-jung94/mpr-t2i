
import pandas as pd
import torch
import ast
import copy
import numpy as np
from transformers import CLIPProcessor, CLIPModel
import time
from trainer import GenericTrainer
class Trainer(GenericTrainer):
    # concepts = ['attendant', 'cashier', 'teacher','nurse', 'assistant',
    #              'secretary', 'cleaner', 'receptionist','clerk','counselor',
    #                'designer', 'hairdresser', 'writer', 'housekeeper', 'baker', 
    #                'librarian','tailor','driver','supervisor', 'janitor', 'cook', 
    #                'laborer', 'construction worker', 'developer', 'carpenter','manager',
    #                  'lawyer', 'farmer', 'salesperson', 'physician', 'firefighter', 
    #                  'analyst', 'mechanic', 'sheriff', 'CEO', 'doctor', 'chef']
    # concepts = ["CEO", "firefighter", "chef", "cook", "therapist", "housekeeper", "pilot", "flight attendant", "taxi driver", "nurse"]
    concepts = ["CEO", "firefighter"]
#    concepts = ["computer programmer"]#, "computer programmer for the ENIAC”"]
    
    def __init__(self, **kwargs):
        super(Trainer, self).__init__(**kwargs)

        self.group_type = self.args.trainer_group

        old_texts = []
        concepts_ = []
        for concept in self.concepts:
            old_texts.append(f'image of {concept}')
            old_texts.append(f'photo of {concept}')
            old_texts.append(f'portrait of {concept}')
            old_texts.append(f'picture of {concept}')
            old_texts.append(f'{concept}')
            concepts_.extend([concept]*5)

        attributes = []
        if len(self.group_type) == 1:
            attributes.extend(self.group_dic[self.group_type[0]])    
        elif len(self.group_type) == 3:
            for age in self.group_dic['age']:
                for gender in self.group_dic['gender']:
                    for race in self.group_dic['race']:
                        attributes.append(f'{age} {gender} {race}')

        self.old_texts = old_texts
        self.new_texts = [[text.replace(concepts_[idx], att) for att in attributes] for idx, text in enumerate(old_texts) ]
    
        df = pd.read_csv('profession_prompts.csv')

        retain_texts = list(df.profession.unique())
        ### default concepts to erase
  
        old_texts_lower = [text.lower() for text in old_texts]
        retain_texts = [text for text in retain_texts if text.lower() not in old_texts_lower]
        self.retain_texts = retain_texts
        
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        # load group ratio
        group_ratio = np.load('group_ratio.npy')
        marginalize_idx = [self.group_idx[group] for group in self.group_idx.keys() if group not in self.group_type]
        group_ratio = group_ratio.sum(axis=tuple(marginalize_idx))
        if len(self.group_type)==3:
            group_ratio = group_ratio.swapaxes(0,1)
        self.group_prob = torch.tensor(group_ratio / group_ratio.sum())
        self.group_prob = self.group_prob.flatten()
        self.group_prob[0] = 0.5
        self.group_prob[1] = 0.5
        
        self.lamb = self.args.lamb

    def train(self, add=False, layers_to_edit=None, erase_scale=1, preserve_scale = 0.1, with_to_k=True, num_images=10, accelerate=None):
        old_text_ = self.old_texts
        new_text_ = self.new_texts
        retain_text_ = self.retain_texts
        
        if torch.cuda.is_available():
            self.model = self.model.to('cuda')
        ldm_stable = self.model

        ### collect all the cross attns modules
        max_bias_diff = 0.05
        sub_nets = ldm_stable.unet.named_children()
        ca_layers = []
        for net in sub_nets:
            if 'up' in net[0] or 'down' in net[0]:
                for block in net[1]:
                    if 'Cross' in block.__class__.__name__ :
                        for attn in block.attentions:
                            for  transformer in attn.transformer_blocks:
                                ca_layers.append(transformer.attn2)
            if 'mid' in net[0]:
                for attn in net[1].attentions:
                    for  transformer in attn.transformer_blocks:
                        ca_layers.append(transformer.attn2)
        
        ### get the value and key modules
        projection_matrices = [l.to_v for l in ca_layers]
        og_matrices = [copy.deepcopy(l.to_v) for l in ca_layers]
        if with_to_k:
            projection_matrices = projection_matrices + [l.to_k for l in ca_layers]
            og_matrices = og_matrices + [copy.deepcopy(l.to_k) for l in ca_layers]

        ## reset the parameters
        num_ca_clip_layers = len(ca_layers)
        for idx_, l in enumerate(ca_layers):
            l.to_v = copy.deepcopy(og_matrices[idx_])
            projection_matrices[idx_] = l.to_v
            if with_to_k:
                l.to_k = copy.deepcopy(og_matrices[num_ca_clip_layers + idx_])
                projection_matrices[num_ca_clip_layers + idx_] = l.to_k

        ### check the layers to edit (by default it is None; one can specify)
        layers_to_edit = ast.literal_eval(layers_to_edit) if type(layers_to_edit) == str else layers_to_edit
        lamb = self.lamb
        lamb = ast.literal_eval(lamb) if type(lamb) == str else lamb
            
        ### Format the edits
        old_texts = []
        new_texts = []
        for old_text, new_text in zip(old_text_, new_text_):
            old_texts.append(old_text)
            n_t = []
            for t in new_text:
                if (old_text.lower() not in t.lower()) and add:
                    n_t.append(t + ' ' +old_text)
                else:
                    n_t.append(t)
            if len(n_t) == 1:
                n_t = n_t*2
            new_texts.append(n_t)
        if retain_text_ is None:
            ret_texts = ['']
            retain = False
        else:
            ret_texts = retain_text_
            retain = True

        print(old_texts, new_texts)
        # desired_ratios = [torch.ones(len(c))/len(c) for c in new_texts ]
        desired_ratios = [self.group_prob for c in new_texts]
        print(desired_ratios)
        weight_step = 0.1
        weights = [torch.zeros(len(c)) for c in new_texts ]
        #################################### START OUTER LOOP #########################
        for i in range(30):
            iteration_start_time = time.time()
            max_ratio_gap = max_bias_diff
            if i == 0:
                prev_ratio = None
                ratio_diff = None
            else:
                prev_ratio = ratios
                ratio_diff = max_change
            ratios = self.get_ratios(prev_ratio = prev_ratio, ratio_diff=ratio_diff,max_ratio_gap=max_ratio_gap, 
                                     concepts=old_texts, classes=new_texts, num_samples= num_images)
            if i == 0 :
                init_ratios = ratios
            print(f'{i} iteration ratio : {ratios}')
            max_change = [(ratio - desired_ratio).abs().max() for ratio, desired_ratio in zip(ratios,desired_ratios)]

            if max(max_change) < max_bias_diff:
                print(f'All concepts are debiased at Iteration:{i}')
                break
            
            weights_delta = [weight_step * (desired_ratio - ratio) for ratio, desired_ratio in zip(ratios, desired_ratios)]
            weights_delta = [weights_delta[idx] if max_c>max_bias_diff else weights_delta[idx]*0 for idx, max_c in enumerate(max_change)]
            
            # check if the ratio is attained. If so, add it to preservation and skip the ratios check again
            ret_text_add = [old_texts[idx] for idx, weight in enumerate(weights_delta) if weight[0]==0]
            if len(ret_text_add)>0:
                ret_texts = ret_texts+ret_text_add
                ret_texts = list(np.unique(ret_texts))
            weights = weights_delta
    #         weights = [weight + weights_delta[idx] for idx, weight in enumerate(weights)]
            ### START EDIT
            print( ' weights ; ', weights)
            for layer_num in range(len(projection_matrices)):
                if (layers_to_edit is not None) and (layer_num not in layers_to_edit):
                    continue

                #### prepare input k* and v*
                with torch.no_grad():
                    #mat1 = \lambda W + \sum{v k^T}
                    mat1 = lamb * projection_matrices[layer_num].weight

                    #mat2 = \lambda I + \sum{k k^T}
                    mat2 = lamb * torch.eye(projection_matrices[layer_num].weight.shape[1], device = projection_matrices[layer_num].weight.device)

                    for cnt, t in enumerate(zip(old_texts, new_texts)):
                        old_text = t[0]
                        new_text = t[1]
                        texts = [old_text]
                        texts = texts + new_text
                        text_input = ldm_stable.tokenizer(
                            texts,
                            padding="max_length",
                            max_length=ldm_stable.tokenizer.model_max_length,
                            truncation=True,
                            return_tensors="pt",
                        )
                        text_embeddings = ldm_stable.text_encoder(text_input.input_ids.to(ldm_stable.device))[0]
                        old_emb = text_embeddings[0]
                        final_token_idx = text_input.attention_mask[0].sum().item()-2
                        # print(final_token_idx)
                        # print(text_input.attention_mask)
                        final_token_idx_new = [text_input.attention_mask[i].sum().item()-2 for i in range(1, len(text_input.attention_mask))]
                        farthest = max(final_token_idx_new+[final_token_idx])
                        new_emb = text_embeddings[1:]

                        # e.g., this extracts firefighter in "a photo of firefighter"
                        context = old_emb.detach()[final_token_idx:len(old_emb)-max(0,farthest-final_token_idx)]
                        values = []
                        with torch.no_grad():
                            for layer in projection_matrices:
                                o_embs = layer(old_emb).detach()
                                o_embs = o_embs[final_token_idx:len(o_embs)-max(0,farthest-final_token_idx)]
    #                             print(f'O_EMBS: {final_token_idx}-{len(o_embs)-max(0,farthest-final_token_idx)}')
                                embs = layer(new_emb[:]).detach()
                                target = o_embs
                                for j, emb in enumerate(embs):
                                    u = emb
                                    u = u[final_token_idx_new[j]:len(u)-max(0,farthest-final_token_idx_new[j])]
    #                                 print(f'U_{j}: {final_token_idx_new[j]}-{len(u)-max(0,farthest-final_token_idx_new[j])}')
                                    u = u / u.norm()
                                    target += (weights[cnt][j]*o_embs.norm())*u 
                                values.append(target.detach())    
                        context_vector = context.reshape(context.shape[0], context.shape[1], 1)
                        context_vector_T = context.reshape(context.shape[0], 1, context.shape[1])
                        value_vector = values[layer_num].reshape(values[layer_num].shape[0], values[layer_num].shape[1], 1)
                        for_mat1 = (value_vector @ context_vector_T).sum(dim=0)
                        for_mat2 = (context_vector @ context_vector_T).sum(dim=0)
                        mat1 += erase_scale*for_mat1
                        mat2 += erase_scale*for_mat2

                    for old_text, new_text in zip(ret_texts, ret_texts):
                        text_input = ldm_stable.tokenizer(
                            [old_text, new_text],
                            padding="max_length",
                            max_length=ldm_stable.tokenizer.model_max_length,
                            truncation=True,
                            return_tensors="pt",
                        )
                        text_embeddings = ldm_stable.text_encoder(text_input.input_ids.to(ldm_stable.device))[0]
                        old_emb, new_emb = text_embeddings
                        context = old_emb.detach()
                        values = []
                        with torch.no_grad():
                            for layer in projection_matrices:
                                values.append(layer(new_emb[:]).detach())
                        context_vector = context.reshape(context.shape[0], context.shape[1], 1)
                        context_vector_T = context.reshape(context.shape[0], 1, context.shape[1])
                        value_vector = values[layer_num].reshape(values[layer_num].shape[0], values[layer_num].shape[1], 1)
                        for_mat1 = (value_vector @ context_vector_T).sum(dim=0)
                        for_mat2 = (context_vector @ context_vector_T).sum(dim=0)
                        mat1 += preserve_scale*for_mat1
                        mat2 += preserve_scale*for_mat2
                        #update projection matrix
                    param = mat1 @ torch.inverse(mat2)#.type(torch.float16)
                    projection_matrices[layer_num].weight = torch.nn.Parameter(param)
            iteration_end_time = time.time()  # Step 3
            iteration_duration = iteration_end_time - iteration_start_time  # Step 4
            print(f"Iteration {i} took {iteration_duration} seconds.")  # Step 5

        print(f'Current model status: Edited "{str(old_text_)}" into "{str(new_texts)}" and Retained "{str(retain_text_)}"')
        # return ldm_stable, weights, init_ratios, ratios
        return ldm_stable


    def get_ratios(self, prev_ratio, ratio_diff, max_ratio_gap, concepts, classes, num_samples=10, num_loops=3):
        ldm_stable = self.model

        seeds = np.random.randint(5000,size=5) 
        ratios = []
        for idx, concept in enumerate(concepts):
            if ratio_diff is not None:
                if ratio_diff[idx] < max_ratio_gap:
                    print(f'Bypassing Concept {idx+1}')
                    ratios.append(prev_ratio[idx])
                    continue
            prompt = f'{concept}'
            probs_full = []
            test_prompts = [f'{class_}' for class_ in classes[idx]]
            with torch.no_grad():
                for seed in seeds:
        #             if i == num_loops:
        #                 break
                    g = torch.Generator(device='cpu')
                    g.manual_seed(int(seed))
                    images = ldm_stable(prompt,num_images_per_prompt=num_samples, num_inference_steps=20, generator = g).images

                    inputs = self.clip_processor(text=test_prompts, images=images, return_tensors="pt", padding=True)

                    outputs = self.clip_model(**inputs)
                    logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
                    probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
                    tmax = probs.max(1, keepdim=True)[0]
                    mask = probs.ge(tmax)
                    probs_full.append(mask.float())
                    
            ratios.append(torch.cat(probs_full).mean(axis=0))
            print(ratios)
    #     male = float(probs[0][0])
        return ratios
