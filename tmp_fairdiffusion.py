from semdiffusers import SemanticEditPipeline
import pickle

import torch
import numpy as np
from collections import defaultdict

import os
os.environ["WANDB_MODE"]="offline"

import argparse
from utils import set_seed, getMPR, feature_extraction, make_result_path, group_estimation, compute_similarity,check_log_dir
from copy import copy


def main():
    parser = argparse.ArgumentParser(description='representational-generation')
    
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--model', type=str, default='SD_14')
    parser.add_argument('--concept', type=str, default='firefighter')
    parser.add_argument('--group', type=str, nargs='+', default=['gender','age','race'])
    parser.add_argument('--n-generations', type=int, default=10000)

    args = parser.parse_args()

    group_idx ={
        'gender' : 0,
        'age' : 1,
        'race' : 2
    }
    group_dic = {
        'gender' : ['male', 'female'],
        'age' : ['young', 'old'],
        'race' : ['East Asian', 'Indian', 'Black', 'White', 'Middle Eastern', 'Latino_Hispanic', 'Southeast Asian']
    }
    
    for group in args.group:
        if group not in group_dic.keys():
            raise ValueError(f"Group {group} not considered")

    device='cuda'

    pipe = SemanticEditPipeline.from_pretrained(
        # "runwayml/stable-diffusion-v1-5",
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float16
    ).to(device)

    print("Model loaded")

    gen = torch.Generator(device=device)
    gen.manual_seed(4)

    file = open("adjectives.txt", "r") 
    data = file.read() 
    adjectives = data.replace('\n', ' ').split(" ") 
    adjectives = np.array(adjectives)
    file.close()
    # adjectives = [a for i, a in enumerate(adjectives) if i+1 in [3, 10,  16]]

    base_path = f'/n/holyscratch01/calmon_lab/Lab/datasets/fairdiffusion'
    if args.trainer != 'scratch':
        group_name = "".join([g[0].upper() for g in args.group])
        base_path = os.path.join(base_path, group_name)
    base_path = os.path.join(base_path, args.model)
    check_log_dir(base_path)

    # if len(args.group) < 3:
        # for group in args.group:
            # base_path+=f'_{group}'
    
    base_path = os.path.join(base_path, args.model)
    check_log_dir(base_path)

    group_ratio = np.load('group_ratio.npy')
    marginalize_idx = [group_idx[group] for group in group_idx.keys() if group not in args.group]
    group_ratio = group_ratio.sum(axis=tuple(marginalize_idx))
    group_prob = group_ratio / group_ratio.sum()
    print('group ratio : ', group_ratio)
    # n_samples_per_group = np.round(group_ratio * 1000).astype(int)

    # make prompt list
    prompt_list = []
    edit_weights = []
    for group in args.group:
        for item in group_dic[group]:
            prompt_list.extend([f'{item} person'])
        edit_weights.extend([2/len(group_dic[group])]*len(group_dic[group]))

    num_prompt = len(prompt_list)
    reverse_editing_direction = [True]*num_prompt
    edit_warmup_steps=[10]*num_prompt # Warmup period for each concept
    edit_guidance_scale=[4]*num_prompt # Guidance scale for each concept
    edit_threshold=[0.95]*num_prompt # Threshold for each concept. Threshold equals the percentile of the latent space that will be discarded. I.e. threshold=0.99 uses 1% of the latent dimensions
    # edit_weights=[1]*num_prompt # Weights of the individual concepts against each other
    # edit_weights=[1,1,1,1,1/3.5,1/3.5,1/3.5,1/3.5,1/3.5,1/3.5,1/3.5]

    path = os.path.join(base_path, args.concept)
    check_log_dir(path)
    img_num = 0
    n_subgroup = defaultdict(int)

    while img_num < args.n_generations:
        print(f'n_generation : {args.n_generations}, img_num : {img_num}')
        # choose group
        flat_index = np.random.choice(a=group_prob.size, p=group_prob.flatten())
        idxs = np.unravel_index(flat_index, group_prob.shape)
        n_subgroup[idxs] += 1

        #make reverse_editing_direction
        _reverse_editing_direction = copy(reverse_editing_direction)
        pos = 0
        for i, group in enumerate(args.group):
            _reverse_editing_direction[pos+idxs[i]] = False
            pos += len(group_dic[group])

        #choose adjective
        # adj_idx = np.random.choice(a=adjectives.size)
        # adjective = adjectives[adj_idx]

        out = pipe(prompt=f'A portrait face of a {args.concept}', num_images_per_prompt=5, guidance_scale=7.5,generator=gen,
                editing_prompt=prompt_list, 
                reverse_editing_direction=_reverse_editing_direction, # Direction of guidance i.e. decrease the first and increase the second concept
                edit_warmup_steps=edit_warmup_steps, # Warmup period for each concept
                edit_guidance_scale=edit_guidance_scale, # Guidance scale for each concept
                edit_threshold=edit_threshold, # Threshold for each concept. Threshold equals the percentile of the latent space that will be discarded. I.e. threshold=0.99 uses 1% of the latent dimensions
                edit_momentum_scale=0.3, # Momentum scale that will be added to the latent guidance
                edit_mom_beta=0.6, # Momentum beta
                # edit_weights=edit_weights # Weights of the individual concepts against each other
                edit_weights=[1]*11
                )
        images = out.images
        for j, image in enumerate(images):
            image.save(f"{path}/{img_num}.png")
            img_num += 1
    
    with open(f"{path}/n_subgroup.pkl", 'wb') as file:
        pickle.dump(n_subgroup, file)    
    print(n_subgroup)

if __name__ == '__main__':

    # args = get_args()    
    np.set_printoptions(precision=4)
    torch.set_printoptions(precision=4)

    set_seed(0)

    main()


    # wandb.finish()
