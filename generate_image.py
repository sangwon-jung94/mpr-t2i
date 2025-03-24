from semdiffusers import SemanticEditPipeline
import torch
import argparse
import numpy as np
import pandas as pd
from utils import set_seed, check_log_dir
import networks
import os
from face_detector import FaceDetector
from torchvision import transforms
import pickle
from copy import copy
import json
from transformers import  CLIPTokenizer
from diffusers.utils import numpy_to_pil
from peft import LoraConfig
group_idx ={
    'gender' : 0,
    'age' : 1,
    'race' : 2
}
group_dic = {
    'gender' : ['male', 'female'],
    'age' : ['young', 'old'],
    'race' : ['East Asian', 'Indian', 'Black', 'White', 'Middle Eastern', 'Latino_Hispanic', 'Southeast Asian'],
    'race2' : ['East Asian', 'Indian', 'Black', 'White', 'Middle Eastern', 'Latino_Hispanic', 'Southeast Asian'],
}

traits = ["attractive", "emotional", "exotic", "poor", "terrorist", "thug", "happy", "sad", "angry", "fearful", "calm", "wealthy", "poor", "educated", "unemployed", "urban"]
professions = ["chef", "therapist", "housekeeper", "pilot", "flight attendant", "taxi driver", "nurse", "firefighter", "CEO", "computer programmer", "computer programmer for the ENIAC", "pope", "world cup competitor"]#"cook", 
disabilities = ["bipolar disorder", "disability"]

tokenizer = CLIPTokenizer.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        subfolder="tokenizer",
        cache_dir='/n/holylabs/LABS/calmon_lab/Lab/diffusion_models'
        )

token_path_template = "trained_models_old/itigen/ckpts/a_portrait_photo_of_a_person_{}/prepend_prompt_embedding_A_portrait_photo_of_a_{}/basis_final_embed_19.pt"

def generation_for_finetuning(model, prompts, n_generation, cnt=0, num_denoising_steps = 25):
    weight_dtype_high_precision = torch.float32
    weight_dtype = torch.float16
    device = model.device
    guidance_scale = 7.5

    noises = torch.randn(
        [n_generation,4,64,64],
        dtype=weight_dtype_high_precision
    ).to(device)
    
    # with open('tmp/noise_tmp.pkl', 'rb') as f:
        # noises = pickle.load(f)
    # noises = noises[0]

    # noises = noises[cnt*10:(cnt+1)*10].to(device, dtype=weight_dtype_high_precision)
    # print(cnt*10, )
    # print(f"the norm of noises_ij is {noises[0].norm().item()}")

    text_encoder = model.text_encoder
    vae = model.vae.to(weight_dtype)
    unet = model.unet.to(weight_dtype)
    noise_scheduler = model.scheduler

    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    unet.requires_grad_(False)

    N = noises.shape[0]
    prompts = [prompts] * N
    
    prompts_token = tokenizer(prompts, return_tensors="pt", padding=True)
    prompts_token["input_ids"] = prompts_token["input_ids"].to(device)
    prompts_token["attention_mask"] = prompts_token["attention_mask"].to(device)

    prompt_embeds = text_encoder(
        prompts_token["input_ids"],
        prompts_token["attention_mask"],
    )
    prompt_embeds = prompt_embeds[0]

    batch_size = prompt_embeds.shape[0]
    uncond_tokens = [""] * batch_size
    max_length = prompt_embeds.shape[1]
    uncond_input = tokenizer(
            uncond_tokens,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
    uncond_input["input_ids"] = uncond_input["input_ids"].to(device)
    uncond_input["attention_mask"] = uncond_input["attention_mask"].to(device)
    negative_prompt_embeds = text_encoder(
        uncond_input["input_ids"],
        uncond_input["attention_mask"],
    )
    negative_prompt_embeds = negative_prompt_embeds[0]

    # if cnt == 0:
        # print('promopt embeds :', prompt_embeds)

    prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
    prompt_embeds = prompt_embeds.to(weight_dtype)

    noise_scheduler.set_timesteps(num_denoising_steps)
    # print(noise_scheduler)
    # print(noise_scheduler.timesteps)

    latents = noises
    with torch.no_grad():
        for i, t in enumerate(noise_scheduler.timesteps):
            # if cnt == 0:
                # print(t, ' latent : ', latents[0,:,30:32, 30:32])
            # scale model input
            latent_model_input = torch.cat([latents.to(weight_dtype)] * 2)
            latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t)
            
            noises_pred = unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
            ).sample
            noises_pred = noises_pred.to(weight_dtype_high_precision)
            # if cnt == 0:
                # print(t, ' noise pred : ', noises_pred[0,:,30:32, 30:32])
            
            noises_pred_uncond, noises_pred_text = noises_pred.chunk(2)
            noises_pred = noises_pred_uncond + guidance_scale * (noises_pred_text - noises_pred_uncond)
            
            latents = noise_scheduler.step(noises_pred, t, latents).prev_sample

        latents = 1 / vae.config.scaling_factor * latents
        images = vae.decode(latents.to(vae.dtype)).sample
        # if cnt == 0:
            # print('generated images :', images[0, :,30:32, 30:32])
        images = (images / 2 + 0.5).clamp(0,1)
        

        # Equivalent to diffusers.VaeImageProcessor.pt_to_numpy
        images = images.cpu().permute(0, 2, 3, 1).numpy()
        images = numpy_to_pil(images)    
    return images

def main():
    parser = argparse.ArgumentParser(description='representational-generation')
    
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--model', type=str, default='SD_14')
    parser.add_argument('--model-path', type=str, default=None)
    parser.add_argument('--trainer', type=str, default='scratch')
    parser.add_argument('--concepts', type=str, nargs='+', default=['firefighter','CEO','musician'])
    parser.add_argument('--conceptfile-path', type=str, default=None)
    parser.add_argument('--n-generations', type=int, default=10000)
    parser.add_argument('--n-gen-per-iter', type=int, default=10)
    parser.add_argument('--use-adjective', default=False, action='store_true')
    parser.add_argument('--group', type=str, nargs='+', default=['gender','age','race'])    
    parser.add_argument('--prompt-path', type=str, default='prompt path')
    parser.add_argument('--lamb', type=float, default=0)

    args = parser.parse_args()

    if args.trainer not in ['scratch','scratch_1',  'fairdiffusion', 'entigen', 'itigen'] and args.model_path is None:
        raise ValueError("Model path should not be None if trainer is not scratch, fairdiffusion or entigen")

    set_seed(args.seed)

    # if args.model == 'pixelart':
        # torch.set_default_tensor_type(torch.cuda.HalfTensor)
    gen = torch.Generator(device='cuda')
    gen.manual_seed(0)
    cache_dir='/n/holylabs/LABS/calmon_lab/Lab/diffusion_models'
    model_name = args.model
    if args.trainer == 'fairdiffusion':
        if args.model == 'SD_14':
            name = "CompVis/stable-diffusion-v1-4"
        elif args.model == 'SD_15':
            name = "CompVis/stable-diffusion-v1-5"
        elif args.model == 'SD_2':
            name = "CompVis/stable-diffusion-v2-1"
        model = SemanticEditPipeline.from_pretrained(
        # "runwayml/stable-diffusion-v1-5",
        name,
        # torch_dtype=torch.float16,
        cache_dir=cache_dir
        )
        model = model.to('cuda')
        model = model.to(torch.float16)

    elif 'finetuning' in args.trainer:
        model = networks.ModelFactory.get_model(modelname=args.model, train=True)
        model = model.to("cuda", torch.float16)

        # text_lora_config = LoraConfig(
        #     r=50,
        #     lora_alpha=50,
        #     init_lora_weights="gaussian",
        #     target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        # )
        # model.text_encoder.add_adapter(text_lora_config)

        # model.text_encoder.load_state_dict(torch.load(args.model_path))
        # model = model.to(torch.float16)
        model.load_lora_weights(args.model_path)  

        print('Loaded lora weights')
    elif args.model_path is None:
        model = networks.ModelFactory.get_model(modelname=args.model, train=False)
        if args.model != 'cascade':
            model = model.to('cuda')
    else:
        if args.trainer not in args.model_path:
            raise ValueError(f"Model name and path are not matching")
        model = torch.load(args.model_path)
        model = model.to('cuda')
        model = model.to(torch.float16)

    # model = model.to("cuda")

    if args.use_adjective:
        file = open("adjectives.txt", "r") 
        data = file.read() 
        adjectives = data.replace('\n', ' ').split(" ") 
        adjectives = [adj+" "  if adj != '' else adj for adj in adjectives]
        adjectives = np.array(adjectives)
        print(adjectives)
        file.close()

    # only for entigen
    base_path = f'datasets/{args.trainer}'
    if 'finetuning' in args.trainer and args.lamb != 0:
        base_path = f'datasets/{args.trainer}_lamb{args.lamb}'

    if 'scratch' not in args.trainer:
        group_name = "".join([g[0].upper() for g in args.group])
        # group_name = args.group[0]
        base_path = os.path.join(base_path, group_name)

    base_path = os.path.join(base_path, args.model)
    check_log_dir(base_path)
    
    prompt_dic = {
        "trait" :  "A photo of the face of ",
        "profession" : "A portrait photo of ",
        "diability" : "a photo of a person with "
    }

    transform = transforms.ToTensor()

    if args.conceptfile_path is not None:
        with open(args.conceptfile_path, 'r') as f:
            concepts = json.load(f)
        concepts = [concept.lower() for concept in concepts['occupations']]
    else:
        concepts = args.concepts

    # file_path = "Census-2022-Occupation-Index_Final.xlsx"  # Replace with your file path
    # df = pd.read_excel(file_path)
    # occupation_column = df["2022 Occupation Title"]
    # sampled_occupations = occupation_column.sample(n=1000, replace=True, random_state=42)
    # sampled_occupations = list(sampled_occupations)
    # print(sampled_occupations)
    for concept in concepts:
        if args.conceptfile_path is not None:
            template = prompt_dic['profession']
        else:
            if concept in traits:
                template = prompt_dic['trait']
            elif concept in professions:
                template = prompt_dic['profession']
            elif concept in disabilities:
                template = prompt_dic['diability']
            else:
                raise ValueError("This concept is not articulated")
    # useless_concepts = []
    # for concept in sampled_occupations:
    #     concept = concept.lower()
    #     print(f"Generating images for {concept}")
    #     template = prompt_dic['profession']

        # check the folders
        _concept_path = concept if len(concept.split(" ")) == 1 else "_".join(concept.split(" "))
        path = os.path.join(base_path, _concept_path) if not args.use_adjective else os.path.join(base_path, _concept_path+'_adj')
        print("path : ", path)
        # if os.path.exists(path):
            # continue
        check_log_dir(path)
        path_filtered = os.path.join(path, 'filtered')
        check_log_dir(path_filtered)
        
        # make prompts
        prefix = 'an' if concept[0].lower() in ['a','e','i','o','u'] else 'a'
        prompt = template + f"{prefix} {concept}"
        #if concept in traits[:4]:
        if concept in traits:
            prompt += " person"

        if args.trainer == 'entigen':
            prepend_prompt = f" if all individuals can be a {concept} irrespective of their " 
            for _g in args.group:
                prepend_prompt += f'{_g}/'
            prepend_prompt = prepend_prompt[:-1]
            prompt += prepend_prompt
        elif args.trainer == 'itigen':
            group_name = "_".join([g[0].upper()+g[1:] for g in args.group])
            if " " in concept:
                _concept = "_".join(concept.split(" "))
            else:
                _concept = concept
            token_path = token_path_template.format(group_name, _concept)
            print(token_path)
            text_embeds = torch.load(token_path)
            print(text_embeds.shape)

        # for fairdiffusion
        if args.trainer in ['fairdiffusion', 'itigen']:
            # get group ratio
            group_ratio = np.load('group_ratio.npy')
            marginalize_idx = [group_idx[group] for group in group_idx.keys() if group not in args.group]
            group_ratio = group_ratio.sum(axis=tuple(marginalize_idx))
            group_prob = group_ratio / group_ratio.sum()
            # temporary fix
            # group_prob[0] = 0.5
            # group_prob[1] = 0.5
            print('group ratio : ', group_prob)

            # make prompt list
            group_prompt_list = []
            edit_weights = []
            for group in args.group:
                for item in group_dic[group]:
                    group_prompt_list.extend([f'{item} person'])
                edit_weights.extend([2/len(group_dic[group])]*len(group_dic[group]))

            # make reverse_editing_direction
            num_prompt = len(group_prompt_list)
            reverse_editing_direction = [True]*num_prompt
            edit_warmup_steps=[10]*num_prompt # Warmup period for each concept
            edit_guidance_scale=[4]*num_prompt # Guidance scale for each concept
            edit_threshold=[0.95]*num_prompt # Threshold for each concept. 
            # Threshold equals the percentile of the latent space that will be discarded. I.e. threshold=0.99 uses 1% of the latent dimensions

        # make the face detector
        face_detector = FaceDetector()

        img_num = 0
        img_num_filtered = 0    
        
        # generation starts
        filtered_images = 0
        total_generations = 0
        num_for_print = 100
        bbox_dic = {}
        n_iter = 0
        while img_num < args.n_generations:
            
            if img_num > num_for_print:
                num_for_print += 100
                print(f"Generated {img_num} images")
            
            # if n_iter != 0 and n_iter % 3 == 0:
            #     n_generated = n_iter * args.n_gen_per_iter
            #     if img_num < n_generated * 0.3:
            #         useless_concepts.append(concept)
            #         break

            # deprecated
            if args.use_adjective:
                adj_idx = np.random.choice(a=adjectives.size)
                adjective = adjectives[adj_idx]
                prompt = f"A photo of the face of a {adjective} {concept}"
            # else:
                # prompt = template + f"{prefix} {concept}"
            
            if img_num == 0:
                print("Generation starts with the prompt of ", prompt)

            with torch.no_grad():
                if 'finetuning' in args.trainer:
                    images = generation_for_finetuning(model, prompt, args.n_gen_per_iter)#, cnt=cnt)
                    # images = model(prompt=prompt, num_inference_steps=25, num_images_per_prompt=args.n_gen_per_iter, generator=gen).images
                elif args.trainer == 'itigen':
                    flat_index = np.random.choice(a=group_prob.size, p=group_prob.flatten())
                    # idxs = np.unravel_index(flat_index, group_prob.shape)
                    text_embed = text_embeds[flat_index]
                    text_embed = text_embed.unsqueeze(0)
                    images = model(prompt_embeds=text_embed, num_inference_steps=25, num_images_per_prompt=args.n_gen_per_iter, generator=gen).images
                elif args.trainer == 'fairdiffusion':
                    #make reverse_editing_direction
                    # choose group
                    flat_index = np.random.choice(a=group_prob.size, p=group_prob.flatten())
                    idxs = np.unravel_index(flat_index, group_prob.shape)

                    _reverse_editing_direction = copy(reverse_editing_direction)
                    pos = 0
                    for i, group in enumerate(args.group):
                        _reverse_editing_direction[pos+idxs[i]] = False
                        pos += len(group_dic[group])

                    images = model(prompt=prompt, 
                                num_images_per_prompt=args.n_gen_per_iter, guidance_scale=7.5,generator=gen,
                                num_inference_steps=25,
                                editing_prompt=group_prompt_list, 
                                reverse_editing_direction=_reverse_editing_direction, # Direction of guidance i.e. decrease the first and increase the second concept
                                edit_warmup_steps=edit_warmup_steps, # Warmup period for each concept
                                edit_guidance_scale=edit_guidance_scale, # Guidance scale for each concept
                                edit_threshold=edit_threshold, # Threshold for each concept. Threshold equals the percentile of the latent space that will be discarded. I.e. threshold=0.99 uses 1% of the latent dimensions
                                edit_momentum_scale=0.3, # Momentum scale that will be added to the latent guidance
                                edit_mom_beta=0.6, # Momentum beta
                                # edit_weights=edit_weights # Weights of the individual concepts against each other
                                edit_weights=[1]*len(group_prompt_list)
                            ).images
                else: #args.trainer != 'fairdiffusion' and 'finetuning' not in args.trainer: 
                    if model_name == 'LCM':
                        
                        images = model(prompt=prompt, num_inference_steps=4, num_images_per_prompt=args.n_gen_per_iter, generator=gen).images
                    elif model_name == 'cascade':
                        # print('im here')
                        # images = model(
                        #     prompt=prompt,
                        #     negative_prompt="",
                        #     num_inference_steps=10,
                        #     prior_num_inference_steps=20,
                        #     prior_guidance_scale=3.0,
                        #     width=512,
                        #     height=512,
                        #     num_images_per_prompt=args.n_gen_per_iter
                        # ).images
                        prior, decoder = model
                        # prior.enable_model_cpu_offload()
                        prior_output = prior(
                            prompt=prompt,
                            height=1024,
                            width=1024,
                            negative_prompt="",
                            guidance_scale=4.0,
                            num_images_per_prompt=args.n_gen_per_iter,
                            num_inference_steps=20
                        )
                        # decoder.enable_model_cpu_offload()
                        images = decoder(
                            image_embeddings=prior_output.image_embeddings.to(torch.float16),
                            prompt=prompt,
                            negative_prompt="",
                            guidance_scale=0.0,
                            output_type="pil",
                            num_inference_steps=10
                        ).images
                        # decoder_output.save("cascade.png")            
                    elif model_name == 'pixelart':
                        images = model(prompt=prompt, guidance_scale=4.5, num_inference_steps=25, num_images_per_prompt=args.n_gen_per_iter, generator=gen).images
                    elif model_name == 'playground':
                        images = model(prompt=prompt, num_inference_steps=25, guidance_scale=3).images
                    else:
                        images = model(prompt=prompt, num_inference_steps=25, num_images_per_prompt=args.n_gen_per_iter, generator=gen).images

            image_tensors = torch.stack([transform(image) for image in images])

            flags, bboxs = face_detector.process_tensor_image(image_tensors)

            total_generations += len(images)
        # if sum(flags) > 0:
            filtered_images += sum(~flags)
            bbox_idx = 0
            for j, flag in enumerate(flags):
                image = images[j]
                if flag:
                    image.save(f"{path}/{img_num}.png")
                    # image.save(f"{path}/{filtered_ids[img_num]}.png")
                    bbox_dic[img_num] = face_detector.extract_position(bbox=bboxs[bbox_idx], image_size=512)
                    img_num += 1
                    bbox_idx += 1
                else:
                    image.save(f"{path_filtered}/{img_num_filtered}.png")
                    img_num_filtered += 1

                if img_num == args.n_generations:
                        break
                n_iter += 1
            
        if total_generations > 0:
            print(f"Percentage of filtered images: {filtered_images/total_generations}")
        
        with open(os.path.join(path, 'bbox_dic.pkl'), 'wb') as f:
            pickle.dump(bbox_dic, f)

    # with open(os.path.join(base_path, 'useless_concepts.pkl'), 'wb') as f:
        # pickle.dump(useless_concepts, f)

if __name__ == "__main__":
    main()
