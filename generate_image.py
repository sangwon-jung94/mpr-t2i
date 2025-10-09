import torch
import argparse
from utils import set_seed, check_log_dir
import networks
import os
from face_detector import FaceDetector
from torchvision import transforms
import pickle
import json
from transformers import  CLIPTokenizer
from diffusers.utils import numpy_to_pil

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
    parser.add_argument('--group', type=str, nargs='+', default=['gender','age','race'])    
    parser.add_argument('--prompt-path', type=str, default='prompt path')
    parser.add_argument('--lamb', type=float, default=0)

    args = parser.parse_args()

    if args.trainer not in ['scratch','scratch_1',  'fairdiffusion', 'entigen', 'itigen'] and args.model_path is None:
        raise ValueError("Model path should not be None if trainer is not scratch, fairdiffusion or entigen")

    set_seed(args.seed)

    gen = torch.Generator(device='cuda')
    gen.manual_seed(0)

    if 'finetuning' in args.trainer:
        model = networks.ModelFactory.get_model(modelname=args.model, train=True)
        model = model.to("cuda", torch.float16)

        model.load_lora_weights(args.model_path)  
        print('Loaded lora weights')
        
    elif args.model_path is None:
        model = networks.ModelFactory.get_model(modelname=args.model, train=False)
        model = model.to('cuda')

    # only for entigen
    base_path = f'datasets/{args.trainer}'
    if 'finetuning' in args.trainer and args.lamb != 0:
        base_path = f'datasets/{args.trainer}_lamb{args.lamb}'

    if 'scratch' not in args.trainer:
        group_name = "".join([g[0].upper() for g in args.group])
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

        # check the folders
        _concept_path = concept if len(concept.split(" ")) == 1 else "_".join(concept.split(" "))
        path = os.path.join(base_path, _concept_path)
        print("path : ", path)
        check_log_dir(path)
        path_filtered = os.path.join(path, 'filtered')
        check_log_dir(path_filtered)
        
        # make prompts
        prefix = 'an' if concept[0].lower() in ['a','e','i','o','u'] else 'a'
        prompt = template + f"{prefix} {concept}"
        
        #if concept in traits[:4]:
        if concept in traits:
            prompt += " person"

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
            
            if img_num == 0:
                print("Generation starts with the prompt of ", prompt)

            with torch.no_grad():
                if 'finetuning' in args.trainer:
                    images = generation_for_finetuning(model, prompt, args.n_gen_per_iter)#, cnt=cnt)
                else:  
                    images = model(prompt=prompt, num_inference_steps=25, num_images_per_prompt=args.n_gen_per_iter, generator=gen).images

            image_tensors = torch.stack([transform(image) for image in images])

            flags, bboxs = face_detector.process_tensor_image(image_tensors)

            total_generations += len(images)
            filtered_images += sum(~flags)
            bbox_idx = 0
            for j, flag in enumerate(flags):
                image = images[j]
                if flag:
                    image.save(f"{path}/{img_num}.png")
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
        
def generation_for_finetuning(model, prompts, n_generation, cnt=0, num_denoising_steps = 25):
    weight_dtype_high_precision = torch.float32
    weight_dtype = torch.float16
    device = model.device
    guidance_scale = 7.5

    noises = torch.randn(
        [n_generation,4,64,64],
        dtype=weight_dtype_high_precision
    ).to(device)

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

    prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
    prompt_embeds = prompt_embeds.to(weight_dtype)

    noise_scheduler.set_timesteps(num_denoising_steps)

    latents = noises
    with torch.no_grad():
        for i, t in enumerate(noise_scheduler.timesteps):
            latent_model_input = torch.cat([latents.to(weight_dtype)] * 2)
            latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t)
            
            noises_pred = unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
            ).sample
            noises_pred = noises_pred.to(weight_dtype_high_precision)
            
            noises_pred_uncond, noises_pred_text = noises_pred.chunk(2)
            noises_pred = noises_pred_uncond + guidance_scale * (noises_pred_text - noises_pred_uncond)
            
            latents = noise_scheduler.step(noises_pred, t, latents).prev_sample

        latents = 1 / vae.config.scaling_factor * latents
        images = vae.decode(latents.to(vae.dtype)).sample
        images = (images / 2 + 0.5).clamp(0,1)
        
        # Equivalent to diffusers.VaeImageProcessor.pt_to_numpy
        images = images.cpu().permute(0, 2, 3, 1).numpy()
        images = numpy_to_pil(images)    
    return images
        

if __name__ == "__main__":
    main()
