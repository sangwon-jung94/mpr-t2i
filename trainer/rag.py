#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import os, sys

from trainer import GenericTrainer
from trainer.image_utils import FaceFeatsModel, image_grid, plot_in_grid, expand_bbox, crop_face, image_pipeline, get_face_feats, get_face
import data_handler
import networks
from mpr.mpr import oracle_function
from mpr.mpr import getMPR

import itertools
import logging
import math
import shutil
import json
import pytz
import random
from datetime import datetime
from tqdm.auto import tqdm
import copy
import yaml
from pathlib import Path
import pickle
import wandb

import torch.nn.functional as F

import torch
from torch import nn
from torchvision.models.mobilenetv3 import mobilenet_v3_large, MobileNet_V3_Large_Weights
from torch.utils.data import Dataset
from torchvision import transforms

import numpy as np
import scipy

import kornia

import transformers
import diffusers
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor, CLIPVisionModelWithProjection
from accelerate.logging import get_logger

import diffusers
from diffusers.optimization import get_scheduler

# you MUST import torch before insightface
# otherwise onnxruntime, used by FaceAnalysis, can only use CPU
from insightface.app import FaceAnalysis
from copy import deepcopy

my_timezone = pytz.timezone("Asia/Singapore")

os.environ["WANDB__SERVICE_WAIT"] = "300"  # set to DETAIL for runtime logging.

logger = get_logger(__name__)

class Trainer(GenericTrainer):
    def __init__(self, **kwargs):
        super(Trainer, self).__init__(**kwargs)
        self.modelname = self.args.functionclass

    # def _load_model(self):        
    
    def train(self, accelerator=None, loader=None):
        # loader = data_handler.get_loader(self.args)
        self.accelerator = accelerator
        self.weight_dtype_high_precision = torch.float32
        self.weight_dtype = torch.float32
        if self.accelerator != None:
            if self.accelerator.mixed_precision == "fp16":
                self.weight_dtype = torch.float16
            elif self.accelerator.mixed_precision == "bf16":
                self.weight_dtype = torch.bfloat16

        model = self.model
        
        self.tokenizer = CLIPTokenizer.from_pretrained(
            model.name_or_path,
            subfolder="tokenizer"
            )
        self.text_encoder = model.text_encoder.to(self.accelerator.device,dtype=self.weight_dtype)
        self.vae = model.vae.to(self.accelerator.device, dtype=self.weight_dtype)
        self.unet = model.unet.to(self.accelerator.device, dtype=self.weight_dtype)
        self.noise_scheduler = model.scheduler
        print('just ckecking the noise_scheduler:', self.noise_scheduler)

        self.vision_encoder = networks.ModelFactory.get_model(modelname='CLIP').to(self.accelerator.device)
        
        # uncond_tokens = [""] 
        # max_length = prompt_embeds.shape[1]
        # print(max_length)
        # uncond_input = self.tokenizer(
        #         uncond_tokens,
        #         padding="max_length",
        #         max_length=max_length,
        #         truncation=True,
        #         return_tensors="pt",
        #     )
        # uncond_input["input_ids"] = uncond_input["input_ids"].to(self.accelerator.device)
        # uncond_input["attention_mask"] = uncond_input["attention_mask"].to(self.accelerator.device)
        # negative_prompt_embeds = self.text_encoder(
        #     uncond_input["input_ids"],
        #     uncond_input["attention_mask"],
        # )
        # negative_prompt_embeds = negative_prompt_embeds[0]
        # print(negative_prompt_embeds.shape)
        # prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds]).to(self.weight_dtype)
        
        # We only train the additional adapter LoRA layers
        self.text_encoder.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.vision_encoder.requires_grad_(False)
        self.unet.enable_gradient_checkpointing()
        self.vae.enable_gradient_checkpointing()

        # For mixed precision training we cast all non-trainable weigths (self.vae, non-lora text_encoder and non-lora unet) to half-precision
        # as these weights are only used for inference, keeping weights in full precision is not required.

        # Make linear projectors
        prompt = "a photo of person" 
        prompts_token = self.tokenizer([prompt], return_tensors="pt", padding=True)
        prompts_token["input_ids"] = prompts_token["input_ids"].to(self.accelerator.device)
        prompts_token["attention_mask"] = prompts_token["attention_mask"].to(self.accelerator.device)

        prompt_embeds = self.text_encoder(
            prompts_token["input_ids"],
            prompts_token["attention_mask"],
        )
        prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds.to(self.weight_dtype)
        print(prompt_embeds.shape)
        max_length = prompt_embeds.shape[1]

        token_embedding_dim = prompt_embeds.shape[-1]
        self.linear_projector = linear_projector = nn.Sequential(
            nn.Linear(512, token_embedding_dim),
            nn.ReLU()
        ).to(self.accelerator.device, dtype=self.weight_dtype_high_precision)
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        self.linear_projector.apply(init_weights)
        # nn.Linear(512, token_embedding_dim).to(self.accelerator.device, dtype=self.weight_dtype_high_precision)
        # torch.nn.init.xavier_uniform_(self.linear_projector.weight)
        self.linear_projector.requires_grad_(True)
        params_to_optimize = self.linear_projector.parameters()
        print(params_to_optimize)
        optimizer = torch.optim.AdamW(
            params_to_optimize,
            lr=self.args.learning_rate,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            weight_decay=self.args.adam_weight_decay,
            eps=self.args.adam_epsilon,
        )
        
        # self-make a simple dataloader
        # the created train_dataloader_idxs should be identical across devices
        random.seed(self.args.seed+1)

        lr_scheduler = get_scheduler(
            self.args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=self.args.lr_warmup_steps * self.accelerator.num_processes,
            num_training_steps=self.args.max_train_steps * self.accelerator.num_processes,
            num_cycles=self.args.lr_num_cycles,
            power=self.args.lr_power,
        )
        
        self.linear_projector, optimizer, lr_scheduler = self.accelerator.prepare(self.linear_projector, optimizer, lr_scheduler)
        
        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num Iterations = {self.args.iterations}")
        logger.info(f"  Total optimization steps = {self.args.max_train_steps}")
        self.global_step = 0

        # Only show the progress bar once on each machine.
        self.wandb_tracker = self.accelerator.get_tracker("wandb", unwrap=True)
        
        self.global_step = 0
        for name, param in self.linear_projector.named_parameters():
            print(name, param.requires_grad)
        torch.autograd.set_detect_anomaly(True)
        self.text_encoder.train()
        self.unet.train()
        self.vae.train()
        while self.global_step < self.args.iterations:
            # num_denoising_steps = random.choices(range(19,24), k=1)
            # torch.distributed.broadcast_object_list(num_denoising_steps, src=0)
            # num_denoising_steps = num_denoising_steps[0]
            # self.noise_scheduler.set_timesteps(num_denoising_steps)

            # grad_coefs = []
            # for i, t in enumerate(self.noise_scheduler.timesteps):
            #     grad_coefs.append( self.noise_scheduler.alphas_cumprod[t].sqrt().item() * (1-self.noise_scheduler.alphas_cumprod[t]).sqrt().item() / (1-self.noise_scheduler.alphas[t].item()) )
            # grad_coefs = np.array(grad_coefs)
            # grad_coefs /= (math.prod(grad_coefs)**(1/len(grad_coefs)))
            train_loss = 0.0
            for image1, image2, idxs in tqdm(loader):
                self.global_step += 1
                if self.global_step > self.args.iterations:
                    break
                image2 = image2.to(self.accelerator.device, dtype=self.weight_dtype)
                image1 = image1.to(self.accelerator.device)
                latents = self.vae.encode(image2).latent_dist.sample()
                latents = latents * self.vae.config.scaling_factor
                # latents = latents.to(self.weight_dtype)
                # Sample noise that we'll add to the latents   
                
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
                
                # Generate image tokens
                image_embeds = self.vision_encoder.encode_image(image1).to(self.weight_dtype) 

                image_token_embeds = self.linear_projector(image_embeds)
                image_token_embeds = image_token_embeds / image_token_embeds.norm(dim=-1, keepdim=True)
                image_token_embeds = image_token_embeds.to(self.weight_dtype)
                # print(image_token_embeds.dtype)
                
                # Concatenate the image tokens with text tokens
                image_token_embeds = image_token_embeds.unsqueeze(1)
                _prompt_embeds = prompt_embeds.repeat(bsz, 1, 1)
                # repeaet prompt_embeds as much as the number of batch size
                
                text_image_embeds = torch.cat([_prompt_embeds, image_token_embeds], dim=1)
                text_image_embeds = text_image_embeds.to(self.weight_dtype)

                # repeat text_image as much as the number of batch size
                model_pred = self.unet(noisy_latents, timesteps, text_image_embeds,  return_dict=False)[0]

                target = noise
                # loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                loss = F.mse_loss(model_pred, target, reduction="mean")
                print(f"loss: {loss.item()}")

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(self.args.train_GPU_batch_size)).mean()
                train_loss += avg_loss.item() #/ args.gradient_accumulation_steps

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(self.linear_projector.parameters(), self.args.max_grad_norm)
                self.model_sanity_print(self.linear_projector, "check No.1, text_encoder: after self.accelerator.backward()")
                #for p in self.linear_projector.parameters():
                #    print(p.grad)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                # if self.accelerator.is_main_process:
                    # if self.global_step % self.args.checkpointing_steps == 0:
                    #     # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                    #     if self.args.checkpoints_total_limit is not None:
                    #         name = "checkpoint_tmp"
                    #         clean_checkpoint(self.args.ckpts_save_dir, name, self.args.checkpoints_total_limit)

                    #     save_path = os.path.join(self.args.ckpts_save_dir, f"checkpoint_tmp-{self.global_step}")
                    #     self.accelerator.save_state(save_path)
                    #     torch.save(self.linear_projector.state_dict(), save_path)
                    
                    #     logger.info(f"Accelerator checkpoint saved to {save_path}")
            print(f"train_loss: {train_loss/len(loader)}")
            if self.accelerator.is_main_process:
                #if self.global_step % self.args.checkpointing_steps_long == 0:
                    # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`

                save_path = os.path.join(self.args.ckpts_save_dir, f"checkpoint-{self.global_step}")
                self.accelerator.save_state(save_path)
                # save linear projector parameter
                torch.save(self.linear_projector.state_dict(), save_path+'.pt')

                logger.info(f"Accelerator checkpoint saved to {save_path}")

    def model_sanity_print(self, model, state):
        params = [p for p in model.parameters()]
        print(f"\t{self.accelerator.device}; {state};\n\t\tparam[0]: {params[0].flatten()[0].item():.8f};\tparam[0].grad: {params[0].grad.flatten()[0].item():.8f}")


def clean_checkpoint(ckpts_save_dir, name, checkpoints_total_limit):
    checkpoints = os.listdir(ckpts_save_dir)
    checkpoints = [d for d in checkpoints if d.startswith(name)]
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

    # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
    if len(checkpoints) >= checkpoints_total_limit:
        num_to_remove = len(checkpoints) - checkpoints_total_limit + 1
        removing_checkpoints = checkpoints[0:num_to_remove]

        logger.info(
            f"chekpoint name:{name}, {len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
        )
        logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

        for removing_checkpoint in removing_checkpoints:
            removing_checkpoint = os.path.join(ckpts_save_dir, removing_checkpoint)
            shutil.rmtree(removing_checkpoint)

