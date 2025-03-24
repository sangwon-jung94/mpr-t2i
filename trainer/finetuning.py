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


import torch
from torch import nn
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
from diffusers import (
    UNet2DConditionModel,
)
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
# 
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.utils import convert_state_dict_to_diffusers

from peft import LoraConfig
from peft.utils import get_peft_model_state_dict

# you MUST import torch before insightface
# otherwise onnxruntime, used by FaceAnalysis, can only use CPU
# from insightface.app import FaceAnalysis
from copy import deepcopy
from utils import get_current_device, AverageMeter, gpu_mem_usage

my_timezone = pytz.timezone("Asia/Singapore")

os.environ["WANDB__SERVICE_WAIT"] = "300"  # set to DETAIL for runtime logging.


class PromptsDataset(Dataset):
    def __init__(
        self,
        prompts,
    ):
        self.prompts = prompts
    def __len__(self):
        return len(self.prompts)
    def __getitem__(self, i):
        return self.prompts[i]


# def parse_args(input_args=None):
#     parser = argparse.ArgumentParser(description="Script to finetune Stable Diffusion for debiasing purposes.")

#     env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
#     if env_local_rank != -1 and env_local_rank != args.local_rank:
#         args.local_rank = env_local_rank

#     return args

logger = get_logger(__name__)

class Trainer(GenericTrainer):
    def __init__(self, **kwargs):
        super(Trainer, self).__init__(**kwargs)
        self.modelname = self.args.functionclass
        if "boolean" in self.modelname:
            self.model_depth = int(self.modelname[7:])
        self.ver = self.args.finetuning_ver
        self.temp = self.args.temp
        self.n_cs = self.args.n_cs
        self.eniac = self.args.eniac
        self.cache_dir = '/n/holylabs/LABS/calmon_lab/Lab/diffusion_models'

    def _set_group_clfs(self, groups):
        path = '/n/holylabs/LABS/calmon_lab/Lab/datasets/mpr_stuffs/'
        self.group_classifier_dic = {}
        vision_encoder = networks.ModelFactory.get_model(modelname=self.args.vision_encoder)  
        self.group_classifier_dic['vision_encoder'] = vision_encoder
        self.group_classifier_dic['vision_encoder'].to(self.accelerator.device, dtype=self.weight_dtype)
        self.group_classifier_dic['vision_encoder'].requires_grad_(False)
        self.group_classifier_dic['vision_encoder'].eval()
        if self.args.vision_encoder == 'CLIP':
            # _, transform = clip.load("ViT-L/14", device= 'cpu')
            self.clip_img_mean_for_group = torch.tensor([0.48145466, 0.4578275, 0.40821073]).reshape([-1,1,1]).to(self.accelerator.device, dtype=self.weight_dtype) # mean is based on range [0,1]
            self.clip_img_std_for_group = torch.tensor([0.26862954, 0.26130258, 0.27577711]).reshape([-1,1,1]).to(self.accelerator.device, dtype=self.weight_dtype) # std is based on range [0,1]

        output_dim = 0
        for group in groups:
            if group in ['gender', 'age','race']:
                with open(os.path.join(path,'clfs',f'fairface_{self.args.vision_encoder}_clf_{group}.pkl'), 'rb') as f:
                    clf = pickle.load(f)
                
                in_features = clf.best_estimator_.coef_.shape[1]
                out_features = clf.best_estimator_.coef_.shape[0]
                
                linear_clf = nn.Linear(in_features, out_features)
                linear_clf.weight.data = torch.tensor(clf.best_estimator_.coef_, dtype=self.weight_dtype)
                linear_clf.bias.data = torch.tensor(clf.best_estimator_.intercept_, dtype=self.weight_dtype)

                self.group_classifier_dic[group] = linear_clf
                self.group_classifier_dic[group].to(self.accelerator.device, dtype=self.weight_dtype)
                self.group_classifier_dic[group].requires_grad_(False)
                self.group_classifier_dic[group].eval()
                
                if out_features == 1:
                    out_features += 1
                output_dim += out_features

            elif group == 'face':
                continue
        self.output_dim = output_dim

    def _make_cartesian(self, dataset):
        dim = dataset.shape[1]

        if isinstance(dataset, np.ndarray):
            for d in range(2, self.model_depth + 1):
                for idxs in itertools.combinations(range(dim), d):
                    dataset = np.concatenate((dataset, np.prod(dataset[:, idxs], axis=1).reshape(-1, 1)), axis=1)
        elif isinstance(dataset, torch.Tensor):
            for d in range(2, self.model_depth + 1):
                for idxs in itertools.combinations(range(dim), d):
                    dataset = torch.cat((dataset, torch.prod(dataset[:, idxs], dim=1).unsqueeze(1)), dim=1)
        else:
            raise TypeError("Unsupported dataset type. Must be numpy array or torch tensor.")
        
        return dataset
    
    def _load_model(self):
        # Load the model
        # Convert the data type of model weights to fp16 if required

        model = self.model
        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.model.name_or_path,
            subfolder="tokenizer",
            cache_dir = self.cache_dir
            )
        self.text_encoder = model.text_encoder
        self.vae = model.vae
        self.unet = model.unet
        self.noise_scheduler = model.scheduler
        
        # We only train the additional adapter LoRA layers
        self.text_encoder.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.unet.enable_gradient_checkpointing()
        self.vae.enable_gradient_checkpointing()

        # For mixed precision training we cast all non-trainable weigths (self.vae, non-lora text_encoder and non-lora unet) to half-precision
        # as these weights are only used for inference, keeping weights in full precision is not required.
        self.weight_dtype_high_precision = torch.float32
        self.weight_dtype = torch.float32
        if self.accelerator != None:
            if self.accelerator.mixed_precision == "fp16":
                self.weight_dtype = torch.float16
            elif self.accelerator.mixed_precision == "bf16":
                self.weight_dtype = torch.bfloat16

        # Move self.unet, self.vae and text_encoder to device and cast to self.weight_dtype
        self.text_encoder.to(self.accelerator.device, dtype=self.weight_dtype)
        self.unet.to(self.accelerator.device, dtype=self.weight_dtype)
        self.vae.to(self.accelerator.device, dtype=self.weight_dtype)

    def _set_lora_params(self):
        if self.args.train_text_encoder:
            self.eval_text_encoder = CLIPTextModel.from_pretrained(
                self.model.name_or_path, 
                subfolder="text_encoder",
                cache_dir = self.cache_dir
                )
            self.eval_text_encoder.requires_grad_(False)
            self.eval_text_encoder.to(self.accelerator.device, dtype=self.weight_dtype)
            
        if self.args.train_unet:        
            self.eval_unet = UNet2DConditionModel.from_pretrained(
            self.model.name_or_path, 
            subfolder="unet",
            )
            self.eval_unet.requires_grad_(False)
            self.eval_unet.to(self.accelerator.device, dtype=self.weight_dtype)

        if self.args.train_unet:
            unet_lora_config = LoraConfig(
                r=self.args.rank,
                lora_alpha=self.args.rank,
                init_lora_weights="gaussian",
                target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            )

            self.unet.add_adapter(unet_lora_config, a=None)
            unet_lora_layers = list(filter(lambda p: p.requires_grad, self.unet.parameters()))
            for p in unet_lora_layers:
                p.data = p.data.float()
            self.unet_lora_layers = unet_lora_layers
            
            for p in unet_lora_layers.parameters():
                torch.distributed.broadcast(p, src=0)
            
            self.unet_lora_ema = EMAModel(filter(lambda p: p.requires_grad, self.unet.parameters()), decay=self.args.EMA_decay)
            self.unet_lora_ema.to(self.accelerator.device)
            
            # print to check whether unet lora & ema is identical across devices
            print(f"{self.accelerator.device}; unet lora init to: {self.unet_lora_layers[0].flatten()[1]:.6f}; unet lora ema init to: {self.unet_lora_ema.shadow_params[0].flatten()[1]:.6f}")
            
            # print(f"{self.accelerator.device}; unet lora init to: {list(unet_lora_layers)[0].flatten()[1]:.6f}; unet lora ema init to: {unet_lora_ema.shadow_params[0].flatten()[1]:.6f}")

        if self.args.train_text_encoder:
            text_lora_config = LoraConfig(
                r=self.args.rank,
                lora_alpha=self.args.rank,
                init_lora_weights="gaussian",
                target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
            )
            self.text_encoder.add_adapter(text_lora_config)
            text_encoder_lora_layers = {}
            for name, param in self.text_encoder.named_parameters():
                if param.requires_grad:
                    text_encoder_lora_layers[name] = param
            for name, p in text_encoder_lora_layers.items():
                p.data = p.data.float()

            for name, p in text_encoder_lora_layers.items():
                torch.distributed.broadcast(p, src=0)

            text_encoder_lora_layers = CustomModel(text_encoder_lora_layers)
            # text_encoder_lora_layers = list(filter(lambda p: p.requires_grad, self.text_encoder.parameters()))
            # ensure that dtype is float32, even if rest of the model that isn't trained is loaded in fp16

            self.text_encoder_lora_layers = text_encoder_lora_layers
                        
            # self.text_encoder_lora_ema = EMAModel(filter(lambda p: p.requires_grad, self.text_encoder.parameters()), decay=self.args.EMA_decay)
            # self.text_encoder_lora_ema.to(self.accelerator.device)

            # print to check whether text_encoder lora & ema is identical across processes
            # print(f"{self.accelerator.device}; TE lora init to: {self.text_encoder_lora_layers[0].flatten()[1]:.6f}; TE lora ema init to: {self.text_encoder_lora_ema.shadow_params[0].flatten()[1]:.6f}")

    def _set_optimizer(self):
        if self.args.train_text_encoder and self.args.train_unet:
            params_to_optimize = self.unet_lora_layers + self.text_encoder_lora_layers
        elif self.args.train_text_encoder and not self.args.train_unet:
            params_to_optimize = self.text_encoder_lora_layers.parameters()
        elif not self.args.train_text_encoder and self.args.train_unet:
            # params_to_optimize = unet_lora_layers.parameters()
            params_to_optimize = self.unet_lora_layers

        optimizer = torch.optim.AdamW(
            params_to_optimize,
            lr=self.args.learning_rate,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            weight_decay=self.args.adam_weight_decay,
            eps=self.args.adam_epsilon,
        )
        
        lr_scheduler = get_scheduler(
            self.args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=self.args.lr_warmup_steps * self.accelerator.num_processes,
            num_training_steps=self.args.iterations * self.accelerator.num_processes,
            num_cycles=self.args.lr_num_cycles,
            power=self.args.lr_power,
        )
        
        optimizer, lr_scheduler = self.accelerator.prepare(
                optimizer, lr_scheduler
            )
        return optimizer, lr_scheduler
    
    def _set_face_feat_models(self):
        #######################################################
        # set up face_recognition and face_app on all devices
        # face_app = FaceAnalysis(
        #     name="buffalo_l",
        #     allowed_modules=['detection'], 
        #     providers=['CUDAExecutionProvider'], 
        #     provider_options=[{'device_id': self.accelerator.device.index}]
        #     )
        # face_app.prepare(ctx_id=0, det_size=(640, 640))
        
        clip_image_processoor = CLIPImageProcessor.from_pretrained(
            "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
            cache_dir = self.cache_dir
        )
        self.clip_vision_model_w_proj = CLIPVisionModelWithProjection.from_pretrained(
            "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
            cache_dir = self.cache_dir
        )
        self.clip_vision_model_w_proj.vision_model.to(self.accelerator.device, dtype=self.weight_dtype)
        self.clip_vision_model_w_proj.visual_projection.to(self.accelerator.device, dtype=self.weight_dtype)
        self.clip_vision_model_w_proj.requires_grad_(False)
        self.clip_vision_model_w_proj.gradient_checkpointing_enable()
        self.clip_img_mean = torch.tensor(clip_image_processoor.image_mean).reshape([-1,1,1]).to(self.accelerator.device, dtype=self.weight_dtype) # mean is based on range [0,1]
        self.clip_img_std = torch.tensor(clip_image_processoor.image_std).reshape([-1,1,1]).to(self.accelerator.device, dtype=self.weight_dtype) # std is based on range [0,1]
        

        self.dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        self.dinov2.to(self.accelerator.device, dtype=self.weight_dtype)
        self.dinov2.requires_grad_(False)
        self.dinov2_img_mean = torch.tensor([0.485, 0.456, 0.406]).reshape([-1,1,1]).to(self.accelerator.device, dtype=self.weight_dtype)
        self.dinov2_img_std = torch.tensor([0.229, 0.224, 0.225]).reshape([-1,1,1]).to(self.accelerator.device, dtype=self.weight_dtype)
        
        CE_loss = nn.CrossEntropyLoss(reduction="none")   

        # build opensphere model
        sys.path.append(Path(__file__).parent.parent.__str__())
        sys.path.append(Path(__file__).parent.parent.joinpath("opensphere").__str__())
        from opensphere.builder import build_from_cfg
        from opensphere.utils import fill_config

        with open(self.args.opensphere_config, 'r') as f:
            opensphere_config = yaml.load(f, yaml.SafeLoader)
        opensphere_config['data'] = fill_config(opensphere_config['data'])
        self.face_feats_net = build_from_cfg(
            opensphere_config['model']['backbone']['net'],
            'model.backbone',
        )
        self.face_feats_net = nn.DataParallel(self.face_feats_net)
        self.face_feats_net.load_state_dict(torch.load(self.args.opensphere_model_path))
        self.face_feats_net = self.face_feats_net.module
        self.face_feats_net.to(self.accelerator.device)
        self.face_feats_net.requires_grad_(False)
        self.face_feats_net.to(self.weight_dtype)
        self.face_feats_net.eval()
        
        self.face_feats_model = FaceFeatsModel(self.args.face_feats_path)
        self.face_feats_model.to(self.weight_dtype_high_precision)
        self.face_feats_model.to(self.accelerator.device)
        self.face_feats_model.eval()   
    

    def train(self, accelerator=None, loader=None):
        
        self.accelerator = accelerator

        self._load_model()
        self._set_lora_params()

        # Enable TF32 for faster training on Ampere GPUs,
        # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
        if self.args.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True

        self.optimizer, self.lr_scheduler = self._set_optimizer()

        # set the accelerator to use the model
        if self.args.train_text_encoder:
            # self.text_encoder_lora_layers, self.text_encoder_lora_ema = self.accelerator.prepare(self.text_encoder_lora_layers, self.text_encoder_lora_ema)
            self.text_encoder_lora_layers = self.accelerator.prepare(self.text_encoder_lora_layers)
            # self.accelerator.register_for_checkpointing(self.text_encoder_lora_ema)
        if self.args.train_unet:
            self.unet_lora_layers, self.unet_lora_ema = self.accelerator.prepare(self.unet_lora_layers, self.unet_lora_ema)
            self.accelerator.register_for_checkpointing(self.unet_lora_ema)

        # Dataset and DataLoaders creation:
        with open(self.args.prompt_occupation_path, 'r') as f:
            experiment_data = json.load(f)
        prompts_train = [prompt.format(occupation=occupation) for prompt in experiment_data["prompt_templates_train"] for occupation in experiment_data["occupations_train_set"]]
        
        train_dataset = PromptsDataset(prompts=prompts_train)
        self.args.num_update_steps_per_epoch = train_dataset.__len__()
        self.args.num_train_epochs = math.ceil(self.args.iterations / self.args.num_update_steps_per_epoch)

        # construct clip and dino for regualzrarization
        self._set_face_feat_models()
        # self-make a simple dataloader
        # the created train_dataloader_idxs should be identical across devices
        random.seed(self.args.seed+1)
        train_dataloader_idxs = []
        for epoch in range(self.args.num_train_epochs):
            idxs = list(range(train_dataset.__len__()))
            random.shuffle(idxs)
            train_dataloader_idxs.append(idxs)

        prompts_val = [prompt.format(occupation=occupation) for prompt in experiment_data["prompt_templates_test"] for occupation in experiment_data["occupations_val_set"]]
        
        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num prompts = {train_dataset.__len__()}")
        logger.info(f"  Num images per prompt = {self.args.train_images_per_prompt_GPU} (per GPU) * {self.accelerator.num_processes} (GPU)")
        logger.info(f"  Num Epochs = {self.args.num_train_epochs}")
        logger.info(f"  Total optimization steps = {self.args.max_train_steps}")
        self.global_step = 0
        first_epoch = 0
#        for param in self.unet.parameters():
#            if param.requires_grad:
#                param.data = param.data.float()

        # Potentially load in the weights and states from a previous save
        if self.args.resume_from_checkpoint:

            if not os.path.exists(self.args.resume_from_checkpoint):
                self.accelerator.print(
                    f"Checkpoint '{self.args.resume_from_checkpoint}' does not exist. Starting a new training run."
                )
                self.args.resume_from_checkpoint = None
            else:
                self.accelerator.print(f"Resuming from checkpoint {self.args.resume_from_checkpoint}")
                for param in self.text_encoder_lora_layers:
                    print(param.norm())
                self.accelerator.load_state(self.args.resume_from_checkpoint)
                print('after loading')
                for param in self.text_encoder_lora_layers:
                    print(param.norm())
                self.global_step = int(os.path.basename(self.args.resume_from_checkpoint).split("-")[1])

                resume_global_step = self.global_step
                first_epoch = self.global_step // self.args.num_update_steps_per_epoch
                resume_step = resume_global_step % (self.args.num_update_steps_per_epoch)
                # self.model.load_lora_weights(self.args.resume_from_checkpoint)
                # self.text_encoder_lora_layers = []
                # for name, param in self.model.text_encoder.named_parameters():
                    # self.text_encoder_lora_layers.append(param)
                        # param.data = param.data.float()
                # if self.args.train_text_encoder:
                    # self.text_encoder.load_lora_weights(self.args.resume_from_checkpoint)
                    # self.text_encoder_lora_ema.to(self.accelerator.device)
                    
                    # # need to recreate text_encoder_lora_ema_dict
                    # text_encoder_lora_ema_dict = {}
                    # for name, shadow_param in itertools.zip_longest(text_encoder_lora_params_name_order, self.text_encoder_lora_ema.shadow_params):
                    #     text_encoder_lora_ema_dict[name] = shadow_param
                    # assert text_encoder_lora_ema_dict.__len__() == text_encoder_lora_dict.__len__(), "length does not match! something wrong happened while converting lora params to a state dict."
                
                if self.args.train_unet:
                    self.unet_lora_ema.to(self.accelerator.device)

        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(self.global_step, self.args.max_train_steps), disable=not self.accelerator.is_local_main_process)
        progress_bar.set_description("Steps")
        self.wandb_tracker = self.accelerator.get_tracker("wandb", unwrap=True)

        curation_set = get_curation_set(self.args)
        if self.eniac:
            curation_set = np.array(([[-1,1]]))

        self.curation_set = curation_set
        if "boolean" in self.args.functionclass:
            curation_set = self._make_cartesian(curation_set)
        curation_set_tensor = torch.tensor(curation_set, dtype=self.weight_dtype, device=self.accelerator.device)
        mpr_set = {}
        self.oracle_function_set = {}
        for epoch in range(first_epoch, self.args.num_train_epochs):
            #torch.zeros([len(train_dataloader_idxs[0])*self.args.mpr_num_batches, self.args.num_group_attributes], dtype=self.weight_dtype, device=self.accelerator.device)
            for step, data_idx in enumerate(train_dataloader_idxs[epoch]):
                if accelerator.is_main_process:
                    print("At start, mam mem: {:.1f} GB ".format(gpu_mem_usage()))
                # Skip steps until we reach the resumed step
                if self.args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                    progress_bar.update(1)
                    continue

                if self.global_step == 0:
                    self.evaluation_step(prompts_val, self.global_step)

                # get prompt, should be identical across processes
                prompt_i = train_dataset.__getitem__(data_idx)
                if prompt_i not in mpr_set.keys():
                    mpr_set[prompt_i] = None
                    self.oracle_function_set[prompt_i] = []
                
                # generate noises, should differ by processes
                noises_i = torch.randn(
                    [self.args.train_images_per_prompt_GPU,4,64,64],
                    dtype=self.weight_dtype_high_precision
                    ).to(self.accelerator.device)

                self.accelerator.wait_for_everyone()
                self.optimizer.zero_grad()
                # logs = []
                # log_imgs = []

                # print noise to check if they are different by device
                noises_i_all = [noises_i.detach().clone() for i in range(self.accelerator.num_processes)]
                torch.distributed.all_gather(noises_i_all, noises_i)
                if self.accelerator.is_main_process:
                    now = datetime.now(my_timezone)
                    self.accelerator.print(
                        f"{now.strftime('%Y/%m/%d - %H:%M:%S')} --- epoch: {epoch}, step: {step}, global_step: {self.global_step}, prompt: {prompt_i}\n" +
                        " ".join([f"\tprocess idx: {idx}; noise: {noises_i_all[idx].flatten()[-1].item():.4f};" for idx in range(len(noises_i_all))])
                        )
                
                if self.accelerator.is_main_process:
                    logs_i = {
                        "loss_fair": [],
                        "loss_face": [],
                        "loss_CLIP": [],
                        "loss_DINO": [],
                        "loss": [],
                        "mpr": [],
                    }
                    log_imgs_i = {}

                num_denoising_steps = random.choices(range(19,24), k=1)
                torch.distributed.broadcast_object_list(num_denoising_steps, src=0)
                num_denoising_steps = num_denoising_steps[0]

                with torch.no_grad():
                    ################################################
                    # step 1: generate all images using the diffusion model being finetuned
                    images = []
                    N = math.ceil(noises_i.shape[0] / self.args.val_GPU_batch_size)
                    for j in range(N):
                        noises_ij = noises_i[self.args.val_GPU_batch_size*j:self.args.val_GPU_batch_size*(j+1)]
                        images_ij = self.generate_image_no_gradient(prompt_i, noises_ij, num_denoising_steps, which_text_encoder=self.text_encoder, which_unet=self.unet)
                        images.append(images_ij)
                    images = torch.cat(images)

                    face_indicators, face_bboxs, face_chips, face_landmarks, aligned_face_chips = get_face(images, self.args)
                    
                    preds_group, probs_group = self._get_group_predictions(face_chips, selector=face_indicators, fill_value=-1)
                    # print('in line 551, the size of probs_group :', probs_group.size())
                    n_faces_detected = face_indicators.sum().item()
                    probs_group = 2 * probs_group - 1

                    # update the MPR set
                    if n_faces_detected != 0:
                        # probs_group = probs_group[face_indicators]
                        if mpr_set[prompt_i] is None:
                            mpr_set[prompt_i] = probs_group ## probs_attributes is a tensor of shape [num_faces, num_group_attributes]
                        else:
                            mpr_set[prompt_i] = torch.cat([mpr_set[prompt_i], probs_group], dim=0)
                        
                        ## remove the oldest batch if the number of faces detected exceeds the number of faces that can be stored
                        if mpr_set[prompt_i].shape[0] >= self.args.mpr_num_batches*noises_i.shape[0]:
                            mpr_set[prompt_i] = mpr_set[prompt_i][-self.args.mpr_num_batches*noises_i.shape[0]:] 
                        print(f'the size of mpr_set for {prompt_i} is :', mpr_set[prompt_i].size())

                    face_feats = torch.ones([aligned_face_chips.shape[0],512], dtype=self.weight_dtype_high_precision, device=aligned_face_chips.device) * (-1)
                    if sum(face_indicators)>0:
                        face_feats_ = get_face_feats(self.face_feats_net, aligned_face_chips[face_indicators])
                        face_feats[face_indicators] = face_feats_

                    # _, face_real_scores = self.face_feats_model.semantic_search(face_feats, selector=face_indicators, return_similarity=True)

                    face_indicators_all, face_indicators_others = customized_all_gather(face_indicators, self.accelerator, return_tensor_other_processes=True)
                    self.accelerator.print(f"\tNum faces detected: {face_indicators_all.sum().item()}/{face_indicators_all.shape[0]}.")
                    if face_indicators_all.sum().item() == 0:
                        continue
                    
                    images_all = customized_all_gather(images, self.accelerator, return_tensor_other_processes=False)
                    face_bboxs_all = customized_all_gather(face_bboxs, self.accelerator, return_tensor_other_processes=False)
                    preds_group_all = customized_all_gather(preds_group, self.accelerator, return_tensor_other_processes=False)
                    probs_group_all = customized_all_gather(probs_group, self.accelerator, return_tensor_other_processes=False)

                    if self.accelerator.is_main_process:
                        probs_tmp = probs_group_all[(probs_group_all!=-1).all(dim=-1)]
                        logs_i["mpr"], _ = getMPR(self.args.trainer_group, mpr_set[prompt_i].cpu().numpy(), 
                                                curation_set=self.curation_set, 
                                                modelname=self.modelname, 
                                                normalize=self.args.normalize)                        

                    ################################################
                    # Step 3: generate all original images using the original diffusion model
                    # note that only targets from above will be used to compute loss
                    # all other variables will not be used below
                    images_ori = []
                    N = math.ceil(noises_i.shape[0] / self.args.val_GPU_batch_size)
                    for j in range(N):
                        noises_ij = noises_i[self.args.val_GPU_batch_size*j:self.args.val_GPU_batch_size*(j+1)]
                        if self.args.train_text_encoder and self.args.train_unet:
                            images_ij = self.generate_image_no_gradient(prompt_i, noises_ij, num_denoising_steps, which_text_encoder=self.eval_text_encoder, which_unet=self.eval_unet)
                        elif self.args.train_text_encoder and not self.args.train_unet:
                            images_ij = self.generate_image_no_gradient(prompt_i, noises_ij, num_denoising_steps, which_text_encoder=self.eval_text_encoder, which_unet=self.unet)
                        elif not self.args.train_text_encoder and self.args.train_unet:
                            images_ij = self.generate_image_no_gradient(prompt_i, noises_ij, num_denoising_steps, which_text_encoder=self.text_encoder, which_unet=self.eval_unet)
                        images_ori.append(images_ij)
                    images_ori = torch.cat(images_ori)

                    face_indicators_ori, face_bboxs_ori, face_chips_ori, face_landmarks_ori, aligned_face_chips_ori = get_face(images_ori, self.args)
                    preds_group_ori, probs_group_ori = self._get_group_predictions(face_chips_ori, selector=face_indicators_ori, fill_value=-1)
                    # print('the size of probs_group_ori :', probs_group_ori.size())
                    
                    images_small_ori = transforms.Resize(self.args.img_size_small)(images_ori)
                    clip_feats_ori = self.get_clip_feat(images_small_ori, normalize=True, to_high_precision=True)
                    DINO_feats_ori = self.get_dino_feat(images_small_ori, normalize=True, to_high_precision=True)

                    images_ori_all = customized_all_gather(images_ori, self.accelerator, return_tensor_other_processes=False)
                    face_indicators_ori_all = customized_all_gather(face_indicators_ori, self.accelerator, return_tensor_other_processes=False)
                    face_bboxs_ori_all = customized_all_gather(face_bboxs_ori, self.accelerator, return_tensor_other_processes=False)
                    preds_group_ori_all = customized_all_gather(preds_group_ori, self.accelerator, return_tensor_other_processes=False)
                    probs_group_ori_all = customized_all_gather(probs_group_ori, self.accelerator, return_tensor_other_processes=False)

                    face_feats_ori = get_face_feats(self.face_feats_net, aligned_face_chips_ori)
                    
                    if self.accelerator.is_main_process:
                        if step % self.args.train_plot_every_n_iter == 0:
                            concat_images = torch.cat((images_ori_all, images_all), dim=0)
                            concat_face_indicators = torch.cat((face_indicators_ori_all, face_indicators_all), dim=0)
                            concat_face_bboxs = torch.cat((face_bboxs_ori_all, face_bboxs_all), dim=0)
                            concat_preds_group = torch.cat((preds_group_ori_all, preds_group_all), dim=0)
                            concat_probs_group = torch.cat((probs_group_ori_all, probs_group_all), dim=0)
                            save_to = os.path.join(self.args.imgs_save_dir, f"train-{self.global_step}_concat.png")
                            plot_in_grid(concat_images, save_to, face_indicators=concat_face_indicators, face_bboxs=concat_face_bboxs, preds_group=concat_preds_group, probs_group=concat_probs_group, group=self.args.trainer_group)

                            log_imgs_i["img_concat"] = [save_to]

                ################################################
                # Step 4: compute loss
                loss_fair_i = torch.ones(noises_i.shape, dtype=self.weight_dtype, device=self.accelerator.device) *(-1)
                loss_CLIP_i = torch.ones(noises_i.shape[0], dtype=self.weight_dtype, device=self.accelerator.device) *(-1)
                loss_DINO_i = torch.ones(noises_i.shape[0], dtype=self.weight_dtype, device=self.accelerator.device) *(-1)
                loss_i = torch.ones(noises_i.shape[0], dtype=self.weight_dtype, device=self.accelerator.device) *(-1)
                
                idxs_i = list(range(noises_i.shape[0]))
                N_backward = math.ceil(noises_i.shape[0] / self.args.train_GPU_batch_size)
                if self.accelerator.is_main_process:
                    print("Before step 4, mam mem: {:.1f} GB ".format(gpu_mem_usage()))

                for j in range(N_backward):
                    if face_indicators[j*self.args.train_GPU_batch_size:(j+1)*self.args.train_GPU_batch_size].sum().item() == 0:
                        continue
                    idxs_ij = idxs_i[j*self.args.train_GPU_batch_size:(j+1)*self.args.train_GPU_batch_size]
                    noises_ij = noises_i[idxs_ij]
                    clip_feats_ori_ij = clip_feats_ori[idxs_ij]
                    DINO_feats_ori_ij = DINO_feats_ori[idxs_ij]
                    preds_group_ori_ij = preds_group_ori[idxs_ij]
                    probs_group_ori_ij = probs_group_ori[idxs_ij]
                    face_bboxs_ori_ij = face_bboxs_ori[idxs_ij]
                    face_feats_ori_ij = face_feats_ori[idxs_ij]
                    
                    images_ij = self.generate_image_w_gradient(prompt_i, noises_ij, num_denoising_steps, which_text_encoder=self.text_encoder, which_unet=self.unet)
                    face_indicators_ij, face_bboxs_ij, face_chips_ij, face_landmarks_ij, aligned_face_chips_ij = get_face(images_ij, self.args)
                    
                        # detached_images = images_ij.detach()
                        # face_indicators_ij.detach()
                        # face_bboxs_ij.detach()
                        # face_chips_ij.detach()
                        # face_landmarks_ij.detach()
                        # aligned_face_chips_ij.detach()
                        # continue
                    
                    preds_group_ij, probs_group_ij = self._get_group_predictions(face_chips_ij, selector=face_indicators_ij, fill_value=-1)
                    # print(f" in line 672 , the size of probs_group_ij is {probs_group_ij.size()}")

                    images_ij = self.apply_grad_hook_face(images_ij, face_bboxs_ij, face_bboxs_ori_ij, preds_group_ori_ij, probs_group_ori_ij, factor=self.args.factor2)
                    images_small_ij = transforms.Resize(self.args.img_size_small)(images_ij)
                    clip_feats_ij = self.get_clip_feat(images_small_ij, normalize=True, to_high_precision=True)
                    DINO_feats_ij = self.get_dino_feat(images_small_ij, normalize=True, to_high_precision=True)

                    loss_CLIP_ij = - (clip_feats_ij * clip_feats_ori_ij).sum(dim=-1) + 1
                    loss_DINO_ij = - (DINO_feats_ij * DINO_feats_ori_ij).sum(dim=-1) + 1
                    if face_indicators_ij.sum().item() != 0:
                        probs_group_ij = 2 * probs_group_ij - 1
                        mpr_set_with_gradients = mpr_set[prompt_i].clone()
                        # print(probs_group_ij.shape)
                        # print(mpr_set_with_gradients[-n_faces_detected:])
                        ## insert computation graph for attributes for batch i, device j 
                        mpr_set_with_gradients[-noises_i.shape[0]:][j*self.args.train_GPU_batch_size:(j+1)*self.args.train_GPU_batch_size] = probs_group_ij 

                        loss_fair_ij = 0
                        if self.modelname == "linear" or "boolean" in self.modelname:
                            # compute oracle
                            generative_set_no_grad = mpr_set[prompt_i]
                            generative_set_no_grad = generative_set_no_grad[(generative_set_no_grad!=-3).all(dim=-1)]
                            generative_set_grad = mpr_set_with_gradients[(mpr_set_with_gradients!=-3).all(dim=-1)]
                            if "boolean" in self.modelname:
                                generative_set_no_grad = self._make_cartesian(generative_set_no_grad)
                                generative_set_grad = self._make_cartesian(generative_set_grad)
                            if self.ver == "ver1" or self.ver == "ver3":
                                with torch.no_grad():
                                    generative_set = generative_set_no_grad
                                    mean_left = generative_set.mean(dim=0)
                                    mean_right = curation_set_tensor.mean(dim=0)
                                    diff = mean_left - mean_right
                                    _oracle = diff / (torch.norm(diff, p=2))

                            elif self.ver == "ver2":
                                generative_set = generative_set_grad
                                mean_left = generative_set.mean(dim=0)
                                mean_right = curation_set_tensor.mean(dim=0)
                                diff = mean_left - mean_right
                                _oracle = diff / (torch.norm(diff, p=2))

                            # compute MPR
                            mean_left = generative_set_grad.mean(dim=0) if (self.ver == "ver1" or self.ver == "ver3") else mean_left
                            if self.ver == "ver1" or self.ver== "ver3":
                                if j == 0:
                                    if len(self.oracle_function_set[prompt_i]) == self.n_cs:   
                                        self.oracle_function_set[prompt_i].pop(0)
                                    self.oracle_function_set[prompt_i].append(_oracle.detach().clone())
                                else:
                                    self.oracle_function_set[prompt_i][-1] = _oracle.detach().clone()
                                    
                                print(f'the size of oracle_function_set for {prompt_i} is :', len(self.oracle_function_set[prompt_i]))
                                for _oracle in self.oracle_function_set[prompt_i]:
                                    _tmp_mpr = torch.abs((mean_left - mean_right) @ _oracle.detach())
                                    loss_fair_ij += _tmp_mpr
                                    # loss_fair_ij += torch.abs(torch.sum((1/k)*c[:generative_set.shape[0]]) - torch.sum((1/m)*c[generative_set.shape[0]:]))
                                loss_fair_ij /= len(self.oracle_function_set[prompt_i])
                            
                            elif self.ver == "ver2":
                                loss_fair_ij = torch.abs((mean_left - mean_right) @ _oracle)
                    else:
                        loss_fair_ij = torch.tensor(0, dtype=self.weight_dtype, device=self.accelerator.device)

                    loss_ij = loss_fair_ij + self.args.weight_loss_img * (loss_CLIP_ij + loss_DINO_ij).mean() 
                    # if self.accelerator.is_main_process:
                        # print(f"Right before backward of {j}-batch, mem: {gpu_mem_usage():.1f} GB ")
                    self.accelerator.backward(loss_ij)

                    with torch.no_grad():
                        # loss_fair_i[idxs_ij] = loss_fair_ij.to(loss_fair_i.dtype)
                        loss_fair_i[idxs_ij] = loss_fair_ij.to(loss_fair_i.dtype)
                        # loss_face_i[idxs_ij] = loss_face_ij.to(loss_face_i.dtype)
                        loss_CLIP_i[idxs_ij] = loss_CLIP_ij.to(loss_CLIP_i.dtype)
                        loss_DINO_i[idxs_ij] = loss_DINO_ij.to(loss_DINO_i.dtype)
                        loss_i[idxs_ij] = loss_ij.to(loss_i.dtype)
                    
                    if self.accelerator.is_main_process:
                        print(f"After backward of {j}-batch, mam mem: {gpu_mem_usage():.1f} GB ")

                        
                # for logging purpose, gather all losses to main_process
                self.accelerator.wait_for_everyone()
                loss_fair_all = customized_all_gather(loss_fair_i, self.accelerator)
                # loss_face_all = customized_all_gather(loss_face_i, self.accelerator)
                loss_CLIP_all = customized_all_gather(loss_CLIP_i, self.accelerator)
                loss_DINO_all = customized_all_gather(loss_DINO_i, self.accelerator)
                loss_all = customized_all_gather(loss_i, self.accelerator)

                if self.accelerator.is_main_process:
                    # logs_i["loss_fair"].append(loss_fair_all)
                    # logs_i["loss_face"].append(loss_face_all)
                    logs_i["loss_fair"].append(loss_fair_all)
                    logs_i["loss_CLIP"].append(loss_CLIP_all)
                    logs_i["loss_DINO"].append(loss_DINO_all)
                    logs_i["loss"].append(loss_all)
                
                # process logs
                if self.accelerator.is_main_process:
                    for key in ["loss_fair", "loss_CLIP", "loss_DINO", "loss"]:#wandb_tracker.log({f"train_{key}"
                        if logs_i[key] == []:
                            logs_i.pop(key)
                        else:
                            logs_i[key] = torch.cat(logs_i[key])
                    # for key in ["gender_gap", "gender_gap_abs", "gender_pred_between_0.2_0.8"]:
                    for key in ["mpr"]:
                        if logs_i[key] == []:
                            logs_i.pop(key)

                ##########################################################################
                # log process for training
                if self.accelerator.is_main_process:
                    for key, values in logs_i.items():
                        if isinstance(values, list):
                            self.wandb_tracker.log({f"train_{key}": np.mean(values)}, step=self.global_step)
                        else:
                            self.wandb_tracker.log({f"train_{key}": values.mean().item()}, step=self.global_step)

              #      for key, values in log_imgs_i.items():
              #          self.wandb_tracker.log({f"train_{key}":wandb.Image(
              #                  data_or_path=values[0],
              #                  caption=prompt_i,
              #              )
              #              },
              #              step=self.global_step
              #              )

                if self.args.train_text_encoder:
                    self.model_sanity_print(self.text_encoder_lora_layers, "check No.1, text_encoder: after self.accelerator.backward()")
                if self.args.train_unet:
                    self.model_sanity_print(self.unet_lora_layers, "check No.1, unet: after self.accelerator.backward()")

                # note that up till now grads are not synced
                # we mannually sync grads
                # self.accelerator.wait_for_everyone()
                grad_is_finite = True
                with torch.no_grad():
                    if self.args.train_text_encoder:
                        for p in self.text_encoder_lora_layers.parameters():
                            if not torch.isfinite(p.grad).all():
                                grad_is_finite = False
                            torch.distributed.all_reduce(p.grad, torch.distributed.ReduceOp.SUM)
                            p.grad = p.grad / self.accelerator.num_processes / N_backward
                    if self.args.train_unet:
                        for p in self.unet_lora_layers:
                            if not torch.isfinite(p.grad).all():
                                grad_is_finite = False
                            torch.distributed.all_reduce(p.grad, torch.distributed.ReduceOp.SUM)
                            p.grad = p.grad / self.accelerator.num_processes / N_backward
                    
                if self.args.train_text_encoder:
                    self.model_sanity_print(self.text_encoder_lora_layers, "check No.2, text_encoder: after gradients allreduce & average")
                if self.args.train_unet:
                    self.model_sanity_print(self.unet_lora_layers, "check No.2, unet: after gradients allreduce & average")

                if grad_is_finite:
                    self.optimizer.step()
                else:
                    self.accelerator.print(f"grads are not finite, skipped!")
            
                if self.accelerator.is_main_process:
                    print(f"After gradient steps, mam mem: {gpu_mem_usage():.1f} GB ")
                    self.lr_scheduler.step()
                    
                    if grad_is_finite:
                        # if self.args.train_text_encoder:
                            # self.text_encoder_lora_ema.step(self.text_encoder_lora_layers)
                        if self.args.train_unet:
                            self.unet_lora_ema.step(self.unet_lora_layers )

                    progress_bar.update(1)
                    # self.global_step += 1

                    if self.accelerator.is_main_process:
                        with torch.no_grad():
                            if self.args.train_text_encoder:
                                param_norm = np.mean([p.norm().item() for p in self.text_encoder_lora_layers.parameters()])
                                # param_ema_norm = np.mean([p.norm().item() for p in self.text_encoder_lora_ema.shadow_params])
                                self.wandb_tracker.log({f"train_TE_lora_norm": param_norm}, step=self.global_step)
                                # self.wandb_tracker.log({f"train_TE_lora_ema_norm": param_ema_norm}, step=self.global_step)
                            if self.args.train_unet:
                                param_norm = np.mean([p.norm().item() for p in self.unet_lora_layers])
                                param_ema_norm = np.mean([p.norm().item() for p in self.unet_lora_ema.shadow_params])
                                self.wandb_tracker.log({f"train_unet_lora_norm": param_norm}, step=self.global_step)
                                self.wandb_tracker.log({f"train_unet_lora_ema_norm": param_ema_norm}, step=self.global_step)

                    if self.global_step % self.args.evaluate_every_n_iter == 0:
                        if self.global_step != 0:
                            self.evaluation_step(prompts_val, self.global_step)

                    if self.accelerator.is_main_process:
                        # if self.global_step % self.args.checkpointing_steps == 0:
                        #     # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        #     if self.args.checkpoints_total_limit is not None:
                        #         name = "checkpoint_tmp"
                        #         clean_checkpoint(self.args.ckpts_save_dir, name, self.args.checkpoints_total_limit)

                        #     save_path = os.path.join(self.args.ckpts_save_dir, f"checkpoint_tmp-{self.global_step}")
                        #     self.accelerator.save_state(save_path)
                        
                        #     logger.info(f"Accelerator checkpoint saved to {save_path}")

                        if self.global_step % self.args.checkpointing_steps_long == 0:
                            # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                            
                            save_path = os.path.join(self.args.ckpts_save_dir, f"checkpoint-{self.global_step}")
                            self.accelerator.save_state(save_path)
                                                    
                            unet_state_dict = None
                            if self.args.train_unet:    
                                unwrapped_model = self.unwrap_model(self.unet)
                                unet_state_dict = convert_state_dict_to_diffusers(
                                    get_peft_model_state_dict(unwrapped_model)
                                )
                            text_encoder_state_dict = None
                            if self.args.train_text_encoder:
                                unwrapped_model = self.unwrap_model(self.text_encoder)
                                
                                text_encoder_state_dict = convert_state_dict_to_diffusers(
                                    get_peft_model_state_dict(unwrapped_model)
                                )
                            
                            StableDiffusionPipeline.save_lora_weights(
                                save_directory=save_path,
                                unet_lora_layers=unet_state_dict,
                                text_encoder_lora_layers=text_encoder_state_dict,
                                safe_serialization=True,
                            )
                            # for name, param in self.text_encoder.named_parameters():
                                # print(name, param.norm().item())
                            # torch.save(self.text_encoder.state_dict(), save_path+f"/text_encoder.pth")
                            # torch.save(self.unet.state_dict(), save_path+f"/unet.pth")
                            # torch.save(self.vae.state_dict(), save_path+f"/vae.pth")

                            logger.info(f"Accelerator checkpoint saved to {save_path}")
                    self.global_step += 1
                    torch.cuda.empty_cache()
        self.accelerator.end_training()
    
        #######################################################
    
    @torch.no_grad()
    def generate_image_no_gradient(self, prompt, noises, num_denoising_steps, which_text_encoder, which_unet, flag=False):
        """
        prompts: str
        noises: [N,4,64,64], N is number images to be generated for the prompt
        """
        N = noises.shape[0]
        prompts = [prompt] * N
        
        prompts_token = self.tokenizer(prompts, return_tensors="pt", padding=True)
        prompts_token["input_ids"] = prompts_token["input_ids"].to(self.accelerator.device)
        prompts_token["attention_mask"] = prompts_token["attention_mask"].to(self.accelerator.device)

        prompt_embeds = which_text_encoder(
            prompts_token["input_ids"],
            prompts_token["attention_mask"],
        )
        prompt_embeds = prompt_embeds[0]

        batch_size = prompt_embeds.shape[0]
        uncond_tokens = [""] * batch_size
        max_length = prompt_embeds.shape[1]
        uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )
        uncond_input["input_ids"] = uncond_input["input_ids"].to(self.accelerator.device)
        uncond_input["attention_mask"] = uncond_input["attention_mask"].to(self.accelerator.device)
        negative_prompt_embeds = which_text_encoder(
            uncond_input["input_ids"],
            uncond_input["attention_mask"],
        )
        negative_prompt_embeds = negative_prompt_embeds[0]
        # if flag:
        #     print('promopt embeds :', prompt_embeds)

        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
        prompt_embeds = prompt_embeds.to(self.weight_dtype)
        # if flag:
        #     print(self.noise_scheduler)
        self.noise_scheduler.set_timesteps(num_denoising_steps)
        # if flag:
        #     print(self.noise_scheduler.timesteps)

        latents = noises
        for i, t in enumerate(self.noise_scheduler.timesteps):
            # if flag:
            #     print(t, ' latent : ', latents[0,:,30:32, 30:32])
            # scale model input
            latent_model_input = torch.cat([latents.to(self.weight_dtype)] * 2)
            latent_model_input = self.noise_scheduler.scale_model_input(latent_model_input, t)
            
            noises_pred = which_unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
            ).sample
            noises_pred = noises_pred.to(self.weight_dtype_high_precision)
            # if flag:
            #     print(t, ' noise pred : ', noises_pred[0,:,30:32, 30:32])
            
            noises_pred_uncond, noises_pred_text = noises_pred.chunk(2)
            noises_pred = noises_pred_uncond + self.args.guidance_scale * (noises_pred_text - noises_pred_uncond)
            
            latents = self.noise_scheduler.step(noises_pred, t, latents).prev_sample


        latents = 1 / self.vae.config.scaling_factor * latents
        # images = self.vae.decode(latents.to(self.vae.dtype)).sample.clamp(-1,1) # in range [-1,1]
        images = self.vae.decode(latents.to(self.vae.dtype)).sample
        # if flag:
        #     print('generated images :', images[0, :,30:32, 30:32])
        images = (images / 2 + 0.5).clamp(0,1)
        
        return images
    
    def generate_image_w_gradient(self, prompt, noises, num_denoising_steps, which_text_encoder, which_unet):
        """
        prompts: str
        noises: [N,4,64,64], N is number images to be generated for the prompt
        """
        # to enable gradient_checkpointing, unet must be set to train()
        self.unet.train()
        
        N = noises.shape[0]
        prompts = [prompt] * N
        
        prompts_token = self.tokenizer(prompts, return_tensors="pt", padding=True)
        prompts_token["input_ids"] = prompts_token["input_ids"].to(self.accelerator.device)
        prompts_token["attention_mask"] = prompts_token["attention_mask"].to(self.accelerator.device)

        prompt_embeds = which_text_encoder(
            prompts_token["input_ids"],
            prompts_token["attention_mask"],
        )
        prompt_embeds = prompt_embeds[0]

        batch_size = prompt_embeds.shape[0]
        uncond_tokens = [""] * batch_size
        max_length = prompt_embeds.shape[1]
        uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )
        uncond_input["input_ids"] = uncond_input["input_ids"].to(self.accelerator.device)
        uncond_input["attention_mask"] = uncond_input["attention_mask"].to(self.accelerator.device)
        negative_prompt_embeds = which_text_encoder(
            uncond_input["input_ids"],
            uncond_input["attention_mask"],
        )
        negative_prompt_embeds = negative_prompt_embeds[0]

        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds]).to(self.weight_dtype)
        
        self.noise_scheduler.set_timesteps(num_denoising_steps)
        grad_coefs = []
        for i, t in enumerate(self.noise_scheduler.timesteps):
            grad_coefs.append( self.noise_scheduler.alphas_cumprod[t].sqrt().item() * (1-self.noise_scheduler.alphas_cumprod[t]).sqrt().item() / (1-self.noise_scheduler.alphas[t].item()) )
        grad_coefs = np.array(grad_coefs)
        grad_coefs /= (math.prod(grad_coefs)**(1/len(grad_coefs)))
            
        latents = noises
        for i, t in enumerate(self.noise_scheduler.timesteps):
        
            # scale model input
            latent_model_input = torch.cat([latents.detach().to(self.weight_dtype)]*2)
            latent_model_input = self.noise_scheduler.scale_model_input(latent_model_input, t)
            
            noises_pred = which_unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
            ).sample
            noises_pred = noises_pred.to(self.weight_dtype_high_precision)
            
            noises_pred_uncond, noises_pred_text = noises_pred.chunk(2)
            noises_pred = noises_pred_uncond + self.args.guidance_scale * (noises_pred_text - noises_pred_uncond)
            
            hook_fn = make_grad_hook(grad_coefs[i])
            noises_pred.register_hook(hook_fn)
            
            latents = self.noise_scheduler.step(noises_pred, t, latents).prev_sample

        latents = 1 / self.vae.config.scaling_factor * latents
        # images = self.vae.decode(latents.to(self.vae.dtype)).sample.clamp(-1,1) # in range [-1,1]
        images = self.vae.decode(latents.to(self.vae.dtype)).sample
        images = (images / 2 + 0.5).clamp(0,1)
        
        return images
    
    
    def get_clip_feat(self, images, normalize=True, to_high_precision=True):
        """get clip features

        self.args:
            images (torch.tensor): shape [N,3,H,W], in range [-1,1]
            normalize (bool):
            to_high_precision (bool):

        Returns:
            embeds (torch.tensor)
        """
        images_preprocessed = (images - self.clip_img_mean) / self.clip_img_std
        embeds = self.clip_vision_model_w_proj(images_preprocessed).image_embeds
        
        if to_high_precision:
            embeds = embeds.to(torch.float)
        if normalize:
            embeds = torch.nn.functional.normalize(embeds, dim=-1)
        return embeds
    
    def get_dino_feat(self, images, normalize=True, to_high_precision=True):
        """get dino features

        self.args:
            images (torch.tensor): shape [N,3,H,W], in range [-1,1]
            normalize (bool):
            to_high_precision (bool):

        Returns:
            embeds (torch.tensor)
        """
        images_preprocessed = (images - self.dinov2_img_mean) / self.dinov2_img_std
        embeds = self.dinov2(images_preprocessed)
        
        if to_high_precision:
            embeds = embeds.to(torch.float)
        if normalize:
            embeds = torch.nn.functional.normalize(embeds, dim=-1)
        return embeds

    @torch.no_grad()
    def generate_dynamic_targets(self, probs, target_ratio=0.5, w_uncertainty=False):
        """generate dynamic targets for the distributional alignment loss

        Args:
            probs (torch.tensor): shape [N,2], N points in a probability simplex of 2 dims
            target_ratio (float): target distribution, the percentage of class 1 (male)
            w_uncertainty (True/False): whether return uncertainty measures
        
        Returns:
            targets_all (torch.tensor): target classes
            uncertainty_all (torch.tensor): uncertainty of target classes
        """
        idxs_2_rank = (probs!=-1).all(dim=-1)
        probs_2_rank = probs[idxs_2_rank]

        rank = torch.argsort(torch.argsort(probs_2_rank[:,1]))
        targets = (rank >= (rank.shape[0]*target_ratio)).long()

        targets_all = torch.ones([probs.shape[0]], dtype=torch.long, device=probs.device) * (-1)
        targets_all[idxs_2_rank] = targets
        
        if w_uncertainty:
            uncertainty = torch.ones([probs_2_rank.shape[0]], dtype=probs.dtype, device=probs.device) * (-1)
            uncertainty[targets==1] = torch.tensor(
                1 - scipy.stats.binom.cdf(
                    (rank[targets==1]).cpu().numpy(), 
                    probs_2_rank.shape[0], 
                    1-target_ratio
                    )
                ).to(probs.dtype).to(probs.device)
            uncertainty[targets==0] = torch.tensor(
                scipy.stats.binom.cdf(
                    rank[targets==0].cpu().numpy(), 
                    probs_2_rank.shape[0], 
                    target_ratio
                    )
                ).to(probs.dtype).to(probs.device)
            
            uncertainty_all = torch.ones([probs.shape[0]], dtype=probs.dtype, device=probs.device) * (-1)
            uncertainty_all[idxs_2_rank] = uncertainty
            
            return targets_all, uncertainty_all
        else:
            return targets_all
        
    def evaluation_step(self, prompts_val, current_step):
        noises_val = torch.randn(
        [len(prompts_val), self.args.val_images_per_prompt_GPU,4,64,64],
        dtype=self.weight_dtype_high_precision
        ).to(self.accelerator.device)
        self.evaluate_process(self.text_encoder, self.unet, "main", prompts_val, noises_val, current_step)

        # evaluate EMA as well
        # if self.args.train_text_encoder:
        #     text_encoder_lora_layers = copy.deepcopy(self.text_encoder_lora_layers)
        #     # load_state_dict_results = self.text_encoder.load_state_dict(self.text_encoder_lora_ema_dict, strict=False)
        
        # if self.args.train_unet:
        #     with torch.no_grad():
        #         unet_lora_layers_copy = copy.deepcopy(self.unet_lora_layers)
        #         for p, p_from in itertools.zip_longest(list(self.unet_lora_layers), self.unet_lora_ema.shadow_params):
        #             p.data = p_from.data

    @torch.no_grad()
    def evaluate_process(self, which_text_encoder, which_unet, name, prompts, noises, current_global_step):
        logs = []
        log_imgs = []
        num_denoising_steps = 25

        with open('tmp/noise_tmp.pkl', 'wb') as f:
            pickle.dump(noises, f)

        for prompt_i, noises_i in itertools.zip_longest(prompts, noises):
            if self.accelerator.is_main_process:
                logs_i = {
                    "mpr": [],
                }
                log_imgs_i = {}
            ################################################
            # step 1: generate all ori images
            images_ori = []
            N = math.ceil(noises_i.shape[0] / self.args.val_GPU_batch_size)
            for j in range(N):
                noises_ij = noises_i[self.args.val_GPU_batch_size*j:self.args.val_GPU_batch_size*(j+1)]
                if self.args.train_text_encoder and self.args.train_unet:
                    images_ij = self.generate_image_no_gradient(prompt_i, noises_ij, num_denoising_steps, which_text_encoder=self.eval_text_encoder, which_unet=self.eval_unet)
                elif self.args.train_text_encoder and not self.args.train_unet:
                    images_ij = self.generate_image_no_gradient(prompt_i, noises_ij, num_denoising_steps, which_text_encoder=self.eval_text_encoder, which_unet=self.unet)
                elif not self.args.train_text_encoder and self.args.train_unet:
                    images_ij = self.generate_image_no_gradient(prompt_i, noises_ij, num_denoising_steps, which_text_encoder=self.text_encoder, which_unet=self.eval_unet)
                images_ori.append(images_ij)
            images_ori = torch.cat(images_ori)
            face_indicators_ori, face_bboxs_ori, face_chips_ori, face_landmarks_ori, aligned_face_chips_ori = get_face(images_ori, self.args)
            preds_group_ori, probs_group_ori = self._get_group_predictions(face_chips_ori, selector=face_indicators_ori, fill_value=-1)
            
            face_feats_ori = get_face_feats(self.face_feats_net, aligned_face_chips_ori)
            _, face_real_scores_ori = self.face_feats_model.semantic_search(face_feats_ori, selector=face_indicators_ori, return_similarity=True)

            images_ori_all = customized_all_gather(images_ori, self.accelerator, return_tensor_other_processes=False)
            face_indicators_ori_all = customized_all_gather(face_indicators_ori, self.accelerator, return_tensor_other_processes=False)
            face_bboxs_ori_all = customized_all_gather(face_bboxs_ori, self.accelerator, return_tensor_other_processes=False)
            preds_group_ori_all = customized_all_gather(preds_group_ori, self.accelerator, return_tensor_other_processes=False)
            probs_group_ori_all = customized_all_gather(probs_group_ori, self.accelerator, return_tensor_other_processes=False)
            face_real_scores_ori_all = customized_all_gather(face_real_scores_ori, self.accelerator, return_tensor_other_processes=False)

            if self.accelerator.is_main_process:
                save_to = os.path.join(self.args.imgs_save_dir, f"eval_{name}_{self.global_step}_{prompt_i}_ori.png")
                plot_in_grid(
                    images_ori_all, 
                    save_to, 
                    face_indicators=face_indicators_ori_all, face_bboxs=face_bboxs_ori_all, 
                    preds_group=preds_group_ori_all, 
                    probs_group=probs_group_ori_all,
                    group=self.args.trainer_group
                    # face_real_scores=face_real_scores_ori_all
                )

                log_imgs_i["img_ori"] = [save_to]
            
            images = []
            N = math.ceil(noises_i.shape[0] / self.args.val_GPU_batch_size)
            for j in range(N):
                noises_ij = noises_i[self.args.val_GPU_batch_size*j:self.args.val_GPU_batch_size*(j+1)]
                # if j == 0:
                #     print(f"the norm of noises_ij is {noises_ij[0].norm().item()}")
                    # images_ij = self.generate_image_no_gradient(prompt_i, noises_ij, num_denoising_steps, which_text_encoder=which_text_encoder, which_unet=which_unet, flag=True)
                # else:
                images_ij = self.generate_image_no_gradient(prompt_i, noises_ij, num_denoising_steps, which_text_encoder=which_text_encoder, which_unet=which_unet)
                images.append(images_ij)
            images = torch.cat(images)
            
            face_indicators, face_bboxs, face_chips, face_landmarks, aligned_face_chips = get_face(images, self.args, eval=True)
            # face_chips = ((face_chips*0.5 + 0.5) * 255).to(torch.uint8).to(self.weight_dtype)
            
            preds_group, probs_group = self._get_group_predictions(face_chips, selector=face_indicators, fill_value=-1, eval=True)
            # print('in line 1180 the size of probs :', probs_group.size())
            
            # face_feats = get_face_feats(self.face_feats_net, aligned_face_chips)
            # _, face_real_scores = self.face_feats_model.semantic_search(face_feats, selector=face_indicators, return_similarity=True)

            images_all = customized_all_gather(images, self.accelerator, return_tensor_other_processes=False)
            face_indicators_all = customized_all_gather(face_indicators, self.accelerator, return_tensor_other_processes=False)
            face_bboxs_all = customized_all_gather(face_bboxs, self.accelerator, return_tensor_other_processes=False)
            preds_group_all = customized_all_gather(preds_group, self.accelerator, return_tensor_other_processes=False)
            probs_group_all = customized_all_gather(probs_group, self.accelerator, return_tensor_other_processes=False)
            # face_real_scores = customized_all_gather(face_real_scores, self.accelerator, return_tensor_other_processes=False)

            print("the number of face : ", face_indicators_all.sum())
            # print which images don't have faces
            if self.accelerator.is_main_process:
                tmp_prompt = prompt_i.replace(" ", "_")
                save_to = os.path.join(self.args.imgs_save_dir, f"eval_{name}_{self.global_step}_{tmp_prompt}_generated.png")
                plot_in_grid(
                    images_all, 
                    save_to, 
                    face_indicators=face_indicators_all, 
                    face_bboxs=face_bboxs_all, 
                    preds_group=preds_group_all, 
                    probs_group=probs_group_all,
                    group=self.args.trainer_group,
                    save_images=True
                    )

                log_imgs_i["img_generated"] = [save_to]
            
            # probs_tmp = probs_group_ori_all[(probs_group_ori_all!=-1).all(dim=-1)]
            # tmp = probs_tmp[:,:2]
            # entropy = -torch.sum(tmp * torch.log(tmp), dim=-1)
            # print('original gender entropy :', torch.mean(entropy).item())
            # tmp = probs_tmp[:,2:4]
            # entropy = -torch.sum(tmp * torch.log(tmp), dim=-1)
            # print('original age entropy :', torch.mean(entropy).item())
            # tmp = probs_tmp[:,4:]
            # entropy = -torch.sum(tmp * torch.log(tmp), dim=-1)
            # print('original race entropy :', torch.mean(entropy).item())

            # probs_tmp = probs_group_all[(probs_group_all!=-1).all(dim=-1)]
            # tmp = probs_tmp[:,:2]
            # entropy = -torch.sum(tmp * torch.log(tmp), dim=-1)
            # print('gender entropy :', torch.mean(entropy).item())
            # tmp = probs_tmp[:,2:4]
            # entropy = -torch.sum(tmp * torch.log(tmp), dim=-1)
            # print('age entropy :', torch.mean(entropy).item())
            # tmp = probs_tmp[:,4:]
            # entropy = -torch.sum(tmp * torch.log(tmp), dim=-1)
            # print('race entropy :', torch.mean(entropy).item())

            if self.accelerator.is_main_process:
                probs_tmp = probs_group_all[(probs_group_all!=-1).all(dim=-1)]
                # gender_gap = (((probs_tmp[:,1]>=0.5)*(probs_tmp[:,1]<=1)).float().mean() - ((probs_tmp[:,1]>=0)*(probs_tmp[:,1]<=0.5)).float().mean()).item()
                probs_tmp = 2 * probs_tmp - 1
                with open(f"{self.args.imgs_save_dir}/probs_{name}_{self.global_step}_{prompt_i}.pkl", "wb") as f:
                    pickle.dump(probs_tmp.cpu().numpy(), f)
                
                probs_01 = probs_tmp.cpu().numpy()*0.5 + 0.5
                curation_01 = self.curation_set*0.5 + 0.5
                probs_onehot, curation_onehot = [], []
                for (p,p_new) in [(probs_01, probs_onehot), (curation_01,curation_onehot)]:
                    pos = 0
                    if len(self.args.trainer_group) == 3:
                        for idx in [2,2,7]:
                            tmp = p[:, pos:pos+idx] 
                            one_hot_indices = np.argmax(tmp, axis=1)
                            p_new.append(np.eye(tmp.shape[1])[one_hot_indices])
                            pos += idx
                    else:
                        tmp = p
                        one_hot_indices = np.argmax(tmp, axis=1)
                        p_new.append(np.eye(tmp.shape[1])[one_hot_indices])
                        # pos += idx
                probs_onehot = np.concatenate(probs_onehot, axis=1)*2 - 1
                curation_onehot = np.concatenate(curation_onehot, axis=1)*2 - 1

                mpr = getMPR(self.args.trainer_group, probs_tmp.cpu().numpy(), curation_set=self.curation_set, modelname=self.modelname, normalize=self.args.normalize)[0]
                mpr_onehot = getMPR(self.args.trainer_group, probs_onehot, curation_set=curation_onehot, modelname=self.modelname, normalize=self.args.normalize)[0]
                
                # # temporary debugging
                # curation_set_tensor = torch.tensor(self.curation_set, dtype=self.weight_dtype, device=self.accelerator.device)
                # extended_mat = torch.cat([probs_tmp, curation_set_tensor], dim=0)
                # mpr_c = 0
                # if prompt_i in self.oracle_function_set.keys():
                #     for idx, _oracle in enumerate(self.oracle_function_set[prompt_i]):
                #         c = extended_mat @ _oracle.unsqueeze(-1)

                #         k = probs_tmp.shape[0]
                #         m = self.curation_set.shape[0]
                #         _tmp_mpr = torch.abs(torch.sum((1/k)*c[:k]) - torch.sum((1/m)*c[k:]))
                #         mpr_c += _tmp_mpr
                #         print(idx, _oracle.cpu().numpy(), probs_tmp.mean(dim=0).cpu().numpy(), np.mean(self.curation_set, axis=0), _tmp_mpr.cpu().numpy())
                    
                #     if len(self.oracle_function_set[prompt_i])>0:
                #         mpr_c /= len(self.oracle_function_set[prompt_i])

                # gender_pred_between_02_08 = ((probs_tmp[:,1]>=0.2)*(probs_tmp[:,1]<=0.8)).float().mean().item()
                print(f'{prompt_i} MPR: {mpr}')
                print(f'{prompt_i} MPR_onehot: {mpr_onehot}')
                # print(f'{prompt_i} MPR_c:  {mpr_c}')
                logs_i["mpr"].append(mpr)
            
            if self.accelerator.is_main_process:
                log_imgs.append(log_imgs_i)
                logs.append(logs_i)
        
        if self.accelerator.is_main_process:
            for prompt_i, logs_i in itertools.zip_longest(prompts, logs):
                for key, values in logs_i.items():
                    if isinstance(values, list):
                        self.wandb_tracker.log({f"eval_{name}_{key}_{prompt_i}": np.mean(values)}, step=current_global_step)
                    else:
                        self.wandb_tracker.log({f"eval_{name}_{key}_{prompt_i}": values.mean().item()}, step=current_global_step)
                
                for key in list(logs[0].keys()):
                    avg = np.array([log[key] for log in logs]).mean()
                    self.wandb_tracker.log({f"eval_{name}_{key}": avg}, step=current_global_step)

            imgs_dict = {}
        #    for prompt_i, log_imgs_i in itertools.zip_longest(prompts, log_imgs):
        #        for key, values in log_imgs_i.items():
        #            if key not in imgs_dict.keys():
        #                imgs_dict[key] = [wandb.Image(
        #                    data_or_path=values[0],
        #                    caption=prompt_i,
        #                )]
        #            else:
        #                imgs_dict[key].append(wandb.Image(
        #                    data_or_path=values[0],
        #                    caption=prompt_i,
        #                ))
        #    for key, imgs in imgs_dict.items():
        #        self.wandb_tracker.log(
        #            {f"eval_{name}_{key}": imgs},
        #            step=current_global_step
        #            ) 
        return logs, log_imgs
    
    def unwrap_model(self, model):
        model = self.accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model 

    def _get_group_predictions(self, face_chips, selector=None, fill_value=-1, eval=False):
        groups = self.args.trainer_group

        if selector != None:
            face_chips_w_faces = face_chips[selector]
        else:
            face_chips_w_faces = face_chips

        if not hasattr(self, "group_classifier_dic"):
            self._set_group_clfs(groups)

        def softmax_with_temperature(logits, temperature=1.0):
            # Step 1: Temperature scaling
            scaled_logits = logits / temperature

            # Step 2: Exponentiate each logit (avoid overflow by subtracting max logit)
            max_logit = torch.max(scaled_logits, dim=-1, keepdim=True).values
            exp_logits = torch.exp(scaled_logits - max_logit)
            
            # Step 3: Calculate the sum of exponentiated logits
            sum_exp_logits = torch.sum(exp_logits, dim=-1, keepdim=True)
            
            # Step 4: Normalize by dividing each exponentiated logit by the sum
            softmax_probs = exp_logits / sum_exp_logits
            return softmax_probs
            
        def sigmoid_with_temperature(logits, temperature=1.0):
            # Step 1: Temperature scaling
            scaled_logits = logits / temperature
            
            # Step 2: Apply the Sigmoid function manually
            sigmoid_probs = 1 / (1 + torch.exp(-scaled_logits))
            
            return sigmoid_probs

        if face_chips_w_faces.shape[0] == 0: # if there is no face from generated images,
            probs_group = torch.empty([0,self.output_dim], dtype=face_chips.dtype, device=face_chips.device)
            preds_group = torch.empty([0], dtype=face_chips.dtype, device=face_chips.device)
        else:
            with torch.autocast("cuda"):
                face_chips_w_faces = (face_chips_w_faces - self.clip_img_mean_for_group) / self.clip_img_std_for_group
                embeddings = self.group_classifier_dic['vision_encoder'].encode_image(face_chips_w_faces)
            probs_group = []
            preds_group = []
            for group in groups:
                if group == 'face':
                    continue
                clf = self.group_classifier_dic[group]
                logits = clf(embeddings)
                # if logits.shape[-1] == 1:
                if group in ['gender', 'age']:
                    if self.ver == "ver3":
                        probs = sigmoid_with_temperature(logits, self.temp)
                    else:
                        probs = torch.sigmoid(logits)
                    probs = torch.cat([1-probs, probs], dim=-1)
                else:
                    if self.ver == "ver3":
                        probs = softmax_with_temperature(logits, self.temp)
                    else:
                        probs = torch.softmax(logits, dim=-1)
                # with torch.no_grad():
                #     print(probs)
                #     print(torch.softmax(logits, dim=-1))
                probs_group.append(probs)
                temp = probs.max(dim=-1)
                preds = temp.indices
                preds_group.append(preds)
                # print('group:', group, 'probs:', probs.size())

            probs_group = torch.cat(probs_group, dim=-1)
            preds_group = torch.stack(preds_group, dim=-1)
        
        if selector != None:
            probs_group_new = torch.ones(
                [selector.shape[0]]+list(probs_group.shape[1:]),
                dtype=probs_group.dtype, 
                device=probs_group.device
                ) * (fill_value)
            probs_group_new[selector] = probs_group
            preds_group_new = torch.ones(
                [selector.shape[0]]+list(preds_group.shape[1:]), 
                dtype=preds_group.dtype, 
                device=preds_group.device
                ) * (fill_value)
            preds_group_new[selector] = preds_group

        return preds_group_new.to(self.weight_dtype), probs_group_new.to(self.weight_dtype)

    def apply_grad_hook_face(self, images, face_bboxs, face_bboxs_ori, preds_group_ori, probs_group_ori, factor=0.1):
        """apply gradient hook on non-face regions of the generated images
        """
        images_new = []
        # for image, face_bbox, face_bbox_ori, target, pred_gender_ori, prob_gender_ori in itertools.zip_longest(images, face_bboxs, face_bboxs_ori, targets, preds_gender_ori, probs_gender_ori):
        for image, face_bbox, face_bbox_ori, pred_group_ori, prob_group_ori in itertools.zip_longest(images, face_bboxs, face_bboxs_ori, preds_group_ori, probs_group_ori):
            if (face_bbox == -1).all():
                images_new.append(image.unsqueeze(dim=0))
            else:
                img_width, img_height = image.shape[1:]
                idx_left = max(face_bbox[0], face_bbox_ori[0], 0)
                idx_right = min(face_bbox[2], face_bbox_ori[2], img_width)
                idx_bottom = max(face_bbox[1], face_bbox_ori[1], 0)
                idx_top = min(face_bbox[3], face_bbox_ori[3], img_height)

                img_face = image[:,idx_bottom:idx_top,idx_left:idx_right].clone()
                # if target==-1:
                #     grad_hook = make_grad_hook(factor)
                # elif target==pred_gender_ori:
                grad_hook = make_grad_hook(1)
                # elif target!=pred_gender_ori:
                #     grad_hook = make_grad_hook(factor)
                img_face.register_hook(grad_hook)

                img_add = torch.zeros_like(image)
                img_add[:,idx_bottom:idx_top,idx_left:idx_right] = img_face

                mask = torch.zeros_like(image)
                mask[:,idx_bottom:idx_top,idx_left:idx_right] = 1

                image = mask*img_add + (1-mask)*image
                images_new.append(image.unsqueeze(dim=0))

        images_new = torch.cat(images_new)
        return images_new
    
    def gen_dynamic_weights(self, face_indicators, targets, preds_gender_ori, probs_gender_ori, factor=0.2):
        weights = []
        for face_indicator, target, pred_gender_ori, prob_gender_ori in itertools.zip_longest(face_indicators, targets, preds_gender_ori, probs_gender_ori):
            if (face_indicator == False).all():
                weights.append(1)
            else:
                if target==-1:
                    weights.append(factor)
                elif target==pred_gender_ori:
                    weights.append(1)
                elif target!=pred_gender_ori:
                    weights.append(factor)

        weights = torch.tensor(weights, dtype=probs_gender_ori.dtype, device=probs_gender_ori.device)
        return weights

    def model_sanity_print(self, model, state):
        params = [p for p in model.parameters()]
        # for i, p in enumerate(model):
            # print(params[0].grad)
        print(f"\t{self.accelerator.device}; {state};\n\t\/: {params[0].flatten()[0].item():.8f};\tparam[0].grad: {params[0].grad.flatten()[0].item():.8f}")
        

class CustomModel(torch.nn.Module):
    def __init__(self, dict):
        """
        In the constructor we instantiate four parameters and assign them as
        member parameters.
        """
        super().__init__()
        self.param_names = list(dict.keys())
        self.params = nn.ParameterList()
        for name in self.param_names:
            self.params.append( dict[name] )
    def forward(self, x):
        """
        no forward function
        """
        return None
    
def get_curation_set(args):
    from mpr.preprocessing import identity_embedding
    vision_encoder = networks.ModelFactory.get_model(modelname=args.vision_encoder)  
    _, refer_loader = data_handler.DataloaderFactory.get_dataloader(dataname=args.refer_dataset, args=args)
    refer_embedding, _ = identity_embedding(args, vision_encoder, refer_loader, args.trainer_group, query=False)
    print('refer_embedding:', refer_embedding.shape)
    return refer_embedding

def make_grad_hook(coef):
    return lambda x: coef * x

def customized_all_gather(tensor, accelerator, return_tensor_other_processes=False):
    tensor_all = [tensor.detach().clone() for i in range(accelerator.num_processes)]
    torch.distributed.all_gather(tensor_all, tensor)
    
    if return_tensor_other_processes:
        if accelerator.num_processes>1:
            tensor_others = torch.cat([tensor_all[idx] for idx in range(accelerator.num_processes) if idx != accelerator.local_process_index], dim=0)
        else:
            tensor_others = torch.empty([0,]+ list(tensor_all[0].shape[1:]), device=accelerator.device, dtype=tensor_all[0].dtype)
    tensor_all = torch.cat(tensor_all, dim=0)
    
    if return_tensor_other_processes:
        return tensor_all, tensor_others
    else:
        return tensor_all


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


    # # deprecated
    # def export_checkpoint(self, accelerator=None):
    #     self.accelerator = accelerator

    #     self._load_model()
    #     self._set_lora_params()
        
    #     self._set_optimizer()
    #     if self.args.train_text_encoder:
    #         print(self.text_encoder_lora_layers[0])
    #         self.text_encoder, self.text_encoder_lora_ema = self.accelerator.prepare(self.text_encoder, self.text_encoder_lora_ema)
    #         self.accelerator.register_for_checkpointing(self.text_encoder_lora_ema)
    #     if self.args.train_unet:
    #         self.unet_lora_layers, self.unet_lora_ema = self.accelerator.prepare(self.unet_lora_layers, self.unet_lora_ema)
    #         self.accelerator.register_for_checkpointing(self.unet_lora_ema)
    #         # Potentially load in the weights and states from a previous save
        
    #     if not self.args.resume_from_checkpoint:
    #         raise ValueError("resume_from_checkpoint must be provided.")
    #     if self.args.resume_from_checkpoint:
    #         if not os.path.exists(self.args.resume_from_checkpoint):
    #             raise ValueError(f"{self.args.resume_from_checkpoint}' does not exist.")
            
    #         self.args.export_dir = str(Path(self.args.resume_from_checkpoint).parent / (Path(self.args.resume_from_checkpoint).name + "_exported"))
    #         if not os.path.exists(self.args.export_dir):
    #             os.makedirs(self.args.export_dir)

            
    #         if self.args.train_unet:
    #             self.unet_lora_ema.to(accelerator.device)

    #         if self.args.train_text_encoder:
    #             self.text_encoder_lora_ema.to(accelerator.device)
    #             text_encoder_lora_dict = {}
    #             text_encoder_lora_params_name_order = []
    #             for lora_param in self.text_encoder_lora_layers:
    #                 for name, param in self.text_encoder.named_parameters():
    #                     if param is lora_param:
    #                         # param.data = lora_param.data
    #                         print(name, param.data.dtype)
    #                         text_encoder_lora_dict[name] = lora_param
    #                         text_encoder_lora_params_name_order.append(name)
    #                         break
                
    #             # need to recreate text_encoder_lora_ema_dict
    #             text_encoder_lora_ema_dict = {}
    #             for name, shadow_param in itertools.zip_longest(text_encoder_lora_params_name_order, self.text_encoder_lora_ema.shadow_params):
    #                 text_encoder_lora_ema_dict[name] = shadow_param
    #             assert text_encoder_lora_ema_dict.__len__() == text_encoder_lora_dict.__len__(), "length does not match! something wrong happened while converting lora params to a state dict."

    #         accelerator.print(f"Resuming from checkpoint {self.args.resume_from_checkpoint}")
    #         accelerator.load_state(self.args.resume_from_checkpoint)
    #         print(self.text_encoder_lora_layers[0])
    #         unet_state_dict = None
    #         if self.args.train_unet:    
    #             unwrapped_model = self.unwrap_model(self.unet)
    #             unet_state_dict = convert_state_dict_to_diffusers(
    #                 get_peft_model_state_dict(unwrapped_model)
    #             )
    #         text_encoder_state_dict = None
    #         if self.args.train_text_encoder:
    #             text_encoder_lora_dict = {key: value.to('cpu') for key, value in text_encoder_lora_dict.items()}
    #             torch.save(text_encoder_lora_dict, self.args.resume_from_checkpoint + "/text_encoder_lora_dict.pth")
    #             # self.text_encoder.load_state_dict(text_encoder_lora_dict, strict=False)
    #             # unwrapped_model = self.unwrap_model(self.text_encoder)
    #             # text_encoder_state_dict = convert_state_dict_to_diffusers(
    #             #     get_peft_model_state_dict(unwrapped_model)
    #             # )
    #             # for key in text_encoder_state_dict.keys():
    #             #     print(key)            
    #         # StableDiffusionPipeline.save_lora_weights(
    #         #     save_directory=self.args.resume_from_checkpoint,
    #         #     unet_lora_layers=unet_state_dict,
    #         #     text_encoder_lora_layers=text_encoder_lora_dict,
    #         #     safe_serialization=True,
    #         # )


    #     print("Finished exporting checkpoint.")
