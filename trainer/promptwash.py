import os

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch

from transformers import CLIPProcessor, CLIPTokenizer #, CLIPModel
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
# from diffusers import LMSDiscreteScheduler
# from diffusers.utils.import_utils import is_xformers_available

from tqdm import tqdm

import torch.optim as optim

import torch
import torch.nn.functional as F

import torch.distributed as dist
from collections import OrderedDict

from trainer import embedding, matcher
from utils import get_current_device, AverageMeter, gpu_mem_usage
# , set_random_seed,  get_entropy, image_grid, , image_grid_plt
# from loss import loss
# _fn_v1, loss_fn_v2, LossV2, LossV3
from trainer.reduced_clip import forward_clip, CLIPModel, forward_clip_text

from trainer import GenericTrainer

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import CrossEntropyLoss

class Trainer(GenericTrainer):
    concepts = ['firefighter']
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rank = 0 # rank of the current process
        self.cur_device = get_current_device()

        self.group_type = self.args.trainer_group
        
        self.start_iter = 0
        self._load_models()
        self._init_prompt()
        self.embed_builder, self.embed_builder_noddp = self._build_embed_builder()
        self.criterion = self._build_criterion()
        self.optimizer = self._build_optimizer(self.embed_builder)
        self.lr_scheduler = self._build_scheduler(self.optimizer)
        self.scaler = torch.cuda.amp.GradScaler()
        self.meters = {'train': [{'normal': AverageMeter('normal_ratio', n_attr=len(self.attr)),
                                  'weighted': AverageMeter('weighted_ratio', weight=0.5, n_attr=len(self.attr))} for _ in
                                 self.concepts],
                       'test': [{'normal': AverageMeter('normal_ratio', n_attr=len(self.attr)),
                                 'weighted': AverageMeter('weighted_ratio', weight=0.5, n_attr=len(self.attr))} for _ in
                                self.concepts]}
        # assert len(self.args.target_ratio) == len(self.attr)

        # load group ratio
        group_ratio = np.load('group_ratio.npy')
        marginalize_idx = [self.group_idx[group] for group in self.group_idx.keys() if group not in self.group_type]
        group_ratio = group_ratio.sum(axis=tuple(marginalize_idx))
        self.group_prob = torch.tensor(group_ratio / group_ratio.sum())
        self.group_prob = self.group_prob.flatten()

        self.target_ratio = torch.Tensor(self.group_prob)
        self.args.report_memory = True
        self.args.use_amp = True
        self.args.n_image_inference = 100
        self.args.grad_mode = 'last'
        self.args.pre_history = True
        self.args.reg_weight = self.args.lamb
        self.args.num_inference_steps = 50
        if self.args.report_memory:
            print("After initialization of Tranier, max mem: {:.1f} GB ".format(gpu_mem_usage()))
        # self._resume()
    def _init_prompt(self):
        train_prompt, test_prompt = [], []
        for cls in self.concepts:
            cls_train_prompt = []
            tp = 'A photo of cls'
            cls_train_prompt.append(tp.replace('cls', f'an {cls}' if cls[0] in ['a', 'e', 'i', 'o',
                                                                                                'u'] else f'a {cls}'))
            test_prompt.append(tp.replace('cls', f'an {cls}' if cls[0] in ['a', 'e', 'i', 'o',
                                                                                              'u'] else f'a {cls}'))
            train_prompt.append(cls_train_prompt)

        self.train_prompt = [item for sublist in train_prompt for item in sublist]
        self.test_prompt = test_prompt

        print("Prompt for training: {}".format(train_prompt))
        print("Prompt for testing: {}".format(test_prompt))

        tokenizer, text_model, processor = self.models['tokenizer'], self.models['model'].text_model, self.models['processor']

        attr_train_prompt = []
        if self.group_type == ['gender']:
            for _cls, cls_prompt in zip(self.concepts, train_prompt):
                cls = f'an {_cls}' if _cls[0] in ['a', 'e', 'i', 'o', 'u'] else f'a {_cls}'
                for prompt in cls_prompt:
                    attr_train_prompt += [prompt.replace(cls, f'a male {_cls}'), prompt.replace(cls, f'a female {_cls}')]

            candidate_prompt, n_candidate_prompt = [], []
            for cls, cls_prompt in zip(self.concepts, train_prompt):
                for prompt in cls_prompt:
                    candidate_prompt += [prompt.replace(cls, f'male'), prompt.replace(cls, f'female')]
                    n_candidate_prompt.append((len(candidate_prompt) - 2, len(candidate_prompt)))
            self.attr = ['male', 'female']
        elif self.args.group == ['race']:
            for cls, cls_prompt in zip(self.concepts, train_prompt):
                for prompt in cls_prompt:
                    attr_train_prompt += [prompt.replace(cls, f'White {cls}'), prompt.replace(cls, f'Black {cls}'),
                                          prompt.replace(f'a {cls}', f'an Indian {cls}'),
                                          prompt.replace(f'a {cls}', f'an Asian {cls}'),
                                          prompt.replace(f'{cls}', f'Latino {cls}')]
            candidate_prompt, n_candidate_prompt = [], []
            for cls, cls_prompt in zip(self.concepts, train_prompt):
                for prompt in cls_prompt:
                    candidate_prompt += [prompt.replace(cls, f'White person'), prompt.replace(cls, f'Black person'),
                                            prompt.replace(f'A {cls}',
                                                        f'An Indian') if f'A {cls}' in prompt else prompt.replace(
                                                f'a {cls}', f'an Indian'),
                                            prompt.replace(f'A {cls}',
                                                        f'An Asian') if f'A {cls}' in prompt else prompt.replace(
                                                f'a {cls}', f'an Asian'),
                                            prompt.replace(cls, f'Latino')]
                    n_candidate_prompt.append((len(candidate_prompt) - 5, len(candidate_prompt)))
            self.attr = ['White', 'Black', 'Indian', 'Asian', 'Latino']
        elif self.args.group == 'age':
            candidate_prompt, n_candidate_prompt = [], []
            for cls, cls_prompt in zip(self.concepts, train_prompt):
                for prompt in cls_prompt:
                    candidate_prompt += [prompt.replace(cls, f'young person'),
                                            # prompt.replace(cls, f'middle aged {cls}'),
                                            prompt.replace(f'A {cls}', f'An old person') if f'A {cls}' in prompt else prompt.replace(f'a {cls}', f'an old person')]
                    n_candidate_prompt.append((len(candidate_prompt) - 2, len(candidate_prompt)))
            # self.attr = ['young', 'middle aged', 'old']
            self.attr = ['young person', 'old person']
        else:
            raise NotImplementedError
        
        if self.rank == 0:
            print("Attribute Prompt: {}".format(candidate_prompt))
            # Attribute Prompt: ['A photo of a male', 'A photo of a female', 'A photo of a male', 'A photo of a female']

        print(attr_train_prompt)
        # ['A photo of a male firefighter', 'A photo of a female firefighter', 'A photo of a male CEO', 'A photo of a female CEO']

        self.attr_train_prompt = attr_train_prompt
        candidate_inputs_clip = processor(text=candidate_prompt, return_tensors="pt", padding=True).to(self.cur_device)
        self.candidate_inputs_clip = [{k: v[n_s:n_e] for k, v in candidate_inputs_clip.items()} for (n_s, n_e) in n_candidate_prompt]

    def _load_models(self):
        model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        
        # vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
        # unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")
        # scheduler = LMSDiscreteScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")        
        vae = self.model.vae
        unet = self.model.unet
        scheduler = self.model.scheduler

        models = {'model': model, 'processor': processor, 'tokenizer': tokenizer, 'vae': vae, 'unet': unet, 'scheduler': scheduler}
        for k, v in models.items():
            if k in ['processor', 'tokenizer', 'scheduler']:
                continue
            _model = v.to(self.cur_device)
            models[k] = _model
        self.models = models

    def _build_embed_builder(self): # build the embedding builder (Soft Embedding)
        tokenizer, text_model = self.models['tokenizer'], self.models['model'].text_model

        # Define Soft Embedding
        init, init_range = None, 1.
        wash_size = 2
        try:
            attribute_vocab_idxes = tokenizer(self.attr, return_tensors='pt').input_ids[:, 1:-1].squeeze()
            with torch.no_grad():
                init = text_model.embeddings.token_embedding(attribute_vocab_idxes.to(self.cur_device))
                init = init.mean(dim=0, keepdim=True).expand(wash_size, -1)
        except:
            init = []
            for att in self.attr:
                attribute_vocab_idxes = tokenizer(att, return_tensors='pt').input_ids[:, 1:-1].squeeze()
                with torch.no_grad():
                    init.append(text_model.embeddings.token_embedding(attribute_vocab_idxes.to(self.cur_device)))
            init = [_it.view(-1, _it.shape[-1]) for _it in init]
            max_dim = max([_it.shape[0] for _it in init])
            init = [_it.expand(max_dim, -1) for _it in init]
            init = torch.stack(init, dim=0).mean(dim=0).expand(wash_size, -1)

        text_input = tokenizer(self.train_prompt, padding="max_length",
                               max_length=tokenizer.model_max_length - wash_size, return_tensors="pt")
        text_input_nowash = tokenizer(self.train_prompt, padding="max_length",
                                      max_length=tokenizer.model_max_length, return_tensors="pt")
        text_input_nowash_attr = tokenizer(self.attr_train_prompt, padding="max_length",
                                      max_length=tokenizer.model_max_length, return_tensors="pt")
        test_text_input = tokenizer(self.test_prompt, padding="max_length",
                                    max_length=tokenizer.model_max_length - wash_size, return_tensors="pt")
        test_text_input_nowash = tokenizer(self.test_prompt, padding="max_length",
                                           max_length=tokenizer.model_max_length, return_tensors="pt")
        uncond_input = tokenizer([""], padding="max_length", max_length=tokenizer.model_max_length,
                                 return_tensors="pt")

        with torch.no_grad():
            text_embeddings = text_model.embeddings.token_embedding(text_input.input_ids.to(self.cur_device)) # embedding of text encoder
            test_text_embeddings = text_model.embeddings.token_embedding(test_text_input.input_ids.to(self.cur_device))
            orig_text_input = {k: torch.cat([v[:,:1]]*wash_size+[v], dim=1).to(self.cur_device) for k, v in text_input.items()}
            self.ori_embeddings = text_model(orig_text_input['input_ids'])[0].detach() # This does not include attention_mask
                
            uncond_embeddings = text_model(uncond_input.input_ids.to(self.cur_device))[0]
            text_embeddings_nowash = text_model(text_input_nowash.input_ids.to(self.cur_device))[0]
            text_embeddings_nowash_attr = text_model(text_input_nowash_attr.input_ids.to(self.cur_device))[0]
            self.test_text_embeddings = test_text_embeddings
            self.test_text_embeddings_nowash = text_model(test_text_input_nowash.input_ids.to(self.cur_device))[0]

        embed_builder = embedding.SoftEmbedding(uncond_embeddings, text_embeddings, wash_size,
                                                        n_attr=len(self.attr),
                                                        init=init, init_range=init_range, 
                                                        text_embeddings_nowash=text_embeddings_nowash,
                                                        text_embeddings_nowash_attr=text_embeddings_nowash_attr,
                                                        before=True
                                                        # prompt_location=self.args.prompt_location,
                                                        ).to(self.cur_device)
        embed_builder_noddp = embed_builder
        # if self.args.distributed:
            # embed_builder = DistributedDataParallel(embed_builder, device_ids=[self.cur_device], output_device=self.cur_device)
        if self.rank == 0:
            print("Text: ", embed_builder_noddp.text_embeddings.size())
            # print("Uncond: ", self.uncond_embedding.size())
            print("Soft: ", embed_builder_noddp.soft_embedding.size())
        return embed_builder, embed_builder_noddp
    

    def _build_criterion(self):
        return LossV2(cls_type='ce', update_gather=True,
                        no_sampling=False, weight_no_update=10000., no_matching=False).to(self.cur_device)

    def _build_optimizer(self, embed_builder):
        lr = self.args.lr
        params_list = []
        for name, params in embed_builder.named_parameters():
            if name == 'soft_embedding':
                # if not (self.args.before and self.args.no_update_before):
                params_list.append(params)
                print(name)
            else:
                params_list.append(params)
                print(name)
        update_params = [{'params': params_list, 'lr': lr}]
        if self.args.optimizer == 'adam':
            optimizer = optim.Adam(update_params, weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'adamw':
            optimizer = optim.AdamW(update_params, weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'sgd':
            optimizer = optim.SGD(update_params, momentum=0., weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'adagrad':
            optimizer = optim.Adagrad(update_params, weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'rmsprop':
            optimizer = optim.RMSprop(update_params, weight_decay=self.args.weight_decay)
        return optimizer

    def _build_scheduler(self, optimizer):
        if self.args.lr_scheduler is None:
            return None
        if self.args.lr_scheduler == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.n_iters)
        elif self.args.lr_scheduler != 'step-tt':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
        
        return None
    
    def train(self, stage='stage2'):
        best_diff, best_diff_train = 1., 0.7
        initial_argmax = None
        cls_weight = [1.] * len(self.concepts)

        diff_list, best, dis_list = [], False, []

        for cls_idx, cls in enumerate(self.concepts):
            # torch.set_grad_enabled(False)
            self.embed_builder.eval()
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=self.args.use_amp):
                    self.run_one(is_train=False, cls_idx=cls_idx, prompt_idx=0,
                                    n_images=self.args.n_image_inference,
                                    num_inference_steps=50, it=0,
                                    stage=f'{stage}-eval-{cls}', world_size=1)
            diff = self.meters['test'][cls_idx]['weighted'].get_mse(self.target_ratio)
            discrepancy = self.meters['test'][cls_idx]['weighted'].get_discrepancy(self.target_ratio)
            diff_list.append(diff * cls_weight[cls_idx])
            dis_list.append(discrepancy)
            if self.rank == 0:
                print("mse", diff)
        diff_avg = np.array(diff_list).mean()
        if best_diff >= diff_avg:
            best_diff = diff_avg
        if initial_argmax is None:
            obtained_vals = [self.meters['test'][cls_idx]['weighted'].get_val() - self.target_ratio for cls_idx, _
                                in enumerate(self.concepts)]
            initial_argmax = [obtained_vals[cls_idx].sign() for cls_idx, _ in enumerate(self.concepts)]
        for cls_idx, cls in enumerate(self.concepts):
            self.meters['test'][cls_idx]['weighted'].step()
            self.meters['test'][cls_idx]['normal'].step()

        # Stage for updating
        for it in range(self.args.n_iters):
            loss_dict_all = {}
            for cls_idx, cls in enumerate(self.concepts):
                # prompt_idx = it % len(self.args.train_prompt)
                prompt_idx = 0
                # torch.set_grad_enabled(True)
                self.embed_builder.train()

                with torch.cuda.amp.autocast(enabled=self.args.use_amp):
                    no_update_batch_size = 9
                    loss_dict  = self.run_one(is_train=True, cls_idx=cls_idx, prompt_idx=prompt_idx, n_images=self.args.batch_size + no_update_batch_size,
                                                       num_inference_steps=self.args.num_inference_steps, it=it, stage=f'{stage}-{cls}',
                                                       world_size=1)
                # print(loss_dict)
                if self.args.report_memory:
                    print("After running one for training, max mem: {:.1f} GB ".format(gpu_mem_usage()))
                # torch.autograd.set_detect_anomaly(True)
                if not self.args.use_amp:
                    (loss_dict['total'] * cls_weight[cls_idx]).backward()
                else:
                    self.scaler.scale((loss_dict['total'] * cls_weight[cls_idx])).backward()
                    # self.scaler.unscale_(self.optimizer)
                    # print(self.embed_builder_noddp.soft_embedding.grad)

                for k, v in loss_dict.items():
                    if k in loss_dict_all.keys():
                        loss_dict_all[k].append(v)
                    else:
                        loss_dict_all[k] = [v]

            # Grad Accumulation and evaluation
            grad_ac_steps=5
            if (it + 1) % grad_ac_steps == 0:
                clip_grad_norm = 1
                torch.nn.utils.clip_grad_norm_(self.embed_builder.parameters(), clip_grad_norm)

                if self.args.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()

                if self.args.report_memory:
                    print("After updating learnable parameters, max mem: {:.1f} GB ".format(gpu_mem_usage()))

                # if self.rank == 0:
                    # self.save_embedding(it, best=False)

                diff_train = torch.Tensor([self.meters['train'][cls_idx]['weighted'].get_mse(self.target_ratio)
                                                  for cls_idx, cls in enumerate(self.concepts)]).mean()

                if diff_train < best_diff_train:
                    best_diff_train = diff_train
                if getattr(self, 'lr_scheduler', None) is not None:
                    if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        if it >= 10:
                            self.lr_scheduler.step(diff_train)
                    elif self.args.lr_scheduler != 'step-tt':
                        self.lr_scheduler.step()

                # if True or (it + 1) % 10 == 0 or excute_eval:
                diff_list, best, dis_list = [], False, []
                for cls_idx, cls in enumerate(self.concepts):
                    # torch.set_grad_enabled(False)
                    self.embed_builder.eval()
                    with torch.no_grad():
                        with torch.cuda.amp.autocast(enabled=self.args.use_amp):
                            self.run_one(is_train=False, cls_idx=cls_idx, prompt_idx=0, n_images=self.args.n_image_inference,
                                            num_inference_steps=self.args.num_inference_steps, it=it,
                                            stage=f'{stage}-eval-{cls}', world_size=1)
                    diff = self.meters['test'][cls_idx]['weighted'].get_mse(self.target_ratio)
                    discrepancy = self.meters['test'][cls_idx]['weighted'].get_discrepancy()
                    diff_list.append(diff * cls_weight[cls_idx])
                    dis_list.append(discrepancy * cls_weight[cls_idx])
                    if self.rank == 0:
                        print("mse", diff)

                    # if self.args.compare_unbiased:
                    #     self.run_one_with_undebiased(cls_idx=cls_idx, n_images=10,
                    #                                  num_inference_steps=self.args.num_inference_steps, it=it,
                    #                                  stage=f'{stage}-eval-{cls}', world_size=self.args.world_size)
                print(self.args.lr_scheduler, initial_argmax)
                if initial_argmax is None:
                    obtained_vals = [self.meters['test'][cls_idx]['weighted'].get_val() - self.target_ratio for cls_idx, _ in enumerate(self.concepts)]
                    # initial_argmax = [(obtained_vals[cls_idx].argmax().item(), obtained_vals[cls_idx].argmin().item()) for cls_idx, _ in enumerate(self.concepts)]
                    initial_argmax = [obtained_vals[cls_idx].sign() for cls_idx, _ in enumerate(self.concepts)]
                elif self.args.lr_scheduler == 'step-tt':
                    print('im here')
                    obtained_vals = [self.meters['test'][cls_idx]['weighted'].get_val() - self.target_ratio for
                                        cls_idx, _ in enumerate(self.concepts)]
                    current_argmax = [obtained_vals[cls_idx].sign() for cls_idx, _ in enumerate(self.concepts)]
                    # current_argmax = [
                    #     (obtained_vals[cls_idx].argmax().item(), obtained_vals[cls_idx].argmin().item()) for
                    #     cls_idx, _ in enumerate(self.concepts)]
                    print(initial_argmax, current_argmax)
                    n_same = 0
                    for a, b in zip(initial_argmax, current_argmax):
                        # if self.args.bias in ['race5', 'age']:
                        #     if a[0] == b[0]:
                        #         n_same += 1
                        # else:
                        #     if (a[0] == b[0] or a[1] == b[1]) and b[0] != b[1]:
                        #         n_same += 1
                        if (a * b).sum() > 0:
                            n_same += 1
                    if n_same == 0:
                        drop_r = 0.2
                        # drop_r = 0.1 if self.args.bias == 'gender' else 0.5
                        self.optimizer.param_groups[0]['lr'] = self.optimizer.param_groups[0]['lr'] * drop_r
                        print("Reduce lr")
                        print("change", initial_argmax, current_argmax)
                        initial_argmax = current_argmax

                for cls_idx, cls in enumerate(self.concepts):
                    self.meters['train'][cls_idx]['weighted'].step()
                    self.meters['train'][cls_idx]['normal'].step()
                    self.meters['test'][cls_idx]['weighted'].step()
                    self.meters['test'][cls_idx]['normal'].step()

                if self.rank == 0:
                    diff_avg = np.array(diff_list).mean()
                    if best_diff >= diff_avg:
                        best_diff = diff_avg
                        best = True
                #    self.save_embedding(it, best=best)

                if self.args.report_memory:
                    print("After evaluation, max mem: {:.1f} GB ".format(gpu_mem_usage()))

                if self.optimizer.param_groups[0]['lr'] < 1e-8:
                    break

        return self.model

                # if (it + 1) % (self.args.grad_ac_steps * 2) == 0 and self.args.compare_unbiased:
                #     for cls_idx, cls in enumerate(self.concepts):
                #         # torch.set_grad_enabled(False)
                #         self.embed_builder.eval()
                #         self.run_one_with_undebiased(cls_idx=cls_idx, prompt_idx=0, n_images=10,
                #                                      num_inference_steps=self.args.num_inference_steps, it=it,
                #                                      stage=f'{stage}-eval-{cls}', world_size=1)    

    def run_one(self, is_train, cls_idx, prompt_idx, n_images, num_inference_steps, it, stage, world_size=1):
        model, vae, unet, scheduler, processor, tokenizer = \
            self.models['model'], self.models['vae'], self.models['unet'], self.models['scheduler'], self.models['processor'], self.models['tokenizer']
        meters = self.meters['train'][cls_idx] if (is_train or 'stage1' in stage) else self.meters['test'][cls_idx]

        height = 512  # default height of Stable Diffusion
        width = 512  # default width of Stable Diffusion
        guidance_scale = 7.5  # Scale for classifier-free guidance
        generator = torch.manual_seed((self.args.seed + 1) * (self.rank + 1) * (it + 1) - 1)
        # generator = torch.manual_seed(self.rank)
        batch_size = min(self.args.batch_size, 4) if is_train else 10

        # prt_idx = cls_idx * len(self.args.train_prompt) + prompt_idx
        prt_idx = cls_idx + prompt_idx
        print(prt_idx, is_train)
        ##### DEBIAS #####
        text_embeddings = self.embed_builder(prt_idx, 1,
                                             text_embeddings=self.test_text_embeddings[prt_idx:prt_idx+1] if not (is_train or 'stage1' in stage) else None,
                                             text_model=model.text_model)
        # text_embeddings[batch_size:] = self.ori_embeddings.expand(batch_size, *self.ori_embeddings.shape[1:])

        all_images = []
        grid_images = []
        n_images_update = self.args.batch_size
        init_latents_update = []

        for i in tqdm(range(0, n_images, batch_size)) if self.rank == 0 else range(0, n_images, batch_size):
            bs = min(batch_size, n_images - i)
            t_embeddings = torch.stack([text_embeddings] * bs, dim=1).view(-1, *text_embeddings.shape[1:])

            with torch.enable_grad() if is_train and i < n_images_update else torch.no_grad():
                with torch.cuda.amp.autocast(enabled=False if is_train and i < n_images_update else self.args.use_amp):
                    # Generate Initial Noise
                    in_channels = unet.in_channels
                    init_latents = torch.randn((bs, in_channels, height // 8, width // 8), generator=generator
                                          ).to(self.cur_device)
                    if i < n_images_update:
                        init_latents_update.append(init_latents.detach())
                    scheduler.set_timesteps(num_inference_steps)
                    latents = init_latents * scheduler.init_noise_sigma

                    for t_idx, t in enumerate(scheduler.timesteps):

                        with torch.no_grad():
                            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                            latent_model_input = torch.cat([latents] * 2)
                            latent_model_input = scheduler.scale_model_input(latent_model_input, t)

                        # predict the noise residual
                        grad_mode = is_train and ((self.args.grad_mode == 'full') or (
                                self.args.grad_mode == 'half' and t_idx >= num_inference_steps // 2) or (
                                self.args.grad_mode == 'last' and t_idx == num_inference_steps - 1) or (
                                self.args.grad_mode == 'last2' and t_idx > num_inference_steps - 3))

                        if grad_mode:
                            with torch.no_grad():
                                noise_pred_uncond = unet(latent_model_input[:bs], t,
                                                         encoder_hidden_states=t_embeddings[:bs]).sample
                            with torch.enable_grad():
                                print('here')
                                noise_pred_text = unet(latent_model_input[bs:], t,
                                                       encoder_hidden_states=t_embeddings[bs:]).sample
                        else:
                            with torch.no_grad():
                                # predict the noise residual
                                noise_pred = unet(latent_model_input, t, encoder_hidden_states=t_embeddings).sample

                            # perform guidance
                            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                        # compute the previous noisy sample x_t -> x_t-1
                        latents = scheduler.step(noise_pred, t, latents).prev_sample
                    if i == 0:
                        final_latents_update = latents

                    # scale and decode the image latents with vae
                    latents = 1 / 0.18215 * latents

                    image = vae.decode(latents).sample

                    image = (image / 2 + 0.5).clamp(0, 1)
                    all_images.append(image)

            # if i != batch_size or self.args.reg_version != 'v3':
            # Save images
            image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
            images = (image * 255).round().astype("uint8")
            pil_images = [Image.fromarray(image) for image in images]
            grid_images.extend(pil_images)

            if self.args.report_memory:
                print("After generation of images within one batch, max mem: {:.1f} GB ".format(gpu_mem_usage()))

        init_latents_update = torch.cat(init_latents_update, dim=0)
        all_images = torch.cat(all_images, dim=0)
        all_images = F.interpolate(all_images, size=(224, 224), mode='bilinear', align_corners=False)
        all_images -= torch.Tensor(processor.feature_extractor.image_mean).to(self.cur_device).view(1, -1, 1, 1)
        all_images /= torch.Tensor(processor.feature_extractor.image_std).to(self.cur_device).view(1, -1, 1, 1)

        with torch.cuda.amp.autocast(enabled=False):
            logits_all_image = forward_clip(model, all_images, **self.candidate_inputs_clip[prt_idx])
        if self.args.pre_history:
            if world_size > 1:
                _logits_all_image = logits_all_image.contiguous()
                with torch.no_grad():
                    _all_logits_all_image = [torch.zeros_like(_logits_all_image) for _ in range(world_size)]
                    torch.distributed.all_gather(_all_logits_all_image, _logits_all_image)
                _logits_all_image = torch.cat(_all_logits_all_image, dim=0)
                probs = F.softmax(_logits_all_image, dim=-1).detach()
            else:
                probs = F.softmax(logits_all_image, dim=-1).detach()
            preds = F.one_hot(probs.argmax(dim=-1), num_classes=len(self.attr)).cpu().float().sum(0)
            print(preds)
            meters['weighted'].update(preds)
            meters['normal'].update(preds)

        sampling_prob_smoothing = 0
        t_sampling_prob = (meters['weighted'].avg + sampling_prob_smoothing) / (1 + len(self.attr) * sampling_prob_smoothing)
        t_sampling_prob = 1 / (t_sampling_prob + 1e-8)
        t_sampling_prob /= t_sampling_prob.sum()
        # sampling_prob = self.embed_builder_noddp.dsampling_prob if self.args.train_sampling_prob else t_sampling_prob.detach().clone().to(
            # self.cur_device)
        sampling_prob = t_sampling_prob.detach().clone().to(self.cur_device)
        sampling_prob.data = self.target_ratio.data

        # Save images with classified attributes
        # pred = torch.argmax(logits_all_image, dim=-1).detach().cpu()
        # if len(pred) < len(grid_images):
        #     pred = torch.cat([pred[:batch_size]] * (len(add_img_for_reg) + 1) + [pred[batch_size:]])

        # CHECK LATER
        # if self.rank == 0:
        #     for img_i, img in enumerate(grid_images):
        #         draw = ImageDraw.Draw(img)
        #         draw.text((10, 10), self.attr[pred[img_i]], fill=(255, 255, 255), font=ImageFont.truetype("arial.ttf", size=30))
        #         draw.text((10, 30), self.attr[pred[img_i]], fill=(0, 0, 0),
        #                   font=ImageFont.truetype("arial.ttf", size=30))
        #     self.save_images(grid_images, f"img_grid_{stage}_{it}")

        # with torch.no_grad():
        #     logits_all_image_prompt = forward_clip(model, all_images, **self.prompt_inputs_clip[cls_idx])
        if self.args.report_memory:
            print("After forwarding images to CLIP, max mem: {:.1f} GB ".format(gpu_mem_usage()))

        loss_dict = {}
        if is_train:
            if self.rank == 0:
                val = ' '.join(['{:.3f}'.format(v) for v in sampling_prob])
                print("Iter {} sampling prob for attr with {}: {}".format(it, self.concepts[cls_idx], val))

            loss, probs, targets = self.criterion(logits_all_image, sampling_prob=sampling_prob, update_batch=n_images_update,
                                                  rank=self.rank, world_size=1)
            loss_dict['total'] = loss * 1 #self.args.cls_weight if self.args.cls_weight > 0 else torch.zeros_like(loss)
            loss_dict['debias'] = loss.item()
            loss_dict['total_log'] = loss_dict['total'].item()
            loss_dict['total'] *= n_images / n_images_update # scale loss when n_images > n_images_update

            # if meters['weighted'].count > 0 and self.args.train_sampling_prob:
            #     loss_sampling = F.mse_loss(sampling_prob, t_sampling_prob.detach().clone().to(self.cur_device))
            #     loss_dict['total'] += loss_sampling
            #     loss_dict['total_log'] += loss_sampling.item()
            #     loss_dict['sampling'] = loss_sampling.item()

            # if not self.args.update_gather and world_size > 1:
            #     # Synchrnize preds across device
            #     with torch.no_grad():
            #         all_probs = [torch.zeros_like(probs) for _ in range(world_size)]
            #         torch.distributed.all_gather(all_probs, probs)
            #     probs = torch.cat(all_probs, dim=0)

        else:
            # if world_size > 1:
            #     logits_all_image = logits_all_image.contiguous()
            #     with torch.no_grad():
            #         all_logits_all_image = [torch.zeros_like(logits_all_image) for _ in range(world_size)]
            #         torch.distributed.all_gather(all_logits_all_image, logits_all_image)
            #     logits_all_image = torch.cat(all_logits_all_image, dim=0)
            probs = F.softmax(logits_all_image, dim=-1).detach()
        # if world_size > 1:
        #     logits_all_image_prompt = logits_all_image_prompt.contiguous()
        #     with torch.no_grad():
        #         all_logits_all_image_prompt = [torch.zeros_like(logits_all_image_prompt) for _ in range(world_size)]
        #         torch.distributed.all_gather(all_logits_all_image_prompt, logits_all_image_prompt)
        #     logits_all_image_prompt = torch.cat(all_logits_all_image_prompt, dim=0)

        if is_train:
            t_embeddings = self.embed_builder_noddp.get_undebised_embeddings(prt_idx, len(init_latents_update),
                                                                                attr=len(self.attr) * cls_idx + targets[:len(init_latents_update)].long())
            final_latents = self._generate_image(model, vae, unet, scheduler, init_latents_update, num_inference_steps, t_embeddings,
                                                    guidance_scale)
            loss_reg = F.mse_loss(final_latents_update, final_latents)
            if self.args.reg_weight > 0:
                loss_dict['total'] += loss_reg * self.args.reg_weight
                loss_dict['total_log'] += loss_reg.item() * self.args.reg_weight
            loss_dict['prompt_reg'] = loss_reg.item()

        if self.rank == 0:
            print(str(meters['weighted']))
        return loss_dict#, {'entropy': get_entropy(probs).mean()}

    def _generate_image(self, model, vae, unet, scheduler, init_latents, num_inference_steps, text_embeddings, guidance_scale):
        with torch.no_grad():
            with torch.cuda.amp.autocast(self.args.use_amp):
                # Generate Initial Noise
                scheduler.set_timesteps(num_inference_steps)
                latents = init_latents * scheduler.init_noise_sigma

                for t_idx, t in enumerate(scheduler.timesteps):

                    with torch.no_grad():
                        # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                        latent_model_input = torch.cat([latents] * 2)
                        latent_model_input = scheduler.scale_model_input(latent_model_input, t)

                    # predict the noise residual
                    noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

                    # perform guidance
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                    # compute the previous noisy sample x_t -> x_t-1
                    latents = scheduler.step(noise_pred, t, latents).prev_sample

        return latents.detach() #, image.detach()


class LossV2(torch.nn.Module):
    def __init__(self, cls_type='bce', weight_no_update=1., update_gather=False, no_sampling=False, no_matching=False):
        super(LossV2, self).__init__()
        if not no_matching:
            self.matcher = matcher.HungarianMatcher(weight_no_update=weight_no_update)
        if cls_type == 'ce':
            self.cls_loss = CrossEntropyLoss(reduction='none')
        else:
            raise NotImplementedError
        self.update_gather = update_gather
        self.no_sampling = no_sampling

    def forward(self, outputs, sampling_prob, update_batch=None, rank=None, world_size=None):
        """
        This is based on gumbel softmax
        :param outputs: (batch_size, 2) logits
        :param targets: (2,) ratio of negative and positive samples
        :return: loss
        """
        r = torch.rand(outputs.shape[0]).to(outputs.device)
        targets = torch.zeros_like(r)
        current_device_indices = torch.ones(outputs.shape[0])
        for i in range(len(sampling_prob)):
            targets += (r > sampling_prob[:i+1].sum()).float()
        if self.update_gather and world_size > 1:
            assert rank is not None
            outputs = outputs.contiguous()
            with torch.no_grad():
                all_outputs = [torch.zeros_like(outputs) for _ in range(world_size)]
                torch.distributed.all_gather(all_outputs, outputs)
                current_device_indices = [torch.ones(outputs.shape[0]) if ws == rank else torch.zeros(outputs.shape[0]) for ws in range(world_size)]
                if not self.no_sampling:
                    all_targets = [torch.zeros_like(targets) for _ in range(world_size)]
                    torch.distributed.all_gather(all_targets, targets)
            all_outputs[rank] = outputs
            all_outputs = torch.cat(all_outputs, dim=0)
            current_device_indices = torch.cat(current_device_indices, dim=0)
            if not self.no_sampling:
                all_targets = torch.cat(all_targets, dim=0)
            else:
                n0 = int(sampling_prob * all_outputs.shape[0])
                all_targets = torch.ones(all_outputs.shape[0]).to(outputs.device)
                all_targets[:n0] = 0
        else:
            all_outputs, all_targets = outputs, targets
            if self.no_sampling:
                n0 = int(sampling_prob * all_outputs.shape[0])
                all_targets = torch.ones(all_outputs.shape[0]).to(outputs.device)
                all_targets[:n0] = 0

        if hasattr(self, 'matcher'):
            pred_ind, targ_ind = self.matcher(all_outputs, all_targets, update_batch=update_batch)
            all_outputs = all_outputs[pred_ind]
            all_targets = all_targets[targ_ind]

            current_targets = all_targets[pred_ind[current_device_indices == 1]].detach()
        else:
            current_targets = all_targets[current_device_indices == 1].detach()

        outputs_prob = F.softmax(all_outputs, dim=-1)
        loss = self.cls_loss(all_outputs, all_targets.long())
        loss_total = loss.mean()
        # loss_update = loss[:update_batch].mean() if update_batch is not None else loss_total
        # loss = (loss_total - loss_update).detach() + loss_update

        return loss_total, outputs_prob.detach(), current_targets

