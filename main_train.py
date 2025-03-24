

import argparse
import itertools
import logging
import math
import shutil
import json
import pytz
import random

import torch
import numpy as np
import datetime
from argument_train import get_args
from pathlib import Path
import pickle
import wandb

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed, GradScalerKwargs

import transformers
import diffusers

import sys
import os
os.environ["WANDB_MODE"]="offline"

import data_handler
import networks
import trainer
from utils import check_log_dir
import utils

def main(args):
    dm = networks.ModelFactory.get_model(modelname=args.target_model, train=args.train)
    print(dm.name_or_path)
    _trainer = trainer.TrainerFactory.get_trainer(trainername=args.trainer, model=dm, args=args)    

    if args.trainer in ['finetuning', 'rag']:
        logger = get_logger(__name__)
        logging_dir = Path(args.output_dir, args.logging_dir)

        kwargs = GradScalerKwargs(
            init_scale = 2.**0,
            growth_interval=99999999, 
            backoff_factor=0.5,
            growth_factor=2,
            )

        accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

        accelerator = Accelerator(
            gradient_accumulation_steps=1, # we did not implement gradient accumulation
            mixed_precision=args.mixed_precision, # default : fp16
            log_with=args.report_to, 
            project_config=accelerator_project_config,
            kwargs_handlers=[kwargs]
        )

        # if args.report_to == "wandb":
        #     if not is_wandb_available():
        #         raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
        #     import wandb
        # else:
        #     raise ValueError("--report_to must be set to 'wanb', others are not implemented.")
        
        # Make one log on every process with the configuration for debugging.
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        logger.info(accelerator.state, main_process_only=False)
        if accelerator.is_local_main_process:
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()
        else:
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()

        # Handle the repository creation
        if accelerator.is_main_process:
            if args.output_dir is not None:
                os.makedirs(args.output_dir, exist_ok=True)
                
        # We need to initialize the trackers we use, and also store our configuration.
        # The trackers initializes automatically on the main process.

        # folder_name = f"{args.trainer}/{args.train_images_per_prompt_GPU*accelerator.num_processes}_wImg-{args.weight_loss_img}-{args.factor1}-{args.factor2}_wFace-{args.weight_loss_face}_Th-{args.uncertainty_threshold}_loraR-{args.rank}_lr-{args.learning_rate}"#_{timestring}"
        group_name = "".join([g[0].upper() for g in args.trainer_group])
        # group_name = args.group[0]

        prompt_file_name = args.prompt_occupation_path.split('/')[-1].split('.')[0]
        folder_name = f"{args.train_images_per_prompt_GPU*accelerator.num_processes}_wImg-{args.weight_loss_img}-loraR-{args.rank}_lr-{args.learning_rate}_{prompt_file_name}"#_{timestring}"
        if args.finetuning_ver == 'ver3':
            folder_name += f'_{args.temp}'
        if args.finetuning_ver == 'ver1':
            folder_name += f'_{args.n_cs}_{args.mpr_num_batches}'
        if args.trainer == 'finetuning':
            method_name = args.trainer + '_' + args.finetuning_ver 
        else:
            method_name = args.trainer
        args.imgs_save_dir = os.path.join('/n/netscratch/calmon_lab/Lab/datasets', method_name, group_name, "imgs_in_training", folder_name)
        args.ckpts_save_dir = os.path.join('/n/netscratch/calmon_lab/Lab/trained_models', method_name, group_name, folder_name, "ckpts")

        if accelerator.is_main_process:
            os.makedirs(args.imgs_save_dir, exist_ok=True)
            os.makedirs(args.ckpts_save_dir, exist_ok=True)
            accelerator.init_trackers(
                args.trainer, 
                init_kwargs = {
                    "wandb": {
                        "name": folder_name, 
                        "dir": args.output_dir
                            }
                    }
                )

        set_seed(args.seed, device_specific=True)
    else:
        accelerator = None
        utils.set_seed(args.seed)


    if args.training_dataset is not None:
        train_loader, _ = data_handler.DataloaderFactory.get_dataloader(dataname=args.training_dataset, args=args)
    else:
        train_loader = None
    model =_trainer.model
    _trainer.train(accelerator, train_loader)

    # save model
    save_dir = f'trained_models/{args.trainer}'
    check_log_dir(save_dir)

    groupname = [_g[0] for _g in args.trainer_group]
    groupname = "".join(groupname)
    filename = f'{args.date}_{groupname}'
    torch.save(model, os.path.join(save_dir, f'{filename}.pt'))
        
    # Get the required model

    # Train the model

    wandb.finish()

if __name__ == '__main__':

    print(" ".join(sys.argv))

    # check gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    print()

    # Print out additional information when using CUDA
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Reserved: ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
        print()

    args = get_args()    

    np.set_printoptions(precision=4)
    torch.set_printoptions(precision=4)

    # set_seed(args.seed)


    now = datetime.datetime.now()
    # Format as 'ddmmyyHMS'
    formatted_time = now.strftime('%H%M')
    if args.date == 'default':
        args.date = now.strftime('%m%d%y')
    args.time = formatted_time

    run = wandb.init(
            project='mpr_generative',
            entity='sangwonjung-Harvard University',
            name=args.date+'_'+formatted_time,
            settings=wandb.Settings(start_method="fork")
    )
    print('wandb mode : ',run.settings.mode)
    
    wandb.config.update(args)

    main(args)
    
    wandb.finish()
