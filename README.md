# Multi-Group Proportional Representation in Text-to-Image Models
---
Codebase for the paper ["Multi-Group Proportional Representation in Text-to-Image Models"] by Sangwon Jung, Alex Oesterling, Claudio Mayrink Verdun, Sajani Vithana, Taesup Moon, and Flavio P. Calmon

## Installation
Install necessary packages:

```
pip install -r requirement.yml
```

## Usage

### Precomputing Sensitive Attribute Classifiers 
If you use sensitive attribute classifiers, you should train them first. In our experiments, we probe a linear classifier on the top of the CLIP embedding space for gender, age and race or utilize VQA systems like BLIP for the presence of wheelchair.

For the case of gender, age and race, probe a linear classifier using the following command:

```
python train_linear_probes.py --train --vision-encoder CLIP
```

### Generate images from SD 1.4v
You can generate images from Stable Diffusion v1.4 using the following command.
The generated images will be saved in ./dataset/{method_name}.

If you want to generate images using a fine-tuned model, specify the path to the fine-tuned checkpoint and set --trainer to 'finetuning'.
```
python generate_image.py --trainer scratch --n-generations 1000 --concepts CEO --n-gen-per-iter 10 --group gender --model SD_14 
```

### Compute MPR score
To compute MPR, you need to prepare both the generated images and a reference dataset (or precomputed statistics).
Specify the path to the generated images using the argument --dataset-path.
For the reference dataset, place the dataset files inside the ./dataset directory.

You can choose the function class by setting --functionclass to either 'linear' or 'dtX', where 'dtX' denotes a decision tree of depth X (e.g., 'dt2' for a depth-2 tree).

The --pool-size argument specifies the total number of generated images, while --resampling-size determines how many images are resampled during the bootstrapping process. The resampling procedure is repeated --n-resampling times.
```
python main_eval.py --dataset-path ../representational-generation/datasets/scratch/SD_14/CEO --refer-dataset fairface --functionclass dt1 --mpr-group gender --target-concept CEO --pool-size 500  --bootstrapping --resampling-size 500 --n-resampling 10 --mpr-onehot --no-wandb
```

### Finetune SD 1.4v with MPR regularization

To finetune Stable Diffusion v1.4 with MPR regularization, prepare a JSON file containing the occupations of interest.
You should also specify which attributes (e.g., gender, age, race), function class (e.g., linear or decision tree), and regularization strength (λ) to use, as well as the total number of training iterations.
```
accelerate launch --config_file configs/accelerate_config.yaml main_train.py --prompt_occupation_path occupation.json --train_text_encoder --trainer-group gender age race --functionclass linear --val_images_per_prompt_GPU 50 --weight_loss_img 0.5 --iteration 2000 --evaluate_every_n_iter 100 --n-cs 4
```

