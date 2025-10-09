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

### Compute MPR score
To compute MPR, you need to prepare both the generated images and a reference dataset (or precomputed statistics).
Specify the path to the generated images using the argument --dataset-path.
For the reference dataset, place the dataset files inside the ./dataset directory.

You can choose the function class by setting --functionclass to either 'linear' or 'dtX', where 'dtX' denotes a decision tree of depth X (e.g., 'dt2' for a depth-2 tree).

The --pool-size argument specifies the total number of generated images, while --resampling-size determines how many images are resampled during the bootstrapping process. The resampling procedure is repeated --n-resampling times.
```
python main_eval.py --dataset-path ../representational-generation/datasets/scratch/SD_14/CEO --refer-dataset fairface --functionclass dt1 --mpr-group gender --target-concept CEO --pool-size 500  --bootstrapping --resampling-size 500 --n-resampling 10 --mpr-onehot --no-wandb
```



