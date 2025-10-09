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
To compute MPR, you need to prepare generate images and reference datasets or statistics. You should set the dataset path for the generated images in the argument "dataset-path". For the reference dataset, put the dataset in the "./dataset". 
You can set the function class using 'linear' or 'dt2' for the linear functions or decision trees. The number following 'dt' means the depth of decision trees. 
The "pool-size" is the number of generated images and the "resampling-sze" is the number of images resampled when using the bootstrapping algorithm: the images are samples with "n-resampling" times.
```
python main_eval.py --dataset-path ../representational-generation/datasets/scratch/SD_14/CEO --refer-dataset fairface --functionclass dt1 --mpr-group gender --target-concept CEO --pool-size 500  --bootstrapping --resampling-size 500 --n-resampling 10 --mpr-onehot --no-wandb
```


