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

For the case of gender, age and race, probe a linear classifier:

```
python train_linear_probes.py --train --vision-encoder CLIP
```

'--refer-dataset
### Compute MPR score

```
python main_eval.py --dataset-path ../representational-generation/datasets/scratch/SD_14/CEO --refer-dataset fairface --functionclass dt1 --target-concept CEO --pool-size 500  --bootstrapping --resampling-size 500 --n-resampling 10 --mpr-onehot --mpr-group gender
```

