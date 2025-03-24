# Multi-Group Proportional Representation in Text-to-Image Models
---
Codebase for the paper ["Multi-Group Proportional Representation in Text-to-Image Models"] by Sangwon Jung, Alex Oesterling, Claudio Mayrink Verdun, Sajani Vithana, Taesup Moon, and Flavio P. Calmon

## Installation
Install necessary packages:

```
pip install -r requirement.yml
```

## Usage

## Precomputing Sensitive Attribute Classifiers 
If you use sensitive attribute classifiers, you should train them first. In our experiments, we probe a linear classifier on the top of the CLIP embedding space for gender, age and race or utilize VQA systems like BLIP for the presence of wheelchair.

For the case of gender, age and race, probe a linear classifier:

```
python experiments/train_linear_probes.py -device "cuda" -dataset "celeba" --data-path "path/to/dataset" --embed-path "path/to/load/embeddings"
```

## Replicating Experiments

### Base 
