import fiftyone
dir = "/n/holyscratch01/calmon_lab/Lab/datasets"
dataset = fiftyone.zoo.load_zoo_dataset("open-images-v6", split="train", dataset_dir = dir)