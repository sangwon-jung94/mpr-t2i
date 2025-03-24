import torch 
n_generation = 100
weight_dtype_high_precision = torch.float

noises = torch.randn(
        [n_generation,4,64,64],
        dtype=weight_dtype_high_precision
    )

print(noises.shape)
import pickle
with open('noise_vis.pkl', 'wb') as f:
    pickle.dump(noises, f)
