import json
import torch
import clip
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from mpr.preprocessing import group_estimation
import numpy as np
import pickle

json_file_path = 'datasets/occupation_list.json'

with open(json_file_path, 'r') as file:
    data = json.load(file)

formatted_occupations = []
for occupation in data['occupations']:
    formatted_occupation = occupation.lower()
    formatted_occupations.append(f'{formatted_occupation}')

print("Occupations:", formatted_occupations)

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

text_inputs = torch.cat([clip.tokenize(occupation) for occupation in formatted_occupations]).to(device)

batch_size = 128
text_features_list = []
for batch in torch.split(text_inputs, batch_size):
    with torch.no_grad():
        text_features_batch = model.encode_text(batch)
        text_features_list.append(text_features_batch)
text_features = torch.cat(text_features_list)
print("Text Features Shape:", text_features.shape)

text_features_np = text_features.cpu().numpy()

# normal similarity matrix
cosine_sim_matrix = cosine_similarity(text_features_np)

print("Cosine Similarity Matrix Shape:", cosine_sim_matrix.shape)
print("Cosine Similarity Matrix:", cosine_sim_matrix)

with open('results/sim_mats/cosine_sim_matrix.pkl', 'wb') as f:
    pickle.dump(cosine_sim_matrix, f)

estimated_groups = []

for group in ['gender','age','race']:
    estimated_group = group_estimation(text_features_np, group=group, vision_encoder_name='CLIP', onehot=False, loader=None)
    estimated_groups.append(estimated_group)
estimated_groups = np.concatenate(estimated_groups, axis=1)

cosine_sim_matrix_gender = np.zeros((len(formatted_occupations), len(formatted_occupations)))
for i in range(len(formatted_occupations)):
    for j in range(len(formatted_occupations)):
        cosine_sim_matrix_gender[i][j] = np.abs(estimated_groups[i][0] - estimated_groups[j][0])

with open('results/sim_mats/cosine_sim_matrix_for_gender.pkl', 'wb') as f:
    pickle.dump(cosine_sim_matrix_gender, f)

cosine_sim_matrix_gar = np.zeros((len(formatted_occupations), len(formatted_occupations)))
for i in range(len(formatted_occupations)):
    for j in range(len(formatted_occupations)):
        print(estimated_groups[i], estimated_groups[j])
        print(np.linalg.norm(estimated_groups[i] - estimated_groups[j], ord=1))
        cosine_sim_matrix_gar[i][j] = np.linalg.norm(estimated_groups[i] - estimated_groups[j], ord=1)


with open('results/sim_mats/cosine_sim_matrix_for_gar.pkl', 'wb') as f:
    pickle.dump(cosine_sim_matrix_gar, f)





