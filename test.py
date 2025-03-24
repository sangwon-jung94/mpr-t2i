

import torch
import numpy as np
import datetime
from argument import get_args
import pickle
import wandb

import sys
import os
os.environ["WANDB_MODE"]="offline"
import data_handler
import networks
import retriever
from utils import set_seed, make_result_path,  compute_similarity, get_statistics
from mpr.mpr import getMPR
from mpr.preprocessing import identity_embedding
from tqdm import tqdm

def eval(args):
    print(args.mpr_group)
    if args.refer_dataset != 'statistics':
        refer_loader = data_handler.DataloaderFactory.get_dataloader(dataname=args.refer_dataset, args=args)
    else:
        statistics = get_statistics(args.target_concept, args.mpr_group)
    query_loader = data_handler.DataloaderFactory.get_dataloader(dataname=args.query_dataset, args=args)

    ## Compute MPR
    vision_encoder = networks.ModelFactory.get_model(modelname=args.vision_encoder, train=args.train)  
    vision_encoder = vision_encoder.cuda() if torch.cuda.is_available() else vision_encoder

    ## Make embedding vectors
    print('extract embeddings from the reference distribution')
    if args.refer_dataset != 'statistics':
        refer_embedding, _ = identity_embedding(args, vision_encoder, refer_loader,args.mpr_group, query=False)
    print('extract embeddings from the query distribution')
    print(query_loader)
    print(len(query_loader.dataset))    
    query_embedding, feature_dic = identity_embedding(args, vision_encoder, query_loader, args.mpr_group, query=True)
    print('Complete estimating group labels')
    trainer = 'scratch' if args.dataset_path.split('/') == 4 else args.dataset_path.split('/')[1]
    base_filename = f'group_labels/{args.target_model}_{args.target_concept}_{args.functionclass}' if trainer == 'scratch' else f'group_labels/{trainer}_{args.target_model}_{args.target_concept}_{args.functionclass}'
    base_filename += '_onehot' if args.mpr_onehot else ''

    idxs_set = []
    # for batch in tqdm(query_loader):
    #     image, label, idxs =  batch
    #     idxs_set.append(idxs)
    # idxs_set = torch.cat(idxs_set).numpy()
    # print(idxs_set)
    # with open(f'group_labels/{args.target_concept}_refer_group_labels.pkl', 'wb') as f:
        # pickle.dump(refer_g_embedding,f)
    # print(args.mpr_onehot)
   # with open(f'group_labels/{args.target_concept}_refer_group_labels_true.pkl', 'wb') as f:
   #     pickle.dump(refer_loader.dataset.labels.numpy(),f)
    with open(base_filename+'_group_labels.pkl', 'wb') as f:
        pickle.dump(query_embedding,f)
        print(base_filename+'_group_labels.pkl is saved')
    # with open(base_filename+'_img_idxs.pkl', 'wb') as f:
        # pickle.dump(query_embedding,f)
        # print(base_filename+'_group_labels.pkl is saved')

    # s = compute_similarity(query_embedding, concept_labels, concept_set, vision_encoder, args.vision_encoder)        
    # with open(base_filename+'_scores.pkl', 'wb') as f:
        # pickle.dump(s,f)
    
#     if args.retrieve: 
#         _retriever = retriever.RetrieverFactory.get_retriever(args.retriever, args)
    
#     MPR_dic = {}
#     norMPR_dic = {}
#     score_dic = {}
#     idx_dic = {}
#     # n_compute_mpr = args.n_compute_mpr if args.pool_size != 21000.0 or args.pool_size!=1.0  else 1
#     n_compute_mpr = 1
#     for _ in range(n_compute_mpr):
#         if args.pool_size != 1.0:
#             total = query_embedding.shape[0]
#             if args.pool_size < 1:
#                 n_samples = int(query_embedding.shape[0]*args.pool_size)
#             else:
#                 n_samples = int(args.pool_size)

#             idx = np.random.choice(total, n_samples, replace=False)
#             # print(idx[:20])
#             print('Pool size is reduced to ', args.pool_size)

#             query_embedding_split = query_embedding[idx]
#             query_g_embedding_split = query_g_embedding[idx]
#             concept_labels_split = concept_labels[idx]

#         else:
#             query_embedding_split = query_embedding
#             query_g_embedding_split = query_g_embedding
#             concept_labels_split = concept_labels

#     #     # compute CLIP similarity
#     #     s = compute_similarity(query_embedding_split, concept_labels_split, concept_set, vision_encoder, args.vision_encoder)        
        
#     # compute MPR
#         for i, concept in enumerate(concept_set):
#             if concept not in ['firefighter', 'CEO']:
#                 continue
#             if concept not in MPR_dic.keys():
#                 MPR_dic[concept] = []
#                 norMPR_dic[concept] = []
#                 score_dic[concept] = []
#                 idx_dic[concept] = []

#             concept_idx = concept_labels_split == i
#             if args.retrieve:
#                 idx, MPR_list, noretrieve_MPR_list, score_list = _retriever.retrieve(query_g_embedding_split[concept_idx], refer_g_embedding, k=args.k, s=s[concept_idx])
#                 # score = np.sum(s[concept_idx][idx.astype(dtype=bool)])
#                 idx_dic[concept].append(idx)
#                 MPR_dic[concept].append(MPR_list)
#                 score_dic[concept].append(score_list)
#                 norMPR_dic[concept].append(noretrieve_MPR_list)
#                 print(f'concept: {concept}, MPR: {noretrieve_MPR_list[-1]}, score: {score_list[-1]}')
                
#             else:
#                 idx = None                
#                 score = np.sum(s[concept_idx])
#                 MPR, reg = getMPR(query_g_embedding_split[concept_idx], curation_set=refer_g_embedding, indices=idx, modelname=args.functionclass)
#                 MPR_dic[concept].append(MPR)
#                 score_dic[concept].append(score)
#                 print(f'concept: {concept}, final MPR: {MPR}, score: {score}')
#                 with open(base_filename+'_weight.pkl', 'wb') as f:
#                     pickle.dump(reg,f)


#     # log_path =  make_result_path(args)
#     # filename = args.time + '_' + wandb.run.id
#     # with open(os.path.join(log_path,filename+'_MPR.pkl'), 'wb') as f:
#     #     pickle.dump(MPR_dic, f)
#     # with open(os.path.join(log_path,filename+'_score.pkl'), 'wb') as f:
#     #     pickle.dump(score_dic, f)
    
#     # if args.retrieve:
#     #     with open(os.path.join(log_path,filename+'_norMPR.pkl'), 'wb') as f:
#     #         pickle.dump(norMPR_dic, f)
#     #     with open(os.path.join(log_path,filename+'_idx.pkl'), 'wb') as f:
#     #         pickle.dump(idx_dic, f)


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

    set_seed(args.seed)

    now = datetime.datetime.now()
    # Format as 'ddmmyyHMS'
    formatted_time = now.strftime('%H%M')
    if args.date == 'default':
        args.date = now.strftime('%m%d%y')
    args.time = formatted_time

    # run = wandb.init(
    #         project='mpr_generative',
    #         entity='sangwonjung-Harvard University',
    #         name=args.date+'_'+formatted_time,
    #         settings=wandb.Settings(start_method="fork")
    # )
    # print('wandb mode : ',run.settings.mode)
    
    # wandb.config.update(args)

    if args.train:
        train(args)
    else:
        eval(args)

    # wandb.finish()
