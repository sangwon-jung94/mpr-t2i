

import torch
import numpy as np
import datetime
from argument import get_args
import pickle
import wandb

import sys
import os
# os.environ["WANDB_MODE"]="offline"

import data_handler
import networks
from utils import set_seed, make_result_path,  compute_similarity, get_statistics
from mpr.mpr import getMPR
from mpr.preprocessing import identity_embedding

def main(args):
    if args.refer_dataset == 'statistics':
        statistics = get_statistics(args.target_concept, args.mpr_group)
        
    elif args.refer_dataset == 'uniform':
        refer_embedding = np.zeros((2,2))
        refer_embedding[0,0] = 1
        refer_embedding[1,1] = 1
        refer_embedding[0,1] = -1
        refer_embedding[1,0] = -1
        
    else:
        refer_loader = data_handler.DataloaderFactory.get_dataloader(args=args, dataname=args.refer_dataset)
        
    query_loader = data_handler.DataloaderFactory.get_dataloader(args=args)

    ## Compute MPR
    vision_encoder = networks.ModelFactory.get_model(modelname=args.vision_encoder, train=args.train)
    vision_encoder = vision_encoder.cuda() if torch.cuda.is_available() else vision_encoder

    ## Make embedding vectors
    print('extract embeddings from the reference distribution')
    if args.refer_dataset not in ['statistics','uniform']:
        refer_embedding, _ = identity_embedding(args, vision_encoder, refer_loader,args.mpr_group, query=False)
    print('extract embeddings from the query distribution')
    query_embedding, feature_dic = identity_embedding(args, vision_encoder, query_loader, args.mpr_group, query=True)

    # Save labels
    if args.save_labels:
        group_string = '_'.join(args.mpr_group)
        with open(args.dataset_path+'/group_labels_'+group_string+'.pkl', 'wb') as f:
            pickle.dump(query_embedding*0.5+0.5, f)
        with open(refer_loader.dataset.dataset_path+'/group_labels_'+group_string+'.pkl', 'wb') as f:
            pickle.dump(refer_embedding*0.5+0.5, f)
            print(refer_loader.dataset.dataset_path+'/group_labels_'+group_string+'.pkl')
        print('Labels are saved ')
        exit()
    
    
    MPR_dic = []
    score_dic = []
    idx_dic = []

    if args.bootstrapping and args.n_resampling == 1:
        raise ValueError('the number of resampling should be larger 1 for bootstrapping')
    
    if args.pool_size != 1.0:
        if args.pool_size < 1:
            n_samples = int(query_embedding.shape[0]*args.pool_size)
        else:
            n_samples = int(args.pool_size)

        if not args.bootstrapping and args.n_resampling > 1:
            assert n_samples > args.n_resampling * args.resampling_size, 'the number of resampling should be smaller than the pool size'

        idx = np.arange(n_samples)
        print('Pool size is reduced to ', args.pool_size)

        query_embedding = query_embedding[idx]
    else:
        idx = np.arange(query_embedding.shape[0])
    s = compute_similarity(feature_dic['normal'], args.target_concept, vision_encoder, args.vision_encoder)        

    for j in range(args.n_resampling):
        print(f'Computation of MPR {j+1}/{args.n_resampling}...')
        refer_embedding_split = refer_embedding if args.refer_dataset != 'statistics' else None

        # sample with replacement in the pool
        if args.bootstrapping:
            n_samples = query_embedding.shape[0] 
            resampling_idx = np.random.choice(n_samples, args.resampling_size, replace=True)
            query_embedding_split = query_embedding[resampling_idx]
            s_split = s[resampling_idx]

            if args.refer_size != 1:
                if args.refer_dataset == 'statistics':
                    raise ValueError('statistics cannot be resampled')
                total_samples = refer_embedding.shape[0]
                n_samples = int(total_samples*args.refer_size) if args.refer_size < 1 else int(args.refer_size)
                split_idx = np.random.choice(n_samples, n_samples, replace=True)
                refer_embedding_split = refer_embedding[split_idx]
                print('Refer size is reduced to ', n_samples)

        # sample without replacement
        elif args.n_resampling > 1:
            split_idx = np.arange(j*args.resampling_size, (j+1)*args.resampling_size)
            query_embedding_split = query_embedding[split_idx]
            s_split = s[split_idx]

            if args.refer_size != 1:
                if args.refer_dataset == 'statistics':
                    raise ValueError('statistics cannot be resampled')
                split_idx = np.arange(j*args.refer_size, (j+1)*args.refer_size)
                split_idx = split_idx.astype(int)
                if max(split_idx) > refer_embedding.shape[0]:
                    break
                refer_embedding_split = refer_embedding[split_idx]
                print('Refer size is reduced to ', args.refer_size)
                
        else:
            query_embedding_split = query_embedding
            s_split = s

        score = np.sum(s_split)

        if args.refer_dataset == 'statistics':
            MPR, c = getMPR(args.mpr_group, query_embedding_split, curation_set=None, statistics=statistics, modelname=args.functionclass, normalize=args.normalize, onehot=args.mpr_onehot)
        else:
            MPR, c = getMPR(args.mpr_group, query_embedding_split, curation_set=refer_embedding_split, modelname=args.functionclass, normalize=args.normalize, onehot=args.mpr_onehot)
        print(f'Concept: {args.target_concept}, final MPR: {MPR}, score: {score/len(query_embedding_split)}')#, fid: {fid_score}')
        
        MPR_dic.append(MPR)
        score_dic.append(score)

    log_path =  make_result_path(args)

    if args.bootstrapping:
        mean = np.mean(MPR_dic)
        std = np.std(MPR_dic)
        print(f'Mean MPR: {mean}, std: {std}')

    if not args.no_wandb:
        filename = args.time + '_' + wandb.run.id
        results = {}
        results['MPR'] = MPR_dic
        results['score'] = score_dic
        results['optimal_c'] = c
        # results['fid'] = fid_dic

        if args.retrieve: 
            results['idx'] = idx_dic
        elif args.bootstrapping:
            results['idx'] = idx[resampling_idx]
        else:
            results['idx'] = idx
            
        with open(os.path.join(log_path,filename+'.pkl'), 'wb') as f:
            pickle.dump(results, f)
        print('Results are saved at ', os.path.join(log_path,filename+'.pkl'))


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

    if not args.no_wandb:
        run = wandb.init(
                project='mpr_generative',
                entity='sangwonjung-Harvard University',
                name=args.date+'_'+formatted_time,
                settings=wandb.Settings(start_method="fork")
        )
        print('wandb mode : ',run.settings.mode)
        
        wandb.config.update(args)

    main(args)

    if not args.no_wandb:    
        wandb.finish()
    