import torch
import numpy as np
import datetime
from argument import get_args
import pickle
import wandb
from tqdm import tqdm
from mpr.preprocessing import CLIPExtractor
import faiss

import sys
import os
os.environ["WANDB_MODE"]="offline"

import data_handler
import networks
import retriever
from utils import set_seed, make_result_path,  compute_similarity, get_concept_embedding
from mpr.mpr import getMPR
from mpr.preprocessing import identity_embedding

def convert_loader(loader, encoder, args):
    query_embedding = get_concept_embedding(encoder, args.target_concept, args.vision_encoder)
    ## note: no saving implemented just because of memory concerns
    # filename = f'{args.vision_encoder}_{ver}_feature.pkl'
    # filepath = os.path.join(path,filename)
    # print(filepath)

    # save_flag = False
    # if os.path.exists(filepath):
    #     with open(os.path.join(path,filename), 'rb') as f:
    #         feature_dic[ver] =  pickle.load(f)
    #     print(f'embedding vectors of {dataset_name} are successfully loaded in {path}')
    #     continue
    # else:
    #     save_flag = True

    ## go through in batches, run FAISS to get the top-k similar images 
    encoder.eval()
    encoder = encoder.cuda() if torch.cuda.is_available() else encoder

    feature_extractor = None

    if args.vision_encoder == 'BLIP':
        raise ValueError('Only CLIP is supported on the query text embedding side so we cannot use BLIP for query-contextual MPR measurements')
    
    elif args.vision_encoder == 'CLIP':
        feature_extractor =  CLIPExtractor(encoder, args)

    res = faiss.StandardGpuResources()  # use a single GPU
    # build a flat (CPU) index
    index_flat = faiss.IndexFlatL2(query_embedding.shape[0])
    # make it into a gpu index
    gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
    
    features = torch.zeros((args.query_contextual_dataset_size*2, query_embedding.shape[0]))
    idx = 0
    for batch in tqdm(loader, desc="searching openface for top-k similar images to measure MPR over"):
        if args.refer_dataset == "openface":
            image = batch
        else:  
            raise ValueError('Only openface is supported for query-contextual reference set for now')

        if torch.cuda.is_available():
            image = image.cuda()

        feature = feature_extractor.extract(image)
        features[idx:idx+feature.shape[0]] = feature
        # if features is None:
        #     features = feature
        # else:
        #     features = torch.cat([features, feature], dim=0)
        if idx >= args.query_contextual_dataset_size*2: #every time our temporary dataset is 10 times our target size, run the similarity search and repeat
            gpu_index_flat.add(features)

            _, I = gpu_index_flat.search(query_embedding, args.query_contextual_dataset_size)
            features.cpu() # just in case old features stay on GPU memory, clear them
            features = torch.zeros((args.query_contextual_dataset_size*2, query_embedding.shape[0]))
            features[:feature.shape[0]] = I[0]
            print(features.shape)
            print(I[0].shape)
            idx = 1
        else:
            idx += image.shape[0]
    
    features = features.detach().cpu()
    return torch.data.utils.DataLoader(torch.data.utils.TensorDataset(features))
        
def main(args):
    print(args.query_dataset)
    refer_loader = data_handler.DataloaderFactory.get_dataloader(dataname=args.refer_dataset, args=args)
    query_loader = data_handler.DataloaderFactory.get_dataloader(dataname=args.query_dataset, args=args)

    ## Compute MPR
    vision_encoder = networks.ModelFactory.get_model(modelname=args.vision_encoder, train=args.train)  
    vision_encoder = vision_encoder.cuda() if torch.cuda.is_available() else vision_encoder

    refer_loader_contextual = convert_loader(refer_loader, vision_encoder, args)
    for i in refer_loader_contextual:
        print(i)
        exit()
    
    ## Make embedding vectors and search for top-k similar images in reference dataset
    print('extract embeddings from the reference distribution')
    refer_embedding, _ = identity_embedding(args, vision_encoder, refer_loader,args.mpr_group, query=False) ## TODO
    print('extract embeddings from the query distribution')
    query_embedding, feature_dic = identity_embedding(args, vision_encoder, query_loader, args.mpr_group, query=True)

    print('Complete estimating group labels')
    
    if args.retrieve:
        _retriever = retriever.RetrieverFactory.get_retriever(args.retriever, args)
    
    MPR_dic = []
    norMPR_dic =[]
    score_dic = []
    idx_dic = []

    if args.bootstrapping and args.n_resampling == 1:
        raise ValueError('the number of resampling should be larger 1 for bootstrapping')
    
    if args.pool_size != 1.0:
        total = query_embedding.shape[0]
        if args.pool_size < 1:
            n_samples = int(query_embedding.shape[0]*args.pool_size)
        else:
            n_samples = int(args.pool_size)

        if not args.bootstrapping and args.n_resampling > 1:
            assert n_samples < args.n_resampling * args.resampling_size, 'the number of resampling should be smaller than the pool size'

        # idx = np.random.choice(total, n_samples, replace=False)
        idx = np.arange(n_samples)
        print('Pool size is reduced to ', args.pool_size)

        query_embedding = query_embedding[idx]
        query_embedding = query_embedding[idx]
    else:
        idx = np.arange(query_embedding.shape[0])

    s = compute_similarity(feature_dic['normal'], args.target_concept, vision_encoder, args.vision_encoder)        
    
    for j in range(args.n_resampling):
        refer_embedding_split = refer_embedding
        if args.bootstrapping:
            n_samples = query_embedding.shape[0] 
            resampling_idx = np.random.choice(n_samples, n_samples, replace=True)
            query_embedding_split = query_embedding[resampling_idx]
            
            s_split = s[resampling_idx]
        elif args.n_resampling > 1:
            split_idx = np.arange(j*args.resampling_size, (j+1)*args.resampling_size)
            query_embedding_split = query_embedding[split_idx]
            # query_embedding_split = query_embedding
            s_split = s[split_idx]

            if args.refer_size != 1:
                n_samples = int(refer_embedding.shape[0]*args.refer_size) if args.refer_size < 1 else int(args.refer_size)
                split_idx = np.arange(j*n_samples, (j+1)*n_samples)
                refer_embedding_split = refer_embedding[split_idx]
                print('Refer size is reduced to ', args.refer_size)

        else:
            query_embedding_split = query_embedding
            s_split = s

        # compute CLIP similarity

        # compute MPR

        if args.retrieve:
            retrieved_idx, _, MPR, score = _retriever.retrieve(query_embedding_split, refer_embedding_split, k=args.k, s=s)
            # score = np.sum(s[concept_idx][idx.astype(dtype=bool)])
            idx_dic.append(idx[retrieved_idx])
            # norMPR_dic.append(noretrieve_MPR_list)
            print(f'Concept: {args.target_concept}, MPR: {noretrieve_MPR_list[-1]}, score: {score_list[-1]}')
            
        else:
            score = np.sum(s_split)
            MPR, c = getMPR(args, query_embedding_split, curation_set=refer_embedding_split, modelname=args.functionclass)
            print(f'Concept: {args.target_concept}, final MPR: {MPR}, score: {score}')
        MPR_dic.append(MPR)
        score_dic.append(score)

    log_path =  make_result_path(args)
    filename = args.time + '_' + wandb.run.id
    results = {}
    results['MPR'] = MPR_dic
    results['score'] = score_dic
    results['optimal_c'] = c

    if args.retrieve: 
        results['idx'] = idx_dic
    elif args.bootstrapping:
        results['idx'] = idx[resampling_idx]
    else:
        results['idx'] = idx

    if args.bootstrapping:
        mean = np.mean(MPR_dic)
        std = np.std(MPR_dic)
        print(f'Mean MPR: {mean}, std: {std}')
        
    with open(os.path.join(log_path,filename+'.pkl'), 'wb') as f:
        pickle.dump(results, f)
    # with open(os.path.join(log_path,filename+'_MPR.pkl'), 'wb') as f:
    #     pickle.dump(MPR_dic, f)
    # with open(os.path.join(log_path,filename+'_score.pkl'), 'wb') as f:
    #     pickle.dump(score_dic, f)
    # with open(os.path.join(log_path,filename+'_idx.pkl'), 'wb') as f:
    #     pickle.dump(idx_dic, f)

    ## Compute FID

    ## IS


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

    run = wandb.init(
            project='mpr_generative',
            entity='sangwonjung-Harvard University',
            name=args.date+'_'+formatted_time,
            settings=wandb.Settings(start_method="fork")
    )
    print('wandb mode : ',run.settings.mode)
    
    wandb.config.update(args)

    main(args)
    
    wandb.finish()
