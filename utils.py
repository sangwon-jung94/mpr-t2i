
import torch
import numpy as np 
import os
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import random 
import itertools
import pickle
from tqdm import tqdm
import clip
import sys
import logging
import collections
import networks
from torchvision import transforms
from copy import deepcopy
from torch import distributed as dist
from mpr.preprocessing import CLIPExtractor

def is_dist() -> bool:
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size() -> int:
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()

def get_rank() -> int:
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def merge(dict1, dict2):
    ''' Return a new dictionary by merging
        two dictionaries recursively.
    '''
    result = deepcopy(dict1)
    for key, value in dict2.items():
        if isinstance(value, collections.abc.Mapping):
            result[key] = merge(result.get(key, {}), value)
        else:
            result[key] = deepcopy(dict2[key])
    return result

def fill_config(config):
    #config = copy.deepcopy(config)
    base_cfg = config.pop('base', {})
    for sub, sub_cfg in config.items():
        if isinstance(sub_cfg, dict):
            config[sub] = merge(base_cfg, sub_cfg)
        elif isinstance(sub_cfg, list):
            config[sub] = [merge(base_cfg, c) for c in sub_cfg]
    return config


class IterLoader:

    def __init__(self, dataloader):
        self._dataloader = dataloader
        self.iter_loader = iter(self._dataloader)
        self._epoch = 0

    @property
    def epoch(self):
        return self._epoch

    def __next__(self):
        try:
            data = next(self.iter_loader)
        except StopIteration:
            self._epoch += 1
            if hasattr(self._dataloader.sampler, 'set_epoch'):
                self._dataloader.sampler.set_epoch(self._epoch)
            self.iter_loader = iter(self._dataloader)
            data = next(self.iter_loader)

        return data

    def __len__(self):
        return len(self._dataloader)

class LoggerBuffer():
    def __init__(self, name, path, headers, screen_intvl=1):
        self.logger = self.get_logger(name, path)
        self.history = []
        self.headers = headers
        self.screen_intvl = screen_intvl

    def get_logger(self, name, path):
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
    
        # set log level
        msg_fmt = '[%(levelname)s] %(asctime)s, %(message)s'
        time_fmt = '%Y-%m-%d_%H-%M-%S'
        formatter = logging.Formatter(msg_fmt, time_fmt)
    
        # define file handler and set formatter
        file_handler = logging.FileHandler(path, 'w')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(logging.INFO)
        logger.addHandler(stream_handler)

        # to avoid duplicated logging info in PyTorch >1.9
        if len(logger.root.handlers) == 0:
            stream_handler = logging.StreamHandler(sys.stdout)
            stream_handler.setFormatter(formatter)
            logger.root.addHandler(stream_handler)
        # to avoid duplicated logging info in PyTorch >1.8
        for handler in logger.root.handlers:
            handler.setLevel(logging.WARNING)


        return logger

    def clean(self):
        self.history = {}

    def update(self,  msg):
        # get the iteration
        n = msg.pop('Iter')
        self.history.append(msg)

        # header expansion
        novel_heads = [k for k in msg if k not in self.headers]
        if len(novel_heads) > 0:
            self.logger.warning(
                    'Items {} are not defined.'.format(novel_heads))

        # missing items
        missing_heads = [k for k in self.headers if k not in msg]
        if len(missing_heads) > 0:
            self.logger.warning(
                    'Items {} are missing.'.format(missing_heads))

        if self.screen_intvl != 1:
            doc_msg = ['Iter: {:5d}'.format(n)]
            for k, fmt in self.headers.items():
                v = self.history[-1][k]
                doc_msg.append(('{}: {'+fmt+'}').format(k, v))
            doc_msg = ', '.join(doc_msg)
            self.logger.debug(doc_msg)

        '''
        construct message to show on screen every `self.screen_intvl` iters
        '''
        if n % self.screen_intvl == 0:
            screen_msg = ['Iter: {:5d}'.format(n)]
            for k, fmt in self.headers.items():
                vals = [msg[k] for msg in self.history[-self.screen_intvl:]
                        if k in msg]
                v = sum(vals) / len(vals)
                screen_msg.append(('{}: {'+fmt+'}').format(k, v))
                    
            screen_msg = ', '.join(screen_msg)
            self.logger.info(screen_msg)

def get_statistics(concept, groups):
    if concept == 'disability':
        path = '/n/holylabs/LABS/calmon_lab/Lab/datasets/mpr_stuffs/statistics/'
        if groups == ['wheelchair', 'race2']:
            group_name = 'wheelchair_race'
        filename = f'{concept}_{group_name}.pkl'
        with open(os.path.join(path, filename), 'rb') as f:
            statistics = pickle.load(f)
    elif 'ENIAC' in concept:
        statistics = {}
        statistics['group'] = ['female','male']
        statistics['prob'] = torch.zeros((2,2))
        statistics['prob'][0,0] = 1
    return statistics


def make_result_path(args):
    # filename = f'{args.query_dataset}_{args.refer_dataset}_{args.vision_encoder}_{args.target_model}_{args.functionclass}'
    # if args.retriever != 'random_ratio':
    #     filename += f'_{args.retriever}_{args.k}' if args.retrieve else ''
    # else:
    #     filename += f'_{args.retriever}_{args.ratio}' if args.retrieve else ''
    save_dir = os.path.join(args.save_dir, args.date)
    check_log_dir(save_dir)
    return save_dir

def check_log_dir(log_dir):
    try:
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
            return 1
        else:
            return 0
    except OSError:
        print("Failed to create directory!!")

def set_seed(seed): 
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_current_device():
    if torch.cuda.is_available():
        # Determine the GPU used by the current process
        cur_device = torch.cuda.current_device()
    else:
        cur_device = torch.device('cpu')
    return cur_device


def compute_similarity(visual_features, target_concept, vision_encoder, vision_encoder_name='clip'):
    if vision_encoder_name != 'CLIP':
        raise ValueError('Only CLIP is supported for now')
    similarity = np.zeros(visual_features.shape[0])

    visual_features = visual_features / np.linalg.norm(visual_features, axis=-1, keepdims=True) 
    vision_encoder.eval()
    with torch.no_grad():        
        with torch.autocast("cuda"):
            prompt = f'photo portrait of {target_concept}'
            text = clip.tokenize(prompt)
            if torch.cuda.is_available():
                text = text.cuda()
            text_embedding = vision_encoder.encode_text(text).float()
            text_embedding = text_embedding.cpu().numpy().squeeze()
            text_embedding = text_embedding / np.linalg.norm(text_embedding)
            similarity = visual_features @ text_embedding
    return similarity

def make_embeddings(loader, vision_encoder=None, args=None, query=True):
    dataset_name = 'mscoco' if not query else args.query_dataset
    path = loader.dataset.dataset_path
    ver = 'normal'
    filename = f'fid_feature.pkl'
    filepath = os.path.join(path,filename)

    features = []
    if os.path.exists(filepath):
        with open(os.path.join(path,filename), 'rb') as f:
            feature =  pickle.load(f)
        if feature.shape[0] == len(loader.dataset):
            print(f'embedding vectors of {dataset_name} are successfully loaded in {path}')
            feature = torch.tensor(feature)
            # feature =  feature / feature.norm(dim=-1, keepdim=True)
            return feature.cpu()

    transform = transforms.Compose(
                    [
                    transforms.Resize((224,224)),
                    transforms.CenterCrop(224),
                    transforms.ToTensor()]
                    )   
    loader.dataset.transform = transform

    # with torch.no_grad():
    #     if vision_encoder is None:
    #         vision_encoder = networks.ModelFactory.get_model(modelname='CLIP', train=False)  
    #         vision_encoder.eval()
    #         vision_encoder = vision_encoder.cuda() if torch.cuda.is_available() else vision_encoder
    #     feature_extractor = CLIPExtractor(vision_encoder)
        
    total_samples = 0
    features = []
    for batch in tqdm(loader):
        image, label, idxs =  batch
        total_samples += len(idxs)
        # if torch.cuda.is_available():
        #     image = image.cuda()
        # feature = feature_extractor.extract(image)
        feature = image
        features.append(feature)
        if total_samples >= 1000:
            break
    features = torch.cat(features, dim=0)
    # features = features / features.norm(dim=-1, keepdim=True)
    features = features.cpu()
    with open(filepath, 'wb') as f:
        pickle.dump(features, f)
    return features

def compute_mmd(x, y):
    """A memory-efficient MMD implementation in PyTorch.

    This implements the minimum-variance/biased version of the estimator described
    in Eq.(5) of https://jmlr.csail.mit.edu/papers/volume13/gretton12a/gretton12a.pdf.
    As described in Lemma 6's proof in that paper, the unbiased estimate and the
    minimum-variance estimate for MMD are almost identical.

    Note that this function may consume significant memory if x has many rows.

    Args:
        x: The first set of embeddings of shape (n, embedding_dim).
        y: The second set of embeddings of shape (n, embedding_dim).

    Returns:
        The MMD distance between x and y embedding sets.
    """
    # The bandwidth parameter for the Gaussian RBF kernel.
    _SIGMA = 10
    # Used to make the metric more human-readable.
    _SCALE = 100

    x = torch.as_tensor(x, dtype=torch.float32)
    y = torch.as_tensor(y, dtype=torch.float32)

    # Compute squared norms of x and y
    x_sqnorms = torch.sum(x * x, dim=1)
    y_sqnorms = torch.sum(y * y, dim=1)
    gamma = 1 / (2 * _SIGMA**2)

    # Compute squared distances and kernel matrices
    D_xx = -2 * torch.matmul(x, x.T) + x_sqnorms.unsqueeze(1) + x_sqnorms.unsqueeze(0)
    print(torch.matmul(x, x.T).shape)
    K_xx = torch.exp(-gamma * D_xx)
    k_xx = torch.mean(K_xx)

    D_xy = -2 * torch.matmul(x, y.T) + x_sqnorms.unsqueeze(1) + y_sqnorms.unsqueeze(0)
    K_xy = torch.exp(-gamma * D_xy)
    k_xy = torch.mean(K_xy)

    D_yy = -2 * torch.matmul(y, y.T) + y_sqnorms.unsqueeze(1) + y_sqnorms.unsqueeze(0)
    K_yy = torch.exp(-gamma * D_yy)
    k_yy = torch.mean(K_yy)

    mmd =  _SCALE * (k_xx + k_yy - 2 * k_xy)
    # return torch.matmul(x, y.T).mean()
    return mmd.item()

def get_concept_embedding(vision_encoder, target_concept, vision_encoder_name='clip'):
    """
    function that embeds a text query. Alex wrote this to allow us to query the reference dataset in eval_contextual.py
    """
    if vision_encoder_name != 'CLIP':
        raise ValueError('Only CLIP is supported for now')
    with torch.no_grad():        
        prompt = f'photo portrait of {target_concept}'
        text = clip.tokenize(prompt)
        if torch.cuda.is_available():
            text = text.cuda()
        text_embedding = vision_encoder.encode_text(text).float()
        text_embedding = text_embedding.cpu().numpy().squeeze()
        text_embedding = text_embedding / np.linalg.norm(text_embedding)
    return text_embedding
# for PBM
def fon(l):
    try:
        return l[0]
    except:
        return None

def statEmbedding(embeddings):
    distances = []
    for i in range(len(embeddings)):
        for j in range(i+1, len(embeddings)):
            # print(embeddings[i], embeddings[j])
            distance = np.linalg.norm(embeddings[i] - embeddings[j])
            distances.append(distance)
    distances = np.array(distances)
    mean_embedding = np.mean(distances)
    std_embedding = np.std(distances)
    return mean_embedding, std_embedding

# def getMPR(indices, labels, oracle, k, m):


def gpu_mem_usage():
    """Computes the GPU memory usage for the current device (GB)."""
    if not torch.cuda.is_available():
        return 0
    # Number of bytes in a megabyte
    _B_IN_GB = 1024 * 1024 * 1024

    mem_usage_bytes = torch.cuda.max_memory_allocated()
    return mem_usage_bytes / _B_IN_GB


def print_intersectional_probs(dataset):
    # Combine dataset and curation_set
    probs = np.zeros((2,2,7))
    # Count occurrences of each unique intersectional group
    # unique, counts = np.unique(dataset, axis=0, return_counts=True)
    for i, gender in enumerate(group_dic['gender']):
        for j, age in enumerate(group_dic['age']):
            for k, race in enumerate(group_dic['race']):
                prob = np.mean((dataset[:,i] == 1) & (dataset[:,j+2] == 1) & (dataset[:,k+4] == 1))
                probs[i,j,k] = prob
    refer_prob = np.load('group_ratio.npy')
    np.save('fairdiffusion_ratio.npy', probs)   
    print(probs)
    print(refer_prob)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, weight=1.0, n_attr=2, fmt=':f'):
        self.name = name
        self.n_attr = n_attr
        self.fmt = fmt
        self.reset()
        self.weight = weight
        self.updated_step = -1

    def reset(self):
        self.val = torch.Tensor([0] * self.n_attr)
        self.avg = torch.Tensor([0] * self.n_attr)
        self.sum = torch.Tensor([0] * self.n_attr)
        self.count = 0
        self.cur_count = 0
        self.updated_step = -1

    def update(self, val):
        self.val += val
        self.sum += val
        self.count += val.sum()
        self.cur_count += val.sum()
        self.avg = self.sum / self.count

    def step(self):
        self.val = torch.Tensor([0] * self.n_attr)
        self.sum *= self.weight
        self.count *= self.weight
        self.cur_count = 0
        self.avg = self.sum / self.count

    def get_val(self):
        return self.val / (self.val.sum() + 1e-8)

    def get_discrepancy(self, target_ratio=None):
        if target_ratio is None:
            target_ratio = 1 / self.n_attr
        return max(self.get_val()-target_ratio) - min(self.get_val()-target_ratio)

    def get_mse(self, target_ratio=None):
        if target_ratio is None:
            target_ratio = 1 / self.n_attr
        return torch.mean((self.get_val() - target_ratio)**2)

    def __str__(self):
        val = ' '.join(['{:.3f}'.format(v) for v in self.val / self.val.sum()])
        avg = ' '.join(['{:.3f}'.format(a) for a in self.avg])
        log_str = "{} r: {} (# of current images: {}) (weighted r: {})"\
            .format(self.name, val, self.cur_count, avg)
        return log_str

    def copy(self, meter):
        self.val = meter.val
        self.avg = meter.avg
        self.sum = meter.sum
        self.count = meter.count
