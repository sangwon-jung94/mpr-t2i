
import torch
import numpy as np 
import os
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import random 
import itertools
import pickle
from tqdm import tqdm

group_dic = {
    'gender' : ['female', 'male'],
    'age' : ['young', 'old'],
    'race' : ['East Asčian', 'Indian', 'Black', 'White', 'Middle Eastern', 'Latino_Hispanic', 'Southeast Asian'],
    'race2' : ['White', 'Black', 'Latino_Hispanic', 'Asian'],
    'face' : ['Bald', 'Bangs', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Double_Chin', 'Eyeglasses',
                'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'Mustache', 'No_Beard', 'Sideburns', 'Smiling', 'Wearing_Hat'],
    'wheelchair' : ['not wheelchair', 'wheelchair'],
    'emotion' : ['male_emotion', 'female_emotion'],
    'eyeglasses' : ['eyeglsses', 'no_eyeglasses'],
}

def make_dnf_feature(modelname, dataset, curation_set):
    # preprocessing features for the case of boolean series
    if 'boolean' in modelname:
        try:
            depth = int(modelname[7:])
        except:
            raise ValueError('Please specify the depth of the decision tree')
        
        dim = dataset.shape[1]
        for d in range(2,depth+1):
            for idxs in itertools.combinations(range(dim), d):
                dataset = np.concatenate((dataset, np.prod(dataset[:,idxs], axis=1).reshape(-1,1)), axis=1)
                curation_set = np.concatenate((curation_set, np.prod(curation_set[:,idxs], axis=1).reshape(-1,1)), axis=1)

    return dataset, curation_set

def getMPR(groups, dataset, k=0, curation_set=None, statistics=None, modelname=None, indices=None, normalize=False, onehot=False):
    if indices is None:
        indices = np.ones(dataset.shape[0])
    k = int(np.sum(indices))

    # normalizing the dataset, we don't need to normalize the curation set for decision tree
    if normalize and 'dt' not in modelname:
        dataset = dataset / np.linalg.norm(dataset, axis=-1, keepdims=True)
        curation_set = curation_set / np.linalg.norm(curation_set, axis=-1, keepdims=True)

    #linear function & boolean series
    if 'dt' not in modelname:
        reg = oracle_function(indices, dataset, curation_set=curation_set, modelname=modelname)
        
        if curation_set is not None:
            expanded_dataset = np.concatenate((dataset, curation_set), axis=0)
            m = curation_set.shape[0]
            if modelname != 'linear' and 'boolean' not in modelname:
                c = reg.predict(expanded_dataset)
            else:
                c = np.dot(expanded_dataset, reg)
            mpr = np.abs(np.sum((indices/k)*c[:dataset.shape[0]]) - np.sum((1/m)*c[dataset.shape[0]:]))

    # Decision tree
    elif 'dt' in modelname and onehot:
        group_list = []
        for group in groups:
            if group != 'face':
                group_list.extend(group_dic[group])
            elif group == 'face':
                tmp = []
                for attr in group_dic['face']:
                    tmp.append(attr)
                    tmp.append('not ' + attr)
                group_list.extend(tmp)

        group_list = np.array(group_list)
        try:
            depth = int(modelname[2:])
            print('depth:', depth)
        except:
            raise ValueError('Please specify the depth of the decision tree')
        dim = dataset.shape[1]
        if depth > dim:
            raise ValueError('The depth of the decision tree should be smaller than the dimension of the dataset')
        
        max_mpr = 0
        max_idxs = None
        subgroup_info = {
            'pos' : [],
            'neg' : []
        }
        for idxs in itertools.combinations(range(dim), depth):
            idxs = list(idxs)
            idxs.sort()
            idxs = np.array(idxs)
            # should we condition cases where male and female are selected at the same time?
            # dataset = np.concatenate((dataset, np.prod(dataset[:,idxs], axis=1).reshape(-1,1)), axis=1)
            p = _compute_intersectional_probabilities(dataset[:,idxs], depth)
            if statistics is not None:
                q =  _marginalize(statistics, group_list[idxs])
            else:
                q = _compute_intersectional_probabilities(curation_set[:,idxs], depth)
            mpr = 0.5 * np.sum(np.abs(p - q))
            # print(group_list[idxs], mpr, p, q)
            if max_mpr < mpr:
                max_mpr = mpr
                max_idxs = idxs
                binary_vectors = list(itertools.product([0, 1], repeat=depth))
                subgroup_info['pos'] = []
                subgroup_info['neg'] = []
                
                for i, (prob_p, prob_q) in enumerate(zip(p, q)):
                    subgroup_name = [group_list[idxs][k] if j == 1 else 'non ' + group_list[idxs][k] for k, j in enumerate(binary_vectors[i])]
                    for i, name in enumerate(subgroup_name):
                        if name == 'non not wheelchair':
                            subgroup_name[i] = 'wheelchair'
                    if prob_p > prob_q or prob_p == prob_q:
                        subgroup_info['pos'].append(subgroup_name)
                    elif prob_q > prob_p:
                        subgroup_info['neg'].append(subgroup_name)
                print(subgroup_info)
                print(p, q)
        mpr = max_mpr

        reg = {'subgroup_info' : subgroup_info, 'max_mpr' : max_mpr, 'max_idxs' : max_idxs}
    
    elif 'dt' in modelname and not onehot:
        oracle = RandomForestRegressor(max_depth=2)
            
    return mpr, reg

def oracle_function(indices, dataset, curation_set=None, modelname='linear'):
    if modelname == 'linear' or 'boolean' in modelname:
        # model = LinearRegression()
        k = int(np.sum(indices))
        dataset_mean = np.mean(dataset, axis=0)
        curation_mean = np.mean(curation_set, axis=0)
        diff = dataset_mean - curation_mean
        reg = diff / np.linalg.norm(diff)

        # expanded_dataset = np.concatenate((dataset, curation_set), axis=0)
        # curation_indicator = np.concatenate((np.zeros(dataset.shape[0]), np.ones(curation_set.shape[0])))
        # a_expanded = np.concatenate((indices, np.zeros(curation_set.shape[0])))
        # m = curation_set.shape[0]
        # alpha = (a_expanded/k - curation_indicator/m)
        # reg = np.dot(alpha, expanded_dataset)
        # reg = reg/np.linalg.norm(reg)
        return reg

    elif modelname == 'l2':
        return alpha
    else:
        raise ValueError('Only linear and dt are supported for now')

    # k = int(np.sum(indices))
    # if curation_set is not None:
    #     m = curation_set.shape[0]
    #     expanded_dataset = np.concatenate((dataset, curation_set), axis=0)
    #     curation_indicator = np.concatenate((np.zeros(dataset.shape[0]), np.ones(curation_set.shape[0])))
    #     a_expanded = np.concatenate((indices, np.zeros(curation_set.shape[0])))
    #     m = curation_set.shape[0]
    #     alpha = (a_expanded/k - curation_indicator/m)
    #     reg = model.fit(expanded_dataset, alpha) 
    # else:
    #     m = dataset.shape[0]
    #     alpha = (indices/k - 1/m)
    #     reg = model.fit(dataset, alpha)
    # return reg

def _compute_intersectional_probabilities(dataset, depth):
    # Combine dataset and curation_set
    probs = np.zeros(2**depth)
    # Count occurrences of each unique intersectional group
    unique, counts = np.unique(dataset, axis=0, return_counts=True)
    
    # Compute probabilities
    data_probs = counts / np.sum(counts)
    binary_vectors = list(itertools.product([0, 1], repeat=depth))
    for subgroup, prob in zip(unique, data_probs):
        subgroup = (subgroup+1)/2
        subgroup = subgroup.astype(int)
        idx =binary_vectors.index(tuple(subgroup))
        # idx = np.sum([2**i for i in subgroup if i == 1])
        probs[idx] = prob
    
    return probs

# def nominate_group_numer(dataset, depth):
#     dim = dataset.shape[1]
#     binary_vectors = list(itertools.product([0, 1], repeat=depth))
#     binary_vectors = [np.array(v) for v in binary_vectors]
#     for idxs in itertools.combinations(range(dim), depth): 
#         datasets
#     idxs = list(idxs)
#     idxs.sort()
#     idxs = np.array(idxs)
#     # should we condition cases where male and female are selected at the same time?
#     # dataset = np.concatenate((dataset, np.prod(dataset[:,idxs], axis=1).reshape(-1,1)), axis=1)
#     p = _compute_intersectional_probabilities(dataset[:,idxs], depth)


def _marginalize(statistics, groups):
    idxs_marginzalie = []
    for idx, group in enumerate(statistics['group']):
        if group not in groups:
            idxs_marginzalie.append(idx)
    # print(statistics['prob'], idxs_marginzalie)
    # print(statistics['prob'].sum(axis=idxs_marginzalie))
    # idxs_marginzalie = tuple(idxs_marginzalie)
    return statistics['prob'].sum(axis=idxs_marginzalie).flatten().numpy()
    
