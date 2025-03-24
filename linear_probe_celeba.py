
import torch
import numpy as np
from argument import get_args
import pickle

import data_handler
import networks
from utils import set_seed
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import os

def main():

    train_loader, test_loader = data_handler.DataloaderFactory.get_dataloader(dataname='celeba', args=args)
    
    # Get the required model
    if args.vision_encoder != 'CLIP':
        raise ValueError("Only CLIP is supported for this task.")

    vision_encoder = networks.ModelFactory.get_model(modelname='CLIP')
    vision_encoder = vision_encoder.cuda() if torch.cuda.is_available() else vision_encoder
    print('cuda : ', torch.cuda.is_available())
    embeds = []
    labels = []
    
    # to check whether the saved embeddings exist
    if not os.path.exists('./stuffs/celeba_embedding_dic.pkl'):
        with torch.no_grad():
            for image, label in tqdm(train_loader):
                if torch.cuda.is_available():
                    image = image.cuda()
                    label = label.cuda()
                embed = vision_encoder.encode_image(image)
                embeds.append(embed)
                labels.append(label)
        with open('./stuffs/celeba_embedding_dic.pkl', 'wb') as f:
            _dic = {'embed' : embeds,
                    'label' : labels}
            pickle.dump(_dic, f)
    
    else:
        with open('./stuffs/celeba_embedding_dic.pkl', 'rb') as f:
            _dic = pickle.load(f)
            embeds = _dic['embed']
            labels = _dic['label']
        
    embeds = torch.cat(embeds).cpu().numpy()
    labels = torch.cat(labels).cpu().numpy()
 
    parameters = {'C':[0.01, 0.1, 1, 10, 100]}

    _dic = {'embed' : embeds,
            'label' : labels}
    
    with open('celeba_dic.pkl', 'wb') as f:
        pickle.dump(_dic, f)

    print(embeds.shape)
    print(labels.shape)

    test_embeds, test_labels = [], []
    with torch.no_grad():
        for image, label in tqdm(test_loader):
            if torch.cuda.is_available():
                image = image.cuda()
                label = label.cuda()
            embed = vision_encoder.encode_image(image)
            test_embeds.append(embed)
            test_labels.append(label)
            
    test_embeds = torch.cat(test_embeds).cpu().numpy()
    test_labels = torch.cat(test_labels).cpu().numpy()

    clf_dic = {}
    for i in range(labels.shape[1]):
        # label = 
        attr_name = train_loader.dataset.attr_names[i]
        print('attr_name : ', attr_name)
        _label = labels[:,i]
        # check balanceness of the label
        print(f"Train {attr_name} 1/0: {np.sum(_label)}/{len(_label)-np.sum(_label)}")
        # extract subset to balance the label
        _label_1 = np.where(_label == 1)[0]
        _label_0 = np.where(_label == 0)[0]
        if len(_label_1) > len(_label_0):
            _label_1 = np.random.choice(_label_1, len(_label_0), replace=False)
        else:
            _label_0 = np.random.choice(_label_0, len(_label_1), replace=False)
        _label = np.concatenate((_label[_label_1], _label[_label_0]))
        _embed = np.concatenate((embeds[_label_1], embeds[_label_0]))

        lr_attr = LogisticRegression(penalty="l2", C=1)
        clf_attr = GridSearchCV(lr_attr, parameters)
        clf_attr.fit(_embed, _label)
        clf_dic[attr_name] = clf_attr

        print(f"{attr_name} Train/Test:", clf_attr.score(embeds, labels[:,i]), clf_attr.score(test_embeds, test_labels[:,i]))
        # print(f"{attr_name} Test:", clf_attr.score(test_embeds, test_labels[:,i]))

    # save clf_age, clf_gender and clf_race
    with open('./stuffs/clf_celeba.pkl', 'wb') as f:
        pickle.dump(clf_dic, f)

    
    # print("Train gender:", clf_gender.score(fairface_embeds, fairface_gender_labels))
    # print("Train race:", clf_race.score(fairface_embeds, fairface_race_labels))

if __name__ == '__main__':

    args = get_args()    

    np.set_printoptions(precision=4)
    torch.set_printoptions(precision=4)

    set_seed(args.seed)

    main()