
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

def main():
    
    train_loader, test_loader = data_handler.DataloaderFactory.get_dataloader(dataname='fairface', args=args)
    args.eval = True
    
    # Get the required model
    if args.vision_encoder != 'CLIP':
        raise ValueError("Only CLIP is supported for this task.")

    vision_encoder = networks.ModelFactory.get_model(modelname='CLIP')
    vision_encoder = vision_encoder.cuda() if torch.cuda.is_available() else vision_encoder
    
    # train
    fairface_embeds = []
    fairface_gender_labels = []
    fairface_age_labels = []
    fairface_race_labels = []
    with torch.no_grad():
        for image, labels, idxs in tqdm(train_loader):
            if torch.cuda.is_available():
                image = image.cuda()
                labels = labels.cuda()
            embed = vision_encoder.encode_image(image)
            fairface_embeds.append(embed)
            fairface_gender_labels.append(labels[:, 0])
            fairface_age_labels.append(labels[:, 1])
            fairface_race_labels.append(torch.argmax(labels[:, 2:], dim=-1))
        
    fairface_embeds = torch.cat(fairface_embeds).cpu().numpy()
    fairface_gender_labels = torch.cat(fairface_gender_labels).cpu().numpy()
    fairface_age_labels = torch.cat(fairface_age_labels).cpu().numpy()
    fairface_race_labels = torch.cat(fairface_race_labels).cpu().numpy()

    # Test loader processing
    test_fairface_embeds = []
    test_fairface_gender_labels = []
    test_fairface_age_labels = []
    test_fairface_race_labels = []

    with torch.no_grad():
        for image, labels, idxs in tqdm(test_loader):
            if torch.cuda.is_available():
                image = image.cuda()
                labels = labels.cuda()
            embed = vision_encoder.encode_image(image)
            test_fairface_embeds.append(embed)
            test_fairface_gender_labels.append(labels[:, 0])
            test_fairface_age_labels.append(labels[:, 1])
            test_fairface_race_labels.append(torch.argmax(labels[:, 2:], dim=-1))

    test_fairface_embeds = torch.cat(test_fairface_embeds).cpu().numpy()
    test_fairface_gender_labels = torch.cat(test_fairface_gender_labels).cpu().numpy()
    test_fairface_age_labels = torch.cat(test_fairface_age_labels).cpu().numpy()
    test_fairface_race_labels = torch.cat(test_fairface_race_labels).cpu().numpy()

    parameters = {'C':[0.01, 0.1, 1, 10, 100]}

    fairface_dic = {'embed' : fairface_embeds,
                  'gender' : fairface_gender_labels,
                  'age' : fairface_age_labels,
                  'race' : fairface_race_labels}
    
    # with open('fairface_embeddings.pkl', 'wb') as f:
        # pickle.dump(fairface_dic, f)

    print(fairface_embeds.shape)
    print(fairface_age_labels.shape)
    print(fairface_gender_labels.shape)
    print(fairface_race_labels.shape)

    if not args.eval:
        lr_age = LogisticRegression(penalty="l2", C=1)
        lr_gender = LogisticRegression(penalty="l2", C=1)
        lr_race = LogisticRegression(C=1, multi_class="multinomial", solver="saga")

        clf_age = GridSearchCV(lr_age, parameters)
        clf_gender = GridSearchCV(lr_gender, parameters)
        clf_race = GridSearchCV(lr_race, parameters)

        # clf_age.fit(fairface_embeds, fairface_age_labels)
        # clf_gender.fit(fairface_embeds, fairface_gender_labels)
        clf_race.fit(fairface_embeds, fairface_race_labels)

        # save clf_age, clf_gender and clf_race
        # with open('/n/holylabs/LABS/calmon_lab/Lab/datasets/mpr_stuffs/clf_age.pkl', 'wb') as f:
        #     pickle.dump(clf_age, f)
        # with open('/n/holylabs/LABS/calmon_lab/Lab/datasets/mpr_stuffs/clf_gender.pkl', 'wb') as f:
        #     pickle.dump(clf_gender, f)
        if args.race_reduce:
            with open('/n/holylabs/LABS/calmon_lab/Lab/datasets/mpr_stuffs/clf_race2.pkl', 'wb') as f:
                pickle.dump(clf_race, f)
        else:
            with open('/n/holylabs/LABS/calmon_lab/Lab/datasets/mpr_stuffs/clf_race.pkl', 'wb') as f:
                pickle.dump(clf_race, f)
    else:
        with open(f'/n/holylabs/LABS/calmon_lab/Lab/datasets/mpr_stuffs/clfs/fairface_{args.vision_encoder}_clf_age.pkl', 'rb') as f:
            clf_age = pickle.load(f)
        with open(f'/n/holylabs/LABS/calmon_lab/Lab/datasets/mpr_stuffs/clfs/fairface_{args.vision_encoder}_clf_gender.pkl', 'rb') as f:
            clf_gender = pickle.load(f)
        with open(f'/n/holylabs/LABS/calmon_lab/Lab/datasets/mpr_stuffs/clfs/fairface_{args.vision_encoder}_clf_race.pkl', 'rb') as f:
            clf_race = pickle.load(f)

    print("Train age:", clf_age.score(fairface_embeds, fairface_age_labels))
    print("Train gender:", clf_gender.score(fairface_embeds, fairface_gender_labels))
    print("Train race:", clf_race.score(fairface_embeds, fairface_race_labels))

    print("Test age:", clf_age.score(test_fairface_embeds, test_fairface_age_labels))
    print("Test gender:", clf_gender.score(test_fairface_embeds, test_fairface_gender_labels))
    print("Test race:", clf_race.score(test_fairface_embeds, test_fairface_race_labels))

if __name__ == '__main__':

    args = get_args()    

    np.set_printoptions(precision=4)
    torch.set_printoptions(precision=4)

    set_seed(args.seed)

    main()
