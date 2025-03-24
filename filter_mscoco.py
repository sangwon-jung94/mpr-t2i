from data_handler.dataset_factory import GenericDataset
import torch
from PIL import Image
import json
import argparse
from face_detector import FaceDetector
from aesthetic_scoring import AestheticScorer
from mpr.preprocessing import group_estimation
import clip
import numpy as np
from collections import defaultdict
import json
import os
from tqdm import tqdm
from torchvision import transforms
import time
from mpr.preprocessing import identity_embedding
import pickle 
class MSCOCODataset(GenericDataset):
    dataset_path = './datasets/mscoco/'
    def __init__(self, data_dir, preprocess=None, bbox_dic=None):
        GenericDataset.__init__(self, args=None, split=None)

        self.data_dir = data_dir
        self.img_ids = self.data_dict()
        self.preprocess = preprocess
        # self.rgb_flag = False
        
        self.face_detect = False
        if bbox_dic is not None:
            self.bbox_dic = bbox_dic
            self.face_detect = True
            new_img_ids = []
            for key in self.bbox_dic.keys():
                if key not in self.img_ids:
                    raise ValueError(f"Image id {key} not found in MSCOCO dataset")
            self.img_ids = list(self.bbox_dic.keys())
        
        print("Total number of images: ", len(self.img_ids))
        # print("Total number of images: ", len(list(set(self.img_ids))))

    def data_dict(self):
        f = open(self.data_dir + 'annotations/captions_train2017.json')
        data = json.load(f)
        f.close()

        img_ids = []

        # img_id_set = {}
        for x in data['annotations']:
            img_id = x["image_id"]
            img_ids.append(img_id)
            # if img_id in img_id_set:
            #     img_id_set[img_id].append(x["caption"])
            # else:
            #     img_id_set[img_id] = [x["caption"]]
        img_ids = sorted(img_ids)
        img_ids = list(set(img_ids))
        return img_ids

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_path = self.data_dir + "train2017/" + str(self.img_ids[idx]).zfill(12) + ".jpg"
        image = Image.open(img_path)
        if self.face_detect:
            left, top, right, bottom = self.bbox_dic[self.img_ids[idx]]
            image = image.crop((left, top, right, bottom))

        if self.preprocess is not None:
            image = self.preprocess(image)
        
        return image, 0, self.img_ids[idx]            
        
        # is_face = self.facedetector.process_pil_image(image)
        # print(batch_counter, flush=True)
        # if is_face:
        #     img[batch_counter] = preprocess(image).to(args.device)
        #     ids[batch_counter] = img_ids[id_counter]
        #     batch_counter += 1
        # id_counter += 1
        # return self.preprocess(image), self.img_ids[idx], is_face
        

def data_dict(args):
    f = open(args.data_dir + 'annotations/captions_train2017.json')
    data = json.load(f)
    f.close()

    img_id_set = {}
    for x in data['annotations']:

        img_id = x["image_id"]

        if img_id in img_id_set:
            img_id_set[img_id].append(x["caption"])
        else:
            img_id_set[img_id] = [x["caption"]]

    print(len(img_id_set))
    return img_id_set

def embed_images(clipmodel, facedetector, aestheticscorer, aestheticthreshold, args):
    output = defaultdict(dict)

    # transform = transforms.Compose(
    #         [transforms.ToTensor()]
    # )
    def custom_collate(batch):
        images = [np.array(item[0]) for item in batch]  # List of NumPy images
        images_ori = [item[0] for item in batch]
        ids = torch.tensor([item[2] for item in batch])  # Tensor of labels
        return images, images_ori, ids
    dataset = MSCOCODataset(args.data_dir)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=False, collate_fn=custom_collate, num_workers=2)
    
    for data in tqdm(dataloader):
        with torch.no_grad():
            images, images_ori, ids = data
            start_time = time.time()
            flags, bboxs = facedetector.process_tensor_image(images,torch_tensor=False)
            end_time = time.time()
            processing_time = end_time - start_time
            print(f"Time to process {len(images)} images: {processing_time} seconds")
            bbox_pos = 0
            for i, (id, flag) in enumerate(zip(ids, flags)):
                id = id.item()
                output['flag'][id] = flag.item()
                if flag:
                    output['bbox'][id] = facedetector.extract_position(transforms.ToTensor()(images[i]), bboxs[bbox_pos])
                    # print(images_ori[i], output['bbox'][id])
                    bbox_pos += 1
            end_time = time.time()
    
    # with open(os.path.join(args.data_dif, "bbox_dic.pkl"), "wb") as f:
        # f.dump(bbox_dic, f)

    preprocess = aestheticscorer.get_preprocess()
    dataset.preprocess = preprocess
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=False, num_workers=2)
    
    print('start scoring')
    for data in tqdm(dataloader):
        images, _, ids = data
        images = images.to('cuda')
        start_time = time.time()
        scores = aestheticscorer(images).squeeze()
        end_time = time.time()
        processing_time = end_time - start_time
        print(f"Time to process 512 images: {processing_time} seconds")
        for id, score in zip(ids, scores):
            output['score'][id.item()] = score.item()
    
    with open(os.path.join(args.out_dir, "mscoco_info.pkl"), "wb") as outfile:
        pickle.dump(dict(output), outfile)

    _, clip_transform = clip.load("ViT-B/32", device= 'cpu')    
    dataset = MSCOCODataset(args.data_dir, preprocess=clip_transform, bbox_dic=output['bbox'])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=False, num_workers=2, drop_last=False)
    
    # for compatibility with identity_embedding
    args.mpr_onehot = True 
    args.vision_encoder = 'CLIP'
    args.query_dataset = 'mscoco'
    query_embedding, _ = identity_embedding(args, clipmodel, dataloader, ['gender','age','race'], query=True)
    genders = np.argmax(query_embedding[:,:2], axis=-1)
    ages = np.argmax(query_embedding[:,2:5], axis=-1)
    races = np.argmax(query_embedding[:,5:], axis=-1)
    for i, id in enumerate(dataset.img_ids):
        output['gender'][id] = genders[i]
        output['age'][id] = ages[i]
        output['race'][id] = races[i]
    
        # scores = aestheticscorer(image).squeeze()
        # # print(faces)
        # if not torch.any(torch.logical_and((scores > aestheticthreshold),(faces))):
        #     continue
        # score_indices = torch.where((scores > aestheticthreshold) & (faces))

        # newbatch = image[score_indices]
        # ids = ids[score_indices]

        # embeds = clipmodel.encode_image(newbatch).cpu().numpy()

        # gender_pred = group_estimation(embeds, 'gender')
        # race_pred = group_estimation(embeds, 'race')
        # age_pred = group_estimation(embeds, 'age')

        # # gender_pred = np.round((gender_pred+1)/2)
        # # race_pred = np.round((race_pred+1)/2)
        # # age_pred = np.round((age_pred+1)/2)
        # gender_pred = np.argmax((gender_pred+1)/2, axis=-1)
        # race_pred = np.argmax((race_pred+1)/2, axis=-1)
        # age_pred = np.argmax((age_pred+1)/2, axis=-1)

        # # keylist = ["negative", "positive"]

        # for i in range(len(score_indices)):
        #     if str(gender_pred[i]) not in output["gender"].keys():
        #         output["gender"][str(gender_pred[i])] = []
        #     if str(race_pred[i]) not in output["race"].keys():
        #         output["race"][str(race_pred[i])] = []
        #     if str(age_pred[i]) not in output["age"].keys():
        #         output["age"][str(age_pred[i])] = []
        #     output["gender"][str(gender_pred[i])].append(ids[i].item())
        #     output["race"][str(race_pred[i])].append(ids[i].item())
        #     output["age"][str(age_pred[i])].append(ids[i].item())
            # print(output)
            # exit()
    with open(os.path.join(args.out_dir, "mscoco_info.pkl"), "wb") as outfile:
        pickle.dump(dict(output), outfile)

    print(output)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_dir', type=str, default="./datasets/mscoco/")
    parser.add_argument('-out_dir', type=str, default="./datasets/mscoco/")
    parser.add_argument('-device', type=str, default='cuda', help="cpu, cuda, or mps")
    parser.add_argument('--batch_size', type=int, default=1024)
    args = parser.parse_args()

    clipmodel, _ = clip.load("ViT-B/32", device=args.device)
    facedetector = FaceDetector()
    aestheticscorer = AestheticScorer()
    aestheticscorer.to(args.device)
    
    with torch.no_grad():
        embed_images(clipmodel, facedetector, aestheticscorer, 1, args)

if __name__ == "__main__":
    main()
