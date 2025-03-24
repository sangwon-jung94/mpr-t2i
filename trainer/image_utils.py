import os
import torch
from torch import nn
import math
from PIL import Image, ImageOps, ImageDraw, ImageFont
import pickle as pkl
import torchvision
from torchvision import transforms
from skimage import transform
from sentence_transformers import SentenceTransformer, util
import kornia
import numpy as np
import face_recognition
import itertools
class FaceFeatsModel(torch.nn.Module):
    def __init__(self, face_feats_path):
        super().__init__()
        
        with open(face_feats_path, "rb") as f:
            face_feats, face_genders, face_logits = pkl.load(f)
        
        face_feats = torch.nn.functional.normalize(face_feats, dim=-1)
        self.face_feats = nn.Parameter(face_feats)   
        self.face_feats.requires_grad_(False)               
        
    def forward(self, x):
        """no forward function
        """
        return None
        
    @torch.no_grad()
    def semantic_search(self, query_embeddings, selector=None, return_similarity=False):
        """search the closest face embedding from vector database.
        """
        target_embeddings = torch.ones_like(query_embeddings) * (-1)
        if return_similarity:
            similarities = torch.ones([query_embeddings.shape[0]], device=query_embeddings.device, dtype=query_embeddings.dtype) * (-1)
            
        if selector.sum()>0:
            hits = util.semantic_search(query_embeddings[selector], self.face_feats, score_function=util.dot_score, top_k=1)
            target_embeddings_ = torch.cat([self.face_feats[hit[0]["corpus_id"]].unsqueeze(dim=0) for hit in hits])
            target_embeddings[selector] = target_embeddings_
            if return_similarity:
                similarities_ = torch.tensor([hit[0]["score"] for hit in hits], device=query_embeddings.device, dtype=query_embeddings.dtype)
                similarities[selector] = similarities_

        if return_similarity:
            return target_embeddings.data.detach().clone(), similarities
        else:
            return target_embeddings.data.detach().clone()

def get_face_feats(net, data, flip=True, normalize=True, to_high_precision=True):
    # extract features from the original 
    # and horizontally flipped data
    feats = net(data)
    if flip:
        data = torch.flip(data, [3])
        feats += net(data)
    if to_high_precision:
        feats = feats.to(torch.float)
    if normalize:
        feats = torch.nn.functional.normalize(feats, dim=-1)
    return feats

def get_face(images, args, fill_value=-1, eval=False):
    """
    images:shape [N,3,H,W], in range [-1,1], pytorch tensor
    returns:
        face_indicators: torch tensor of shape [N], only True or False
            True means face is detected, False otherwise
        face_bboxs: torch tensor of shape [N,4], 
            if face_indicator is False, the corresponding face_bbox will be [fill_value,fill_value,fill_value,fill_value]
        face_chips: torch tensor of shape [N,3,224,224]
            if face_indicator is False, the corresponding face_chip will be all fill_value
    """
    # face_indicators_app, face_bboxs_app, face_chips_app, face_landmarks_app, aligned_face_chips_app = get_face_app(face_app, images, args, fill_value=fill_value)


    # if face_indicators_app.logical_not().sum() > 0:
    face_indicators_FR, face_bboxs_FR, face_chips_FR, face_landmarks_FR, aligned_face_chips_FR = get_face_FR(images, args, fill_value=fill_value, eval=eval)

    # face_bboxs_app[face_indicators_app.logical_not()] = face_bboxs_FR
    # face_chips_app[face_indicators_app.logical_not()] = face_chips_FR
    # face_landmarks_app[face_indicators_app.logical_not()] = face_landmarks_FR
    # aligned_face_chips_app[face_indicators_app.logical_not()] = aligned_face_chips_FR

    # face_indicators_app[face_indicators_app.logical_not()] = face_indicators_FR

    return face_indicators_FR, face_bboxs_FR, face_chips_FR, face_landmarks_FR, aligned_face_chips_FR

# def get_face_app(images, args, fill_value=-1):
#     """
#     images:shape [N,3,H,W], in range [-1,1], pytorch tensor
#     returns:
#         face_indicators: torch tensor of shape [N], only True or False
#             True means face is detected, False otherwise
#         face_bboxs: torch tensor of shape [N,4], 
#             if face_indicator is False, the corresponding face_bbox will be [fill_value,fill_value,fill_value,fill_value]
#         face_chips: torch tensor of shape [N,3,224,224]
#             if face_indicator is False, the corresponding face_chip will be all fill_value
#     """        
#     images_np = ((images*0.5 + 0.5)*255).cpu().detach().permute(0,2,3,1).float().numpy().astype(np.uint8)
    
#     face_indicators_app = []
#     face_bboxs_app = []
#     face_chips_app = []
#     face_landmarks_app = []
#     aligned_face_chips_app = []
#     for idx, image_np in enumerate(images_np):
#         # face_app.get input should be [BGR]
#         # faces_from_app = face_app.get(image_np[:,:,[2,1,0]]) # insight face implementation
#         faces_from_app = face_recognition.face_locations(image_np, model="cnn") # face_recognition implementation
#         face_landmarks_from_app = face_recognition.face_landmarks(image_np)

#         if len(faces_from_app) == 0:
#             face_indicators_app.append(False)
#             face_bboxs_app.append([fill_value]*4)
#             face_chips_app.append(torch.ones([1,3,args.size_face,args.size_face], dtype=images.dtype, device=images.device)*(fill_value))
#             face_landmarks_app.append(torch.ones([1,5,2], dtype=images.dtype, device=images.device)*(fill_value))
#             aligned_face_chips_app.append(torch.ones([1,3,args.size_aligned_face,args.size_aligned_face], dtype=images.dtype, device=images.device)*(fill_value))
#         else:
#             max_idx = get_largest_face_app_idx(faces_from_app, dim_max=image_np.shape[0], dim_min=0) ## get index of largest face for face_recognition implementation
#             bbox = expand_bbox(faces_from_app[max_idx], expand_coef=0.5, target_ratio=1) ##use to get face bbox
#             face_chip = crop_face(images[idx], bbox, target_size=[args.size_face,args.size_face], fill_value=fill_value)
            
#             face_landmarks = np.array(face_landmarks_from_app[max_idx]) ##use to get face landmarks. TODO see if these landmarks are the same as insightface
#             aligned_face_chip = image_pipeline(images[idx], face_landmarks)
            
#             face_indicators_app.append(True)
#             face_bboxs_app.append(bbox)
#             face_chips_app.append(face_chip.unsqueeze(dim=0))
#             face_landmarks_app.append(torch.tensor(face_landmarks).unsqueeze(dim=0).to(device=images.device).to(images.dtype))
#             aligned_face_chips_app.append(aligned_face_chip.unsqueeze(dim=0))
    
#     face_indicators_app = torch.tensor(face_indicators_app).to(device=images.device)
#     face_bboxs_app = torch.tensor(face_bboxs_app).to(device=images.device)
#     face_chips_app = torch.cat(face_chips_app, dim=0)
#     face_landmarks_app = torch.cat(face_landmarks_app, dim=0)
#     aligned_face_chips_app = torch.cat(aligned_face_chips_app, dim=0)
    
#     return face_indicators_app, face_bboxs_app, face_chips_app, face_landmarks_app, aligned_face_chips_app
            

def get_face_FR(images, args, fill_value=-1, eval=False):
    """
    images:shape [N,3,H,W], in range [-1,1], pytorch tensor
    returns:
        face_indicators: torch tensor of shape [N], only True or False
            True means face is detected, False otherwise
        face_bboxs: torch tensor of shape [N,4], 
            if face_indicator is False, the corresponding face_bbox will be [fill_value,fill_value,fill_value,fill_value]
        face_chips: torch tensor of shape [N,3,224,224]
            if face_indicator is False, the corresponding face_chip will be all fill_value
    """
    images_np = (images*255).cpu().detach().permute(0,2,3,1).float().numpy().astype(np.uint8)
    # images_np = (images*255).cpu().detach().permute(0,2,3,1).float().numpy().astype(np.uint8)
    
    face_indicators_FR = []
    face_bboxs_FR = []
    face_chips_FR = []
    face_landmarks_FR = []
    aligned_face_chips_FR = []
    for idx, image_np in enumerate(images_np):
        # import pdb; pdb.set_trace()
        faces_from_FR = face_recognition.face_locations(image_np, model="cnn", number_of_times_to_upsample=0)
        if len(faces_from_FR) == 0:
            face_indicators_FR.append(False)
            face_bboxs_FR.append([fill_value]*4)
            face_chips_FR.append(torch.ones([1,3,args.size_face,args.size_face], dtype=images.dtype, device=images.device)*(fill_value))
            face_landmarks_FR.append(torch.ones([1,5,2], dtype=images.dtype, device=images.device)*(fill_value))
            aligned_face_chips_FR.append(torch.ones([1,3,args.size_aligned_face,args.size_aligned_face], dtype=images.dtype, device=images.device)*(fill_value))
        else:
            face_from_FR = get_largest_face_FR(faces_from_FR, dim_max=image_np.shape[0], dim_min=0)
            bbox = face_from_FR
            bbox = np.array((bbox[-1],) + bbox[:-1]) # need to convert bbox from face_recognition to the right order
            bbox = expand_bbox(bbox, expand_coef=1.1, target_ratio=1) # need to use a larger expand_coef for FR
            face_chip = crop_face(images[idx], bbox, target_size=[args.size_face,args.size_face], fill_value=fill_value, eval=eval, idx=idx)
            
            face_landmarks = face_recognition.face_landmarks(image_np, face_locations=[face_from_FR], model="large")

            left_eye = np.array(face_landmarks[0]["left_eye"]).mean(axis=0)
            right_eye = np.array(face_landmarks[0]["right_eye"]).mean(axis=0)
            nose_tip = np.array(face_landmarks[0]["nose_bridge"][-1])
            top_lip_left = np.array(face_landmarks[0]["top_lip"][0])
            top_lip_right = np.array(face_landmarks[0]["top_lip"][6])
            face_landmarks = np.stack([left_eye, right_eye, nose_tip, top_lip_left, top_lip_right])
            
            aligned_face_chip = image_pipeline(images[idx], face_landmarks)
            
            face_indicators_FR.append(True)
            face_bboxs_FR.append(bbox)
            face_chips_FR.append(face_chip.unsqueeze(dim=0))
            face_landmarks_FR.append(torch.tensor(face_landmarks).unsqueeze(dim=0).to(device=images.device).to(images.dtype))
            aligned_face_chips_FR.append(aligned_face_chip.unsqueeze(dim=0))
    
    face_indicators_FR = torch.tensor(face_indicators_FR).to(device=images.device)
    face_bboxs_FR = torch.tensor(face_bboxs_FR).to(device=images.device)
    face_chips_FR = torch.cat(face_chips_FR, dim=0)
    face_landmarks_FR = torch.cat(face_landmarks_FR, dim=0)
    aligned_face_chips_FR = torch.cat(aligned_face_chips_FR, dim=0)
    
    return face_indicators_FR, face_bboxs_FR, face_chips_FR, face_landmarks_FR, aligned_face_chips_FR

def get_largest_face_FR(faces_from_FR, dim_max, dim_min):
    if len(faces_from_FR) == 1:
        return faces_from_FR[0]
    elif len(faces_from_FR) > 1:
        area_max = 0
        idx_max = 0
        for idx, bbox in enumerate(faces_from_FR):
            bbox1 = np.array((bbox[-1],) + bbox[:-1])
            area = (min(bbox1[2],dim_max) - max(bbox1[0], dim_min)) * (min(bbox1[3],dim_max) - max(bbox1[1], dim_min))
            if area > area_max:
                area_max = area
                idx_max = idx
        return faces_from_FR[idx_max]


def get_largest_face_app_idx(face_from_app, dim_max, dim_min):
    if len(face_from_app) == 1:
        return face_from_app[0]
    elif len(face_from_app) > 1:
        area_max = 0
        idx_max = 0
        for idx in range(len(face_from_app)):
            bbox = face_from_app[idx]["bbox"]
            area = (min(bbox[2],dim_max) - max(bbox[0], dim_min)) * (min(bbox[3],dim_max) - max(bbox[1], dim_min))
            if area > area_max:
                area_max = area
                idx_max = idx
        return idx_max

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

def plot_in_grid_gender_race_age(images, save_to, face_indicators=None, face_bboxs=None, preds_gender=None, pred_class_probs_gender=None, preds_race=None, pred_class_probs_race=None, preds_age=None, pred_class_probs_age=None):
    """
    images: torch tensor in shape of [N,3,H,W], in range [-1,1]
    """
    idxs_reordered = []
    for g in [1,0]:
        for r in [0,1,2,3]:
            for a in [0,1]:
                idxs_ = ((preds_gender==g) * (preds_race == r) * (preds_age == a)).nonzero(as_tuple=False).view([-1])
                probs_ = pred_class_probs_gender[idxs_]
                idxs_ = idxs_[probs_.argsort(descending=True)]
                idxs_reordered.append(idxs_)
                
    idxs_no_face = (preds_race == -1).nonzero(as_tuple=False).view([-1])
    idxs_reordered.append(idxs_no_face)    
    idxs_reordered = torch.cat(idxs_reordered) 

    images_to_plot = []
    for idx in idxs_reordered:
        img = images[idx]
        face_indicator = face_indicators[idx]
        face_bbox = face_bboxs[idx]
        pred_gender = preds_gender[idx]
        pred_class_prob_gender = pred_class_probs_gender[idx]
        pred_race = preds_race[idx]
        pred_class_prob_race = pred_class_probs_race[idx]
        pred_age = preds_age[idx]
        pred_class_prob_age = pred_class_probs_age[idx]
        
        if pred_gender == 0:
            gender_border_color = "red"
        elif pred_gender == 1:
            gender_border_color = "blue"
        elif pred_gender == -1:
            gender_border_color = "white"

        if pred_race == 0:
            race_border_color = "limegreen"
        elif pred_race == 1:
            race_border_color = "Black"
        elif pred_race == 2:
            race_border_color = "brown"
        elif pred_race == 3:
            race_border_color = "orange"
        elif pred_race == -1:
            race_border_color = "white"
        
        if pred_age == 0:
            age_border_color = "darkorange"
        elif pred_age == 1:
            age_border_color = "darkgreen"
        elif pred_age == -1:
            age_border_color = "white"

        img_pil = transforms.ToPILImage()(img*0.5+0.5)
        img_pil_draw = ImageDraw.Draw(img_pil)  
        img_pil_draw.rectangle(face_bbox.tolist(), fill =None, outline ="black", width=4)

        img_pil = ImageOps.expand(img_pil_draw._image, border=(50,0,0,0),fill=age_border_color)
        img_pil_draw = ImageDraw.Draw(img_pil)
        if pred_class_prob_race.item() < 1:
            img_pil_draw.rectangle([(0,0),(50,(1-pred_class_prob_age.item())*512)], fill ="white", outline =None)
            
        img_pil = ImageOps.expand(img_pil_draw._image, border=(50,0,0,0),fill=race_border_color)
        img_pil_draw = ImageDraw.Draw(img_pil)
        if pred_class_prob_race.item() < 1:
            img_pil_draw.rectangle([(0,0),(50,(1-pred_class_prob_race.item())*512)], fill ="white", outline =None)
            
        img_pil = ImageOps.expand(img_pil_draw._image, border=(50,0,0,0),fill=gender_border_color)
        img_pil_draw = ImageDraw.Draw(img_pil)
        if pred_class_prob_gender.item() < 1:
            img_pil_draw.rectangle([(0,0),(50,(1-pred_class_prob_gender.item())*512)], fill ="white", outline =None)
            
        fnt = ImageFont.truetype(font="../data/0-utils/arial-bold.ttf", size=100)
        img_pil_draw.text((400, 400), f"{idx.item()}", align ="left", font=fnt)

        img_pil = ImageOps.expand(img_pil_draw._image, border=(10,10,10,10),fill="black")
        
        images_to_plot.append(img_pil)
        
    N_imgs = len(images_to_plot)
    N1 = int(math.sqrt(N_imgs))
    N2 = math.ceil(N_imgs / N1)

    for i in range(N1*N2-N_imgs):
        images_to_plot.append(
            Image.new('RGB', color="white", size=images_to_plot[0].size)
        )
    grid = image_grid(images_to_plot, N1, N2)
    if not os.path.exists(os.path.dirname(save_to)):
        os.makedirs(os.path.dirname(save_to))
    grid.save(save_to, quality=25)

def plot_in_grid(images, save_to, face_indicators=None, face_bboxs=None, preds_group=None, probs_group=None, group=['gender','age','rage'], save_images=False):
    """
    images: torch tensor in shape of [N,3,H,W], in range [-1,1]
    """
    group_dic = {
        'gender' : ['male', 'female'],
        'age' : ['young', 'old'],
        'race' : ['East Asian', 'Indian', 'Black', 'White', 'Middle Eastern', 'Latino_Hispanic', 'Southeast Asian']
    }
    images_w_face = images[face_indicators]
    images_wo_face = images[face_indicators.logical_not()]


    images_to_plot = []
    
    # if save_images:
    #     tmp_folder = save_to.split('.')[-3] + save_to.split('.')[-2] + '/tmp'
    #     print(tmp_folder)
    #     if not os.path.exists(tmp_folder):
    #         os.makedirs(tmp_folder)

    for idx in range(len(images)):
        img = images[idx]
        
        img_pil = transforms.ToPILImage()(img)

        # if idx == 2:
        #     img_tmp = (img * 255).to(torch.uint8).to(torch.float16)/255.0
        #     img = transforms.ToTensor()(img_pil)
        #     print('before saving :', img[:,100:103,100:103])
            # print('before saving2 :', img_tmp[:,100:103,100:103])
        # if save_images:
            # img_pil.save(f"{tmp_folder}/{idx}.png")
        
        images_to_plot.append(img_pil)
        
    N_imgs = len(images_to_plot)
    N1 = int(math.sqrt(N_imgs))
    N2 = math.ceil(N_imgs / N1)

    for i in range(N1*N2-N_imgs):
        images_to_plot.append(
            Image.new('RGB', color="white", size=images_to_plot[0].size)
        )
    grid = image_grid(images_to_plot, N1, N2)
    if not os.path.exists(os.path.dirname(save_to)):
        os.makedirs(os.path.dirname(save_to))
    grid.save(save_to, quality=25)

def expand_bbox(bbox, expand_coef, target_ratio):
    """
    bbox: [width_small, height_small, width_large, height_large], 
        this is the format returned from insightface.app.FaceAnalysis
    expand_coef: 0 is no expansion
    target_ratio: target img height/width ratio
    
    note that it is possible that bbox is outside the original image size
    confirmed for insightface.app.FaceAnalysis
    """
    
    bbox_width = bbox[2] - bbox[0]
    bbox_height = bbox[3] - bbox[1]
    
    current_ratio = bbox_height / bbox_width
    if current_ratio > target_ratio:
        more_height = bbox_height * expand_coef
        more_width = (bbox_height+more_height) / target_ratio - bbox_width
    elif current_ratio <= target_ratio:
        more_width = bbox_width * expand_coef
        more_height = (bbox_width+more_width) * target_ratio - bbox_height
    
    bbox_new = [0,0,0,0]
    bbox_new[0] = int(round(bbox[0] - more_width*0.5))
    bbox_new[2] = int(round(bbox[2] + more_width*0.5))
    bbox_new[1] = int(round(bbox[1] - more_height*0.5))
    bbox_new[3] = int(round(bbox[3] + more_height*0.5))
    return bbox_new

def crop_face(img_tensor, bbox_new, target_size, fill_value, eval=False, idx=-1):
    """
    img_tensor: [3,H,W]
    bbox_new: [width_small, height_small, width_large, height_large]
    target_size: [width,height]
    fill_value: value used if need to pad
    """
    img_height, img_width = img_tensor.shape[-2:]
    
    idx_left = max(bbox_new[0],0)
    idx_right = min(bbox_new[2], img_width)
    idx_bottom = max(bbox_new[1],0)
    idx_top = min(bbox_new[3], img_height)

    pad_left = max(-bbox_new[0],0)
    pad_right = max(-(img_width-bbox_new[2]),0)
    pad_top = max(-bbox_new[1],0)
    pad_bottom = max(-(img_height-bbox_new[3]),0)

    img_face = img_tensor[:,idx_bottom:idx_top,idx_left:idx_right]
    # if eval:
    #     img_face = (img_face * 255).to(torch.uint8).to(torch.float16)
    #     img_face = img_face/255.0 
    #     if idx == 2:
    #         print('before padding :', img_face[:,100:103,100:103])
    if pad_left>0 or pad_top>0 or pad_right>0 or pad_bottom>0:
        img_face = torchvision.transforms.Pad([pad_left,pad_top,pad_right,pad_bottom], fill=0)(img_face)
    # if idx == 2 and eval:
    #     print('after padding :', img_face[:,100:103,100:103])
    img_face = torchvision.transforms.Resize(size=target_size)(img_face)
    # if idx == 2 and eval:
    #     print('after resize :', img_face[:,100:103,100:103])
    return img_face

def image_pipeline(img, tgz_landmark):
    img = (img+1)/2.0 * 255 # map to [0,255]

    crop_size = (112,112)
    src_landmark = np.array(
    [[38.2946, 51.6963], # left eye
    [73.5318, 51.5014], # right eye
    [56.0252, 71.7366], # nose
    [41.5493, 92.3655], # left corner of the mouth
    [70.7299, 92.2041]] # right corner of the mouth
    )

    tform = transform.SimilarityTransform()
    tform.estimate(tgz_landmark, src_landmark)

    M = torch.tensor(tform.params[0:2, :]).unsqueeze(dim=0).to(img.dtype).to(img.device)
    img_face = kornia.geometry.transform.warp_affine(img.unsqueeze(dim=0), M, crop_size, mode='bilinear', padding_mode='zeros', align_corners=False)
    img_face = img_face.squeeze()

    img_face = (img_face/255.0)*2-1 # map back to [-1,1]
    return img_face
