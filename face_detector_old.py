
import torch
import numpy as np 
import os
from tqdm import tqdm
# from insightface.app import FaceAnalysis
import cv2
import torchvision 
from PIL import Image, ImageOps
import argparse
import data_handler
from torchvision import transforms
import dlib
import pickle
import face_recognition
dlib.DLIB_USE_CUDA = True
# from trainer.image_utils import expand_bbox
class FaceDetector:
    def __init__(self):
        # self.app = FaceAnalysis(name='buffalo_l', 
                # providers=[('CUDAExecutionProvider', {'device_id':'0'})],
# )
        # self.app.prepare(ctx_id=0, det_size=(640, 640))

        # self.app = dlib.get_frontal_face_detector()
        self.app = dlib.cnn_face_detection_model_v1('/n/holylabs/LABS/calmon_lab/Lab/datasets/mpr_stuffs/stuffs/dlib_models/mmod_human_face_detector.dat')

    def _expand_bbox(self, bbox, expand_coef, target_ratio):
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

    # def _get_largest_face_app(self, faces, image):
    #     area_max = 0
    #     idx_max = 0
    #     for idx in range(len(faces)):
    #         left, bottom, right, top = faces[idx]
    #         area = (right - left) * (bottom - top)
    #         if area > area_max:
    #             area_max = area
    #             idx_max = idx
    #     return faces[idx_max]

    def __get_largest_face_app(self, faces, image):
        area_max = 0
        idx_max = 0
        for idx in range(len(faces)):
            left, top, right, bottom = self.extract_position(image, faces[idx])
            bbox = faces[idx]
            area = (right - left) * (bottom - top)
            if area > area_max:
                area_max = area
                idx_max = idx
        return faces[idx_max]
    
    def process_tensor_image(self, images, fill_value=-1, torch_tensor=True):
        faces = []
        face_indicators = []
        face_bboxs = []
        images_ori = images
        if torch_tensor:
            images = (images*255).cpu().detach().permute(0,2,3,1).numpy().astype(np.uint8)
        
        # images_np = images.permute(0,2,3,1).float().numpy().astype(np.uint8)
        print(images.shape)
        num_faces_list = []
        for idx, image in enumerate(images):
            # faces_from_app = self.app.get(image_np[:,:,[2,1,0]])
            # faces_from_app = self.app.get(image_np[:,:,[2,1,0]])
            # gray_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            # faces_from_app = self.app(gray_image)

            faces_from_app = self.app(image, 1)
            # faces_from_app = face_recognition.face_locations(image, model="cnn", number_of_times_to_upsample=0)
            # print(faces_from_app)
            num_faces = len(faces_from_app)
            num_faces_list.append(num_faces)
            if num_faces >= 1:
                if num_faces > 1:
                    faces_from_app = self.__get_largest_face_app(faces_from_app, image)
                    faces_from_app = [faces_from_app]
                # faces = self._crop_face(images[idx], bbox, target_size=[224,224], fill_value=fill_value)
                
                # face_landmarks = np.array(face_from_app["kps"])
                # aligned_faces = image_pipeline(images[idx], face_landmarks)

                face_indicators.append(True)
                face_bboxs.extend(faces_from_app)
            else:
                face_indicators.append(False)
        print(f"The number of images filtered : {len(images)-sum(face_indicators)}")
        
        face_indicators = torch.tensor(face_indicators)
        return face_indicators, face_bboxs
    
    # def process_tensor_image(self, images, fill_value=-1, torch_tensor=True):
    #     faces = []
    #     face_indicators = []
    #     face_bboxs = []
    #     images_ori = images
    #     if torch_tensor:
    #         images = (images*255).cpu().detach().permute(0,2,3,1).numpy().astype(np.uint8)
        
    #     # images_np = images.permute(0,2,3,1).float().numpy().astype(np.uint8)

    #     num_faces_list = []
    #     for idx, image in enumerate(images):
    #         # faces_from_app = self.app.get(image_np[:,:,[2,1,0]])
    #         # faces_from_app = self.app.get(image_np[:,:,[2,1,0]])
    #         # gray_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    #         # faces_from_app = self.app(gray_image)

    #         faces_from_app = self.app(image, 1)
    #         # faces_from_app = face_recognition.face_locations(image, model="cnn", number_of_times_to_upsample=0)
    #         # print(faces_from_app)
    #         num_faces = len(faces_from_app)
    #         num_faces_list.append(num_faces)
    #         if num_faces >= 1:
    #             if num_faces > 1:
    #                 faces_from_app = self._get_largest_face_app(faces_from_app, image)
    #             else:
    #                 faces_from_app = faces_from_app[0]
    #             faces_from_app = np.array((faces_from_app[-1],) + faces_from_app[:-1])                    
    #             faces_from_app = self._expand_bbox(faces_from_app, expand_coef=1.1, target_ratio=1)
    #             faces_from_app = [faces_from_app]
    #             face_indicators.append(True)
    #             face_bboxs.extend(faces_from_app)
    #         else:
    #             face_indicators.append(False)
    #     print(f"The number of images filtered : {len(images)-sum(face_indicators)}")
        
    #     face_indicators = torch.tensor(face_indicators)
    #     return face_indicators, face_bboxs

    def process_tensor_image_for_mscoco(self, images, fill_value=-1, torch_tensor=True):
        faces = []
        face_indicators = []
        face_bboxs = []
        if torch_tensor:
            images = (images*255).cpu().detach().permute(0,2,3,1).numpy().astype(np.uint8)
        
        # images_np = images.permute(0,2,3,1).float().numpy().astype(np.uint8)
        num_faces_list = []

        for idx, image in enumerate(images):
            # faces_from_app = self.app.get(image_np[:,:,[2,1,0]])
            # faces_from_app = self.app.get(image_np[:,:,[2,1,0]])
            # gray_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            # faces_from_app = self.app(gray_image)
            faces_from_app = self.app(image, 1)
            num_faces = len(faces_from_app)
            num_faces_list.append(num_faces)
            
            if num_faces >= 1:
                if num_faces > 1:
                    faces_from_app = self.__get_largest_face_app(faces_from_app, image)
                    faces_from_app = [faces_from_app]
                # faces = self._crop_face(images[idx], bbox, target_size=[224,224], fill_value=fill_value)
                
                # face_landmarks = np.array(face_from_app["kps"])
                # aligned_faces = image_pipeline(images[idx], face_landmarks)

                face_indicators.append(True)
                face_bboxs.extend(faces_from_app)
                # facess.append(faces.unsqueeze(dim=0))
                # face_landmarks_app.append(torch.tensor(face_landmarks).unsqueeze(dim=0).to(device=images.device).to(images.dtype))
                # aligned_facess_app.append(aligned_faces.unsqueeze(dim=0))
            # elif num_faces >= 1:
            #     face_indicators.append(True)
            #     max_area = 0
            #     for face_bbox in faces_from_app:
            #         bbox = face_bbox.rect
            #         left, top, right, bottom = bbox.left(), bbox.top(), bbox.right(), bbox.bottom()
            #         area = (right-left)*(bottom-top)
            #         if area > max_area:
            #             max_area = area
            #             max_bbox = bbox
            #     face_bboxs.append(max_bbox)
            else:
                face_indicators.append(False)
        # print(num_faces_list)
        
        print(f"The number of images filtered : {len(images)-sum(face_indicators)}")
        
        if torch_tensor:
            face_indicators = torch.tensor(face_indicators).to(device=images.device)
        else:
            face_indicators = torch.tensor(face_indicators)
        # face_bboxs = torch.tensor(face_bboxs).to(device=images.device)
        # facess = torch.cat(facess, dim=0)
        # face_landmarks_app = torch.cat(face_landmarks_app, dim=0)
        # aligned_facess_app = torch.cat(aligned_facess_app, dim=0)
                
        return face_indicators, face_bboxs
    
    def process_batched_tensor_image(self, images, fill_value=-1):
        faces = []
        face_indicators = []
        face_bboxs = []
        
        # images_np = (images*255).cpu().detach().permute(0,2,3,1).numpy().astype(np.uint8)
        # print(images_np[0])
        # images= dlib.convert_image(images_np)
        
        num_faces_list = []
        face_indicators = []
        face_bboxs = []
        print('start detection')
        faces_from_apps = self.app(images, 1, batch_size=len(images))
        num_images = 0
        for face_from_app in faces_from_apps:
            if len(face_from_app) >= 1:
                face_indicators.append(True)
                face_bboxs.extend(face_from_app)
                num_images += 1
        print(f"Number of images with faces: {num_images}")
        return face_indicators, face_bboxs    

    def process_pil_image(self, image):
        image_np = np.array(image)

        faces_from_app = self.app(image_np, 1)
        num_faces = len(faces_from_app)
        return num_faces >= 1
    
    def extract_position(self, image, bbox):
        bbox = bbox.rect
        # left, top, width, height = bbox.left(), bbox.top(), bbox.width(), bbox.height()
        left, top, right, bottom = bbox.left(), bbox.top(), bbox.right(), bbox.bottom()
        if right < left:
            print(left, top, right, bottom)
        left = max(left, 0)
        top = max(top, 0)
        if type(image) == torch.Tensor or type(image) == np.ndarray:
            right = min(right, image.shape[-1])
            bottom = min(bottom, image.shape[-2])
        # if image is PIL Image
        else:
            right = min(right, image.size[-2])
            bottom = min(bottom, image.size[-1])

        return [left, top, right, bottom]    
    
    # def extract_position(self, image=None, bbox=None, image_size=None):
    #     # bbox = bbox.rect
    #     max_height = image_size if image is None else image.shape[-2]
    #     max_width = image_size if image is None else image.shape[-1]
    #     # left, top, right, bottom = bbox.left(), bbox.top(), bbox.right(), bbox.bottom()
    #     left, top, right, bottom = bbox[0], bbox[1], bbox[2], bbox[3]
    #     pad_left = max(0, -left)
    #     pad_top = max(0, -top)
    #     pad_right = max(0, right - max_width)
    #     pad_bottom = max(0, bottom - max_height)
    #     # if pad_left>0 or pad_top>0 or pad_right>0 or pad_bottom>0:
    #         # print(f"padding: {pad_left}, {pad_top}, {pad_right}, {pad_bottom}")

    #     left, top, right, bottom = bbox[0], bbox[1], bbox[2], bbox[3]

    #     if right < left:
    #         print(left, top, right, bottom)
    #     left = max(left, 0)
    #     top = max(top, 0)
    #     if type(image) == torch.Tensor or type(image) == np.ndarray:
    #         right = min(right, max_width)
    #         bottom = min(bottom, max_height)
    #     # if image is PIL Image
    #     else:
    #         right = min(right, max_height)
    #         bottom = min(bottom, max_width)

    #     return [left, top, right, bottom, pad_left, pad_top, pad_right, pad_bottom]

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='representational-generation')
        
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n-workers', type=int, default=1)
    parser.add_argument('--dataset', type=str, default='general')
    parser.add_argument('--trainer', type=str, default='scratch')
    parser.add_argument('--dataset-path', type=str, default=None, help='it is only used when query-dataset is general')
    parser.add_argument('--target-concept', type=str, default='scratch')

    parser.add_argument('--train', default=False, action='store_true', help='train the model')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--vision-encoder', type=str, default='CLIP',choices = ['BLIP', 'CLIP', 'PATHS'])

    args = parser.parse_args()

    face_detector = FaceDetector()
    loader = data_handler.DataloaderFactory.get_dataloader(dataname=args.dataset, args=args)
    
    if 'mscoco' in args.dataset_path:
        transform = transforms.Compose(
            [transforms.Resize((512,512)),
                transforms.ToTensor()]
        )
    else:
        transform = transforms.Compose(
            [transforms.ToTensor()]
        )
        
    loader.dataset.processor = None
    loader.dataset.transform = transform
        
    filtered_ids = []
    unfiltered_ids = []
    bbox_dic = {}

    if 'original' in args.dataset_path:
        args.new_dataset_path = args.dataset_path.replace('_original', '')

    if not os.path.exists(args.dataset_path+'/new'):
        os.makedirs(args.dataset_path+'/new')
    n_imgs = 0

    for image, _, idxs in loader:
        flags, bboxs = face_detector.process_tensor_image(image)
        print(flags)

        filtered_ids.extend(idxs[~flags].tolist())
        unfiltered_ids.extend(idxs[flags].tolist())
        n_imag_per_iter = 0
        for idx, flag in enumerate(flags):
            # bbox_dic[idx.item()] = face_detector.extract_position(image, bbox)
            if flag:
                bbox_dic[n_imgs] = face_detector.extract_position(image[idx], bboxs[n_imag_per_iter])
                img = image[idx]
                img_pil = transforms.ToPILImage()(img)
                img_pil.save(f"{args.dataset_path}/new/{n_imgs}.png")
                n_imag_per_iter += 1
                n_imgs += 1
            

    # Save the filtered and unfiltered IDs to files
    with open(os.path.join(args.dataset_path,'filtered_ids.txt'), 'w') as f:
        for id in filtered_ids:
            f.write(f"{id}\n")

    with open(os.path.join(args.dataset_path,'unfiltered_ids.txt'), 'w') as f:
        for id in unfiltered_ids:
            f.write(f"{id}\n")

    with open(os.path.join(args.dataset_path,'bbox_dic.pkl'), 'wb') as f:
        pickle.dump(bbox_dic, f)
        # for id, bbox in bbox_dic.items():
            # f.write(f"{id}: {bbox}\n")
    
    # percentage of filtered images
    print(f"Percentage of filtered images: {len(filtered_ids)/(len(filtered_ids)+len(unfiltered_ids))}")
