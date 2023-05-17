import time
import torch
import torch.nn as nn
import numpy as np
import imageio,scipy
from torchvision import transforms
from torchvision import datasets
import torchvision
from pycocotools.coco import COCO
import torch
from tqdm import tqdm
import os
from utils.nsd_utils import get_roi_config, extract_single_roi
import glob
from PIL import Image


import ipdb
st = ipdb.set_trace

class NSDDataloader(torch.utils.data.Dataset):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """
    def __init__(self, args, split, transform, transform2=None):
        super(NSDDataloader, self).__init__()
        
        images = [d for d in tqdm(glob.glob(args.coco_images_path + '/*' + '/*.jpg'))]
        image_ids = [int(images[i].split('/')[-1].split('.')[0]) for i in range(len(images))]
        mapping = {id:im for id,im in zip(image_ids, images)}
        self.args = args
        self.subjects = args.subjects
        self.rois = args.rois
        self.split = split
        print(f'Split is {split}')
        if args.coco_ids_path is not None:
            print(f"Loading coco ids from {args.coco_ids_path}")
        self.transform = transform
        self.transform2 = transform2
        # self.data_mapping = {}
        self.ids = []

        print("Getting COCO and brain data for each subject...")
        for subj in tqdm(self.subjects):
            # get coco image ids
            coco_ids_subj = []
            if split=="train":
                coco_ids_subj_ = np.load(
                    os.path.join(args.subjects_repeat_path, "coco_subject_splits", "coco_ID_of_repeats_subj%02d_train.npy" % (subj))
                )
                coco_ids_subj.append(coco_ids_subj_)
            elif split=="validation":
                coco_ids_subj_ = np.load(
                    os.path.join(args.subjects_repeat_path, "coco_subject_splits", "coco_ID_of_repeats_subj%02d_val.npy" % (subj))
                )
                coco_ids_subj.append(coco_ids_subj_)
            elif split=="test":
                coco_ids_subj_ = np.load(os.path.join(args.subjects_repeat_path, 'coco_ID_of_repeats_shared1000.npy'))
                coco_ids_subj.append(coco_ids_subj_)
            coco_ids_subj = np.concatenate(coco_ids_subj, axis=0)
            self.ids.extend(list(coco_ids_subj))
        
        if args.coco_ids_path is not None:
            coco_ids_subj_ = np.load(args.coco_ids_path)
            self.ids.extend(list(coco_ids_subj_))

        self.ids = list(set(self.ids))

        if args.subsample_images is not None:
            self.ids = self.ids[::args.subsample_images]
        
        self.images = [mapping[id] for id in self.ids]

    def filter_topk_images_from_roi_brain_response(self, responses, topk=1000):
        '''
        responses should be same size as self.ids and self.images
        '''

        # get topk images from brain responses
        mean_responses = np.mean(responses, axis=1)
        mean_responses_argsort = np.argsort(-mean_responses)
        topk_indices = list(mean_responses_argsort[:topk])
        self.ids = [self.ids[idx] for idx in topk_indices]
        self.images = [self.images[idx] for idx in topk_indices]

    def __len__(self):
        return len(self.images) #len(self.ids)

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        out = {}
        
        coco_id = self.ids[index] #self.ids[index]

        image_path = self.images[index]
        image = Image.open(image_path)
        if self.transform2 is not None:
            image2 = self.transform2(image)
            out['images2'] = image2
        image = self.transform(image)
        out['images'] = image
        out['coco_id'] = coco_id

        return out 