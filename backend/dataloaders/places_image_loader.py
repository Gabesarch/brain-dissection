import time
import torch
import torch.nn as nn
import numpy as np
import imageio,scipy
from torchvision import transforms
from torchvision import datasets
import torchvision
# from pycocotools.coco import COCO
import torch
from tqdm import tqdm
import os
# from utils.nsd_utils import get_roi_config, extract_single_roi
import glob
from PIL import Image


import ipdb
st = ipdb.set_trace

class NSDDataloader(torch.utils.data.Dataset):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """
    def __init__(self, args, split, transform, transform2=None, download=True):
        super(NSDDataloader, self).__init__()

        if split is None:
            split = 'val'
        dirname = os.path.join(args.data_directory, 'places/%s' % split)
        if download and not os.path.exists(dirname):
            os.makedirs(args.data_directory, exist_ok=True)
            torchvision.datasets.utils.download_and_extract_archive(
                'http://gandissect.csail.mit.edu/datasets/' +
                'places_%s.zip' % split,
                args.data_directory,
                md5=dict(val='593bbc21590cf7c396faac2e600cd30c',
                         train='')[split])
        
        images = [d for d in tqdm(glob.glob(dirname + '/*' + '/*.jpg'))]
        self.args = args
        self.subjects = args.subjects
        self.rois = args.rois
        self.split = split
        self.transform = transform
        self.transform2 = transform2
        self.images = images

        if args.subsample_images is not None:
            self.images = self.images[::args.subsample_images]

    def filter_topk_images_from_roi_brain_response(self, responses, topk=1000):
        '''
        responses should be same size as self.ids and self.images
        '''

        # get topk images from brain responses
        mean_responses = np.mean(responses, axis=1)
        mean_responses_argsort = np.argsort(-mean_responses)
        topk_indices = list(mean_responses_argsort[:topk])
        self.images = [self.images[idx] for idx in topk_indices]

    def __len__(self):
        return len(self.images)

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        out = {}
        
        image_path = self.images[index] #self.data_mapping[coco_id]['image']
        image = Image.open(image_path).convert('RGB')
        if self.transform2 is not None:
            image2 = self.transform2(image)
            out['images2'] = image2
        image = self.transform(image)
        out['images'] = image

        return out 