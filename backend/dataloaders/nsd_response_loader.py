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
    def __init__(self, args, split, transform):
        super(NSDDataloader, self).__init__()
        
        images = [d for d in tqdm(glob.glob(args.coco_images_path + '/*' + '/*.jpg'))]
        image_ids = [int(images[i].split('/')[-1].split('.')[0]) for i in range(len(images))]
        self.args = args
        self.subjects = args.subjects
        self.rois = args.rois
        self.split = split
        self.transform = transform
        self.data_mapping = {}
        self.roi_sizes = {}

        print("Getting COCO and brain data for each subject...")
        for subj in tqdm(self.subjects):
            # get coco image ids
            if split=="train":
                coco_ids_subj = np.load(
                    os.path.join(args.subjects_repeat_path, "coco_subject_splits", "coco_ID_of_repeats_subj%02d_train.npy" % (subj))
                )
            elif split=="validation":
                coco_ids_subj = np.load(
                    os.path.join(args.subjects_repeat_path, "coco_subject_splits", "coco_ID_of_repeats_subj%02d_val.npy" % (subj))
                )
            elif split=="test":
                coco_ids_subj = np.load(os.path.join(args.subjects_repeat_path, 'coco_ID_of_repeats_shared1000.npy'))
            else:
                assert(False)

            stimulus_order = np.load(
                    os.path.join(args.subjects_repeat_path, "coco_ID_of_repeats_subj%02d.npy" % (subj))
                )
            stimulus_order = list(stimulus_order)

            # get brain data
            brain_path = "%s/averaged_cortical_responses_zscored_by_run_subj%02d.npy" % (
                args.brain_data_dir,
                subj,
            )
            brain_data = np.load(brain_path).astype(np.float32)

            # get roi mask
            meta_name_dict, roi_name_to_meta_name = get_roi_config()
            roi_mask = np.zeros(brain_data.shape[1], dtype=bool)
            for roi in self.rois:
                meta_name = roi_name_to_meta_name[roi]
                output_masks, roi_labels = extract_single_roi(meta_name, meta_name_dict, args.roi_dir, subj)
                roi_ind = roi_labels.index(roi)
                roi_mask_ = output_masks[roi_ind] 
                roi_mask = np.logical_or(roi_mask, roi_mask_)

            brain_data = brain_data[:,roi_mask]

            # filter out voxels with low noise ceiling
            if args.nc_threshold>0.:
                # get noise ceiling
                noise_ceiling_1d = np.load(
                    "%s/subj%01d/noise_ceiling_percent_1d_subj%02d.npy"
                    % (args.noise_ceiling_dir, subj, subj)
                )  
                # above is noise ceiling in percent
                # convert to fraction percent
                noise_ceiling_1d = noise_ceiling_1d / 100
                noise_ceiling_1d = noise_ceiling_1d[roi_mask]
                good = noise_ceiling_1d>args.nc_threshold
                brain_data = brain_data[:,good]
                
            self.roi_sizes[subj] = brain_data.shape[1]

            for coco_id in tqdm(list(coco_ids_subj), leave=False, desc=f"subj {subj} mapping"):
                if args.debug:
                    if len(self.data_mapping.keys())>100:
                        break
                if coco_id not in self.data_mapping.keys():
                    self.data_mapping[coco_id] = {}
                    self.data_mapping[coco_id]['image'] = images[image_ids.index(coco_id)]
                    self.data_mapping[coco_id][subj] = {}
                    self.data_mapping[coco_id][subj]['brain_data'] = brain_data[stimulus_order.index(coco_id)]
                else:
                    self.data_mapping[coco_id][subj] = {}
                    self.data_mapping[coco_id][subj]['brain_data'] = brain_data[stimulus_order.index(coco_id)]

            if args.debug:
                break
        
        self.ids = list(self.data_mapping.keys())
        self.total_voxel_size = sum(list(self.roi_sizes.values()))

    def __len__(self):
        return len(self.ids)

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        
        coco_id = self.ids[index]

        image_path = self.data_mapping[coco_id]['image']
        image = Image.open(image_path)
        image = self.transform(image)

        y_brain = []
        y_mask = []
        for subj in self.subjects:
            if subj in self.data_mapping[coco_id].keys():
                y_brain.append(self.data_mapping[coco_id][subj]['brain_data'])
                y_mask.append(np.ones(self.roi_sizes[subj], dtype=bool))
            else:
                y_brain.append(np.zeros(self.roi_sizes[subj], dtype=np.float32))
                y_mask.append(np.zeros(self.roi_sizes[subj], dtype=bool))

        y_brain = np.concatenate(y_brain, axis=0)
        y_mask = np.concatenate(y_mask, axis=0)

        # check for nan
        not_nan = ~np.isnan(y_brain)
        y_mask = np.logical_and(y_mask, not_nan)

        out = {}
        out['images'] = image
        out['y_brain'] = y_brain
        out['y_mask'] = y_mask

        return out 