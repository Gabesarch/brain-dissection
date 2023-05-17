from arguments import args
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
import pickle
import os
# from backend.dataloaders.gqa_loader import NSDDataloader
from backend.dataloaders.gqa_loader import NSDDataloader
if args.arch=="e2cnn":
    print("Using E(2) CNN")
    from nets.convnet_e2cnn import ConvNet
elif args.arch=="cnn":
    assert(False) # not compatible anymore
    print("Using CNN")
    from nets.convnet import ConvNet
elif args.arch=="cnn_alt":
    print("Using CNN ALT")
    from nets.convnet_alt import ConvNet
else:
    assert(False)
import utils.dist

import wandb
from utils.improc import MetricLogger
from nets.loss import OptRespLoss
from backend import saverloader
import copy

# import warnings
# warnings.filterwarnings('error')

import os
import sys
import argparse
import cv2
import random
import colorsys
import requests
from io import BytesIO

import skimage.io
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from PIL import Image
from sklearn.metrics import jaccard_score

from utils.nsd_utils import get_roi_config, extract_single_roi
import cortex
import nibabel as nib

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

import ipdb
st = ipdb.set_trace
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from dissect.netdissect import nethook, imgviz, show, segmenter, renormalize, upsample, tally, pbar, runningstats
from dissect.netdissect import setting
from models.convnet_eval_baudissect_nsd import Eval

torch.backends.cudnn.benchmark = True
torch.set_grad_enabled(False) # not training anything!

# fix the seed for reproducibility
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

class Eval_VG(Eval):
    def __init__(self):  
        super(Eval_VG, self).__init__()

    @torch.no_grad()
    def run_dissection(self):

        self.model.eval()

        iv = imgviz.ImageVisualizer(128, percent_level=args.activation_threshold, image_size=(self.W, self.H))

        upfn = upsample.upsampler((self.W, self.H), (self.act_W, self.act_H))
        cq = runningstats.RunningConditionalQuantile(r=1024)
        most_common_conditions = set()
        gpu_cache = 64
        renorm_seg = renormalize.renormalizer(source='pt', target='zc')

        segs = None
        responses = None
        images_vis = None
        acts = None
        depths = None

        print(f"Image size eval: {args.image_size_eval}")
        if args.load_dissection_samples:
            # load variables from memory if they are saved
            root, model_name = os.path.split(args.load_model_path)
            _, root_model = os.path.split(root)
            subj_add = '_subject'+str(args.eval_subject) if args.eval_subject is not None else ''
            if os.path.exists(os.path.join(args.tmp_dir, 'dissect', root_model, f"{model_name.split('.pth')[0]}_{args.mode}_dissection_acts{subj_add}.npy")):
                if args.analyze_depth or not args.reduced_eval_memory:
                    acts = torch.from_numpy(np.load(os.path.join(args.tmp_dir, 'dissect', root_model, f"{model_name.split('.pth')[0]}_{args.mode}_dissection_acts{subj_add}.npy")))
                if not args.reduced_eval_memory:
                    responses = torch.from_numpy(np.load(os.path.join(args.tmp_dir, 'dissect', root_model, f"{model_name.split('.pth')[0]}_{args.mode}_dissection_responses{subj_add}.npy")))
                    images_vis = torch.from_numpy(np.load(os.path.join(args.tmp_dir, 'dissect', root_model, f"{model_name.split('.pth')[0]}_{args.mode}_dissection_images_vis{subj_add}.npy")))
                    segs = torch.from_numpy(np.load(os.path.join(args.tmp_dir, 'dissect', root_model, f"{model_name.split('.pth')[0]}_{args.mode}_dissection_segs{subj_add}.npy")))
                if args.analyze_depth:
                    depths = torch.from_numpy(np.load(os.path.join(args.tmp_dir, 'dissect', root_model, f"{model_name.split('.pth')[0]}_{args.mode}_dissection_depths{subj_add}.npy")))
                else:
                    depths = None
                cachefile = os.path.join(args.tmp_dir, 'dissect', root_model, f"{model_name.split('.pth')[0]}_{args.mode}_dissection_cq{subj_add}.npz")
                cq_state_dict = np.load(cachefile, allow_pickle=True)
                cq.set_state_dict(cq_state_dict)
                cachefile = os.path.join(args.tmp_dir, 'dissect', root_model, f"{model_name.split('.pth')[0]}_{args.mode}_dissection_params{subj_add}.npz")
                dat = np.load(cachefile, allow_pickle=True)
                seglabels = dat["seglabels"] 
                segcatlabels = dat["segcatlabels"]
                print('segmenter has', len(seglabels), 'labels')
                self.run_analysis(acts, responses, images_vis, segs, iv, seglabels, segcatlabels, cq, depths=depths)
                return

        seglabels = self.seglabels
        segcatlabels = self.segcatlabels
            
        print('segmenter has', len(seglabels), 'labels')

        num_images = args.max_images if args.max_images is not None else len(self.dataset_loader)*args.batch_size
        num_voxels_keep = self.total_voxel_size
        if self.valid_voxels is not None:
            num_voxels_keep = np.sum(self.valid_voxels)
        if args.max_activation_plots is not None:
            num_voxels_keep = min(num_voxels_keep, args.max_activation_plots)
        if args.subsample_activations is not None:
            num_voxels_keep = len(list(np.arange(num_voxels_keep))[::args.subsample_activations])

        if args.filter_images_by_responses and not args.debug:
            b_tmp = 16
            self.dataset.fetch_seg = False
            dataset_loader = torch.utils.data.DataLoader(self.dataset,
                                    batch_size=b_tmp, 
                                    # sampler=sampler,
                                    num_workers=2, 
                                    # collate_fn=my_collate,
                                    pin_memory=True,
                                    drop_last=False
                                    )
            metric_logger = MetricLogger(delimiter="  ")
            header = f'{args.set_name}|{args.mode}|Getting all responses...'
            responses_ = torch.zeros((num_images, num_voxels_keep), dtype=torch.float32)
            # coco_ids = torch.zeros(num_images, dtype=torch.int32)
            for i_batch, batched_samples in enumerate(metric_logger.log_every(dataset_loader, 10, header)):
                images = batched_samples['images'].to(device)
                images_raw = batched_samples['images2'].to(device)
                # coco_id = batched_samples['coco_id']
                response = self.model(images)
                if self.valid_voxels is not None:
                    response = response[:,self.valid_voxels]
                if args.max_activation_plots is not None:
                    response = response[:,:args.max_activation_plots]
                if args.subsample_activations is not None:
                    response = response[:,::args.subsample_activations]
                responses_[i_batch*b_tmp:(i_batch+1)*b_tmp] = response.cpu()
                # if i_batch>10:
                #     break
                # coco_ids[i_batch*b_tmp:(i_batch+1)*b_tmp] = coco_id
            responses_ = responses_.numpy()
            self.dataset_loader.dataset.filter_topk_images_from_roi_brain_response(responses_, topk=args.topk_filter)
            print("NEW size dataloader:", len(self.dataset_loader))
            self.dataset.fetch_seg = True
            del responses_
        elif args.filter_images_by_responses and args.debug:
            responses_ = np.zeros((num_images, num_voxels_keep))
            self.dataset_loader.dataset.filter_topk_images_from_roi_brain_response(responses_, topk=args.topk_filter)
            print("NEW size dataloader:", len(self.dataset_loader))
            del responses_

        num_images = args.max_images if args.max_images is not None else len(self.dataset_loader)*args.batch_size
        metric_logger = MetricLogger(delimiter="  ")
        header = f'{args.set_name}|{args.mode}|{args.split}&{True if args.coco_ids_path is not None else False}|num_voxels:{num_voxels_keep}|depth?{args.analyze_depth}|subject{args.eval_subject}'
        
        # preallocate
        print(f"num images: {num_images}; num voxels: {num_voxels_keep}")
        if not args.reduced_eval_memory:
            images_vis = torch.zeros((num_images, 3, 128, 128), dtype=torch.uint8) 
            segs = torch.zeros((num_images, 13, args.image_size_eval, args.image_size_eval), dtype=torch.int64) 
            responses = torch.zeros((num_images, num_voxels_keep), dtype=torch.float32) 
        if args.analyze_depth:
            depths = torch.zeros((num_images, args.image_size_eval, args.image_size_eval), dtype=torch.float32)
        if args.analyze_depth or not args.reduced_eval_memory:
            acts = torch.zeros((num_images, num_voxels_keep, self.act_W, self.act_H), dtype=torch.float32)
        for i_batch, batched_samples in enumerate(metric_logger.log_every(self.dataset_loader, 10, header)):

            # print(f"Start batch {i_batch}..")

            images = batched_samples['images'].to(device)
            images_raw = batched_samples['images2'].to(device)
            seg = batched_samples['segms'].to(device).long()

            # st()
            # plt.figure()
            # plt.imshow(images_raw.squeeze().permute(1,2,0).cpu().numpy())
            # plt.savefig('data/images/test1.png')  
            
            activations, response = self.model.get_voxel_feature_maps(images, get_voxel_response=True)

            if args.analyze_depth:
                if self.use_zoe:
                    # assert(args.batch_size==1)
                    image_depth = images_raw.permute(0,2,3,1)[0].cpu().numpy()*255
                    image_depth = image_depth.astype(np.uint8)
                    image_depth = Image.fromarray(image_depth)
                    from zoedepth.utils.misc import pil_to_batched_tensor
                    image_depth = pil_to_batched_tensor(image_depth).to(device)
                    depth = self.zoe.infer(image_depth)
                    # plt.figure()
                    # plt.imshow(depth.squeeze().cpu().numpy())
                    # plt.colorbar()
                    # plt.savefig('data/images/test.png')
                    # plt.figure()
                    # plt.imshow(images_raw.squeeze().permute(1,2,0).cpu().numpy())
                    # plt.savefig('data/images/test1.png')                    
                    depth = torch.nn.functional.interpolate(
                                    depth,
                                    size=(args.image_size_eval, args.image_size_eval),
                                    mode="bicubic",
                                    align_corners=False,
                                ).squeeze(1)
                else:
                    image_depth = images_raw.permute(0,2,3,1).cpu().numpy()*255
                    image_depth = image_depth.astype(np.uint8)
                    input = []
                    for im in image_depth:
                        input.append(self.midas_transforms(im))
                    input = torch.cat(input, dim=0)
                    depth = self.midas(input.to(device))
                    depth = torch.nn.functional.interpolate(
                                    depth.unsqueeze(1),
                                    size=(args.image_size_eval, args.image_size_eval),
                                    mode="bicubic",
                                    align_corners=False,
                                ).squeeze(1)
                depths[i_batch*args.batch_size:(i_batch+1)*args.batch_size] = depth.cpu() #.append(depth.cpu())

            if self.valid_voxels is not None:
                activations = activations[:,self.valid_voxels]
                response = response[:,self.valid_voxels]
            
            if args.max_activation_plots is not None:
                activations = activations[:,:args.max_activation_plots]
                response = response[:,:args.max_activation_plots]

            if args.subsample_activations is not None:
                activations = activations[:,::args.subsample_activations]
                response = response[:,::args.subsample_activations]
            sample_set = tally.conditional_samples(upfn(activations), seg)
            
            for cond, sample in sample_set:
                # Move uncommon conditional data to the cpu before collating.
                if cond not in most_common_conditions:
                    sample = sample.cpu()
                cq.add(cond, sample)

            # Move uncommon conditions off the GPU.
            if i_batch and not i_batch & (i_batch - 1):  # if i is a power of 2:
                common_conditions = set(cq.most_common_conditions(gpu_cache))
                cq.to_('cpu', [k for k in cq.keys()
                        if k not in common_conditions])

            # images_raw = torch.nn.functional.interpolate(
            #         images_raw,
            #         mode='bilinear',
            #         size=128
            #         )
            # images_raw = images_raw * 255
            # images_raw = images_raw.to(torch.uint8)
            # acts[i_batch*args.batch_size:(i_batch+1)*args.batch_size] = activations.cpu() #.append(activations.cpu())
            # images_vis[i_batch*args.batch_size:(i_batch+1)*args.batch_size] = images_raw.cpu() #.append(images_raw.cpu())
            # segs[i_batch*args.batch_size:(i_batch+1)*args.batch_size] = seg.cpu() #.append(seg.cpu())
            # responses[i_batch*args.batch_size:(i_batch+1)*args.batch_size] = response.cpu() #.append(response.cpu())

            if args.analyze_depth or not args.reduced_eval_memory:
                acts[i_batch*args.batch_size:(i_batch+1)*args.batch_size] = activations.cpu() 
            if not args.reduced_eval_memory:
                segs[i_batch*args.batch_size:(i_batch+1)*args.batch_size] = seg.cpu() 
                images_raw = torch.nn.functional.interpolate(
                        images_raw,
                        mode='bilinear',
                        size=128
                        )
                images_raw = images_raw * 255
                images_raw = images_raw.to(torch.uint8)
                images_vis[i_batch*args.batch_size:(i_batch+1)*args.batch_size] = images_raw.cpu() 
                responses[i_batch*args.batch_size:(i_batch+1)*args.batch_size] = response.cpu()

            if args.max_images is not None:
                if (i_batch+1)*args.batch_size>=num_images-1:
                    break

        cq.to_('cpu')

        if args.save_dissection_samples:
            save_dict = {}
            if not args.reduced_eval_memory:
                save_dict["images_vis"] = images_vis
                save_dict["segs"] = segs
                save_dict["responses"] = responses
            if args.analyze_depth or not args.reduced_eval_memory:
                save_dict["responses"] = responses
            if args.analyze_depth:
                save_dict["acts"] = acts
            root, model_name = os.path.split(args.load_model_path)
            _, root_model = os.path.split(root)

            os.makedirs(os.path.join(args.tmp_dir, 'dissect', root_model), exist_ok = True)

            subj_add = '_subject'+str(args.eval_subject) if args.eval_subject is not None else ''
            
            for k in save_dict.keys():
                print(f"Saving dissection info {k}...")
                dissect_path = os.path.join(args.tmp_dir, 'dissect', root_model, f"{model_name.split('.pth')[0]}_{args.mode}_dissection_{k}{subj_add}.npy")
                np.save(dissect_path, save_dict[k].cpu().numpy())
            dat = cq.state_dict()
            cachefile = os.path.join(args.tmp_dir, 'dissect', root_model, f"{model_name.split('.pth')[0]}_{args.mode}_dissection_cq{subj_add}.npz")
            np.savez(cachefile, **dat)
            cachefile = os.path.join(args.tmp_dir, 'dissect', root_model, f"{model_name.split('.pth')[0]}_{args.mode}_dissection_params{subj_add}.npz")
            dat = {}
            dat["valid_voxels"] = self.valid_voxels
            dat["max_activation_plots"] = args.max_activation_plots
            dat["subsample_activations"] = args.subsample_activations
            dat["eval_subject"] = args.eval_subject
            dat["seglabels"] = seglabels
            dat["segcatlabels"] = segcatlabels
            np.savez(cachefile, **dat)

        self.run_analysis(acts, responses, images_vis, segs, iv, seglabels, segcatlabels, cq, depths=depths, rois=self.rois)

    def init_dataloaders(self, args):

        transform = torchvision.transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            # transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        transform_croponly = transforms.Compose([
                    transforms.Resize((args.image_size, args.image_size)),
                    # transforms.CenterCrop(args.image_size),
                    transforms.ToTensor(),
                ])

        print("Getting dataloader...")
        dataset = NSDDataloader(args, args.split, transform=transform, transform2=transform_croponly, fetch_metadata=False)
        self.seglabels = dataset.seglabels
        self.segcatlabels = dataset.segcatlabels
        dataset_loader = torch.utils.data.DataLoader(dataset,
                                                    batch_size=args.batch_size, 
                                                    # sampler=sampler,
                                                    num_workers=args.num_workers, 
                                                    # collate_fn=my_collate,
                                                    pin_memory=True,
                                                    drop_last=False
                                                    )
        self.dataset_loader = dataset_loader
        self.dataset = dataset
        print("Size dataloader:", len(self.dataset_loader))

if __name__ == '__main__':
    Ai2Thor() 