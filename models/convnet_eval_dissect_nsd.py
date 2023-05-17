from arguments import args
import torch
import numpy as np
from torchvision import transforms
import torch.nn.functional as F
import torchvision
from tqdm import tqdm
import pickle
import os

if args.mode=="convnet_nsd_eval_baudissect":
    from backend.dataloaders.nsd_image_loader import NSDDataloader
elif args.mode=="convnet_places_eval_baudissect" or args.mode=="convnet_xtc_eval_baudissect": # or args.mode=="convnet_unitvisual":
    from backend.dataloaders.places_image_loader import NSDDataloader
else:
    assert(False) # wrong mode

if args.arch=="e2cnn":
    print("Using E(2) CNN")
    from nets.convnet_e2cnn import ConvNet
elif args.arch=="cnn_alt":
    print("Using CNN ALT")
    from nets.convnet_alt import ConvNet
else:
    assert(False) # wrong arch
import utils.dist

import wandb
from utils.improc import MetricLogger
from backend import saverloader

import os
import cv2
import random
import colorsys

from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from PIL import Image

from utils.nsd_utils import get_roi_config, extract_single_roi
import cortex

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

import ipdb
st = ipdb.set_trace
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from dissect.netdissect import nethook, imgviz, show, segmenter, renormalize, upsample, tally, pbar, runningstats
from dissect.netdissect import setting

torch.backends.cudnn.benchmark = True
torch.set_grad_enabled(False) # not training anything!

# fix the seed for reproducibility
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

class Eval():
    def __init__(self):  

        print(f"subject is {args.eval_subject}")

        if utils.dist.is_main_process():
            # initialize wandb
            if args.set_name=="test00":
                wandb.init(mode="disabled")
            else:
                wandb.init(project="optimize_response_V1", group=args.group, name=args.set_name, config=args, dir=args.wandb_directory)

            args.images_path = os.path.join(args.images_path, args.set_name)

        self.init_dataloaders(args)

        self.save_images = False
        
        if args.total_voxel_size is None:
            checkpoint = torch.load(args.load_model_path, map_location=device)
            if 'W_spatial' in checkpoint['model_state_dict'].keys():
                self.total_voxel_size = checkpoint['model_state_dict']['W_spatial'].shape[1]
            else:
                self.total_voxel_size = checkpoint['model_state_dict']['readout.spatial'].shape[0]
            self.rois = checkpoint["rois"]
            self.nc_threshold = checkpoint["nc_threshold"]
        else:
            self.total_voxel_size = args.total_voxel_size
        print(f"Total voxel size is {self.total_voxel_size}")

        self.W = args.image_size_eval
        self.H = args.image_size_eval
        model = ConvNet(self.total_voxel_size, args.image_size, args.image_size)
        self.model = model
        self.model.eval()
        self.act_W, self.act_H = self.model.get_last_activation_sizes()

        if args.load_model:
            path = args.load_model_path 

            _, _ = saverloader.load_from_path(
                    path, 
                    self.model, 
                    None, 
                    strict=(not args.load_strict_false), 
                    lr_scheduler=None,
                    device=device,
                    )

        if args.analyze_depth:
            self.use_zoe = True
            if self.use_zoe:
                repo = "isl-org/ZoeDepth"
                self.zoe = torch.hub.load(repo, "ZoeD_NK", pretrained=True)
                self.zoe.to(device) 
                self.zoe.eval()
            else:
                model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
                #model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
                #model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)
                self.midas = torch.hub.load("intel-isl/MiDaS", model_type, pretrained=True).eval()
                self.midas.to(device) 
                self.midas.eval()
                midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
                if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
                    self.midas_transforms = midas_transforms.dpt_transform
                else:
                    self.midas_transforms = midas_transforms.small_transform

        self.model.to(device)
        
        self.model.eval()
        
        # (for debugging) if we want we can keep voxels above a correlation value or keep topk voxels by correlation with nsd test images
        if args.min_test_corr is not None or args.topk_units_from_corr is not None:
            if args.load_model:
                root, model_name = os.path.split(args.load_model_path)
                corr_path = os.path.join(root, model_name.split('.pth')[0] + '_corr_nsd.npy')
                if os.path.exists(corr_path):
                    corrs_test = np.load(corr_path)
                    self.valid_voxels = np.ones_like(corrs_test).astype(bool)
                    if args.min_test_corr is not None:
                        self.valid_voxels = np.logical_and(corrs_test>args.min_test_corr, self.valid_voxels)
                    if args.topk_units_from_corr is not None:
                        argsort_corrs_test = np.argsort(-corrs_test)
                        voxels_to_keep = []
                        for idx in range(len(argsort_corrs_test)):                            
                            if self.valid_voxels[idx]:
                                voxels_to_keep.append(argsort_corrs_test[idx])
                            if len(voxels_to_keep)==args.topk_units_from_corr:
                                break
                        valid_voxels_ = np.zeros_like(self.valid_voxels)
                        valid_voxels_[np.asarray(voxels_to_keep)] = True
                        self.valid_voxels = valid_voxels_
                else:
                    assert(False) # Run get_correlation_test.py
        elif args.eval_subject is not None:
            # get voxels for just one subject
            self.roi_sizes = checkpoint['roi_sizes']
            self.valid_voxels = np.ones(sum(self.roi_sizes.values())).astype(bool)
            start_index = 0
            for subj in self.roi_sizes.keys():
                roi_size_subj = self.roi_sizes[subj]
                if subj==args.eval_subject:
                    self.valid_voxels[start_index:start_index+roi_size_subj] = True
                else:
                    self.valid_voxels[start_index:start_index+roi_size_subj] = False
                start_index += roi_size_subj
            if np.sum(self.valid_voxels)==0:
                assert(False) # no valid voxels for this subjects?
        else:
            self.valid_voxels = None
            

        self.global_step = 0

        
    
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

        if args.mode=="convnet_ade20k_eval_baudissect":
            seglabels = self.dataset.index_ade20k['objectnames']
            seglabels.insert(0, 'background')
        else:
            # segmodel, seglabels, segcatlabels = setting.load_segmenter('netpq')
            segmodel, seglabels, segcatlabels = setting.load_segmenter('netpqc')
            assert(224 % args.image_size_eval == 0) # must be divisable by 256
            downsample = 224//args.image_size_eval
            
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
                # coco_ids[i_batch*b_tmp:(i_batch+1)*b_tmp] = coco_id
            responses_ = responses_.numpy()
            self.dataset_loader.dataset.filter_topk_images_from_roi_brain_response(responses_, topk=args.topk_filter)
            print("NEW size dataloader:", len(self.dataset_loader))
            del responses_
        elif args.filter_images_by_responses and args.debug:
            responses_ = np.ones((len(self.dataset_loader), num_voxels_keep))
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
            segs = torch.zeros((num_images, 6, args.image_size_eval, args.image_size_eval), dtype=torch.int64) 
            responses = torch.zeros((num_images, num_voxels_keep), dtype=torch.float32) 
        if args.analyze_depth:
            depths = torch.zeros((num_images, args.image_size_eval, args.image_size_eval), dtype=torch.float32)
        if args.analyze_depth or not args.reduced_eval_memory:
            acts = torch.zeros((num_images, num_voxels_keep, self.act_W, self.act_H), dtype=torch.float32)
        for i_batch, batched_samples in enumerate(metric_logger.log_every(self.dataset_loader, 10, header)):

            images = batched_samples['images'].to(device)
            images_raw = batched_samples['images2'].to(device)
            
            activations, response = self.model.get_voxel_feature_maps(images, get_voxel_response=True, use_spatial_mask=args.use_spatial_mask)
            if 'seg' in batched_samples.keys():
                seg = batched_samples['seg'].to(device)
                seg = torch.nn.functional.interpolate(
                    seg,
                    size=(args.image_size_eval, args.image_size_eval),
                    mode='nearest',
                    )
                seg = seg.long()
            else:
                images_seg = renorm_seg(images_raw) # pytorch -> zero centered
                seg = segmodel.segment_batch(images_seg, downsample=downsample)
            if args.analyze_depth:
                if self.use_zoe:
                    image_depth = images_raw.permute(0,2,3,1)[0].cpu().numpy()*255
                    image_depth = image_depth.astype(np.uint8)
                    image_depth = Image.fromarray(image_depth)
                    from zoedepth.utils.misc import pil_to_batched_tensor
                    image_depth = pil_to_batched_tensor(image_depth).to(device)
                    depth = self.zoe.infer(image_depth)               
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
                save_dict["depths"] = depths
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

    def run_analysis(self, acts, responses, images_vis, segs, iv, seglabels, segcatlabels, cq, depths=None, rois=None):
        
        if args.analyze_depth:
            self.plot_depth(depths, acts, cache=False if args.debug else True)

        # get IOUs
        print(f"Getting IOUs for threshold {args.activation_threshold}...")
        iou_table = tally.iou_from_conditional_quantile(cq, cutoff=args.activation_threshold)
        if iou_table.shape[1]<len(seglabels):
            iou_table = F.pad(iou_table, (0,len(seglabels)-iou_table.shape[1]))
        unit_list = sorted(enumerate(zip(*iou_table.max(1))), key=lambda k: -k[1][0])
        units_argsort = np.argsort(-np.max(iou_table.numpy(), axis=1))
        median_ious = np.median(iou_table, 0)
        argmedian_concept = np.argsort(-median_ious)
        topk_plot = 20
        topk_medians = argmedian_concept[:topk_plot]

        # # get thresholded IOU
        iou_threshold = 0.02
        iou_table_max = np.zeros_like(iou_table)
        argmax_table = np.argmax(iou_table, axis=1)
        # take max concept only if there are multiple
        for idx in range(len(argmax_table)):
            iou_table_max[idx, argmax_table[idx]] = iou_table[idx, argmax_table[idx]]
        where_units_selective = iou_table_max>iou_threshold
        num_units_concepts = where_units_selective.sum(0)
        argnum_units_concepts = np.argsort(-num_units_concepts)
        topk_selective = argnum_units_concepts[:topk_plot]

        # for meta categories
        labels = {"-":1,"object":1,"part":2,"material":3,"color":4,"relation":2,"attribute":3}
        meta_cats = np.asarray([labels[segcatlabels[idx][1]] for idx in range(len(segcatlabels))])
        iou_table_meta_max = np.zeros((iou_table.shape[0], 4))
        iou_table_meta_median = np.zeros(4)
        iou_table_numpy = iou_table.cpu().numpy()
        for idx in range(max(meta_cats)):
            where_meta_cat = meta_cats==idx+1
            iou_table_meta_max[:,idx] = np.max(iou_table_numpy[:,where_meta_cat], axis=1)
            iou_table_meta_median[idx] = np.mean(median_ious[where_meta_cat])
        iou_table_meta_maxmax = np.zeros_like(iou_table_meta_max)
        argmax_table = np.argmax(iou_table_meta_max, axis=1)
        # take max concept only if there are multiple
        for idx in range(len(argmax_table)):
            iou_table_meta_maxmax[idx, argmax_table[idx]] = iou_table_meta_max[idx, argmax_table[idx]]
        where_units_selective_meta = iou_table_meta_maxmax>iou_threshold
        num_units_concepts_meta = where_units_selective_meta.sum(0)

        cache=False if args.debug else True
        if cache:
            subj = args.eval_subject
            root, model_name = os.path.split(args.load_model_path)
            _, root_model = os.path.split(root)
            os.makedirs(os.path.join(args.tmp_dir, 'dissect', root_model), exist_ok = True)
            name = "iou_table"
            path = os.path.join(args.tmp_dir, 'dissect', root_model, f"roi_data_{model_name.split('.pth')[0]}_{''.join(self.rois)}_{name}_subject{subj}.npy")
            print(f"cacheing {name} to {path}..")
            np.save(path, iou_table)

            # save a few other thresholds
            for thresh in [0.96, 0.97, 0.98, 0.99]:
                print(f"Getting IOUs for threshold {thresh}...")
                iou_table_ = tally.iou_from_conditional_quantile(cq, cutoff=thresh)
                if iou_table_.shape[1]<len(seglabels):
                    iou_table_ = F.pad(iou_table_, (0,len(seglabels)-iou_table_.shape[1]))
                name = f"iou_table_threshold{thresh}"
                path = os.path.join(args.tmp_dir, 'dissect', root_model, f"roi_data_{model_name.split('.pth')[0]}_{''.join(self.rois)}_{name}_subject{subj}.npy")
                print(f"cacheing {name} to {path}..")
                np.save(path, iou_table_)

        self.plot_units_pycortex(iou_table, segcatlabels)

        self.plot_IOU(topk_selective, num_units_concepts, segcatlabels, topk_medians, median_ious, num_units_concepts_meta, iou_table_meta_median, iou_threshold=iou_threshold, rois=rois, cache=False if args.debug else True)

        if not args.reduced_eval_memory:
            # plot best units for each top concept (based on median IOU)
            top_units = min(5, iou_table_max.shape[0])
            top_images = 5
            for c_i in range(5):
                top_concept = int(argnum_units_concepts[c_i])
                concept_name = segcatlabels[top_concept][0]
                print(f"plotting concept {concept_name}...")
                # ious_concept = iou_table[:, top_concept]
                ious_concept = iou_table_max[:, top_concept]
                argsort_ious_concept = np.argsort(-ious_concept)
                images_to_plot = torch.zeros((top_units, top_images)).long()
                for i in range(top_units):
                    unit_idx = argsort_ious_concept[i]
                    # acts_responses_unit = responses[:,unit_idx]
                    acts_responses_unit = acts[:,unit_idx].view(acts.shape[0], -1).max(1)[0]
                    args_sort_responses = torch.argsort(acts_responses_unit, descending=True)
                    images_to_plot[i, :] = args_sort_responses[:top_images]
                self.plot_images(argsort_ious_concept[:top_units], images_to_plot, images_vis, acts, segs, iv, name=f"topk_concepts/{concept_name}")

            self.plot_topunits_topimages(units_argsort, images_vis, acts, segs, iv, responses, topk_images_from_acts=True, top_concepts_units=[segcatlabels[tc][0] for tc in list(np.argmax(iou_table_max, axis=1))])

        plt.close('all')


    def plot_images(self, units_to_plot, images_to_plot, images_vis, acts, segs, iv, name="images", return_only=False):
        '''
        Plots units and images espcified by units_to_plot and images_to_plot
        units_to_plot: length N specifying indices of units
        images_to_plot: length NxM specifying images to plot for each unit
        '''
        vis_list = [
            [
                [iv.masked_image(images_vis[imagenum]/255., acts[imagenum], unitnum) for imagenum in images_to_plot[idx]],
                [iv.heatmap(acts[imagenum], unitnum, mode='nearest') for imagenum in images_to_plot[idx]],
                [iv.segmentation(segs[imagenum][0]) for imagenum in images_to_plot[idx]],
                'unit %d' % unitnum
            ]
            for idx, unitnum in enumerate(units_to_plot)
        ]

        if return_only:
            return vis_list

        print("Plotting Images...")      
        width = int(len(units_to_plot))
        height = int(images_to_plot.shape[1])
        plot_num = 1
        plt.figure(figsize=(36,36), dpi=200)
        for unitnum in tqdm(range(len(units_to_plot)), leave=False):
            for imagenum in range(len(images_to_plot[unitnum])):
                masked_im = np.asarray(vis_list[unitnum][0][imagenum])
                heatmap = np.asarray(vis_list[unitnum][1][imagenum].convert('RGB'))
                seg_vis = np.asarray(vis_list[unitnum][2][imagenum])
                vis_im = np.concatenate([masked_im, heatmap, seg_vis], axis=1)
                plt.subplot(width, height, plot_num)
                plt.imshow(vis_im)
                plt.axis('off')
                plot_num += 1
        # name = f'topk_images/dissect_image{imagenum}'
        plt.title(f"threshold={args.activation_threshold}")
        wandb.log({name: wandb.Image(plt)})
        plt.close('all')

    def plot_units_pycortex(self, iou_table, segcatlabels):

        labels = {"-":1,"object":1,"part":2,"material":3,"color":4,"relation":2,"attribute":3}
        best_cat_per_unit = np.argmax(iou_table, axis=1)
        best_meta_per_unit = np.asarray([labels[segcatlabels[idx][1]] for idx in list(best_cat_per_unit)])

        self.plot_attribute_on_pycortex(best_cat_per_unit, subj=args.eval_subject, valid_mask=None, name="best_cat_per_unit", cache=False if args.debug else True)
        self.plot_attribute_on_pycortex(best_meta_per_unit, subj=args.eval_subject, valid_mask=None, name="best_meta_per_unit", cache=False if args.debug else True)


    def plot_IOU(self, topk_selective, num_units_concepts, segcatlabels, topk_medians, median_ious, num_units_concepts_meta, iou_table_meta_median, iou_threshold=0.04, rois=None, cache=False):
        width = 0.8
        names = [segcatlabels[idx][0] for idx in list(topk_selective)]
        plt.figure(1, figsize=(24,16), dpi=100); plt.clf()
        plt.bar(names, num_units_concepts[topk_selective], width, color="blue")
        plt.gca().yaxis.grid(True)
        plt.xticks(rotation=90)
        plt.xticks(fontsize=25)
        plt.title(f"Ranked IOU>{iou_threshold}; threshold={args.activation_threshold}; rois={rois}")
        wandb.log({f"Ranked IOU>{iou_threshold}; threshold={args.activation_threshold}":wandb.Image(plt)}) 
        plt.close()

        width = 0.8
        names = [segcatlabels[idx][0] for idx in list(topk_medians)]
        plt.figure(1, figsize=(24,16), dpi=100); plt.clf()
        plt.bar(names, median_ious[topk_medians], width, color="blue")
        plt.gca().yaxis.grid(True)
        plt.xticks(rotation=90)
        plt.xticks(fontsize=25)
        plt.title(f"Ranked Median IOU; threshold={args.activation_threshold}; rois={rois}")
        wandb.log({f"Ranked Median IOU; threshold={args.activation_threshold}":wandb.Image(plt)}) 
        plt.close()

        labels = {"object":1,"part":2,"material":3,"color":4}
        width = 0.8
        names = list(labels.keys()) #[segcatlabels[idx][1] for idx in list(topk_selective)]
        plt.figure(1, figsize=(24,16), dpi=100); plt.clf()
        plt.bar(names, num_units_concepts_meta, width, color="blue")
        plt.gca().yaxis.grid(True)
        plt.xticks(rotation=90)
        plt.xticks(fontsize=25)
        plt.title(f"Meta Ranked IOU>0.04; threshold={args.activation_threshold}; rois={rois}")
        wandb.log({f"Meta Ranked IOU>0.04; threshold={args.activation_threshold}":wandb.Image(plt)}) 
        plt.close()

        width = 0.8
        names = list(labels.keys()) #[segcatlabels[idx][1] for idx in list(topk_medians)]
        plt.figure(1, figsize=(24,16), dpi=100); plt.clf()
        plt.bar(names, iou_table_meta_median, width, color="blue")
        plt.gca().yaxis.grid(True)
        plt.xticks(rotation=90)
        plt.xticks(fontsize=25)
        plt.title(f"Meta Ranked Median IOU; threshold={args.activation_threshold}; rois={rois}")
        wandb.log({f"Meta Ranked Median IOU; threshold={args.activation_threshold}":wandb.Image(plt)}) 
        plt.close()

        if cache:
            '''
            cache data for later use
            '''
            subj = args.eval_subject
            root, model_name = os.path.split(args.load_model_path)
            _, root_model = os.path.split(root)
            os.makedirs(os.path.join(args.tmp_dir, 'dissect', root_model), exist_ok = True)

            def save_dict(root_model, model_name, name, subj):
                path = os.path.join(args.tmp_dir, 'dissect', root_model, f"roi_data_{model_name.split('.pth')[0]}_{''.join(self.rois)}_{name}_subject{subj}.p")
                print(f"cacheing {name} data to {path}..")
                with open(path, 'wb') as handle:
                    pickle.dump(dict_save, handle, protocol=pickle.HIGHEST_PROTOCOL)

            name = "CatRankedThresholdedIOU"
            dict_save = {
                "labels": [segcatlabels[idx][0] for idx in list(topk_selective)],
                "values": num_units_concepts[topk_selective],
                "iou_threshold": iou_threshold,
            }
            save_dict(root_model, model_name, name, subj)
            
            name = "CatRankedMedianIOU"
            dict_save = {
                "labels": [segcatlabels[idx][0] for idx in list(topk_medians)],
                "values": median_ious[topk_medians],
                "iou_threshold": iou_threshold,
            }
            save_dict(root_model, model_name, name, subj)

            name = "MetaRankedThresholdedIOU"
            dict_save = {
                "labels": list(labels.keys()),
                "values": num_units_concepts_meta,
                "iou_threshold": iou_threshold,
            }
            save_dict(root_model, model_name, name, subj)

            name = "MetaRankedMedianIOU"
            dict_save = {
                "labels": list(labels.keys()),
                "values": iou_table_meta_median,
                "iou_threshold": iou_threshold,
            }
            save_dict(root_model, model_name, name, subj)

    def plot_topunits_topimages(self, units_argsort, images_vis, acts, segs, iv, responses=None, topk_images_from_acts=True, top_concepts_units=None):

        topk_units_keep = args.topk 
        topk_units = list(units_argsort[:topk_units_keep])

        print("Getting Images...")
        topk_images_keep = args.topk 
        if topk_images_from_acts and responses is not None:
            images_topk = responses.mean(1)
            images_topk = torch.argsort(images_topk, descending=True)[:topk_images_keep]
        else:
            images_topk = list(np.arange(topk_images_keep))
        vis_list = [
            [
                [iv.masked_image(images_vis[imagenum]/255., acts[imagenum], unitnum) for imagenum in images_topk],
                [iv.heatmap(acts[imagenum], unitnum, mode='nearest') for imagenum in images_topk],
                [iv.segmentation(segs[imagenum][0]) for imagenum in images_topk],
                'unit %d' % unitnum
            ]
            for unitnum in topk_units
        ]
        
        print("Plotting topk_images images...")      
        width = int(np.sqrt(len(topk_units)))
        height = int(np.ceil(len(topk_units)/width))  
        for imagenum in tqdm(range(len(images_topk)), leave=False):
            plt.figure(figsize=(24,16), dpi=200)
            for unitnum in range(len(topk_units)):
                masked_im = np.asarray(vis_list[unitnum][0][imagenum])
                heatmap = np.asarray(vis_list[unitnum][1][imagenum].convert('RGB'))
                seg_vis = np.asarray(vis_list[unitnum][2][imagenum])
                vis_im = np.concatenate([masked_im, heatmap, seg_vis], axis=1)
                plt.subplot(width, height, unitnum+1)
                plt.imshow(vis_im)
                plt.axis('off')
            name = f'topk_images/dissect_image{imagenum}'
            plt.title(f"threshold={args.activation_threshold}")
            wandb.log({name: wandb.Image(plt)})
            plt.close('all')

        topk = runningstats.RunningTopK(k=args.topk)
        topk.add(responses)
        topk.to_('cpu')

        top_indexes = topk.result()[1]

        print("Plotting topk_units images...") 
        list_images = []
        for u in tqdm(list(topk_units), leave=False):
            plt.figure(figsize=(24,16), dpi=200)
            width = int(np.sqrt(topk_images_keep))
            height = int(np.ceil(topk_images_keep/width))
            list_images = [
                [np.asarray(iv.masked_image(images_vis[i]/255., acts[i], u)) for i in top_indexes[u, :topk_images_keep]],
                [np.asarray(iv.heatmap(acts[i], u, mode='nearest').convert('RGB')) for i in top_indexes[u, :topk_images_keep]]
            ]
            for idx in range(topk_images_keep):
                plt.subplot(width, height, idx+1)
                plt.imshow(np.concatenate([list_images[0][idx], list_images[1][idx]], axis=1))
                plt.axis('off')
            
            name = f'topk_units/dissect_unit{u}'
            if top_concepts_units is not None:
                name += f'_topconcept={top_concepts_units[u]}'
            plt.title(f"threshold={args.activation_threshold}")
            wandb.log({name: wandb.Image(plt)})
            plt.close('all')

    def plot_depth(self, depths, acts, cache=True, top_images=1000):
        '''
        Get depth selectivity of each unit
        '''

        depth_iv = imgviz.ImageVisualizer(depths.shape[-1], percent_level=args.activation_threshold, image_size=(self.W, self.H))
        rel_depth = depths.clone().flatten(1,2)
        rel_depth = (rel_depth - rel_depth.min(1, keepdim=True)[0]) / rel_depth.max(1, keepdim=True)[0]
        rel_depth = rel_depth.reshape(depths.shape[0], depths.shape[-2], depths.shape[-1])
        pseudo_abs_depth = depths.clone()
        
        median_rel_depths = []
        median_abs_depths = []
        median_diff_rel_depths = []
        median_diff_abs_depths = []

        valid_mask = []
        for u in tqdm(list(np.arange(acts.shape[1])), leave=False):
            median_rel_diff_depths_ = []
            median_abs_diff_depths_ = []
            rel_depths_ = []
            abs_depths_ = []
            acts_responses_unit = acts[:,u].view(acts.shape[0], -1).max(1)[0]
            args_sort_responses = torch.argsort(acts_responses_unit, descending=True).cpu().numpy()
            for im_i in list(args_sort_responses)[:top_images]:
                mask = depth_iv.pytorch_mask(acts[im_i].to(device), u)
                if torch.sum(mask)==0:
                    rel_depths_.append(torch.tensor(torch.nan))
                    abs_depths_.append(torch.tensor(torch.nan))
                    median_rel_diff_depths_.append(torch.tensor(torch.nan))
                    median_abs_diff_depths_.append(torch.tensor(torch.nan))
                    continue
                depth_rel_masked = rel_depth[im_i][mask]
                depth_abs_masked = pseudo_abs_depth[im_i][mask]
                rel_depths_.append(depth_rel_masked.median())
                abs_depths_.append(depth_abs_masked.median())
                median_rel_diff_depths_.append(depth_rel_masked.median() - rel_depth[im_i].flatten().median())
                median_abs_diff_depths_.append(depth_abs_masked.median() - pseudo_abs_depth[im_i].flatten().median())

            check = torch.stack(rel_depths_, dim=0)
            if torch.all(torch.isnan(check)):
                print(f"Skipping {u}")
                valid_mask.append(False)
                continue
            valid_mask.append(True)

            median_rel_depths.append(torch.stack(rel_depths_, dim=0))
            median_abs_depths.append(torch.stack(abs_depths_, dim=0))
            median_diff_rel_depths.append(torch.stack(median_rel_diff_depths_, dim=0))
            median_diff_abs_depths.append(torch.stack(median_abs_diff_depths_, dim=0))
        
        median_rel_depths = torch.stack(median_rel_depths, dim=0).cpu().numpy()
        median_abs_depths = torch.stack(median_abs_depths, dim=0).cpu().numpy()
        median_diff_rel_depths = torch.stack(median_diff_rel_depths, dim=0).cpu().numpy()
        median_diff_abs_depths = torch.stack(median_diff_abs_depths, dim=0).cpu().numpy()

        if cache:
            '''
            cache data for use later
            '''
            subj = args.eval_subject
            root, model_name = os.path.split(args.load_model_path)
            _, root_model = os.path.split(root)
            os.makedirs(os.path.join(args.tmp_dir, 'dissect', root_model), exist_ok = True)
            name = "median_rel_depths_allimages"
            path = os.path.join(args.tmp_dir, 'dissect', root_model, f"roi_data_{model_name.split('.pth')[0]}_{''.join(self.rois)}_{name}_subject{subj}.npy")
            print(f"cacheing {name} to {path}..")
            np.save(path, median_rel_depths)
            name = "median_abs_depths_allimages"
            path = os.path.join(args.tmp_dir, 'dissect', root_model, f"roi_data_{model_name.split('.pth')[0]}_{''.join(self.rois)}_{name}_subject{subj}.npy")
            print(f"cacheing {name} to {path}..")
            np.save(path, median_abs_depths)
            name = "median_diff_rel_depths_allimages"
            path = os.path.join(args.tmp_dir, 'dissect', root_model, f"roi_data_{model_name.split('.pth')[0]}_{''.join(self.rois)}_{name}_subject{subj}.npy")
            print(f"cacheing {name} to {path}..")
            np.save(path, median_diff_rel_depths)
            name = "median_diff_abs_depths_allimages"
            path = os.path.join(args.tmp_dir, 'dissect', root_model, f"roi_data_{model_name.split('.pth')[0]}_{''.join(self.rois)}_{name}_subject{subj}.npy")
            print(f"cacheing {name} to {path}..")
            np.save(path, median_diff_abs_depths)
            name = "valid_mask"
            path = os.path.join(args.tmp_dir, 'dissect', root_model, f"roi_data_{model_name.split('.pth')[0]}_{''.join(self.rois)}_{name}_subject{subj}.npy")
            print(f"cacheing {name} to {path}..")
            np.save(path, np.asarray(valid_mask))

        median_rel_depths_std = np.nanstd(median_rel_depths, axis=1)
        median_abs_depths_std = np.nanstd(median_abs_depths, axis=1) 
        median_diff_rel_depths_std = np.nanstd(median_diff_rel_depths, axis=1) 
        median_diff_abs_depths_std = np.nanstd(median_diff_abs_depths, axis=1) 

        median_rel_depths = np.nanmean(median_rel_depths, axis=1) 
        median_abs_depths = np.nanmean(median_abs_depths, axis=1) 
        median_diff_rel_depths = np.nanmean(median_diff_rel_depths, axis=1) 
        median_diff_abs_depths = np.nanmean(median_diff_abs_depths, axis=1) 

        plt.figure(1, figsize=(24,16), dpi=100); plt.clf()
        plt.plot(median_diff_rel_depths, np.ones(len(median_diff_rel_depths)), 'o', color='blue')
        bins = np.arange(-1,1+0.05,0.05)
        plt.hist(median_diff_rel_depths, 
                alpha=0.7, # the transaparency parameter
                label='diff median relative depths',
                bins=bins,
                )
        # plt.savefig('data/images/test1.png')
        wandb.log({f"depth_relative/MEDAINS_DIFF_relative_depth_allunits":wandb.Image(plt)}) 

        plt.figure(1, figsize=(24,16), dpi=100); plt.clf()
        plt.plot(median_diff_abs_depths, np.ones(len(median_diff_abs_depths)), 'o', color='blue')
        bins = np.arange(-10,10+0.05,0.05)
        plt.hist(median_diff_abs_depths, 
                alpha=0.7, # the transaparency parameter
                label='diff median absolute depths',
                bins=bins,
                )
        # plt.savefig('data/images/test.png')
        wandb.log({f"depth_absolute/MEDIANS_DIFF_absolute_depth_allunits":wandb.Image(plt)}) 

        plt.figure(1, figsize=(24,16), dpi=100); plt.clf()
        plt.plot(median_rel_depths, np.ones(len(median_diff_abs_depths)), 'o', color='blue')
        bins = np.arange(-1,1+0.05,0.05)
        plt.hist(median_rel_depths, 
                alpha=0.7, # the transaparency parameter
                label='median relative depths',
                bins=bins,
                )
        # plt.savefig('data/images/test.png')
        wandb.log({f"depth_relative/MEDIANS_relative_depth_allunits":wandb.Image(plt)}) 

        plt.figure(1, figsize=(24,16), dpi=100); plt.clf()
        plt.plot(median_abs_depths, np.ones(len(median_diff_abs_depths)), 'o', color='blue')
        bins = np.arange(-20,20+0.05,0.05)
        plt.hist(median_abs_depths, 
                alpha=0.7, # the transaparency parameter
                label='median absolute depths',
                bins=bins,
                )
        # plt.savefig('data/images/test.png')
        wandb.log({f"depth_absolute/MEDIANS_absolute_depth_allunits":wandb.Image(plt)}) 

        plt.close('all')

        if args.eval_subject is not None:
            self.plot_attribute_on_pycortex(median_diff_rel_depths, subj=args.eval_subject, valid_mask=valid_mask, name="median_diff_rel_depths", cache=cache)
            self.plot_attribute_on_pycortex(median_diff_abs_depths, subj=args.eval_subject, valid_mask=valid_mask, name="median_diff_abs_depths", cache=cache)
            self.plot_attribute_on_pycortex(median_rel_depths, subj=args.eval_subject, valid_mask=valid_mask, name="median_rel_depths", cache=cache)
            self.plot_attribute_on_pycortex(median_abs_depths, subj=args.eval_subject, valid_mask=valid_mask, name="median_abs_depths", cache=cache)

            self.plot_attribute_on_pycortex(median_diff_rel_depths_std, subj=args.eval_subject, valid_mask=valid_mask, name="median_diff_rel_depths_std", cache=cache)
            self.plot_attribute_on_pycortex(median_diff_abs_depths_std, subj=args.eval_subject, valid_mask=valid_mask, name="median_diff_abs_depths_std", cache=cache)
            self.plot_attribute_on_pycortex(median_rel_depths_std, subj=args.eval_subject, valid_mask=valid_mask, name="median_rel_depths_std", cache=cache)
            self.plot_attribute_on_pycortex(median_abs_depths_std, subj=args.eval_subject, valid_mask=valid_mask, name="median_abs_depths_std", cache=cache)

    def plot_attribute_on_pycortex(self, data, subj=1, valid_mask=None, name="", cache=False, load_mask_root=''):

        rois = self.rois

        init_roi = True
        meta_name_dict, roi_name_to_meta_name = get_roi_config()
        for roi in rois:
            meta_name = roi_name_to_meta_name[roi]
            output_masks, roi_labels = extract_single_roi(meta_name, meta_name_dict, args.roi_dir, subj)
            roi_ind = roi_labels.index(roi)
            roi_mask_ = output_masks[roi_ind] 
            if init_roi:
                roi_mask = roi_mask_
                init_roi = False
            else:
                roi_mask = np.logical_or(roi_mask, roi_mask_)

        if self.nc_threshold>0.:
            # get noise ceiling
            noise_ceiling_1d = np.load(
                "%s/subj%01d/noise_ceiling_percent_1d_subj%02d.npy"
                % (args.noise_ceiling_dir, subj, subj)
            )  
            # above is noise ceiling in percent
            # convert to fraction percent
            noise_ceiling_1d = noise_ceiling_1d / 100
            good = noise_ceiling_1d>self.nc_threshold
            roi_mask = np.logical_and(roi_mask, good)

        if valid_mask is not None:
            valid_mask = np.asarray(valid_mask)
            to_remove = np.where(roi_mask)[0][~valid_mask]
            roi_mask[to_remove] = False

        roi_data = np.zeros_like(roi_mask).astype(np.float32)
        roi_data[roi_mask] = data

        if cache:
            root, model_name = os.path.split(args.load_model_path)
            _, root_model = os.path.split(root)
            os.makedirs(os.path.join(args.tmp_dir, 'dissect', root_model), exist_ok = True)
            path = os.path.join(args.tmp_dir, 'dissect', root_model, f"roi_data_{model_name.split('.pth')[0]}_{''.join(self.rois)}_{name}_subject{subj}.npy")
            print(f"cacheing roi data to {path}..")
            np.save(path, roi_data)

        def load_mask(subj):
            mask = cortex.utils.get_cortical_mask(
                "subj%02d" % subj, "func1pt8_to_anat0pt8_autoFSbbr"
            )

            try:
                cortical_mask = np.load(
                    "%s/subj%d/cortical_mask_subj%02d.npy" % (load_mask_root, subj, subj)
                )
            except FileNotFoundError:
                cortical_mask = np.load(
                    "%s/subj%d/old/cortical_mask_subj%02d.npy" % (load_mask_root, subj, subj)
                )

            sig_mask = None

            return mask, cortical_mask, sig_mask


        def project_vals_to_3d(vals, mask):
            all_vals = np.zeros(mask.shape)
            all_vals[mask] = vals
            all_vals = np.swapaxes(all_vals, 0, 2)
            return all_vals

        roi_data[np.isnan(roi_data)] = 0.
        mask, cortical_mask, sig_mask = load_mask(subj)
        vals_3d = project_vals_to_3d(roi_data, cortical_mask)

        '''
        Note: you may have to adjust the upper & lower limits here
        '''

        if name in ["best_cat_per_unit", "best_meta_per_unit"]:
            cmap='nipy_spectral_r'
            if name=="best_cat_per_unit":
                vals_3d[vals_3d!=0] = vals_3d[vals_3d!=0] + 50
            upper = max(vals_3d[vals_3d!=0]) #int(np.quantile(roi_data, .995))
            lower = -upper/256
        elif name in ["median_diff_abs_depths", "median_diff_rel_depths", "median_rel_depths"]:
            
            cmap='J4s'
            if name=="median_diff_abs_depths":
                upper = np.quantile(roi_data, .995)
                lower = np.quantile(roi_data, .005)
            else:
                upper = np.quantile(roi_data, .9999)
                lower = np.quantile(roi_data, .0001)
            diff = upper - lower
            lower = lower - diff/6
            vals_3d[vals_3d==0] = lower + diff/64
        else:
            cmap='J4s'
            upper = int(np.quantile(roi_data, .995))
            lower = -upper/64

        roi_volume = cortex.Volume(
                vals_3d,
                "subj%02d" % subj,
                "func1pt8_to_anat0pt8_autoFSbbr",
                mask=mask,
                vmin=lower,
                vmax=upper, 
                cmap=cmap, 
            )

        _ = cortex.quickflat.make_figure(roi_volume, with_rois=False, with_colorbar=True)
        plt.title(f"plot_attribute")
        # plt.savefig(f'data/images/{name}_subject{subj}.png')
        wandb.log({f"pycortex/{name}_subject{subj}":wandb.Image(plt)}) 
        plt.close()


    def init_dataloaders(self, args):

        transform = torchvision.transforms.Compose([
            transforms.Resize(args.image_size+32),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        transform_croponly = transforms.Compose([
                    transforms.Resize(args.image_size+32),
                    transforms.CenterCrop(args.image_size),
                    transforms.ToTensor(),
                ])

        print("Getting dataloader...")
        dataset = NSDDataloader(args, args.split, transform=transform, transform2=transform_croponly)
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


def NormalizeDataPycortex(data, lower=30, upper=70, lower_prev=-1, upper_prev=1):
    data = (data - lower_prev) / (upper_prev - lower_prev)
    data = lower + (upper - lower) * data
    return data

def apply_mask(image, mask, color, alpha=0.5):
    for c in range(3):
        image[:, :, c] = image[:, :, c] * (1 - alpha * mask) + alpha * mask * color[c] * 255
    return image


def random_colors(N, bright=True):
    """
    Generate random colors.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def display_instances(image, mask, fname="test", figsize=(5, 5), blur=False, contour=True, alpha=0.5, just_return=False):
    # if not just_return:
    fig = plt.figure(figsize=figsize, frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax = plt.gca()

    N = 1
    mask = mask[None, :, :]
    # Generate random colors
    colors = random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    margin = 0
    ax.set_ylim(height + margin, -margin)
    ax.set_xlim(-margin, width + margin)
    ax.axis('off')
    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]
        _mask = mask[i]
        if blur:
            _mask = cv2.blur(_mask,(10,10))
        # Mask
        masked_image = apply_mask(masked_image, _mask, color, alpha)
        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        if contour:
            padded_mask = np.zeros((_mask.shape[0] + 2, _mask.shape[1] + 2))
            padded_mask[1:-1, 1:-1] = _mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                p = Polygon(verts, facecolor="none", edgecolor=color)
                ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8), aspect='auto')
    if just_return:
        canvas = FigureCanvas(plt.gcf())
        canvas.draw()       # draw the canvas, cache the renderer
        width, height = plt.gcf().get_size_inches() * plt.gcf().get_dpi()
        image = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
        return image
    fig.savefig(fname)
    print(f"{fname} saved.")
    return image