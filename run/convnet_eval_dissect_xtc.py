from arguments import args
import torch
import numpy as np
from torchvision import transforms
import torch.nn.functional as F
import torchvision
from tqdm import tqdm
import os
if args.mode=="convnet_nsd_eval_baudissect":
    from backend.dataloaders.nsd_image_loader import NSDDataloader
elif args.mode=="convnet_places_eval_baudissect":
    from backend.dataloaders.places_image_loader import NSDDataloader
if args.arch=="e2cnn":
    print("Using E(2) CNN")
    from nets.convnet_e2cnn import ConvNet
elif args.arch=="cnn_alt":
    print("Using CNN ALT")
    from nets.convnet_alt import ConvNet
else:
    assert(False) # wrong arch
import wandb
from utils.improc import MetricLogger
import copy

import os
import sys
import cv2
import random
import colorsys

from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from PIL import Image

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

import ipdb
st = ipdb.set_trace
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sys.path.append('dissect')
from dissect.netdissect import nethook, imgviz, show, segmenter, renormalize, upsample, tally, pbar, runningstats
from dissect.experiment import setting

from run.convnet_eval_dissect_nsd import Eval
'''
NOTE: Eval_XTC inherits from run.convnet_eval_baudissect_nsd
'''

torch.backends.cudnn.benchmark = True
torch.set_grad_enabled(False) # not training anything!

# fix the seed for reproducibility
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

# for using XTC, please download the repo and model weights from https://github.com/EPFL-VILAB/XTConsistency
sys.path.append('XTConsistency')
from modules.unet import UNet, UNetReshade
import visualpriors

class Eval_XTC(Eval):
    def __init__(self):  

        super(Eval_XTC, self).__init__()

        map_location = (lambda storage, loc: storage.cuda()) if torch.cuda.is_available() else torch.device('cpu')

        self.normal_model = UNet() 
        normals_path = os.path.join(args.xtc_checkpoint_paths,'rgb2'+'normal'+'_'+'consistency'+'.pth')
        model_state_dict = torch.load(normals_path, map_location=map_location)
        self.normal_model.load_state_dict(model_state_dict)
        self.normal_model.to(device).eval()

        self.reshading_model = UNetReshade(downsample=5)
        reshading_path = os.path.join(args.xtc_checkpoint_paths,'rgb2'+'reshading'+'_'+'consistency'+'.pth')
        model_state_dict = torch.load(reshading_path, map_location=map_location)
        self.reshading_model.load_state_dict(model_state_dict)
        self.reshading_model.to(device).eval()

        self.plot_for_figure = args.plot_for_figure
    
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
        segs = None
        depths = None

        print(f"Image size eval: {args.image_size_eval}")
        loaded=False
        if args.load_dissection_samples:
            '''
            load variables from memory if they are saved
            '''
            root, model_name = os.path.split(args.load_model_path)
            _, root_model = os.path.split(root)
            subj_add = '_subject'+str(args.eval_subject) if args.eval_subject is not None else ''
            tmp_dir_ = args.tmp_dir_load if args.tmp_dir_load is not None else args.tmp_dir # can specify two tmp dirs
            if not os.path.exists(os.path.join(f"{tmp_dir_}", 'dissect', root_model, f"{model_name.split('.pth')[0]}_{args.mode2 if args.mode2 is not None else args.mode}_dissection_acts{subj_add}.npy")):
                tmp_dir_ = args.tmp_dir
            if os.path.exists(os.path.join(f"{tmp_dir_}", 'dissect', root_model, f"{model_name.split('.pth')[0]}_{args.mode2 if args.mode2 is not None else args.mode}_dissection_acts{subj_add}.npy")):
                acts = torch.from_numpy(np.load(os.path.join(f"{tmp_dir_}", 'dissect', root_model, f"{model_name.split('.pth')[0]}_{args.mode2 if args.mode2 is not None else args.mode}_dissection_acts{subj_add}.npy")))
                if not args.reduced_eval_memory:
                    responses = torch.from_numpy(np.load(os.path.join(f"{tmp_dir_}", 'dissect', root_model, f"{model_name.split('.pth')[0]}_{args.mode2 if args.mode2 is not None else args.mode}_dissection_responses{subj_add}.npy")))
                    images_vis = torch.from_numpy(np.load(os.path.join(f"{tmp_dir_}", 'dissect', root_model, f"{model_name.split('.pth')[0]}_{args.mode2 if args.mode2 is not None else args.mode}_dissection_images_vis{subj_add}.npy")))
                    segs = torch.from_numpy(np.load(os.path.join(f"{tmp_dir_}", 'dissect', root_model, f"{model_name.split('.pth')[0]}_{args.mode2 if args.mode2 is not None else args.mode}_dissection_segs{subj_add}.npy")))
                if args.analyze_depth:
                    depths = torch.from_numpy(np.load(os.path.join(f"{tmp_dir_}", 'dissect', root_model, f"{model_name.split('.pth')[0]}_{args.mode2 if args.mode2 is not None else args.mode}_dissection_depths{subj_add}.npy")))
                cachefile = os.path.join(f"{tmp_dir_}", 'dissect', root_model, f"{model_name.split('.pth')[0]}_{args.mode2 if args.mode2 is not None else args.mode}_dissection_cq{subj_add}.npz")
                cq_state_dict = np.load(cachefile, allow_pickle=True)
                cq.set_state_dict(cq_state_dict)
                cachefile = os.path.join(f"{tmp_dir_}", 'dissect', root_model, f"{model_name.split('.pth')[0]}_{args.mode2 if args.mode2 is not None else args.mode}_dissection_params{subj_add}.npz")
                dat = np.load(cachefile, allow_pickle=True)
                seglabels = dat["seglabels"] 
                segcatlabels = dat["segcatlabels"]
                print('segmenter has', len(seglabels), 'labels')

                if os.path.exists(os.path.join(args.tmp_dir, 'dissect', root_model, f"{model_name.split('.pth')[0]}_{args.mode2 if args.mode2 is not None else args.mode}_dissection_normals{subj_add}.npy")) and not self.plot_for_figure:
                    normals = torch.from_numpy(np.load(os.path.join(args.tmp_dir, 'dissect', root_model, f"{model_name.split('.pth')[0]}_{args.mode2 if args.mode2 is not None else args.mode}_dissection_normals{subj_add}.npy")))
                    reshadings = torch.from_numpy(np.load(os.path.join(args.tmp_dir, 'dissect', root_model, f"{model_name.split('.pth')[0]}_{args.mode2 if args.mode2 is not None else args.mode}_dissection_reshadings{subj_add}.npy")))
                    principal_curvatures = torch.from_numpy(np.load(os.path.join(args.tmp_dir, 'dissect', root_model, f"{model_name.split('.pth')[0]}_{args.mode2 if args.mode2 is not None else args.mode}_dissection_principal_curvatures{subj_add}.npy")))
                    self.run_analysis_xtc(acts, responses, images_vis, segs, iv, seglabels, segcatlabels, cq, normals=normals, reshadings=reshadings, principal_curvatures=principal_curvatures, depths=depths, rois=self.rois)
                    return
                else:
                    loaded=True

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
            '''
            Get top images by predicted response
            '''
            b_tmp = 16
            dataset_loader = torch.utils.data.DataLoader(self.dataset,
                                    batch_size=b_tmp, 
                                    num_workers=2, 
                                    pin_memory=True,
                                    drop_last=False
                                    )
            metric_logger = MetricLogger(delimiter="  ")
            header = f'{args.set_name}|{args.mode}|Getting all responses...'
            if self.plot_for_figure:
                responses_ = torch.zeros((num_images, len(self.valid_voxels)), dtype=torch.float32)
            else:
                responses_ = torch.zeros((num_images, num_voxels_keep), dtype=torch.float32)
            for i_batch, batched_samples in enumerate(metric_logger.log_every(dataset_loader, 10, header)):
                images = batched_samples['images'].to(device)
                images_raw = batched_samples['images2'].to(device)
                response = self.model(images)
                if self.valid_voxels is not None and not self.plot_for_figure:
                    response = response[:,self.valid_voxels]
                if args.max_activation_plots is not None:
                    response = response[:,:args.max_activation_plots]
                if args.subsample_activations is not None:
                    response = response[:,::args.subsample_activations]
                responses_[i_batch*b_tmp:(i_batch+1)*b_tmp] = response.cpu()
            responses_ = responses_.numpy()
            self.dataset_loader.dataset.filter_topk_images_from_roi_brain_response(responses_, topk=args.topk_filter)
            print("NEW size dataloader:", len(self.dataset_loader))

        num_images = args.max_images if args.max_images is not None else len(self.dataset_loader)*args.batch_size
        metric_logger = MetricLogger(delimiter="  ")
        header = f'{args.set_name}|{args.mode}|{args.split}&{True if args.coco_ids_path is not None else False}|num_voxels:{num_voxels_keep}|depth?{args.analyze_depth}|subject{args.eval_subject}'
        
        if not loaded:
            # preallocate
            print(f"num images: {num_images}; num voxels: {num_voxels_keep}")
            if not args.reduced_eval_memory:
                images_vis = torch.zeros((num_images, 3, 128, 128), dtype=torch.uint8) 
                segs = torch.zeros((num_images, 6, args.image_size_eval, args.image_size_eval), dtype=torch.int64) 
                responses = torch.zeros((num_images, num_voxels_keep), dtype=torch.float32) 
            if args.analyze_depth:
                depths = torch.zeros((num_images, args.image_size_eval, args.image_size_eval), dtype=torch.float32)
            acts = torch.zeros((num_images, num_voxels_keep, self.act_W, self.act_H), dtype=torch.float32)

        normals = torch.zeros((num_images, 3, args.image_size_eval, args.image_size_eval), dtype=torch.float32)
        reshadings = torch.zeros((num_images, args.image_size_eval, args.image_size_eval), dtype=torch.float32)
        principal_curvatures = torch.zeros((num_images, 2, args.image_size_eval, args.image_size_eval), dtype=torch.float32)
        num_images_figure = 50
        units = [[] for _ in range(args.figure_unit_end-args.figure_unit_start)]
        for i_batch, batched_samples in enumerate(metric_logger.log_every(self.dataset_loader, 10, header)):

            # print(f"Start batch {i_batch}..")

            images = batched_samples['images'].to(device)
            images_raw = batched_samples['images2'].to(device)

            transform = torchvision.transforms.Compose([
                transforms.Resize((256, 256)),
            ])
            images_xtc = transform(images_raw)

            normal = self.normal_model(images_xtc)
            reshading = self.reshading_model(images_xtc)

            # # Transform to principal_curvature feature
            # Transform to principal_curvature feature and then visualize the readout
            principal_curvature = visualpriors.feature_readout(images_xtc * 2 - 1, 'curvature', device=device) / 2. + 0.5
            
            reshading = reshading[:,0:1]
            principal_curvature = principal_curvature[:,:2]

            normal_ds = torch.nn.functional.interpolate(
                                        normal,
                                        size=(args.image_size_eval, args.image_size_eval),
                                        mode="bicubic",
                                        align_corners=False,
                                    ).squeeze(1).clamp(min=0, max=1)
            reshading_ds = torch.nn.functional.interpolate(
                                        reshading,
                                        size=(args.image_size_eval, args.image_size_eval),
                                        mode="bicubic",
                                        align_corners=False,
                                    ).squeeze(1).clamp(min=0, max=1)
            principal_curvature_ds = torch.nn.functional.interpolate(
                                        principal_curvature,
                                        size=(args.image_size_eval, args.image_size_eval),
                                        mode="bicubic",
                                        align_corners=False,
                                    ).squeeze(1).clamp(min=0, max=1)

            normals[i_batch*args.batch_size:(i_batch+1)*args.batch_size] = normal_ds.cpu() 
            reshadings[i_batch*args.batch_size:(i_batch+1)*args.batch_size] = reshading_ds.cpu() 
            principal_curvatures[i_batch*args.batch_size:(i_batch+1)*args.batch_size] = principal_curvature_ds.cpu() 
            
            if not loaded:
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
                        # assert(args.batch_size==1)
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
                    depths[i_batch*args.batch_size:(i_batch+1)*args.batch_size] = depth.cpu() 

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

        if self.plot_for_figure:
            for unit_i in range(len(units)):
                to_plot = units[unit_i]
                image_to_plot = np.concatenate(to_plot, axis=1)
                wandb.log({f"XTC/top_images":wandb.Image(image_to_plot)}) 
            assert(False)

        cq.to_('cpu')

        if args.save_dissection_samples:
            '''
            cache dissection variables
            '''
            save_dict = {}
            if not loaded:
                if not args.reduced_eval_memory:
                    save_dict["images_vis"] = images_vis
                    save_dict["segs"] = segs
                    save_dict["responses"] = responses
                if args.analyze_depth or not args.reduced_eval_memory:
                    save_dict["depths"] = depths
                if args.analyze_depth:
                    save_dict["acts"] = acts
            save_dict["normals"] = normals
            save_dict["reshadings"] = reshadings
            save_dict["principal_curvatures"] = principal_curvatures
            root, model_name = os.path.split(args.load_model_path)
            _, root_model = os.path.split(root)

            os.makedirs(os.path.join(args.tmp_dir, 'dissect', root_model), exist_ok = True)

            subj_add = '_subject'+str(args.eval_subject) if args.eval_subject is not None else ''
            
            for k in save_dict.keys():
                print(f"Saving dissection info {k}...")
                dissect_path = os.path.join(args.tmp_dir, 'dissect', root_model, f"{model_name.split('.pth')[0]}_{args.mode2 if args.mode2 is not None else args.mode}_dissection_{k}{subj_add}.npy")
                np.save(dissect_path, save_dict[k].cpu().numpy())

            if not loaded:
                dat = cq.state_dict()
                cachefile = os.path.join(args.tmp_dir, 'dissect', root_model, f"{model_name.split('.pth')[0]}_{args.mode2 if args.mode2 is not None else args.mode}_dissection_cq{subj_add}.npz")
                np.savez(cachefile, **dat)
                cachefile = os.path.join(args.tmp_dir, 'dissect', root_model, f"{model_name.split('.pth')[0]}_{args.mode2 if args.mode2 is not None else args.mode}_dissection_params{subj_add}.npz")
                dat = {}
                dat["valid_voxels"] = self.valid_voxels
                dat["max_activation_plots"] = args.max_activation_plots
                dat["subsample_activations"] = args.subsample_activations
                dat["eval_subject"] = args.eval_subject
                dat["seglabels"] = seglabels
                dat["segcatlabels"] = segcatlabels
                np.savez(cachefile, **dat)

        self.run_analysis_xtc(acts, responses, images_vis, segs, iv, seglabels, segcatlabels, cq, normals=normals, reshadings=reshadings, principal_curvatures=principal_curvatures, depths=depths, rois=self.rois)

    def run_analysis_xtc(self, acts, responses, images_vis, segs, iv, seglabels, segcatlabels, cq, normals=None, reshadings=None, principal_curvatures=None, depths=None, rois=None, run_original_analysis=True):
        
        self.plot_xtc(normals, reshadings, principal_curvatures, acts, cache=False if args.debug else True)
        
        if run_original_analysis:
            self.run_analysis(acts, responses, images_vis, segs, iv, seglabels, segcatlabels, cq, depths=depths, rois=self.rois)

        plt.close('all')

    def plot_xtc(self, normals, reshadings, principal_curvatures, acts, cache=True):
        '''
        Get surface normals, reshading, principal curvature selectivity of each unit
        '''

        B = normals.shape[0]

        xtc_iv = imgviz.ImageVisualizer(normals.shape[-1], percent_level=args.activation_threshold, image_size=(self.W, self.H))

        normals = normals.flatten(2,3)

        principal_curvatures = principal_curvatures.flatten(2,3)
        guassian_curvatures = principal_curvatures[:,0] * principal_curvatures[:,1]
        mean_curvatures = (principal_curvatures[:,0] + principal_curvatures[:,1]) / 2
        guassian_curvatures = guassian_curvatures.reshape(B, self.W, self.H)
        mean_curvatures = mean_curvatures.reshape(B, self.W, self.H)
        principal_curvatures1 = principal_curvatures[:,0].reshape(B, self.W, self.H)
        principal_curvatures2 = principal_curvatures[:,1].reshape(B, self.W, self.H)

        # bin normals by x,y,z components
        normals = np.transpose(normals, (0,2,1)).flatten(0,1)
        bins = np.array([0.   , 0.125, 0.25 , 0.375, 0.5  , 0.625, 0.75 , 0.875, 1.   ])
        hitx = np.digitize(normals[:, 0], bins)
        hity = np.digitize(normals[:, 1], bins)
        hitz = np.digitize(normals[:, 2], bins)
        hitx[hitx==9] = 8 # if >=1 then it is 9 - just add this to the 8th bin
        hity[hity==9] = 8
        hitz[hitz==9] = 8
        hitx -= 1 # start index at 0
        hity -= 1
        hitz -= 1
        normals_binned = hitx + hity*8 + hitz*(8**2)
        normals_binned = torch.from_numpy(normals_binned).reshape(B, self.W, self.H)

        normals = normals.reshape(B, 3, self.W, self.H)

        normals_medians = []
        reshadings_medians = []
        guassian_curvatures_medians = []
        mean_curvatures_medians = []
        principal_curvatures1_medians = []
        principal_curvatures2_medians = []
        normals_c1_medians = []
        normals_c2_medians = []
        normals_c3_medians = []

        valid_mask = []
        for u in tqdm(list(np.arange(acts.shape[1])), leave=False):

            normals_medians_ = []
            reshadings_medians_ = []
            guassian_curvatures_medians_ = []
            mean_curvatures_medians_ = []
            principal_curvatures1_medians_ = []
            principal_curvatures2_medians_ = []
            normals_c1_medians_ = []
            normals_c2_medians_ = []
            normals_c3_medians_ = []

            acts_responses_unit = acts[:,u].view(acts.shape[0], -1).max(1)[0]
            args_sort_responses = torch.argsort(acts_responses_unit, descending=True).cpu().numpy()
            for im_i in list(args_sort_responses)[:1000]:
                mask = xtc_iv.pytorch_mask(acts[im_i].to(device), u)
                if torch.sum(mask)==0:
                    normals_medians_.append(torch.tensor(torch.nan))
                    reshadings_medians_.append(torch.tensor(torch.nan))
                    guassian_curvatures_medians_.append(torch.tensor(torch.nan))
                    mean_curvatures_medians_.append(torch.tensor(torch.nan))
                    principal_curvatures1_medians_.append(torch.tensor(torch.nan))
                    principal_curvatures2_medians_.append(torch.tensor(torch.nan))
                    normals_c1_medians_.append(torch.tensor(torch.nan))
                    normals_c2_medians_.append(torch.tensor(torch.nan))
                    normals_c3_medians_.append(torch.tensor(torch.nan))
                    continue
                
                normals_c1_masked = normals[im_i][0][mask]
                normals_c2_masked = normals[im_i][1][mask]
                normals_c3_masked = normals[im_i][2][mask]
                normals_masked = normals_binned[im_i][mask]
                reshadings_masked = reshadings[im_i][mask]
                guassian_curvatures_masked = guassian_curvatures[im_i][mask]
                mean_curvatures_masked = mean_curvatures[im_i][mask]
                principal_curvatures1_masked = principal_curvatures1[im_i][mask]
                principal_curvatures2_masked = principal_curvatures2[im_i][mask]

                normals_medians_.append(normals_masked.median())
                reshadings_medians_.append(reshadings_masked.median())
                guassian_curvatures_medians_.append(guassian_curvatures_masked.median())
                mean_curvatures_medians_.append(mean_curvatures_masked.median())
                principal_curvatures1_medians_.append(principal_curvatures1_masked.median())
                principal_curvatures2_medians_.append(principal_curvatures2_masked.median())
                normals_c1_medians_.append(normals_c1_masked.median())
                normals_c2_medians_.append(normals_c2_masked.median())
                normals_c3_medians_.append(normals_c3_masked.median())

            check = torch.stack(normals_medians_, dim=0)
            if torch.all(torch.isnan(check)):
                print(f"Skipping {u}")
                valid_mask.append(False)
                continue
            valid_mask.append(True)

            normals_medians.append(torch.stack(normals_medians_, dim=0))
            reshadings_medians.append(torch.stack(reshadings_medians_, dim=0))
            guassian_curvatures_medians.append(torch.stack(guassian_curvatures_medians_, dim=0))
            mean_curvatures_medians.append(torch.stack(mean_curvatures_medians_, dim=0))
            principal_curvatures1_medians.append(torch.stack(principal_curvatures1_medians_, dim=0))
            principal_curvatures2_medians.append(torch.stack(principal_curvatures2_medians_, dim=0))
            normals_c1_medians.append(torch.stack(normals_c1_medians_, dim=0))
            normals_c2_medians.append(torch.stack(normals_c2_medians_, dim=0))
            normals_c3_medians.append(torch.stack(normals_c3_medians_, dim=0))
        
        normals_medians = torch.stack(normals_medians, dim=0).cpu().numpy()
        reshadings_medians = torch.stack(reshadings_medians, dim=0).cpu().numpy()
        guassian_curvatures_medians = torch.stack(guassian_curvatures_medians, dim=0).cpu().numpy()
        mean_curvatures_medians = torch.stack(mean_curvatures_medians, dim=0).cpu().numpy()
        principal_curvatures1_medians = torch.stack(principal_curvatures1_medians, dim=0).cpu().numpy()
        principal_curvatures2_medians = torch.stack(principal_curvatures2_medians, dim=0).cpu().numpy()
        normals_c1_medians = torch.stack(normals_c1_medians, dim=0).cpu().numpy()
        normals_c2_medians = torch.stack(normals_c2_medians, dim=0).cpu().numpy()
        normals_c3_medians = torch.stack(normals_c3_medians, dim=0).cpu().numpy()

        if cache:
            '''
            cache dissection variables
            '''
            subj = args.eval_subject
            root, model_name = os.path.split(args.load_model_path)
            _, root_model = os.path.split(root)
            os.makedirs(os.path.join(args.tmp_dir, 'dissect', root_model), exist_ok = True)
            name = "normals_medians_allimages"
            path = os.path.join(args.tmp_dir, 'dissect', root_model, f"roi_data_{model_name.split('.pth')[0]}_{''.join(self.rois)}_{name}_subject{subj}.npy")
            print(f"cacheing {name} to {path}..")
            np.save(path, normals_medians)
            name = "normals_c1_medians_allimages"
            path = os.path.join(args.tmp_dir, 'dissect', root_model, f"roi_data_{model_name.split('.pth')[0]}_{''.join(self.rois)}_{name}_subject{subj}.npy")
            print(f"cacheing {name} to {path}..")
            np.save(path, normals_c1_medians)
            name = "normals_c2_medians_allimages"
            path = os.path.join(args.tmp_dir, 'dissect', root_model, f"roi_data_{model_name.split('.pth')[0]}_{''.join(self.rois)}_{name}_subject{subj}.npy")
            print(f"cacheing {name} to {path}..")
            np.save(path, normals_c2_medians)
            name = "normals_c3_medians_allimages"
            path = os.path.join(args.tmp_dir, 'dissect', root_model, f"roi_data_{model_name.split('.pth')[0]}_{''.join(self.rois)}_{name}_subject{subj}.npy")
            print(f"cacheing {name} to {path}..")
            np.save(path, normals_c3_medians)
            name = "reshadings_medians_allimages"
            path = os.path.join(args.tmp_dir, 'dissect', root_model, f"roi_data_{model_name.split('.pth')[0]}_{''.join(self.rois)}_{name}_subject{subj}.npy")
            print(f"cacheing {name} to {path}..")
            np.save(path, reshadings_medians)
            name = "guassian_curvatures_medians_allimages"
            path = os.path.join(args.tmp_dir, 'dissect', root_model, f"roi_data_{model_name.split('.pth')[0]}_{''.join(self.rois)}_{name}_subject{subj}.npy")
            print(f"cacheing {name} to {path}..")
            np.save(path, guassian_curvatures_medians)
            name = "mean_curvatures_medians_allimages"
            path = os.path.join(args.tmp_dir, 'dissect', root_model, f"roi_data_{model_name.split('.pth')[0]}_{''.join(self.rois)}_{name}_subject{subj}.npy")
            print(f"cacheing {name} to {path}..")
            np.save(path, mean_curvatures_medians)
            name = "principal_curvatures1_medians"
            path = os.path.join(args.tmp_dir, 'dissect', root_model, f"roi_data_{model_name.split('.pth')[0]}_{''.join(self.rois)}_{name}_subject{subj}.npy")
            print(f"cacheing {name} to {path}..")
            np.save(path, principal_curvatures1_medians)
            name = "principal_curvatures2_medians"
            path = os.path.join(args.tmp_dir, 'dissect', root_model, f"roi_data_{model_name.split('.pth')[0]}_{''.join(self.rois)}_{name}_subject{subj}.npy")
            print(f"cacheing {name} to {path}..")
            np.save(path, principal_curvatures2_medians)
            name = "valid_mask"
            path = os.path.join(args.tmp_dir, 'dissect', root_model, f"roi_data_{model_name.split('.pth')[0]}_{''.join(self.rois)}_{name}_subject{subj}.npy")
            print(f"cacheing {name} to {path}..")
            np.save(path, np.asarray(valid_mask))

        # normals_medians_std = np.nanstd(normals_medians, axis=1)
        # reshadings_medians_std = np.nanstd(reshadings_medians, axis=1) 
        # guassian_curvatures_medians_std = np.nanstd(guassian_curvatures_medians, axis=1) 
        # mean_curvatures_medians_std = np.nanstd(mean_curvatures_medians, axis=1) 
        # principal_curvatures1_medians_std = np.nanstd(principal_curvatures1_medians, axis=1) 
        # principal_curvatures2_medians_std = np.nanstd(principal_curvatures2_medians, axis=1) 
        # normals_c1_medians_std = np.nanstd(normals_c1_medians, axis=1) 
        # normals_c2_medians_std = np.nanstd(normals_c2_medians, axis=1) 
        # normals_c3_medians_std = np.nanstd(normals_c3_medians, axis=1) 

        normals_medians = np.nanmean(normals_medians, axis=1) 
        reshadings_medians = np.nanmean(reshadings_medians, axis=1) 
        guassian_curvatures_medians = np.nanmean(guassian_curvatures_medians, axis=1) 
        mean_curvatures_medians = np.nanmean(mean_curvatures_medians, axis=1) 
        principal_curvatures1_medians = np.nanmean(principal_curvatures1_medians, axis=1) 
        principal_curvatures2_medians = np.nanmean(principal_curvatures2_medians, axis=1) 
        normals_c1_medians = np.nanmean(normals_c1_medians, axis=1) 
        normals_c2_medians = np.nanmean(normals_c2_medians, axis=1) 
        normals_c3_medians = np.nanmean(normals_c3_medians, axis=1)

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

if __name__ == '__main__':
    Ai2Thor() 