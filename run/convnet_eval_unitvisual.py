from arguments import args
import torch
import numpy as np
from torchvision import transforms
import torch.nn.functional as F
import torchvision
import torch
import os
if args.arch=="e2cnn":
    print("Using E(2) CNN")
    from nets.convnet_e2cnn import ConvNet
elif args.arch=="cnn_alt":
    print("Using CNN ALT")
    from nets.convnet_alt import ConvNet
else:
    assert(False)

from backend.dataloaders.places_image_loader import NSDDataloader

import wandb
from utils.improc import MetricLogger
from backend import saverloader

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

from dissect.netdissect import nethook, imgviz, show, segmenter, renormalize, upsample, tally, pbar, runningstats
from dissect.netdissect import setting

sys.path.append('XTConsistency')

torch.backends.cudnn.benchmark = True
torch.set_grad_enabled(False) # not training anything!

# fix the seed for reproducibility
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

from modules.unet import UNet, UNetReshade
import visualpriors

class Eval_UNIT():
    def __init__(self):  

        if args.set_name=="test00":
            wandb.init(mode="disabled")
        else:
            wandb.init(project="optimize_response_V1", group=args.group, name=args.set_name, config=args, dir=args.wandb_directory)

        self.init_dataloaders(args)

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

        repo = "isl-org/ZoeDepth"
        self.zoe = torch.hub.load(repo, "ZoeD_NK", pretrained=True)
        self.zoe.to(device) 
        self.zoe.eval()

        self.save_images = False
        
        self.models = []
        self.rois = []
        for model_path in args.load_model_paths_unitvisual:
            if args.total_voxel_size is None:
                checkpoint = torch.load(model_path, map_location=device)
                if 'W_spatial' in checkpoint['model_state_dict'].keys():
                    self.total_voxel_size = checkpoint['model_state_dict']['W_spatial'].shape[1]
                else:
                    self.total_voxel_size = checkpoint['model_state_dict']['readout.spatial'].shape[0]
                self.rois.append(checkpoint["rois"])
                self.nc_threshold = checkpoint["nc_threshold"]
            else:
                self.total_voxel_size = args.total_voxel_size
            print(f"Total voxel size is {self.total_voxel_size}")

            self.W = args.image_size_eval
            self.H = args.image_size_eval
            model = ConvNet(self.total_voxel_size, args.image_size, args.image_size)
            self.act_W, self.act_H = model.get_last_activation_sizes()
            if args.load_model:
                path = model_path

                _, _ = saverloader.load_from_path(
                        path, 
                        model, 
                        None, 
                        strict=(not args.load_strict_false), 
                        lr_scheduler=None,
                        device=device,
                        )
            model.to(device)
            model.eval()
            self.models.append(model)
    
    @torch.no_grad()
    def run_dissection(self):

        print(f"Image size eval: {args.image_size_eval}")

        self.inds_sorted = []
        for model_i in range(len(self.models)):
            model_path = args.load_model_paths_unitvisual[model_i]
            # load variables from memory if they are saved
            root, model_name = os.path.split(model_path)
            _, root_model = os.path.split(root)
                
            if args.eval_object in ["best_cat_per_unit"]:
                name = f"iou_table"
                root, model_name = os.path.split(model_path)
                _, root_model = os.path.split(root)
                iou_table = []
                for subj in [1,2,3,4,5,6,7,8]:
                    subj_add = '_subject'+str(subj)
                    path = os.path.join(args.tmp_dir_load, 'dissect', root_model, f"roi_data_{model_name.split('.pth')[0]}_{''.join(self.rois[model_i])}_{name}{subj_add}.npy")
                    iou_table_ = np.load(path)
                    iou_table.append(iou_table_)
                iou_table = np.concatenate(iou_table, axis=0)
                seglabels = list(seglabels)
                # label_ind = seglabels.index('table')
                label_ind = seglabels.index(args.eval_object)
                iou_table_seglabel = iou_table[:,label_ind]
                self.inds_sorted.append(np.argsort(-iou_table_seglabel))
            else:
                '''
                Gets units closest to grand mean
                '''
                # name = "normals_medians_allimages"
                # name = "median_abs_depths_allimages"
                name = args.eval_object
                means = []
                for subj in [1,2,3,4,5,6,7,8]:
                    # subj_add = '_subject'+str(subj)
                    tmp_dir_ = args.tmp_dir_load if args.tmp_dir_load is not None else args.tmp_dir
                    if not os.path.exists(os.path.join(tmp_dir_, 'dissect', root_model, f"roi_data_{model_name.split('.pth')[0]}_{''.join(self.rois[model_i])}_{name}_subject{subj}.npy")):
                        tmp_dir_ = args.tmp_dir
                    path = os.path.join(tmp_dir_, 'dissect', root_model, f"roi_data_{model_name.split('.pth')[0]}_{''.join(self.rois[model_i])}_{name}_subject{subj}.npy")
                    normals_ = np.load(path)
                    name_mask = "valid_mask"
                    path_mask = os.path.join(args.tmp_dir, 'dissect', root_model, f"roi_data_{model_name.split('.pth')[0]}_{''.join(self.rois[model_i])}_{name_mask}_subject{subj}.npy")
                    if not os.path.exists(path_mask):
                        print(f"Skipping subject {subj}. No valid mask")
                        continue
                    valid_mask = np.load(path_mask)
                    mean_normals = np.nanmean(normals_, axis=1)
                    mean_normals_all = np.zeros(len(valid_mask)) * np.nan
                    mean_normals_all[valid_mask] = mean_normals
                    means.append(mean_normals_all)
                means = np.concatenate(means, axis=0)
                grand_mean_normals = np.nanmean(means)
                distances = np.linalg.norm(means[:,None] - grand_mean_normals, axis=1)
                argsort_distances = np.argsort(distances)
                self.inds_sorted.append(argsort_distances)

        segmodel, seglabels, segcatlabels = setting.load_segmenter('netpqc')
        assert(224 % args.image_size_eval == 0) # must be divisable by 256
        downsample = 224//args.image_size_eval
            
        print('segmenter has', len(seglabels), 'labels')

        num_images = args.max_images if args.max_images is not None else len(self.dataset_loader)*args.batch_size

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
            responses_ = [np.zeros((num_images, len(np.arange(args.figure_unit_start,args.figure_unit_end))), dtype=np.float32) for _ in range(len(self.models))]
            for i_batch, batched_samples in enumerate(metric_logger.log_every(dataset_loader, 10, header)):
                images = batched_samples['images'].to(device)
                images_raw = batched_samples['images2'].to(device)
                for model_i in range(len(self.models)):
                    response = self.models[model_i](images)
                    response = response[:,self.inds_sorted[model_i][args.figure_unit_start:args.figure_unit_end]]
                    responses_[model_i][i_batch*b_tmp:(i_batch+1)*b_tmp] = response.cpu().numpy()
            responses_ = np.concatenate(responses_, axis=1)

            # FILTER OUT BY TOP FOR THE UNITS
            self.dataset_loader.dataset.filter_topk_images_from_roi_brain_response(responses_, topk=args.topk_filter)
            print("NEW size dataloader:", len(self.dataset_loader))

        num_images = args.max_images if args.max_images is not None else len(self.dataset_loader)*args.batch_size
        metric_logger = MetricLogger(delimiter="  ")
        header = f'{args.set_name}|{args.mode}'
        
        num_images_figure = 50
        units = [[[] for _ in range(args.figure_unit_end-args.figure_unit_start)] for _ in range(len(self.models))]
        for i_batch, batched_samples in enumerate(metric_logger.log_every(self.dataset_loader, 10, header)):

            images = batched_samples['images'].to(device)
            images_raw = batched_samples['images2'].to(device)

            transform = torchvision.transforms.Compose([
                transforms.Resize((256, 256)),
            ])
            images_xtc = transform(images_raw)

            normal = self.normal_model(images_xtc)
            reshading = self.reshading_model(images_xtc)
            principal_curvature = visualpriors.feature_readout(images_xtc * 2 - 1, 'curvature', device=device) / 2. + 0.5
            
            reshading = reshading[:,0:1]
            principal_curvature = principal_curvature[:,:2]

            for model_i in range(len(self.models)):
                with torch.no_grad():
                    activations, _ = self.models[model_i].get_voxel_feature_maps(images, get_voxel_response=True, use_spatial_mask=args.use_spatial_mask)
                figure_iv = imgviz.ImageVisualizer(224, percent_level=0.95, image_size=(self.W, self.H))
                normal_plot = normal.squeeze(0).permute(1,2,0).cpu().numpy()
                reshading_plot = reshading.squeeze(0).permute(1,2,0).cpu().numpy()
                principal_curvature_plot = np.concatenate([principal_curvature.squeeze(0).permute(1,2,0).cpu().numpy(), np.zeros((256, 256,1))] ,axis=2)
                image_plot = images_xtc.squeeze(0).permute(1,2,0).cpu().numpy()
                activation_plot = activations[0]
                image_vis = torch.nn.functional.interpolate(
                        images_raw,
                        mode='bilinear',
                        size=128
                        )
                image_vis = image_vis * 255
                image_vis = image_vis.to(torch.uint8)

                image_depth = images_raw.permute(0,2,3,1)[0].cpu().numpy()*255
                image_depth = image_depth.astype(np.uint8)
                image_depth = Image.fromarray(image_depth)
                from zoedepth.utils.misc import pil_to_batched_tensor
                image_depth = pil_to_batched_tensor(image_depth).to(device)
                depth = self.zoe.infer(image_depth)
                
                for idx in range(args.figure_unit_end-args.figure_unit_start):
                    unit_i = self.inds_sorted[model_i][idx]
                    masked_image = figure_iv.masked_image(images_raw, activation_plot.to(device), unit_i)
                    masked_surface_normal = figure_iv.masked_image(normal, activation_plot.to(device), unit_i)
                    normalized_depth = (depth - torch.min(depth)) / (torch.max(depth) - torch.min(depth))
                    masked_surface_depth = figure_iv.masked_image(normalized_depth, activation_plot.to(device), unit_i).convert('RGB')
                    heatmap_normal = figure_iv.heatmap(activation_plot.cpu(), unit_i, mode='nearest').convert('RGB')
                    masked_principal_curvature = figure_iv.masked_image(torch.from_numpy(principal_curvature_plot).permute(2,0,1).unsqueeze(0), activation_plot.to(device), unit_i).convert('RGB')
                    masked_shading = figure_iv.masked_image(reshading, activation_plot.to(device), unit_i).convert('RGB')

                    images_plot = np.uint8(images_raw.squeeze(0).permute(1,2,0).cpu().numpy() * 255)
                    image_figure = np.concatenate([images_plot, np.asarray(masked_image), np.asarray(masked_surface_normal), np.asarray(masked_surface_depth), np.asarray(masked_principal_curvature), np.asarray(masked_shading), np.asarray(heatmap_normal)], axis=0)

                    units[model_i][idx].append(image_figure)

            if i_batch==num_images_figure-1:
                break

        for model_i in range(len(self.models)):
            for unit_i in range(len(units[model_i])):
                to_plot = units[model_i][unit_i]
                image_to_plot = np.concatenate(to_plot, axis=1)
                wandb.log({f"XTC/top_images_{''.join(self.rois[model_i])}":wandb.Image(image_to_plot)}) 

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