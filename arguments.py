import argparse
import numpy as np
import ipdb
st = ipdb.set_trace
parser = argparse.ArgumentParser()

parser.add_argument("--set_name", type=str, default="test00", help="name of experiment")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--mode", type=str, help="mode to run, see main.py")
parser.add_argument("--split", type=str, default="val", help="dataloader split")


############## VIP #########
# parser.add_argument("--arch", type=str, default="dino", help="which arch to use for training?")
parser.add_argument("--pretraining", type=str, default="dino", help="which pretraining to load weights from?")
parser.add_argument("--distributed", action="store_true", default=False, help="distributed training")
parser.add_argument('--lr', default=2e-5, type=float) 
parser.add_argument('--clip_max_norm', default=0.1, type=float,
                    help='gradient clipping max norm')

parser.add_argument('--weight_decay', default=1e-2, type=float)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--lr_drop', default=4000, type=int)
parser.add_argument('--readout_sparse_weight', default=0.02, type=float)
parser.add_argument("--sparsity_on_feature_weights", action="store_true", default=False, help="put sparsity of feature weights for what/where")
parser.add_argument("--val_freq", type=int, default=2500, help="how often to run validation")
parser.add_argument("--arch", type=str, default="cnn_alt", help="which arch to use for training?")
parser.add_argument("--patience", type=int, default=19, help="patience for early stopping")
parser.add_argument("--early_stopping", type=int, default=10, help="Stop training if correlation has not improved upon best in X epochs. Patience must also be defined.")


parser.add_argument('--total_voxel_size', default=None, type=int)

parser.add_argument("--max_iters", type=int, default=50000, help="maximum iterations to train")
parser.add_argument("--load_model", action="store_true", default=False, help="Load an existing checkpoint")
parser.add_argument("--load_model_path", type=str, default="", help="Path to existing checkpoint")
parser.add_argument("--load_strict_false", action="store_true", default=False, help="do not load strict checkpoint")
parser.add_argument("--lr_scheduler_from_scratch", action="store_true", default=False, help="do not load LR scheduler from checkpoint if True")
parser.add_argument("--optimizer_from_scratch", action="store_true", default=False, help="do not load optimizer from checkpoint if True")
parser.add_argument("--start_one", action="store_true", default=False, help="start from iteration 0")
parser.add_argument('--batch_size', default=24, type=int, help="batch size for model training")
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--save_freq_epoch', default=5, type=int, help="How often every X epochs to save model")
parser.add_argument('--keep_latest', default=10, type=int, help="number of checkpoints to keep at one time")
parser.add_argument('--max_validation_iters', default=None, type=int, help="maximum validation iters")
parser.add_argument("--checkpoint_path", type=str, default="./checkpoints", help="Path for saving checkpoints")
parser.add_argument("--attention_threshold", type=float, default=None, help="""We visualize masks
    obtained by thresholding the self-attention maps to keep xx% of the mass.""")
parser.add_argument("--images_path", type=str, default="./data/images", help="Path for saving checkpoints")

parser.add_argument("--activation_threshold", type=float, default=0.97, help="""We get IOUs
    by thresholding the activation maps""")

# parser.add_argument('--patch_size', default=16, type=int, help="patch size for vit in pixels")

parser.add_argument("--log_freq", type=int, default=250, help="how often to log to tensorboard in iterations")
# parser.add_argument("--lr_scheduler_freq", type=int, default=500, help="how often to step LR scheduler in iterations")
parser.add_argument("--run_validation", action="store_true", default=False, help="run validation every val_freq iters")
parser.add_argument("--save_freq", type=int, default=500, help="how often to save a checkpoint")

parser.add_argument("--image_size", type=int, default=224, help="")
parser.add_argument("--image_size_eval", type=int, default=256, help="")

parser.add_argument('--readout_sparse_weight_spatial', default=0.0, type=float)
parser.add_argument('--readout_sparse_weight_feature', default=0.0, type=float)

parser.add_argument("--coco_images_path", type=str, default="", help="Path to coco images")
parser.add_argument("--coco_annotations_path", type=str, default="", help="Path to coco annotations")

parser.add_argument("--subjects_repeat_path", type=str, default="", help="Path to subject image indices")
parser.add_argument("--brain_data_dir", type=str, default="", help="Path to nsd brain data")
parser.add_argument("--roi_dir", type=str, default="", help="Path to roi info")
parser.add_argument("--noise_ceiling_dir", type=str, default="", help="Path to noise ceiling info")

parser.add_argument("--nc_threshold", type=float, default=0.1, help="noise ceiling threshold for removal")
parser.add_argument("--debug", action="store_true", default=False, help="debugging mode?")

parser.add_argument("--analyze_depth", action="store_true", default=False, help="add depth analysis to dissection")

parser.add_argument("--rois", type=str, nargs='+', default=['FFA-1', 'FFA-2'], help="which rois to use?")
parser.add_argument("--coco_ids_path", type=str, default="", help="Path to coco ids to filter")

parser.add_argument("--max_activation_plots", type=int, default=None, help="plot every X units")
parser.add_argument("--subsample_activations", type=int, default=None, help="subsample every X activations")

parser.add_argument("--default_args", type=str, default=None, help="set default args")

parser.add_argument("--subjects", type=int, nargs='+', default=[1,2,3,4,5,6,7,8], help="which subjects to use?")

parser.add_argument("--topk", type=int, default=100, help="")

parser.add_argument("--subsample_images", type=int, default=None, help="subsample every X images")

parser.add_argument("--group", type=str, default="default", help="group name")

parser.add_argument("--min_test_corr", type=float, default=None, help="minimum correlation to keep unit")

parser.add_argument("--tmp_dir", type=str, default="", help="Path to nsd tmp folder")
parser.add_argument("--max_images", type=int, default=None, help="max images for debugging")
parser.add_argument("--eval_subject", type=int, default=1, help="which subject to use for eval?")


parser.add_argument("--topk_units_from_corr", type=int, default=None, help="take topk units from test correlations")

# broden dataset
parser.add_argument("--categories", type=int, nargs='+', default=["object", "part","scene","texture","color"], help="categories to fetch for dataset")
parser.add_argument("--data_directory", type=str, default="./dataset/broden1_224", help="Path to images")


parser.add_argument("--images_path_alt", type=str, default=None, help="Path to alternate images")

parser.add_argument("--save_dissection_samples", action="store_true", default=False, help="save dissection data?")
parser.add_argument("--load_dissection_samples", action="store_true", default=False, help="save dissection data?")


parser.add_argument("--filter_images_by_responses", action="store_true", default=False, help="filter images by top preicted responses for each ROI?")
parser.add_argument("--topk_filter", type=int, default=1000, help="topk to keep for filter_images_by_responses")

parser.add_argument("--wandb_directory", type=str, default='./wandb', help="Path to wandb metadata")


parser.add_argument("--gqa_path", type=str, default='./gqa', help="Path to wandb metadata")
parser.add_argument("--shared_depth_maps", action="store_true", default=False, help="shared depth maps across subjects?")
parser.add_argument("--use_spatial_mask", action="store_true", default=False, help="use spatial mask in eval?")

parser.add_argument("--reduced_eval_memory", action="store_true", default=False, help="use spatial mask in eval?")

parser.add_argument("--xtc_checkpoint_paths", type=str, default='./checkpoints', help="Path to wandb metadata")
parser.add_argument("--mode2", default=None, type=str, help="mode to run, see main.py")

parser.add_argument("--tmp_dir_load", default=None, type=str, help="mode to run, see main.py")

parser.add_argument("--eval_object", default=None, type=str, help="mode to run, see main.py")

parser.add_argument("--plot_for_figure", action="store_true", default=False, help="generate plot for figure in XTC eval")

parser.add_argument("--figure_unit_start", type=int, default=0, help="")
parser.add_argument("--figure_unit_end", type=int, default=20, help="")

parser.add_argument("--randomly_shuffle_images", action="store_true", default=False, help="randomly shuffle images in response dataloader?")
parser.add_argument("--randomly_shuffle_labels", action="store_true", default=False, help="randomly shuffle labels in places dataloader?")

parser.add_argument("--load_model_paths_unitvisual", type=str, nargs='+', default=[], help="which model paths to use for unit visualization?")

args = parser.parse_args()

if args.debug and args.mode in ["convnet_nsd_response_optimized", "convnet_get_corrs"]:
    args.subjects = [1]
    args.run_validation = True

if args.debug and args.mode in ["convnet_nsd_eval_baudissect", "convnet_places_eval_baudissect", "convnet_broden_eval_baudissect", "convnet_ade20k_eval_baudissect", "convnet_gqa_eval_baudissect", "convnet_xtc_eval_baudissect"]:   
    args.image_size_eval = 56
    args.topk = 20
    args.max_images = 50
    args.save_dissection_samples = False
    args.load_dissection_samples = True