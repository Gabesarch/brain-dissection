from arguments import args
import time
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
import torch
import os
from backend.dataloaders.nsd_response_loader import NSDDataloader
if args.arch=="e2cnn":
    from nets.convnet_e2cnn import ConvNet
elif args.arch=="cnn_alt":
    from nets.convnet_alt import ConvNet
else:
    assert(False) # wrong arch
import utils.dist
import wandb
from utils.improc import MetricLogger
from nets.loss import OptRespLoss
from backend import saverloader
import matplotlib.pyplot as plt
import random

import ipdb
st = ipdb.set_trace
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# fix the seed for reproducibility
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
torch.backends.cudnn.benchmark = False

class NSD_OPT():
    def __init__(self):  

        if utils.dist.is_main_process():
            # initialize wandb
            if args.set_name=="test00":
                # option to turn off wandb for debugging
                wandb.init(mode="disabled")
            else:
                wandb.init(project="optimize_response", name=args.set_name, group=args.group, config=args, dir=args.wandb_directory)

        print(f"Device: {device}")
        print(f"Set Name: {args.set_name}")
        print(f'LR: {args.lr}')
        print(f'readout_sparse_weight_spatial: {args.readout_sparse_weight_spatial}')

        total_voxel_size = self.init_dataloaders(args)
        self.total_voxel_size = total_voxel_size

        self.W = args.image_size
        self.H = args.image_size
        print(f"W: {self.W}, H: {self.H}")
        model = ConvNet(total_voxel_size, self.W, self.H)
        self.model = model

        self.model.to(device)

        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('number of params:', n_parameters)

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                print(name, "requires grad?", param.requires_grad)

        params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            params.append(param)

        # lr set by arg_parser
        self.optimizer = torch.optim.AdamW([{'params': params}], lr=args.lr,
                                      weight_decay=args.weight_decay)
        lr_drop = args.lr_drop # every X epochs, drop lr by 0.1
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, lr_drop)

        self.global_step = 0
        self.start_epoch = 0
        if args.load_model:
            path = args.load_model_path 

            if args.lr_scheduler_from_scratch:
                print("LR SCHEDULER FROM SCRATCH")
                lr_scheduler_load = None
            else:
                lr_scheduler_load = self.lr_scheduler

            if args.optimizer_from_scratch:
                print("OPTIMIZER FROM SCRATCH")
                optimizer_load = None
            else:
                optimizer_load = self.optimizer
            
            self.global_step, self.start_epoch = saverloader.load_from_path(
                path, 
                self.model, 
                optimizer_load, 
                strict=(not args.load_strict_false), 
                lr_scheduler=lr_scheduler_load,
                )
            self.start_epoch += 1 # need to add one since saving corresponds to trained epoch

        if args.start_one:
            print("Starting at iteration 0 despite checkpoint loaded.")
            self.global_step = 0
            self.start_epoch = 0

        self.checkpoint_path = os.path.join(args.checkpoint_path, args.set_name)
        if not os.path.exists(self.checkpoint_path):
            os.mkdir(self.checkpoint_path)
        print(f'Save path is {self.checkpoint_path}')

        self.log_freq = args.log_freq

        self.loss_func = OptRespLoss()


    def run_train(self):

        self.model.train()

        print("Start training")
        best_corr = torch.tensor(0.0).to(device)
        early_stop_count = 0
        for epoch in range(self.start_epoch, args.epochs):
            
            if args.distributed:
                self.train_dataset_loader.sampler.set_epoch(epoch)
        
            self.epoch = epoch

            print("Begin epoch", epoch)
            print("set name:", args.set_name)

            total_loss = self.train_one_epoch()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            print(f"loss for epoch {epoch} is: {total_loss}")

            if epoch % args.save_freq_epoch == 0 and not args.debug:
                if utils.dist.is_main_process():
                    saverloader.save_checkpoint(
                        self.model, 
                        self.checkpoint_path, 
                        self.global_step, 
                        self.epoch, 
                        self.optimizer, 
                        keep_latest=args.keep_latest, 
                        lr_scheduler=self.lr_scheduler,
                        extras={
                            "roi_sizes":self.roi_sizes,
                            "rois":args.rois,
                            "nc_threshold":args.nc_threshold,
                            }
                        )

            if args.run_validation:
                print("Running FULL validation seen set...")
                with torch.no_grad():
                    total_loss, total_corr = self.run_validation(self.validation_dataset_loader, split="validation", global_step=self.global_step)
                
                if args.patience is not None and epoch>args.patience and not args.debug:
                    if total_corr>best_corr:
                        best_corr = total_corr
                        if utils.dist.is_main_process():
                            saverloader.save_checkpoint(
                                self.model, 
                                self.checkpoint_path, 
                                self.global_step, 
                                self.epoch, 
                                self.optimizer, 
                                keep_latest=args.keep_latest, 
                                lr_scheduler=self.lr_scheduler,
                                best=True,
                                extras={
                                    "roi_sizes":self.roi_sizes,
                                    "rois":args.rois,
                                    "nc_threshold":args.nc_threshold,
                                    }
                                )
                        early_stop_count = 0
                    else:
                        '''
                        Stop training if correlation has not improved upon best in X epochs
                        '''
                        if args.early_stopping is not None:
                            early_stop_count += 1
                            if early_stop_count>=args.early_stopping:
                                print("EARLY STOPPING CRITERIA REACHED. STOPPING TRAINING.")
                                break

    def train_one_epoch(self):

        self.model.train()

        metric_logger = MetricLogger(delimiter="  ")
        header = f'TRAIN | {args.set_name} | Epoch: [{self.epoch}/{args.epochs-1}] | ROIS:{args.rois}'
        for i_batch, batched_samples in enumerate(metric_logger.log_every(self.train_dataset_loader, 10, header)):

            images = batched_samples['images'].to(device)
            y_brain = batched_samples['y_brain'].to(device)
            y_mask = batched_samples['y_mask'].to(device)

            if self.global_step % self.log_freq == 0:
                self.log_iter = True
            else:
                self.log_iter = False
                    
            out = self.model(
                images, 
                )

            total_loss = torch.tensor(0.0).to(device)
            # MSE loss
            mse_loss = self.loss_func(out, y_brain, y_mask)
            total_loss += mse_loss

            if torch.isnan(mse_loss):
                continue

            if args.readout_sparse_weight_spatial>0 or args.readout_sparse_weight_feature>0:
                # sparsity constraints - set to 0 in the paper
                l1_spatial, l1_feature = self.model.get_sparsity_loss()
                l1_spatial_reg = args.readout_sparse_weight_spatial * l1_spatial
                l1_feature_reg = args.readout_sparse_weight_feature * l1_feature
                sparsity_loss = l1_spatial_reg + l1_feature_reg
                total_loss += sparsity_loss
                if utils.dist.is_main_process():
                    wandb.log({"train/sparsity_loss": sparsity_loss, 'epoch': self.epoch})
                    wandb.log({"train/l1_spatial_reg": l1_spatial_reg, 'epoch': self.epoch})
                    wandb.log({"train/l1_feature_reg": l1_feature_reg, 'epoch': self.epoch})

            if total_loss is not None:
                self.optimizer.zero_grad()
                total_loss.backward()
                if args.clip_max_norm > 0:
                    grad_total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.clip_max_norm)
                else:
                    pass
                self.optimizer.step()

            if utils.dist.is_main_process():
                wandb.log({"train/mse_loss": mse_loss, 'epoch': self.epoch})
                wandb.log({"train/total_loss": total_loss, 'epoch': self.epoch})
                
            metric_logger.update(loss=total_loss)

            self.global_step += 1

            if args.val_freq > 0:
                if self.global_step % args.val_freq == 0:
                    self.visualize_spatial_mask() # visualize spatial mask in wandb
                    if args.run_validation:
                        with torch.no_grad():
                            self.run_validation(self.validation_dataset_loader, split="validation", global_step=self.global_step)
        
        return total_loss

    @torch.no_grad()
    def visualize_spatial_mask(self):
        # plot every 20th spatial mask

        W,H = self.model.get_last_activation_sizes()

        spatial_masks = self.model.get_spatial_mask() 
        
        for u_i in np.arange(0,spatial_masks.shape[1],20):
            s_mask = spatial_masks[:,u_i].reshape(W,H)
            s_mask = s_mask.detach().cpu().numpy()
            plt.figure(1); plt.clf()
            plt.imshow(s_mask)
            wandb.log({f"spatial_mask/unit{u_i}":wandb.Image(plt)}) 
            plt.close()

        plt.close('all')

    @torch.no_grad()
    def run_validation(self, dataloader, split, global_step):

        print(f"Evaluating {split}...")

        self.model.eval()

        total_loss = torch.tensor(0.0).to(device)
        total_corr = torch.tensor(0.0).to(device)
        total_mse_loss = torch.tensor(0.0).to(device)
        total_sparsity_loss = torch.tensor(0.0).to(device)
        total_l1_spatial_reg = torch.tensor(0.0).to(device)
        total_l1_feature_reg = torch.tensor(0.0).to(device)

        count = 0

        ind_to_log = np.random.randint(len(dataloader))

        metric_logger = MetricLogger(delimiter="  ")
        header = f'{split} | {args.set_name} | Epoch: [{self.epoch}/{args.epochs-1}]'
        for i_batch, batched_samples in enumerate(metric_logger.log_every(dataloader, 10, header)):
                    
            images = batched_samples['images'].to(device)
            y_brain = batched_samples['y_brain'].to(device)
            y_mask = batched_samples['y_mask'].to(device)
            
            out = self.model(
                images, 
                )

            loss = torch.tensor(0.0).to(device)
            # MSE loss
            mse_loss, corr = self.loss_func(out, y_brain, y_mask, compute_corr=True)
            loss += mse_loss

            if torch.isnan(mse_loss):
                continue

            if args.readout_sparse_weight_spatial>0 or args.readout_sparse_weight_feature>0:
                # sparsity constraints
                l1_spatial, l1_feature = self.model.get_sparsity_loss()
                l1_spatial_reg = args.readout_sparse_weight_spatial * l1_spatial
                l1_feature_reg = args.readout_sparse_weight_feature * l1_feature
                sparsity_loss = l1_spatial_reg + l1_feature_reg
                loss += sparsity_loss
                total_sparsity_loss += sparsity_loss
                total_l1_spatial_reg += l1_spatial_reg
                total_l1_feature_reg += l1_feature_reg

            total_loss += loss
            total_mse_loss += mse_loss

            total_corr += corr

            torch.cuda.synchronize()

            count += 1

            metric_logger.update(loss=total_loss/count)

            if args.max_validation_iters is not None:
                if i_batch>=args.max_validation_iters:
                    break

        if utils.dist.is_main_process():
            wandb.log({f"{split}/total_mse_loss": total_mse_loss / count, 'epoch': self.epoch})
            if args.readout_sparse_weight_spatial>0 or args.readout_sparse_weight_feature>0:
                wandb.log({f"{split}/total_sparsity_loss": total_sparsity_loss / count, 'epoch': self.epoch})
                wandb.log({f"{split}/total_l1_spatial_reg": total_l1_spatial_reg / count, 'epoch': self.epoch})
                wandb.log({f"{split}/total_l1_feature_reg": total_l1_feature_reg / count, 'epoch': self.epoch})
            wandb.log({f"{split}/total_loss": total_loss / count, 'epoch': self.epoch})
            wandb.log({f"{split}/total_corr": total_corr / count, 'epoch': self.epoch})

        print(f"{args.set_name}: {split} loss for iter {self.global_step} is: {total_loss/count}")

        self.model.train()

        return total_loss / count, total_corr / count

    def init_dataloaders(self, args):

        train_transform = transforms.Compose([
                    transforms.RandomResizedCrop(args.image_size, scale=(0.9, 1.0)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ])

        print("Getting dataloader...")
        train_dataset = NSDDataloader(args, "train", transform=train_transform)
        if args.distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
        else:
            sampler = torch.utils.data.RandomSampler(train_dataset)
        train_dataset_loader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=args.batch_size, 
                                                    sampler=sampler,
                                                    num_workers=args.num_workers, 
                                                    # collate_fn=my_collate,
                                                    pin_memory=True,
                                                    drop_last=True
                                                    )
        self.train_dataset_loader = train_dataset_loader
        print("Size train dataloader:", len(self.train_dataset_loader))

        self.roi_sizes = train_dataset.roi_sizes

        if args.run_validation:
            val_transform = transforms.Compose([
                transforms.Resize(args.image_size+32),
                transforms.CenterCrop(args.image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
            validation_dataset = NSDDataloader(args, "validation", transform=val_transform)
            if args.distributed:
                sampler = torch.utils.data.distributed.DistributedSampler(validation_dataset, shuffle=True)
            else:
                sampler = torch.utils.data.RandomSampler(validation_dataset)
            validation_dataset_loader = torch.utils.data.DataLoader(validation_dataset,
                                                        batch_size=args.batch_size, 
                                                        sampler=sampler,
                                                        num_workers=args.num_workers, 
                                                        pin_memory=True,
                                                        drop_last=True
                                                        )
            self.validation_dataset_loader = validation_dataset_loader
            print("Size val dataloader:", len(self.validation_dataset_loader))

        return train_dataset.total_voxel_size