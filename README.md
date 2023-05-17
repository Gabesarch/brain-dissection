<h1 align="center">
    Brain Dissection: fMRI-trained Networks Reveal Spatial Selectivity in the Processing of Natural Images
</h1>

<p align="left">
<!--     <a href="//github.com/allenai/ai2thor-rearrangement/blob/main/LICENSE">
        <!-- ai2thor-rearrangement wasn't identifiable by GitHub (on the day this was added), so using the same one as ai2thor -->
<!--         <img alt="License" src="https://img.shields.io/github/license/allenai/ai2thor.svg?color=blue">
    </a> -->
    <a href="https://tidee-agent.github.io/" target="_blank">
        <img alt="Website" src="https://img.shields.io/badge/website-TIDEE-orange">
    </a>
<!--     <a href="//github.com/allenai/ai2thor-rearrangement/releases">
        <img alt="GitHub release" src="https://img.shields.io/github/release/allenai/ai2thor-rearrangement.svg">
    </a> -->
    <a href="https://arxiv.org/abs/2207.10761" target="_blank">
        <img src="https://img.shields.io/badge/arXiv-2103.16544-<COLOR>">
    </a>
<!--     <a href="//arxiv.org/abs/2103.16544" target="_blank">
        <img src="https://img.shields.io/badge/venue-CVPR 2021-blue">
    </a> -->
    <a href="https://youtu.be/wXJuVKeWZmk" target="_blank">
        <img src="https://img.shields.io/badge/video-YouTube-red">
    </a>
<!--     <a href="https://join.slack.com/t/ask-prior/shared_invite/zt-oq4z9u4i-QR3kgpeeTAymEDkNpZmCcg" target="_blank">
        <img src="https://img.shields.io/badge/questions-Ask PRIOR Slack-blue">
    </a> -->
</p>

This repo contains code and data for running Brain Dissection. 

This repo is heavily based on the methods from [Higher visual areas act like domain-general filters with strong selectivity and functional specialization](https://www.biorxiv.org/content/10.1101/2022.03.16.484578v2)

### Contents
<!--
# To create the table of contents, move the [TOC] line outside of this comment
# and then run the below Python block.
[TOC]
import markdown
with open("README.md", "r") as f:
    a = markdown.markdown(f.read(), extensions=["toc"])
    print(a[:a.index("</div>") + 6])
-->
<div class="toc">
<ul>
<li><a href="#installation"> Installation </a></li><ul>
</ul>
<li><a href="#Data"> Data</a></li><ul>
</ul>
<li><a href="#Response-Optimized-Training"> Response-Optimized Training</a><ul>
</ul>
<li><a href="#Model-Dissection"> Model Dissection</a></li><ul>
</ul>
<li><a href="#citation"> Citation </a></li><ul>
</ul>
</ul>
</div>

## Installation 
<!-- **Note:** We have tested this on a remote cluster with CUDA versions 10.2 and 11.1. The dependencies are for running the full TIDEE system. A reduced environment can be used if only running the tidy task and not the TIDEE networks.  -->

**(1)** For training and dissecting, start by cloning the repository:
```bash
git clone https://github.com/Gabesarch/brain-dissection.git
```

**(2)** Clone dependency repos and download checkpoints. Run:
```bash
sh setup.sh
```

**(3)** (optional) If you are using conda, create an environment: 
```bash
conda create -n brain_dissect python=3.8
```

**(4)** Install [PyTorch](https://pytorch.org/get-started/locally/) with the CUDA version you have. For example, run the following for CUDA 11.1: 
```bash
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
```

**(5)** Install additional requirements: 
```bash
pip install -r requirements.txt
```

# Data

## Natural Scenes Dataset
Please download and process the Natural Scenes Dataset to get the fMRI data and NSD images: [Link](https://naturalscenesdataset.org/)

## PLACES365
Our dataloaders will download the places dataset for you when first running the evaluation.

Alternatively, you can download the Places365 dataset : [Link](http://places2.csail.mit.edu/)

Put the dataset in `./datasets`

## GQA dataset
For evaluation of the GQA dataset, please download the GQA dataset : [Link](https://cs.stanford.edu/people/dorarad/gqa/download.html)

Put the dataset in `./datasets`

# Response-Optimized Training

## Training on NSD
Train a network to predict the responses of NSD. You can see `nets/convnet_alt.py` for the model file and `models/convnet_fit_nsd.py` for the model training code. To train on a model on the NSD data using default hyperparameters (see `arguments.py`), run the following (for example ROI RSC):
```
python main.py \
    --rois RSC \
    --mode convnet_nsd_response_optimized \
    --run_validation \
    --coco_images_path REPLACE_WITH_NSD_IMAGE_DIRECTORY \
    --subjects_repeat_path REPLACE_WITH_NSD_SUBJECT_DATA_DIRECTORY \
    --brain_data_dir REPLACE_WITH_NSD_BRAIN_DATA_DIRECTORY \
    --roi_dir REPLACE_WITH_NSD_ROI_DIRECTORY \
    --noise_ceiling_dir REPLACE_WITH_NSD_NOISE_CEILING_DIRECTORY \
    --set_name train_subjs12345678_RSC
```

# Model Dissection
This section details run the network dissection on the response-optimized network.

## Evaluating on Places365
To evaluate on Places365 for depth, surface normals, shading, guassian curvature, and category, run the following (for example on Subject 1 for model trained from the section above):
```
python main.py \
    --mode convnet_xtc_eval_baudissect \
    --data_directory REPLACE_WITH_PLACES365_IMAGE_DIRECTORY \
    --load_model \
    --load_model_path "./checkpoints/train_subjs12345678_RSC/model-best.pth" \
    --eval_subject 1 \
    --analyze_depth \
    --save_dissection_samples \
    --load_dissection_samples \
    --filter_images_by_responses \
    --batch_size 1 \
    --set_name "EVAL_places365_subject1_RSC" 
```

You can also reduce storage memory by turning off `--save_dissection_samples` and can save cpu/storage memory with `--reduced_eval_memory`

## Evaluating on GQA


# Citation
If you like this paper, please cite us:
```
@inproceedings{,
            title = ,
            author = , 
            booktitle = ,
            year = }
```

