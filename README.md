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
For evaluation of the Places365 dataset, please download the Places365 dataset : [Link](http://places2.csail.mit.edu/)
Put the dataset in ./datasets

## GQA dataset
For evaluation of the GQA dataset, please download the GQA dataset : [Link](https://cs.stanford.edu/people/dorarad/gqa/download.html)
Put the dataset in ./datasets

# Response-Optimized Training

## Running the task
The Tidy Task involves detecting and moving out of place objects to plausible places within the scene without any instructions. You can see `task_base/messup.py` for our data generation code to move objects out of place. See `task_base/example.py` for an example script of running the task with random actions. To run the tidy task, the tidy task dataset must be downloaded (see <a href="#dataset"> Dataset</a>)

## Dataset
Our tidy task dataset contains `8000` training scenes, `200` validation scenes, and `100` testing scenes with five objects in each scene moved out of place. To run the tidy task with the generated scenes, download the scene metadata from [here](https://drive.google.com/file/d/1KFUxxL8KU4H8dxBpjhp1SGAf3qnTtEBM/view?usp=sharing) and place the extracted contents inside of the `data` folder.  


# Model Dissection
This section details how to train the Out of Place Detector.

We first train [SOLQ](https://github.com/megvii-research/SOLQ) with two prediction heads (one for category, one for out of place). See `models/aithor_solq.py` and `models/aithor_solq_base.py` for code details, and `arguments.py` for training argument details. 

```
python main.py --mode solq --S 5 --data_batch_size 5 --lr_drop 7 --run_val --load_val_agent --val_load_dir ./data/val_data/aithor_tidee_oop --plot_boxes --plot_masks --randomize_scene_lighting_and_material --start_startx --do_predict_oop --load_base_solq --mess_up_from_loaded --log_freq 250 --val_freq 250 --set_name TIDEE_solq_oop
```

To train the visual and language detector, you can run the following (see `models/aithor_bert_oop_visual.py` and `models/aithor_solq_base.py` for details): 
```
python main.py --mode visual_bert_oop --do_visual_and_language_oop --S 3 --data_batch_size 3 --run_val --load_val_agent --val_load_dir ./data/val_data/aithor_tidee_oop_VL --n_val 3 --load_train_agent --train_load_dir ./data/train_data/aithor_tidee_oop_VL --n_train 50 --randomize_scene_lighting_and_material --start_startx --do_predict_oop --mess_up_from_loaded  --save_freq 2500 --log_freq 250 --val_freq 250 --max_iters 25000 --keep_latest 5 --start_one --score_threshold_oop 0.0 --score_threshold_cat 0.0 --set_name TIDEE_oop_vis_lang
```
The above will generate training and validation data from the simulator if the data does not already exist. 

# Citation
If you like this paper, please cite us:
```
@inproceedings{sarch2022tidee,
            title = "TIDEE: Tidying Up Novel Rooms using Visuo-Semantic Common Sense Priors",
            author = "Sarch, Gabriel and Fang, Zhaoyuan and Harley, Adam W. and Schydlo, Paul and Tarr, Michael J. and Gupta, Saurabh and Fragkiadaki, Katerina", 
            booktitle = "European Conference on Computer Vision",
            year = "2022"}
```

