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
import sys
import os
import glob
from PIL import Image
import json

import ipdb
st = ipdb.set_trace

class NSDDataloader(torch.utils.data.Dataset):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """
    def __init__(self, args, split, transform, transform2=None, fetch_metadata=True, fetch_seg=True, save_dir=None, skip_if_seg_exists=False):
        super(NSDDataloader, self).__init__()

        self.args = args
        
        images = [d for d in tqdm(glob.glob(args.gqa_path + '/allImages/images' + '/*.jpg'))]

        print("loading json file...")
        if split=="val":
            scene_graph_path = os.path.join(args.gqa_path, 'sceneGraphs', 'val_sceneGraphs.json')
            f = open(scene_graph_path)
            data = json.load(f)
            f.close()
        elif split=="train":
            scene_graph_path = os.path.join(args.gqa_path, 'sceneGraphs', 'train_sceneGraphs.json')
            f = open(scene_graph_path)
            data = json.load(f)
            f.close()
        elif split=="both":
            scene_graph_path = os.path.join(args.gqa_path, 'sceneGraphs', 'val_sceneGraphs.json')
            f = open(scene_graph_path)
            data = json.load(f)
            f.close()
            scene_graph_path = os.path.join(args.gqa_path, 'sceneGraphs', 'train_sceneGraphs.json')
            f = open(scene_graph_path)
            data2 = json.load(f)
            f.close()
            data.update(data2)
            del data2

        # get statistics of dataset
        get_rel_att_statistics = False
        if get_rel_att_statistics:
            '''
            Get statistics of GQA dataset
            '''
            classes = {}
            relations = {}
            attributes = {}
            num_objects = 0
            for k in tqdm(data.keys()):
                objects = data[k]['objects']
                for k_o in objects.keys():
                    class_ = objects[k_o]['name']
                    if objects[k_o]['name'] not in classes.keys():
                        classes[class_] = 0
                    classes[class_] += 1

                    rels = objects[k_o]['relations']
                    rels_ = []
                    for rel in rels:
                        rels_.append(rel['name'])

                    rels_ = list(set(rels_))
                    if 'to the right of' in rels_ and 'to the left of' in rels_:
                        rels_.remove('to the right of')
                        rels_.remove('to the left of')
                    for rel in rels_:
                        if rel not in relations.keys():
                            relations[rel] = 0
                        relations[rel] += 1

                    attrs = list(set(objects[k_o]['attributes']))
                    for attr in attrs:
                        if attr not in attributes.keys():
                            attributes[attr] = 0
                        attributes[attr] += 1
                    
                    num_objects += 1
            for k in classes.keys():
                classes[k] /= num_objects
            for k in relations.keys():
                relations[k] /= num_objects
            for k in attributes.keys():
                attributes[k] /= num_objects
            classes_sorted = sorted(classes.items(), key=lambda x:x[1])
            relations_sorted = sorted(relations.items(), key=lambda x:x[1])
            attributes_sorted = sorted(attributes.items(), key=lambda x:x[1])
            print(classes_sorted[-20:])
            print(relations_sorted[-20:])
            print(attributes_sorted[-20:])

            classes_path = os.path.join(args.gqa_path, 'classes_statistics.json')
            relations_path = os.path.join(args.gqa_path, 'relations_statistics.json')
            attributes_path = os.path.join(args.gqa_path, 'attributes_statistics.json')
            with open(classes_path, 'w') as f:
                json.dump(classes, f)
            with open(relations_path, 'w') as f:
                json.dump(relations, f)
            with open(attributes_path, 'w') as f:
                json.dump(attributes, f)
            st()


        # for saving segmentation labels
        # seg labels should be 
        # seg.shape is 1, 6, 56, 56 -> second dim is multiple
        # have 1 dim for class, 5 for relations, 3 for attributes
        seglabels_path = os.path.join(args.gqa_path, 'seglabels.json')
        if not os.path.exists(seglabels_path):
            classes = set()
            relations = set()
            attributes = set()
            max_relations = [] #0
            max_attributes = []
            for k in tqdm(data.keys()):
                objects = data[k]['objects']
                for k_o in objects.keys():
                    classes.add(objects[k_o]['name'])
                    rels = objects[k_o]['relations']
                    for rel in rels:
                        relations.add(rel['name'])
                    attrs = list(set(objects[k_o]['attributes']))
                    max_attributes.append(len(attrs))
                    for attr in attrs:
                        attributes.add(attr)
            seglabels = list(classes) + list(relations) + list(attributes)
            # num classes=1703, num relations=310, num attributes=617
            with open(seglabels_path, 'w') as f:
                json.dump(seglabels, f)
            # max_attributes = 11
            # unique relations per object + counts (array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13]), array([257655, 562363, 447916, 106049,  21859,   6559,   2122,    637, 214,     65,     17,      5,      3,      1]))
            # unique attributes per object + counts (array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11]), array([793841, 488458,  96676,  19780,   4819,   1286,    416,    126, 40,     17,      5,      1]))
        f = open(seglabels_path)
        seglabels = json.load(f)
        f.close()

        self.save_dir = save_dir
        self.skip_if_seg_exists = skip_if_seg_exists

        self.seglabels = seglabels
        self.seglabels.insert(0, '-') # insert 0 for background
        self.metalabels = ["object"]*(1703+1) + ["relation"]*310 + ["attribute"]*617
        self.segcatlabels = [(s,m) for s,m in zip(self.seglabels, self.metalabels)]
        self.seglabels_dict = {self.seglabels[idx]:idx for idx in range(len(self.seglabels))}
        self.seglabels_dict_r = {v:k for k,v in self.seglabels_dict.items()}
        self.fetch_metadata = fetch_metadata
        self.fetch_seg = fetch_seg

        self.label_to_rgb = {}
        count = 0
        for r in range(0,256,1):
            for g in range(0,256,1):
                for b in range(0,256,1):
                    self.label_to_rgb[count] = (r,g,b)
                    count += 1
        self.rgb_to_label = {v:k for k,v in self.label_to_rgb.items()}

        self.data = data
        self.images = images

        self.data_keys = [im.split('/')[-1].split('.jpg')[0] for im in self.images]
        self.images = [self.images[im_i] for im_i in range(len(self.data_keys)) if self.data_keys[im_i] in self.data.keys()]
        self.data_keys = [self.data_keys[im_i] for im_i in range(len(self.data_keys)) if self.data_keys[im_i] in self.data.keys()]

        print("done.")
        
        self.transform = transform
        self.transform2 = transform2

    def filter_topk_images_from_roi_brain_response(self, responses, topk=1000):
        '''
        responses should be same size as self.ids and self.images
        '''

        # get topk images from brain responses
        mean_responses = np.mean(responses, axis=1)
        mean_responses_argsort = np.argsort(-mean_responses)
        topk_indices = list(mean_responses_argsort[:topk])
        self.data_keys = [self.data_keys[idx] for idx in topk_indices]
        self.images = [self.images[idx] for idx in topk_indices]

    def __len__(self):
        return len(self.images) #len(self.ids)

    def get_metadata(self, index, im_size):
        metadata = {}
        data_key = self.data_keys[index]
        data_dict = self.data[data_key]
        w,h = data_dict['width'], data_dict['height']
        object_metadata = data_dict['objects']
        obj_bboxes = np.zeros((len(object_metadata), 4), dtype=np.float64)
        labels_objs = np.zeros((len(object_metadata), 13), dtype=np.int32) # have 1 dim for class, 5 for relations, 3 for attributes
        object_keys = list(object_metadata.keys())
        for obj_idx in range(len(object_metadata)):
            '''
            x   int	Horizontal position of the object bounding box (top left).
            y	int	Vertical position of the object bounding box (top left).
            w	int	The object bounding box width in pixels.
            h	int	The object bounding box height in pixels.
            '''
            obj_k = object_keys[obj_idx]
            obj = object_metadata[obj_k]
            x1, y1, x2, y2 = obj['x'], obj['y'], obj['x'] + obj['w'], obj['y'] + obj['h']
            obj_bboxes[obj_idx] = np.array([x1/w, y1/h, x2/w, y2/h])
            name = obj['name']
            labels_objs[obj_idx, 0] = self.seglabels_dict[name] # class is first index
            relations = np.asarray([self.seglabels_dict[rel['name']] for rel in obj['relations']])
            relations, counts = np.unique(relations, return_counts=True)
            if (self.seglabels_dict['to the left of'] in relations) and (self.seglabels_dict['to the right of'] in relations):
                # if it contains both left of and right of then remove since it is ambiguous
                relations = np.asarray([value for value in list(relations) if value not in [self.seglabels_dict['to the right of'], self.seglabels_dict['to the left of']]])
                relations, counts = np.unique(relations, return_counts=True)
            relations = relations[np.argsort(-counts)]
            relations = relations[:7] # max of 5 relations
            relations = np.pad(relations, (0, 7-len(relations)))
            labels_objs[obj_idx, 1:8] = relations # relations are 1-8

            attributes = np.asarray([self.seglabels_dict[att] for att in obj['attributes']])
            attributes, counts = np.unique(attributes, return_counts=True)
            attributes = attributes[np.argsort(-counts)]
            attributes = attributes[:5] # max of 3 attributes
            relations = np.pad(attributes, (0, 5-len(attributes)))
            labels_objs[obj_idx, 8:] = relations # relations are 8-13
        metadata["labels_objs"] = labels_objs
        metadata["obj_bboxes"] = obj_bboxes
        metadata["ID"] = data_key
        return metadata

    def fetch_segmentation_maps(self, index):
        # this is a bit slow - can we speed it up?
        data_key = self.data_keys[index]
        segms = np.zeros((13, self.args.image_size_eval, self.args.image_size_eval), dtype=np.int32)
        for seg_idx in range(9):
            seg_path = os.path.join(self.args.gqa_path, 'segmentation', f'{data_key}_{seg_idx}.png')
            segm = np.asarray(Image.open(seg_path).convert('RGB'))
            unique_colors = np.unique(segm.reshape(-1, segm.shape[2]), axis=0)
            for color in unique_colors:
                Y, X = np.where(np.all(segm==color,axis=2))
                segms[seg_idx, Y, X] = self.rgb_to_label[tuple(color)]
        return segms

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        out = {}

        # index = self.data_keys.index('2335811')
        if self.skip_if_seg_exists:
            # used for data generation
            data_key = self.data_keys[index]
            seg_path = os.path.join(self.save_dir, 'segmentation', f'{data_key}_12.png')
            if os.path.exists(seg_path):
                out["skip"] = True
                return out
        
        out["skip"] = False

        image_path = self.images[index]
        image = Image.open(image_path).convert('RGB')
        
        if self.fetch_metadata:
            metadata = self.get_metadata(index, im_size=image.size)
            out['metadata'] = metadata
        
        if self.fetch_seg:
            segms = self.fetch_segmentation_maps(index)
            out['segms'] = segms

        if self.transform2 is not None:
            image2 = self.transform2(image)
            out['images2'] = image2
        image = self.transform(image)

        out['images'] = image

        return out 


if __name__ == '__main__':
    '''
    Saves segmentation images from GQA
    '''
    import torch
    import torch.nn.functional as F
    from PIL import Image
    import requests
    from transformers import SamModel, SamProcessor
    import argparse
    import numpy as np
    import ipdb
    st = ipdb.set_trace
    import matplotlib.pyplot as plt
    import cv2
    parser = argparse.ArgumentParser()
    parser.add_argument("--gqa_path", type=str, default='./gqa', help="Path to gqa data")
    parser.add_argument("--save_path", type=str, default=None, help="Path to gqa data")
    parser.add_argument("--image_size", type=int, default=480, help="")
    parser.add_argument("--image_size_eval", type=int, default=56, help="")
    parser.add_argument("--seed", type=int, default=27, help="")
    parser.add_argument("--use_boxes", action="store_true", default=False, help="")
    parser.add_argument("--debug", action="store_true", default=False, help="")
    args = parser.parse_args()

    # fix the seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.save_path is None:
        args.save_path = args.gqa_path

    # get statistics of dataset
    get_rel_att_statistics = True
    if get_rel_att_statistics:
        '''
        Get statistics of GQA dataset
        '''
        scene_graph_path = os.path.join(args.gqa_path, 'sceneGraphs', 'val_sceneGraphs.json')
        f = open(scene_graph_path)
        data = json.load(f)
        f.close()
        scene_graph_path = os.path.join(args.gqa_path, 'sceneGraphs', 'train_sceneGraphs.json')
        f = open(scene_graph_path)
        data2 = json.load(f)
        f.close()
        data.update(data2)
        del data2
        classes = {}
        relations = {}
        attributes = {}
        num_images = 0
        for k in tqdm(data.keys()):
            classes_ = {}
            relations_ = {}
            attributes_ = {}
            objects = data[k]['objects']
            for k_o in objects.keys():
                class_ = objects[k_o]['name']
                if class_ not in classes_.keys():
                    classes_[class_] = 1

                rels = objects[k_o]['relations']
                rels_ = []
                for rel in rels:
                    rels_.append(rel['name'])

                rels_ = list(set(rels_))
                if 'to the right of' in rels_ and 'to the left of' in rels_:
                    rels_.remove('to the right of')
                    rels_.remove('to the left of')
                for rel in rels_:
                    if rel not in relations_.keys():
                        relations_[rel] = 1

                attrs = list(set(objects[k_o]['attributes']))
                for attr in attrs:
                    if attr not in attributes_.keys():
                        attributes_[attr] = 1

            for k in classes_.keys():
                if k not in classes.keys():
                    classes[k] = 0
                classes[k] += 1

            for k in relations_.keys():
                if k not in relations.keys():
                    relations[k] = 0
                relations[k] += 1

            for k in attributes_.keys():
                if k not in attributes.keys():
                    attributes[k] = 0
                attributes[k] += 1
                
            num_images += 1
        for k in classes.keys():
            classes[k] /= num_images
        for k in relations.keys():
            relations[k] /= num_images
        for k in attributes.keys():
            attributes[k] /= num_images
        classes_sorted = sorted(classes.items(), key=lambda x:x[1])
        relations_sorted = sorted(relations.items(), key=lambda x:x[1])
        attributes_sorted = sorted(attributes.items(), key=lambda x:x[1])
        print(classes_sorted[-20:])
        print(relations_sorted[-20:])
        print(attributes_sorted[-20:])

        classes_path = os.path.join(args.save_path, 'classes_statistics_image.json')
        relations_path = os.path.join(args.save_path, 'relations_statistics_image.json')
        attributes_path = os.path.join(args.save_path, 'attributes_statistics_image.json')
        with open(classes_path, 'w') as f:
            json.dump(classes, f)
        with open(relations_path, 'w') as f:
            json.dump(relations, f)
        with open(attributes_path, 'w') as f:
            json.dump(attributes, f)
    
    rgb = {}
    count = 0
    for r in range(0,256,1):
        for g in range(0,256,1):
            for b in range(0,256,1):
                rgb[count] = [r,g,b]
                count += 1

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
    dataset = NSDDataloader(args, "both", transform=transform_croponly, transform2=None, fetch_seg=False, save_dir=args.save_path, skip_if_seg_exists=False if args.debug else True)
    seglabels_dict = dataset.seglabels_dict
    seglabels_dict_r = dataset.seglabels_dict_r
    dataset_loader = torch.utils.data.DataLoader(dataset,
                                                batch_size=1, 
                                                # sampler=sampler,
                                                num_workers=2, 
                                                # collate_fn=my_collate,
                                                pin_memory=True,
                                                drop_last=False,
                                                shuffle=True,
                                                )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam_model_name = "facebook/sam-vit-huge"
    # sam_model_name = "facebook/sam-vit-large"
    # sam_model_name = "facebook/sam-vit-big"
    SAM_model = SamModel.from_pretrained(sam_model_name).to(device).eval()
    SAM_processor = SamProcessor.from_pretrained(sam_model_name)

    if args.use_boxes:
        save_folder = 'segmentation_boxes2'
    else:
        save_folder = 'segmentation'
    root_seg = os.path.join(args.save_path, save_folder)
    os.makedirs(root_seg, exist_ok = True)

    for i_batch, batched_samples in tqdm(enumerate(dataset_loader)):

        if batched_samples["skip"]:
            continue

        metadata = batched_samples["metadata"]
        obj_boxes = metadata["obj_bboxes"] * args.image_size
        labels_objs = metadata["labels_objs"].squeeze(0).cpu().numpy()
        image = batched_samples["images"].squeeze(0).permute(1,2,0).cpu().numpy()
        image_ID = metadata["ID"][0]

        if obj_boxes.shape[1]>0:
            if not args.use_boxes:
                try:
                    with torch.no_grad():
                        raw_image = Image.fromarray(np.uint8(image*255))
                        inputs = SAM_processor(raw_image, input_boxes=obj_boxes, return_tensors="pt").to(device)
                        outputs = SAM_model(**inputs, multimask_output=False)
                        masks = SAM_processor.image_processor.post_process_masks(
                            outputs.pred_masks.cpu(), torch.tensor([args.image_size_eval, args.image_size_eval]).unsqueeze(0), inputs["reshaped_input_sizes"].cpu()
                        )[0] # can just replace inputs["original_sizes"].cpu() with desired size
                        scores = outputs.iou_scores
                except KeyboardInterrupt:
                    sys.exit(0)
                except:
                    # sometimes SAM_processor fails?
                    obj_boxes_m = (obj_boxes / args.image_size) * args.image_size_eval
                    obj_boxes_m = obj_boxes_m.to(torch.long)
                    obj_boxes_m = obj_boxes_m.squeeze(0)
                    masks = torch.zeros((len(obj_boxes), args.image_size_eval, args.image_size_eval), dtype=torch.bool)
                    for seg_i in range(len(obj_boxes)):
                        masks[seg_i, obj_boxes_m[seg_i, 0]:obj_boxes_m[seg_i, 2], obj_boxes_m[seg_i, 1]:obj_boxes_m[seg_i, 3]] = True
            else:
                obj_boxes_m = (obj_boxes / args.image_size) * args.image_size_eval
                obj_boxes_m = obj_boxes_m.to(torch.long)
                obj_boxes_m = obj_boxes_m.squeeze(0)
                masks = torch.zeros((len(obj_boxes_m), args.image_size_eval, args.image_size_eval), dtype=torch.bool)
                for seg_i in range(len(obj_boxes)):
                    masks[seg_i, obj_boxes_m[seg_i, 0]:obj_boxes_m[seg_i, 2], obj_boxes_m[seg_i, 1]:obj_boxes_m[seg_i, 3]] = True
            # assign each pixel one object only - smallest objects get priority
            num_masks = torch.sum(masks.reshape(masks.shape[0], -1), dim=1)
            largest_to_smallest = torch.argsort(-num_masks).tolist()
        else:
            masks = torch.zeros((0, args.image_size_eval, args.image_size_eval), dtype=torch.bool)
            largest_to_smallest = []
        
        # assign each a color and add to one segmentation image
        seg_mask = np.zeros((13, 3, args.image_size_eval, args.image_size_eval), dtype=np.int32)
        for mask_i in largest_to_smallest:
            mask_ = masks[mask_i].squeeze(0)
            mask_ = mask_.cpu().numpy()
            labels = labels_objs[mask_i]
            colors = [rgb[l] for l in list(labels)]
            colors = np.asarray(colors)
            seg_mask[:,:,mask_] = np.repeat(colors[...,None], np.sum(mask_), axis=-1)

        if not args.debug:
            # save segmentation masks
            for seg_i in range(len(seg_mask)):
                seg = seg_mask[seg_i].transpose(1,2,0)
                im = Image.fromarray(np.uint8(seg))
                path = os.path.join(root_seg, f'{image_ID}_{seg_i}.png')
                im.save(path)

        visualize = False
        if visualize:
            '''
            Visualize mask output
            '''

            rgb_disp = np.uint8(image * 255)
            rgb_disp = rgb_disp.astype(np.int32)

            if (0):
                obj_boxes_ = obj_boxes.squeeze(0).cpu().numpy()
                masks_vis = F.interpolate(masks.to(torch.float), inputs["original_sizes"].cpu().squeeze(0).tolist(), mode="bilinear", align_corners=False)
                masks_vis = masks_vis > 0.
                for mask_i in range(len(masks)):
                    mask_ = masks_vis[mask_i].squeeze()
                    mask_ = mask_.cpu().numpy().copy()

                    print([seglabels_dict_r[l] for l in labels_objs[mask_i]])
                    
                    rgb_ = np.float32(rgb_disp.copy())
                    rect_th = 1
                    color = (0, 255, 0)
                    masked_img = np.where(mask_[...,None], color, rgb_)
                    rgb_ = cv2.addWeighted(rgb_, 0.8, np.float32(masked_img), 0.2,0)
                    box = obj_boxes_[mask_i].astype(np.int32)
                    cv2.rectangle(rgb_, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),color, rect_th)
                    plt.figure()
                    plt.imshow(rgb_/255)
                    plt.savefig(f'../../data/images/test.png')
                    plt.close()

            if (1):
                for mask_i in range(len(labels_objs)):
                    for l in labels_objs[mask_i]:
                        if l!=0:
                            print(seglabels_dict_r[l])
                obj_boxes_ = obj_boxes.squeeze(0).cpu().numpy()
                rgb_ = np.float32(rgb_disp.copy())
                for mask_i in range(len(labels_objs)):
                    rect_th = 1
                    color = (0, 255, 0)
                    box = obj_boxes_[mask_i].astype(np.int32)
                    cv2.rectangle(rgb_, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),color, rect_th)
                rgb_ = rgb_.astype(np.uint8)
                for s_i in range(seg_mask.shape[0]):
                    seg_image = np.transpose(seg_mask[s_i], [1,2,0]) 
                    to_plot = np.concatenate([rgb_, seg_image], axis=1)
                    plt.figure()
                    plt.imshow(to_plot)
                    plt.savefig(f'../../data/images/test.png')
                    plt.close()
                    st()

                # for seg in seg_mask:
                #     mask_seg = np.where(seg[2])
                #     seg[0,mask_seg[0],mask_seg[1]] = 255
                #     plt.figure()
                #     plt.imshow(np.transpose(seg, [1,2,0]))
                #     plt.savefig(f'../../data/images/test.png')

