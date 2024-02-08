import os
import re
import json
import pickle
import random
import time
import itertools
import glob

import numpy as np
from PIL import Image, ImageDraw
import skimage.io as io
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon, Rectangle

from torch.utils.data import Dataset, ConcatDataset
from typing import Dict, Any, Mapping
from jinja2 import Template as JinjaTemplate

class REFER:
    def __init__(self, data_root, vis_root, dataset='refcoco', splitBy='unc'):
        # provide data_root folder which contains refclef, refcoco, refcoco+ and refcocog
        # also provide dataset name and splitBy information
        # e.g., dataset = 'refcoco', splitBy = 'unc'
        dataset = dataset.split('inv')[-1]  # inv dataset is stored in the same path as normal dataset
        print('loading dataset %s into memory...' % dataset)
        self.ann_dir = os.path.join(data_root, dataset)
        if dataset in ['refcoco', 'refcoco+', 'refcocog']:
            self.vis_root = vis_root
        elif dataset == 'refclef':
            raise 'No RefClef image data'
        else:
            raise 'No refer dataset is called [%s]' % dataset

        # load refs from data/dataset/refs(dataset).json
        tic = time.time()
        ref_file = os.path.join(self.ann_dir, 'refs(' + splitBy + ').p')
        self.data = {}
        self.data['dataset'] = dataset
        self.data['refs'] = pickle.load(open(ref_file, 'rb'))

        # load annotations from data/dataset/instances.json
        instances_file = os.path.join(self.ann_dir, 'instances.json')
        instances = json.load(open(instances_file, 'r'))
        self.data['images'] = instances['images']
        self.data['annotations'] = instances['annotations']
        self.data['categories'] = instances['categories']

        # create index
        self.createIndex()
        print('DONE (t=%.2fs)' % (time.time() - tic))

    def createIndex(self):
        # create sets of mapping
        # 1)  Refs: 	 	{ref_id: ref}
        # 2)  Anns: 	 	{ann_id: ann}
        # 3)  Imgs:		 	{image_id: image}
        # 4)  Cats: 	 	{category_id: category_name}
        # 5)  Sents:     	{sent_id: sent}
        # 6)  imgToRefs: 	{image_id: refs}
        # 7)  imgToAnns: 	{image_id: anns}
        # 8)  refToAnn:  	{ref_id: ann}
        # 9)  annToRef:  	{ann_id: ref}
        # 10) catToRefs: 	{category_id: refs}
        # 11) sentToRef: 	{sent_id: ref}
        # 12) sentToTokens: {sent_id: tokens}
        print('creating index...')
        # fetch info from instances
        Anns, Imgs, Cats, imgToAnns = {}, {}, {}, {}
        for ann in self.data['annotations']:
            Anns[ann['id']] = ann
            imgToAnns[ann['image_id']] = imgToAnns.get(ann['image_id'], []) + [ann]
        for img in self.data['images']:
            Imgs[img['id']] = img
        for cat in self.data['categories']:
            Cats[cat['id']] = cat['name']

        # fetch info from refs
        Refs, imgToRefs, refToAnn, annToRef, catToRefs = {}, {}, {}, {}, {}
        Sents, sentToRef, sentToTokens = {}, {}, {}
        for ref in self.data['refs']:
            # ids
            ref_id = ref['ref_id']
            ann_id = ref['ann_id']
            category_id = ref['category_id']
            image_id = ref['image_id']

            # add mapping related to ref
            Refs[ref_id] = ref
            imgToRefs[image_id] = imgToRefs.get(image_id, []) + [ref]
            catToRefs[category_id] = catToRefs.get(category_id, []) + [ref]
            refToAnn[ref_id] = Anns[ann_id]
            annToRef[ann_id] = ref

            # add mapping of sent
            for sent in ref['sentences']:
                Sents[sent['sent_id']] = sent
                sentToRef[sent['sent_id']] = ref
                sentToTokens[sent['sent_id']] = sent['tokens']

        # create class members
        self.Refs = Refs
        self.Anns = Anns
        self.Imgs = Imgs
        self.Cats = Cats
        self.Sents = Sents
        self.imgToRefs = imgToRefs
        self.imgToAnns = imgToAnns
        self.refToAnn = refToAnn
        self.annToRef = annToRef
        self.catToRefs = catToRefs
        self.sentToRef = sentToRef
        self.sentToTokens = sentToTokens
        print('index created.')

    def getRefIds(self, image_ids=[], cat_ids=[], ref_ids=[], split=''):
        image_ids = image_ids if type(image_ids) == list else [image_ids]
        cat_ids = cat_ids if type(cat_ids) == list else [cat_ids]
        ref_ids = ref_ids if type(ref_ids) == list else [ref_ids]

        if len(image_ids) == len(cat_ids) == len(ref_ids) == len(split) == 0:
            refs = self.data['refs']
        else:
            if not len(image_ids) == 0:
                refs = [self.imgToRefs[image_id] for image_id in image_ids]
            else:
                refs = self.data['refs']
            if not len(cat_ids) == 0:
                refs = [ref for ref in refs if ref['category_id'] in cat_ids]
            if not len(ref_ids) == 0:
                refs = [ref for ref in refs if ref['ref_id'] in ref_ids]
            if not len(split) == 0:
                if split in ['testA', 'testB', 'testC']:
                    refs = [ref for ref in refs if
                            split[-1] in ref['split']]  # we also consider testAB, testBC, ...
                elif split in ['testAB', 'testBC', 'testAC']:
                    refs = [ref for ref in refs if ref['split'] == split]  # rarely used I guess...
                elif split == 'test':
                    refs = [ref for ref in refs if 'test' in ref['split']]
                elif split == 'train' or split == 'val':
                    refs = [ref for ref in refs if ref['split'] == split]
                else:
                    raise 'No such split [%s]' % split
        ref_ids = [ref['ref_id'] for ref in refs]
        return ref_ids

    def getAnnIds(self, image_ids=[], cat_ids=[], ref_ids=[]):
        image_ids = image_ids if type(image_ids) == list else [image_ids]
        cat_ids = cat_ids if type(cat_ids) == list else [cat_ids]
        ref_ids = ref_ids if type(ref_ids) == list else [ref_ids]

        if len(image_ids) == len(cat_ids) == len(ref_ids) == 0:
            ann_ids = [ann['id'] for ann in self.data['annotations']]
        else:
            if not len(image_ids) == 0:
                lists = [self.imgToAnns[image_id] for image_id in image_ids if image_id in self.imgToAnns]  # list of [anns]
                anns = list(itertools.chain.from_iterable(lists))
            else:
                anns = self.data['annotations']
            if not len(cat_ids) == 0:
                anns = [ann for ann in anns if ann['category_id'] in cat_ids]
            ann_ids = [ann['id'] for ann in anns]
            if not len(ref_ids) == 0:
                ids = set(ann_ids).intersection(set([self.Refs[ref_id]['ann_id'] for ref_id in ref_ids]))
        return ann_ids

    def getImgIds(self, ref_ids=[]):
        ref_ids = ref_ids if type(ref_ids) == list else [ref_ids]

        if not len(ref_ids) == 0:
            image_ids = list(set([self.Refs[ref_id]['image_id'] for ref_id in ref_ids]))
        else:
            image_ids = self.Imgs.keys()
        return image_ids

    def getCatIds(self):
        return self.Cats.keys()

    def loadRefs(self, ref_ids=[]):
        if type(ref_ids) == list:
            return [self.Refs[ref_id] for ref_id in ref_ids]
        elif type(ref_ids) == int:
            return [self.Refs[ref_ids]]

    def loadAnns(self, ann_ids=[]):
        if type(ann_ids) == list:
            return [self.Anns[ann_id] for ann_id in ann_ids]
        elif type(ann_ids) == int:
            return [self.Anns[ann_ids]]

    def loadImgs(self, image_ids=[]):
        if type(image_ids) == list:
            return [self.Imgs[image_id] for image_id in image_ids]
        elif type(image_ids) == int:
            return [self.Imgs[image_ids]]

    def loadCats(self, cat_ids=[]):
        if type(cat_ids) == list:
            return [self.Cats[cat_id] for cat_id in cat_ids]
        elif type(cat_ids) == int:
            return [self.Cats[cat_ids]]

    def getRefBox(self, ref_id):
        ref = self.Refs[ref_id]
        ann = self.refToAnn[ref_id]
        return ann['bbox']  # [x, y, w, h]

    def showRef(self, ref, seg_box='box'):
        ax = plt.gca()
        # show image
        image = self.Imgs[ref['image_id']]
        I = io.imread(os.path.join(self.vis_root, image['file_name']))
        ax.imshow(I)
        # show refer expression
        for sid, sent in enumerate(ref['sentences']):
            print('%s. %s' % (sid + 1, sent['sent']))
        # show segmentations
        if seg_box == 'seg':
            ann_id = ref['ann_id']
            ann = self.Anns[ann_id]
            polygons = []
            color = []
            c = 'none'
            if type(ann['segmentation'][0]) == list:
                # polygon used for refcoco*
                for seg in ann['segmentation']:
                    poly = np.array(seg).reshape((len(seg) / 2, 2))
                    polygons.append(Polygon(poly, True, alpha=0.4))
                    color.append(c)
                p = PatchCollection(polygons, facecolors=color, edgecolors=(1, 1, 0, 0), linewidths=3, alpha=1)
                ax.add_collection(p)  # thick yellow polygon
                p = PatchCollection(polygons, facecolors=color, edgecolors=(1, 0, 0, 0), linewidths=1, alpha=1)
                ax.add_collection(p)  # thin red polygon
            else:
                # mask used for refclef
                raise NotImplementedError('RefClef is not downloaded')
        # show bounding-box
        elif seg_box == 'box':
            ann_id = ref['ann_id']
            ann = self.Anns[ann_id]
            bbox = self.getRefBox(ref['ref_id'])
            box_plot = Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], fill=False, edgecolor='green', linewidth=3)
            ax.add_patch(box_plot)


class COCODatasetBase(Dataset):
    
    def __init__(self, image_processor, cross_image_processor, text_processor,
                 vis_root, ann_path, templates_path, 
                 dataset='refcoco', splitBy='unc', split="train",
                 answer_digit_regression=False):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """

        self.image_processor = image_processor
        self.cross_image_processor = cross_image_processor
        self.text_processor = text_processor

        self.vis_root = vis_root
        self.split = split
        self.templates_path = templates_path
        self.answer_digit_regression = answer_digit_regression

        self.refer = REFER(ann_path, vis_root, dataset, splitBy)
        self.ref_ids = self.refer.getRefIds(split=split)

    def load_templates(self, template_file_name, for_digit_regression=False):
        with open(os.path.join(self.templates_path, template_file_name) , "r") as f:
            template = f.read()
        return JinjaTemplate(template)
    
    def __len__(self):
        return len(self.ref_ids)

    def __getitem__(self, index):
        data = self.preprocess(index)
        return data

    def get_image(self, index):
        ref_id = self.ref_ids[index]
        ref = self.refer.loadRefs(ref_id)[0]

        if self.split == "train":
            image_file = 'COCO_train2014_{:0>12}.jpg'.format(ref["image_id"])
        elif self.split == "val":
            image_file = re.sub(r'_\d+\.jpg$', '.jpg', ref["file_name"])

        image_path = os.path.join(self.vis_root, image_file)
        image = Image.open(image_path).convert("RGB")

        bbox = self.refer.getRefBox(ref['ref_id'])
        bbox = [
            bbox[0],
            bbox[1],
            (bbox[0] + bbox[2]),
            (bbox[1] + bbox[3])
        ]

        max_size = (1120, 1120)
        scale = min(max_size[0] / image.size[0], max_size[1] / image.size[1])
        image = image.resize((int(image.size[0] * scale), int(image.size[1] * scale)), Image.BILINEAR)
        bbox = [x * scale for x in bbox]

        padded_image = Image.new("RGB", max_size, color="white")
        padded_image.paste(image, (0, 0))
        image = padded_image

        return image, bbox

    def process_img(self, img):
        img_dict = {'vision': self.image_processor(img)}
        if self.cross_image_processor:
            img_dict.update({'cross': self.cross_image_processor(img)})
        return img_dict


class ScreenAgentReferCOCODataset(COCODatasetBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.template_file_name = {
            "send_prompt_en": self.load_templates("refer_send_prompt_en.txt"),
            "answer_click_en": self.load_templates("refer_answer_click_en.txt", for_digit_regression=self.answer_digit_regression),
            "answer_drag_en": self.load_templates("refer_answer_drag_en.txt", for_digit_regression=self.answer_digit_regression),

            "send_prompt_zh": self.load_templates("refer_send_prompt_zh.txt"),
            "answer_click_zh": self.load_templates("refer_answer_click_zh.txt", for_digit_regression=self.answer_digit_regression),
            "answer_drag_zh": self.load_templates("refer_answer_drag_zh.txt", for_digit_regression=self.answer_digit_regression)
        }

    def preprocess(self, index):

        ref_id = self.ref_ids[index]
        ref = self.refer.loadRefs(ref_id)[0]

        image, bbox = self.get_image(index)
        img_dict = self.process_img(image)

        video_width = image.size[0]
        video_height = image.size[1]
        task_prompt = random.choice(ref['sentences'])['sent']

        language = random.choice(["en","zh"])
        send_prompt_template = self.template_file_name[f"send_prompt_{language}"]


        digital_regression_value = []
        action_type = random.choice(["click","drag"])
        if action_type == "click":
            send_prompt = send_prompt_template.render(
                video_width = video_width,
                video_height = video_height, 
                task_prompt = f"click {task_prompt}",
            )

            answer_template = self.template_file_name[f"answer_click_{language}"]
            answer = answer_template.render(
                task_prompt = task_prompt,
                center_width = int((bbox[0] + bbox[2]) / 2), 
                center_height = int((bbox[1] + bbox[3]) / 2),
            )
            if self.answer_digit_regression:
                answer, digital_regression_value = answer

        elif action_type == "drag":
            send_prompt = send_prompt_template.render(
                video_width = video_width,
                video_height = video_height, 
                task_prompt = f"drag draw a box of {task_prompt}",
            )

            answer_template = self.template_file_name[f"answer_drag_{language}"]
            answer = answer_template.render(
                task_prompt = task_prompt,
                drag_start_width = int(bbox[0]),
                drag_start_height = int(bbox[1]),
                drag_end_width = int(bbox[2]),
                drag_end_height = int(bbox[3]),
            )
            if self.answer_digit_regression:
                answer, digital_regression_value = answer

        if self.answer_digit_regression:
            digital_regression_value = [x/1000 for x in digital_regression_value]
            text_dict = self.text_processor(answer, send_prompt, digital_regression_value)
        else:
            text_dict = self.text_processor(answer, send_prompt)


        ret = {**img_dict, **text_dict, "question_id": f"coco_{ref_id}"}
        return ret

