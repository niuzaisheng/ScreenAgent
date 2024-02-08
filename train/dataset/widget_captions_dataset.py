import os
import json
import csv
import random

from PIL import Image, ImageDraw
import torch

from torch.utils.data import Dataset, ConcatDataset
from jinja2 import Template as JinjaTemplate

class WidgetCaptionsDataset(Dataset):
    def __init__(self, image_processor, cross_image_processor, text_processor,
                 widget_captions_file, image_dir, split, split_file_dir, template_dir) -> None:

        super().__init__()
        self.image_processor, self.text_processor, self.cross_image_processor = image_processor, text_processor, cross_image_processor

        self.split=split
        self.split_file_dir = split_file_dir
        self.split_file = os.path.join(self.split_file_dir, self.split + ".txt")
        with open(self.split_file, "r") as f:
            self.split_file = f.readlines()
        self.split_file = [item.strip() for item in self.split_file]

        self.widget_captions_file = widget_captions_file
        # read csv file to dicts:{datasetId,screenId,nodeId,captions}
        with open(widget_captions_file, "r") as f:
            reader = csv.reader(f)
            self.widget_captions = list(reader)[1:]

        processed_widget_captions = []
        for i in range(len(self.widget_captions)):
            screenId=self.widget_captions[i][1]
            if screenId not in self.split_file:
                continue
            captions = self.widget_captions[i][3]
            captions = captions.split("|")
            item = {
                "datasetId": self.widget_captions[i][0],
                "screenId": screenId,
                "nodeId": self.widget_captions[i][2],
                "captions": captions,
            }
            processed_widget_captions.append(item)

        self.widget_captions = processed_widget_captions

        self.image_dir = image_dir
        self.template_dir = template_dir

        self.template_file_name = {
            "send_prompt_en": self.load_templates("send_prompt_en.txt"),
            "answer_click_en": self.load_templates("answer_click_en.txt"),
            "send_prompt_zh": self.load_templates("send_prompt_zh.txt"),
            "answer_click_zh": self.load_templates("answer_click_zh.txt"),
        }

        super().__init__()

    def load_templates(self, template_file_name):
        with open(os.path.join(self.template_dir, "prompts", template_file_name) , "r") as f:
            template = f.read()
        return JinjaTemplate(template)
    
    def process_img(self, img):
        img_dict = {'vision': self.image_processor(img)}
        if self.cross_image_processor:
            img_dict.update({'cross': self.cross_image_processor(img)})
        return img_dict


    def get_node_box(self, screen_id, node_id):
        index_list = [int(i) for i in node_id.split(".")[1:]]
        with open(os.path.join(self.image_dir, screen_id + ".json")) as f:
            view = json.load(f)
        curr_node = view["activity"]["root"]
        for index in index_list:
            curr_node = curr_node["children"][index]
        bounds = curr_node["bounds"]
        return bounds
    
    def get_image(self, index):

        widget_caption = self.widget_captions[index]
        screen_id = widget_caption["screenId"]

        image_path = os.path.join(self.image_dir, screen_id + ".jpg")
        image = Image.open(image_path).convert("RGB")

        now_size = image.size

        bbox = self.get_node_box(screen_id, widget_caption["nodeId"])
        # 注意这个数据集图像的尺寸并不与标注的尺寸一致，标注是以1440*2560为基准的，而图像的尺寸是压缩过的。
        bbox[0] = int(bbox[0] / 1440 * now_size[0])
        bbox[1] = int(bbox[1] / 2560 * now_size[1])
        bbox[2] = int(bbox[2] / 1440 * now_size[0])
        bbox[3] = int(bbox[3] / 2560 * now_size[1])

        max_size = (1120, 1120)
        scale = min(max_size[0] / image.size[0], max_size[1] / image.size[1])
        image = image.resize((int(image.size[0] * scale), int(image.size[1] * scale)), Image.BILINEAR)
        bbox = [x * scale for x in bbox]

        input_size = image.size

        # 将图像paddding到 1120x1120
        padded_image = Image.new("RGB", max_size, color="white")
        padded_image.paste(image, (0, 0))
        image = padded_image

        # For Debug 将bbox 画在图片上
        # draw = ImageDraw.Draw(image)
        # draw.rectangle(bbox, outline='red', width=5)
        # image.save(f"WidgetCaptions_{index}.png")

        return image, bbox, input_size

    def __len__(self):
        return len(self.widget_captions)

    def __getitem__(self, index):

        question_id = f"{self.widget_captions[index]['datasetId']}_{self.widget_captions[index]['screenId']}"
        widget_caption = self.widget_captions[index]
        image, bbox, input_size = self.get_image(index)

        caption = random.choice(widget_caption["captions"])

        language = random.choice(["en","zh"])
        send_prompt_template = self.template_file_name[f"send_prompt_{language}"]

        send_prompt = send_prompt_template.render(
            video_width = input_size[0],
            video_height = input_size[1],
            task_prompt = caption
        )

        answer_template = self.template_file_name[f"answer_click_{language}"]
        answer = answer_template.render(
            task_prompt = caption,
            center_width = int((bbox[0] + bbox[2]) / 2), 
            center_height = int((bbox[1] + bbox[3]) / 2),
        )

        img_dict = self.process_img(image)

        text_dict = self.text_processor(answer, send_prompt)

        ret = {**img_dict, **text_dict, "question_id": question_id}
        return ret

