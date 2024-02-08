import os
import json
import glob
import random

import torch
from PIL import Image
from torch.utils.data import Dataset

class ScreenAgentDataset(Dataset):

    def __init__(self, image_processor, cross_image_processor, text_processor, data_dirs,  **kwargs):
        super().__init__()
        self.image_processor, self.text_processor, self.cross_image_processor = image_processor, text_processor, cross_image_processor
    
        self.data_dirs = data_dirs

        self.data = []
        for json_path in glob.glob(os.path.join(data_dirs, "*", "*.json")):
            session_id = json_path.split("/")[-2]
            with open(json_path, 'r') as f:
                one_example = json.load(f)
                one_example["session_id"] = session_id
                one_example["saved_image_path"] = os.path.join(data_dirs, session_id, "images", one_example["saved_image_name"])
                self.data.append(one_example)

    def process_img(self, img):
        img_dict = {'vision': self.image_processor(img)}
        if self.cross_image_processor:
            img_dict.update({'cross': self.cross_image_processor(img)})
        return img_dict

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        info = self.data[index]
        question_id = info["session_id"] + "_" + info["saved_image_name"].split(".")[0]

        image_path = info["saved_image_path"]
        image = Image.open(image_path).convert("RGB")

        max_size = (1120, 1120)
        padded_image = Image.new("RGB", max_size, color="white")
        padded_image.paste(image, (0, 0))
        image = padded_image


        img_dict = self.process_img(image)

        language = random.choice(["en","zh"])
        if language == "en":
            send_prompt = info["send_prompt_en"]
            answer = info["LLM_response_editer_en"]
        else:
            send_prompt = info["send_prompt_zh"]
            answer = info["LLM_response_editer_zh"]

        text_dict = self.text_processor(answer, send_prompt)

        ret = {**img_dict, **text_dict, "question_id": question_id}
        return ret
