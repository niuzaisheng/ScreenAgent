import random
from collections import namedtuple
from torch.utils.data import Dataset, ConcatDataset, Subset

from .coco_dataset import ScreenAgentReferCOCODataset
from .screenagent_dataset import ScreenAgentDataset
from .mind2web_dataset import Mind2WebFilteredDataset, Mind2WebConstructDataset
from .widget_captions_dataset import WidgetCaptionsDataset


class LimitedSubset(Subset):
    def __init__(self, dataset, n):
        super().__init__(dataset, range(min(n, len(dataset))))


class MultiMixedDataset(Dataset):
    def __init__(self, datasets, ratios, name = None):
        # first dataset is main dataset
        self.datasets = datasets
        self.ratios = ratios
        self.lengths = [len(dataset) for dataset in datasets]
        main_set_len = self.lengths[0]
        main_set_ratio = self.ratios[0]
        self.total_len = int(main_set_len / main_set_ratio)

        print(f"MultiMixedDataset: {name} total_len: {self.total_len}, main_set_len: {main_set_len}, main_set_ratio: {main_set_ratio}")

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        dataset_idx = random.choices(range(len(self.datasets)), weights=self.ratios, k=1)[0]
        dataset = self.datasets[dataset_idx]
        sample_idx = random.randint(0, len(dataset) - 1)
        return dataset[sample_idx]

def make_supervised_data_module(image_processor, cross_image_processor, text_processor, do_eval=True):

    answer_digit_regression = False
    coco_data_args = {
        "train_data_path": 'data/COCO/train2014',
        "eval_data_path": 'data/COCO/train2014',
        "ann_path": 'data/COCO/annotations',
        "templates_path": "data/COCO/prompts",
        "dataset": 'refcoco',
        "splitBy": 'unc',
    }
    coco_data_args = namedtuple('ARGS', coco_data_args.keys())(*coco_data_args.values())

    ScreenAgent_data_args = {
        "train_data_path": 'data/ScreenAgent/train',
        "eval_data_path": 'data/ScreenAgent/test',
    }
    ScreenAgent_data_args = namedtuple('ARGS', ScreenAgent_data_args.keys())(*ScreenAgent_data_args.values())

    Mind2WebConstruct_data_args = {
        "data_dir": "data/Mind2Web/processed_dataset"
    }
    Mind2WebConstruct_data_args = namedtuple('ARGS', Mind2WebConstruct_data_args.keys())(*Mind2WebConstruct_data_args.values())

    widget_captions_data_args = {
        "widget_captions_file": 'data/Rico/widget-caption/widget_captions.csv',
        "image_dir": 'data/Rico/combined',
        "split_file_dir": "data/Rico/widget-caption/split",
        "template_dir": "data/Rico",
    }
    widget_captions_data_args = namedtuple('ARGS', widget_captions_data_args.keys())(*widget_captions_data_args.values())


    screen_agent_dataset = ScreenAgentDataset(image_processor, cross_image_processor, text_processor, ScreenAgent_data_args.train_data_path)
    screen_agent_ReferCOCODataset = ScreenAgentReferCOCODataset(image_processor, cross_image_processor, text_processor,
                                        vis_root=coco_data_args.train_data_path, ann_path=coco_data_args.ann_path, templates_path=coco_data_args.templates_path,
                                        dataset=coco_data_args.dataset, splitBy=coco_data_args.splitBy, split="train", answer_digit_regression=answer_digit_regression)
    mind2web_ConstructDataset =  Mind2WebConstructDataset(image_processor, cross_image_processor, text_processor, Mind2WebConstruct_data_args.data_dir)
    widget_captionsDataset = WidgetCaptionsDataset(image_processor, cross_image_processor, text_processor,
                                        widget_captions_file = widget_captions_data_args.widget_captions_file,
                                        image_dir = widget_captions_data_args.image_dir,
                                        split = "train",
                                        split_file_dir = widget_captions_data_args.split_file_dir,
                                        template_dir = widget_captions_data_args.template_dir)

    train_dataset = ConcatDataset([
        MultiMixedDataset([
            screen_agent_dataset, 
            screen_agent_ReferCOCODataset,
            mind2web_ConstructDataset,
            widget_captionsDataset
        ], ratios = [0.3, 0.2, 0.3, 0.2]), # stage 1
        MultiMixedDataset([
            screen_agent_dataset, 
            screen_agent_ReferCOCODataset,
            mind2web_ConstructDataset,
            widget_captionsDataset
        ], ratios = [0.4, 0.1, 0.4, 0.1]), # stage 2
        MultiMixedDataset([
            screen_agent_dataset,
            mind2web_ConstructDataset
        ], ratios = [0.5, 0.5]), # stage 3
        MultiMixedDataset([
            screen_agent_dataset,
            mind2web_ConstructDataset
        ], ratios = [0.7, 0.3]), # stage 4
    ])

    eval_dataset = None
    each_eval_sample_num = 1
    if do_eval:
        eval_dataset = ConcatDataset([
            LimitedSubset(ScreenAgentReferCOCODataset(image_processor, cross_image_processor, text_processor,
                                    vis_root=coco_data_args.train_data_path, ann_path=coco_data_args.ann_path, templates_path=coco_data_args.templates_path,
                                    dataset=coco_data_args.dataset, splitBy=coco_data_args.splitBy, split="val", 
                                    answer_digit_regression=answer_digit_regression), each_eval_sample_num),
            # LimitedSubset(Mind2WebFilteredDataset(image_processor, cross_image_processor, text_processor, Mind2WebFiltered_data_args.data_dir, Mind2WebFiltered_data_args.eval_ann_file), each_eval_sample_num),
            LimitedSubset(Mind2WebConstructDataset(image_processor, cross_image_processor, text_processor, Mind2WebConstruct_data_args.data_dir), each_eval_sample_num), # TODO: eval
            LimitedSubset(WidgetCaptionsDataset(image_processor, cross_image_processor, text_processor,
                                    widget_captions_file = widget_captions_data_args.widget_captions_file,
                                    image_dir = widget_captions_data_args.image_dir,
                                    split = "dev",
                                    split_file_dir = widget_captions_data_args.split_file_dir,
                                    template_dir = widget_captions_data_args.template_dir), each_eval_sample_num),
            LimitedSubset(ScreenAgentDataset(image_processor, cross_image_processor, text_processor, ScreenAgent_data_args.eval_data_path), each_eval_sample_num),
        ])

    if train_dataset is not None:
        print("train_dataset: ", len(train_dataset))

    if eval_dataset is not None:
        print("eval_dataset: ", len(eval_dataset))

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)
