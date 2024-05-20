import os

import json
import copy
import random
import logging
import argparse
import numpy as np
from PIL import Image
from argparse import Namespace
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Sequence

import torch
from torch.utils.data import Dataset

import transformers
from transformers import TrainerCallback
from transformers import HfArgumentParser, TrainingArguments

from llava.model import *
from llava.utils import disable_torch_init
from llava.model_post.builder import load_pretrained_model
from llava.model_post.language_model.llava_llama import LlavaLlamaPIBForCausalLM
from llava.constants import IGNORE_INDEX
from llava import conversation as conversation_lib
from llava.train.train import preprocess_multimodal, preprocess

from peft.peft_model import PeftModelForCausalLM
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)

import sys
sys.path.append('.')

from post_interaction_block.trainer.llava_dpo_trainer_post import LlavaDPOTrainer

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)

def convert_dict_to_tensor(results, device):
    part_tensor = json.dumps(results)
    part_tensor = torch.Tensor([ord(part_tensor[i]) for i in range(len(part_tensor))]).long().to(device)
    return part_tensor

def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                #data_path=data_args.data_path,
                                vg_path=data_args.vg_path,
                                desc_data_path=data_args.desc_data_path,
                                pope_data_path=data_args.pope_data_path,
                                data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)
    
class DataArguments:
    vg_path: str = field(default=None, metadata={"help": "Path to the Visual Genome data."})
    desc_data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    pope_data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default="")
    image_aspect_ratio: str = 'square'
    
class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, 
        #data_path: str,
        vg_path: str,
        desc_data_path: str,
        pope_data_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        data_args: DataArguments,
        sample_strategy: str = "offline",
        seed: int = 42,
    ):
        super(LazySupervisedDataset, self).__init__()
        
        vg_image_data = json.load(open(os.path.join(vg_path, "image_data.json")))
        self.id2path = {
            _data["image_id"]:os.path.join(vg_path, _data["url"].split("/")[-2], _data["url"].split("/")[-1]) 
            for _data in vg_image_data
        }
        
        # preprocess
        desc_data = json.load(open(desc_data_path, "r"))
        pope_data = json.load(open(pope_data_path, "r"))
        random.seed(seed)
        desc_data_dict = self.desc_process(desc_data, sample_strategy)
        pope_data_dict = self.pope_process(pope_data)
        list_data_dict = pope_data_dict + desc_data_dict*2
        
        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args

    def desc_process(self, desc_data, sample_strategy):
        desc_data_dict = []
        for image_id in desc_data.keys():
            if sample_strategy == "offline":
                desc_data[image_id]['chosen'] = [random.choice(desc_data[image_id]['chosen'])]
                desc_data[image_id]['rejected'] = [random.choice(desc_data[image_id]['rejected'])]
            for chosen in desc_data[image_id]['chosen']:
                for rejected in desc_data[image_id]['rejected']:
                    question = random.choice([
                        "Describe this image in detail.",
                        "Take a look at this image and describe what you notice.",
                        "Please provide a detailed description of the picture.",
                        "Could you describe the contents of this image for me?",
                    ])
                    question = "<image>\n" + question
                    desc_data_dict.append({
                        "id": int(image_id),
                        "image": self.id2path[int(image_id)],
                        "chosen_conversations": [
                            {"from": "human", "value": question},
                            {"from": "gpt", "value": chosen},
                        ],
                        "reject_conversations": [
                            {"from": "human", "value": question},
                            {"from": "gpt", "value": rejected},
                        ],
                    })
        return desc_data_dict
        
    def pope_process(self, pope_data):
        pope_data_dict = []
        for idx in range(len(pope_data)):
            if pope_data[idx]['correct']:
                continue
            image_id = pope_data[idx]["image_id"]
            chosen = pope_data[idx]["chosen"]
            reject = pope_data[idx]["reject"]
            answer = pope_data[idx]["answer"]
            question = pope_data[idx]["question"]
            question = "<image>\n" + question
            pope_data_dict.append({
                "id": int(image_id),
                "image": self.id2path[int(image_id)],
                "chosen_conversations": [
                    {"from": "human", "value": question},
                    {"from": "gpt", "value": chosen},
                ],
                "reject_conversations": [
                    {"from": "human", "value": question},
                    {"from": "gpt", "value": reject},
                ],
            })
        return pope_data_dict
        
    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'images' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if 'image' in sources[0]:
            image_file = self.list_data_dict[i]['image']
            image_folder = self.data_args.image_folder
            processor = self.data_args.image_processor
            image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
            if self.data_args.image_aspect_ratio == 'pad':
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result
                image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            else:
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            chosen_sources = preprocess_multimodal(
                copy.deepcopy([e["chosen_conversations"] for e in sources]),
                self.data_args)
            reject_sources = preprocess_multimodal(
                copy.deepcopy([e["reject_conversations"] for e in sources]),
                self.data_args)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
        chosen_data_dict = preprocess(
            chosen_sources,
            self.tokenizer,
            has_image=('image' in self.list_data_dict[i]))
        reject_data_dict = preprocess(
            reject_sources,
            self.tokenizer,
            has_image=('image' in self.list_data_dict[i]))
        if isinstance(i, int):
            data_dict = dict(
                chosen_input_ids=chosen_data_dict["input_ids"][0],
                chosen_labels=chosen_data_dict["labels"][0],
                reject_input_ids=reject_data_dict["input_ids"][0],
                reject_labels=reject_data_dict["labels"][0],
            )

        # image exist in the data
        if 'image' in self.list_data_dict[i]:
            data_dict['images'] = image
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict['images'] = torch.zeros(3, crop_size['height'], crop_size['width'])
        return data_dict
    
@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        chosen_input_ids, chosen_labels, reject_input_ids, reject_labels = tuple([instance[key] for instance in instances]
            for key in ("chosen_input_ids", "chosen_labels", "reject_input_ids", "reject_labels"))
        chosen_input_ids = torch.nn.utils.rnn.pad_sequence(
            chosen_input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        chosen_labels = torch.nn.utils.rnn.pad_sequence(chosen_labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        reject_input_ids = torch.nn.utils.rnn.pad_sequence(
            reject_input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        reject_labels = torch.nn.utils.rnn.pad_sequence(reject_labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        chosen_input_ids = chosen_input_ids[:, :self.tokenizer.model_max_length]
        chosen_labels = chosen_labels[:, :self.tokenizer.model_max_length]
        reject_input_ids = reject_input_ids[:, :self.tokenizer.model_max_length]
        reject_labels = reject_labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            chosen_input_ids=chosen_input_ids,
            chosen_labels=chosen_labels,
            reject_input_ids=reject_input_ids,
            reject_labels=reject_labels,
            chosen_attention_mask=chosen_input_ids.ne(self.tokenizer.pad_token_id),
            reject_attention_mask=reject_input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'images' in instances[0]:
            images = [instance['images'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

        return batch

def main():
    # Model
    disable_torch_init()
    
    local_rank = int(os.environ["LOCAL_RANK"])
    device = f"cuda:{local_rank}"
    # model_name = get_model_name_from_path(args.model_path)
    model_name = "llava-post-decoder"
    train_lora = False
    if args.model_base is None:
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path=args.model_path, 
            model_base=args.model_base, 
            model_name=model_name,
            load_8bit=args.load_8bit, 
            load_4bit=args.load_4bit, 
            device=device
        )
    elif args.model_base is not None and train_lora:
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path=args.model_path,
            model_base=args.model_base, 
            model_name="llava_lora_model",
            load_8bit=args.load_8bit, 
            load_4bit=args.load_4bit, 
            device=device
        )
    else:
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path=args.model_path,
            model_base=args.model_base, 
            model_name=model_name,
            load_8bit=args.load_8bit, 
            load_4bit=args.load_4bit, 
            device=device
        )
        
    conv_mode = "llava_v1"
    

    # Load datasets
    parser = transformers.HfArgumentParser(
        (DataArguments))
    _, _, model_args, data_args = parser.parse_args_into_dataclasses()
    
    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args)
    
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    
    parser.add_argument("--pope_path", type=str, required=True)
    parser.add_argument("--coco_path", type=str, required=True)
    parser.add_argument("--set", type=str, required=True)
    args = parser.parse_args()
    
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    args.dist_backend = "nccl"
    print(
        "| distributed init (rank {}, world {}): {}".format(
            int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"]), "env://"
        ),
        flush=True,
    )
    torch.distributed.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=int(os.environ["WORLD_SIZE"]),
        rank=int(os.environ["LOCAL_RANK"]),
        timeout=datetime.timedelta(
            days=365
        ),  # allow auto-downloading and de-compressing
    )
    
    args = parser.parse_args()
    main(args)