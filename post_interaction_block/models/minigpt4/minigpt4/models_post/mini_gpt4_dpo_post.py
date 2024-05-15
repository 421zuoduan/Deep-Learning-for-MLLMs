import os
import random
import logging
from typing import List, Dict, Union, Any

import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast

from minigpt4.common.registry import registry
from minigpt4.models_post.blip2 import Blip2Base, disabled_train
from minigpt4.conversation.conversation import StoppingCriteriaSub

from transformers import LlamaTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList
# from transformers.models.llama.modeling_llama import LlamaForCausalLM
from minigpt4.models_post.modeling_llama_post import LlamaForCausalLM


@registry.register_model("mini_gpt4_dpo_post")
class MiniGPT4DPOPIB(Blip2Base):
    """
    BLIP2 GPT-LLAMA model.
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_vicuna0": "configs/models/minigpt4_vicuna0.yaml",
        "pretrain_llama2": "configs/models/minigpt4_llama2.yaml",
        "pretrain_llama2_post": "configs/models/minigpt4_llama2.yaml",
    }

    def __init__(
        self,
        vit_model="eva_clip_g",
        q_former_model="/home/cuiruochen/model/minigpt4/blip2_pretrained_flant5xxl/blip2_pretrained_flant5xxl.pth",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        has_qformer=True,
        freeze_qformer=True,
        num_query_token=32,
        llama_model="",
        prompt_path="",
        prompt_template="",
        max_txt_len=32,
        end_sym='\n',
        low_resource=False,  # use 8 bit and put vit in cpu
        device_8bit=0,  # the device of 8bit model should be set when loading and cannot be changed anymore.
        freeze_llama_proj=False,
        tune_post_interaction_block=True,
    ):
        super().__init__()
        
        # if we are in a distributed setting, we need to set the device map and max memory per device
        if os.environ.get('LOCAL_RANK') is not None:
            local_rank = int(os.environ.get('LOCAL_RANK', '0'))
            device_map = {'': local_rank}
        else:
            device_map={'':torch.cuda.current_device()}
        
        self.tokenizer = self.init_tokenizer()
        self.low_resource = low_resource

        print('Loading VIT')
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            for name, param in self.ln_vision.named_parameters():
                param.requires_grad = False
            self.ln_vision = self.ln_vision.eval()
            self.ln_vision.train = disabled_train
            logging.info("freeze vision encoder")
        print('Loading VIT Done')

        self.has_qformer = has_qformer
        if self.has_qformer:
            print('Loading Q-Former')
            self.Qformer, self.query_tokens = self.init_Qformer(
                num_query_token, self.visual_encoder.num_features
            )
            self.Qformer.cls = None
            self.Qformer.bert.embeddings.word_embeddings = None
            self.Qformer.bert.embeddings.position_embeddings = None
            for layer in self.Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None
            self.load_from_pretrained(url_or_filename=q_former_model)

            if freeze_qformer:
                for name, param in self.Qformer.named_parameters():
                    param.requires_grad = False
                self.Qformer = self.Qformer.eval()
                self.Qformer.train = disabled_train
                self.query_tokens.requires_grad = False
                logging.info("freeze Qformer")

            img_f_dim = self.Qformer.config.hidden_size
            print('Loading Q-Former Done')
        else:
            img_f_dim = self.visual_encoder.num_features * 4
            print('Do not use Q-Former here.')

        print('Loading LLAMA')
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model, use_fast=False)
        self.llama_tokenizer.pad_token = "$$"

        self.llama_model = LlamaForCausalLM.from_pretrained(
            llama_model,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            load_in_4bit=True,
            device_map=device_map,
        )
                    
        self.config = self.llama_model.config
        
        for name, param in self.llama_model.named_parameters():
            param.requires_grad = False
        print('Loading LLAMA Done')
        
        self.llama_proj = nn.Linear(
            img_f_dim, self.llama_model.config.hidden_size
        )
        if freeze_llama_proj:
            for name, param in self.llama_proj.named_parameters():
                print ("freeze {} for finetuning post interaction block".format(name))
                param.requires_grad = False
        self.freeze_llama_proj = freeze_llama_proj
                
        # convert uint to float16 and set gradient backward
        if tune_post_interaction_block:
            # for name, param in self.llama_model.post_interaction_block.named_parameters():
            #     if torch.isnan(param.data).any():
            #         nan_indices = torch.isnan(param.data)
            #         param.data[nan_indices] = torch.randn(nan_indices.sum(), dtype=torch.float16, device=param.data.device)
            #         print("NaNs detected in param.data before conversion in train")
            #         print(f"name: {name}, {param.data}")
            #         # import sys
            #         # sys.exit(1)
            #     print(f"name, {name}, {param}")
            #     if param.data.dtype == torch.uint8:
            #         print("convert {} to float16".format(name))
            #         param.data = param.data.float()
            #         if torch.isnan(param.data).any():
            #             print("nan in post_interaction_block3")
            #         param.data = param.data / 255.0
            #         if torch.isnan(param.data).any():
            #             print("nan in post_interaction_block3")
            #         param.data = param.data.to(torch.float16)
            #         if torch.isnan(param.data).any():
            #             print("nan in post_interaction_block3")
            #         # if "align" in name:
            #         #     print(f"name in pib: {name}")
            #         #     print(f"param.data: {param.data}")
            for name, param in self.llama_model.post_interaction_block.named_parameters():
                if param.data.dtype == torch.uint8:
                    print("convert {} to float16 and reinitialize".format(name))
                    # 将参数转换为 float 并重新初始化
                    param.data = torch.randn_like(param.data, dtype=torch.float16, device=param.data.device)
                    if torch.isnan(param.data).any():
                        print("NaNs detected in reinitialized param.data (converted from uint8)")
                elif param.data.dtype == torch.float16:
                    print("reinitialize {} float16 parameter".format(name))
                    # 直接重新初始化为 float16
                    param.data = torch.randn_like(param.data, dtype=torch.float16, device=param.data.device)
                    if torch.isnan(param.data).any():
                        print("NaNs detected in reinitialized param.data (float16)")
                param.requires_grad = True
        self.tune_post_interaction_block = tune_post_interaction_block
        
        self.max_txt_len = max_txt_len
        self.end_sym = end_sym

        self.prompt_template = prompt_template
        
        if prompt_path:
            with open(prompt_path, 'r') as f:
                raw_prompts = f.read().splitlines()
            filted_prompts = [raw_prompt for raw_prompt in raw_prompts if "<ImageHere>" in raw_prompt]
            self.prompt_list = [prompt_template.format(p) for p in filted_prompts]
            print('Load {} training prompts'.format(len(self.prompt_list)))
            print('Prompt Example \n{}'.format(random.choice(self.prompt_list)))
        else:
            self.prompt_list = []

    def vit_to_cpu(self):
        self.ln_vision.to("cpu")
        self.ln_vision.float()
        self.visual_encoder.to("cpu")
        self.visual_encoder.float()

    def encode_img(self, image):
        device = image.device
        if self.low_resource:
            self.vit_to_cpu()
            image = image.to("cpu")

        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image)).to(device)
            if self.has_qformer:
                image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)

                query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
                query_output = self.Qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )

                inputs_llama = self.llama_proj(query_output.last_hidden_state)
            else:
                image_embeds = image_embeds[:, 1:, :]
                bs, pn, hs = image_embeds.shape
                image_embeds = image_embeds.view(bs, int(pn / 4), int(hs * 4))

                inputs_llama = self.llama_proj(image_embeds)
            atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(image.device)
        return inputs_llama, atts_llama

    def get_context_emb(self, prompt, img_list):
        device = img_list[0].device
        prompt_segs = prompt.split('<ImageHere>')
        assert len(prompt_segs) == len(img_list) + 1, "Unmatched numbers of image placeholders and images."
        seg_tokens = [
            self.llama_tokenizer(
                seg, return_tensors="pt", add_special_tokens=i == 0).to(device).input_ids
            # only add bos to the first seg
            for i, seg in enumerate(prompt_segs)
        ]
        seg_embs = [self.embed_tokens(seg_t) for seg_t in seg_tokens]

        mixed_embs = [emb for pair in zip(seg_embs[:-1], img_list) for emb in pair] + [seg_embs[-1]]
        mixed_embs = torch.cat(mixed_embs, dim=1)
        return mixed_embs

    def prompt_wrap(self, img_embeds, atts_img, prompts):
        if prompts:
            emb_lists = []
            if isinstance(prompts, str):
                prompts = [prompts] * len(img_embeds)

            for each_img_embed, each_prompt in zip(img_embeds, prompts):
                p_before, p_after = each_prompt.split('<ImageHere>')

                p_before_tokens = self.llama_tokenizer(
                    p_before, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
                p_after_tokens = self.llama_tokenizer(
                    p_after, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
                p_before_embed = self.embed_tokens(p_before_tokens.input_ids)
                p_after_embed = self.embed_tokens(p_after_tokens.input_ids)
                wrapped_emb = torch.cat([p_before_embed, each_img_embed[None], p_after_embed], dim=1)
                emb_lists.append(wrapped_emb)
            emb_lens = [emb.shape[1] for emb in emb_lists]
            pad_emb = self.embed_tokens(torch.tensor(self.llama_tokenizer.pad_token_id, device=img_embeds.device))
            wrapped_embs = pad_emb.expand(len(emb_lens), max(emb_lens), -1).clone()
            wrapped_atts = torch.zeros([len(emb_lens), max(emb_lens)], dtype=torch.int, device=img_embeds.device)
            for i, emb in enumerate(emb_lists):
                wrapped_embs[i, :emb_lens[i]] = emb
                wrapped_atts[i, :emb_lens[i]] = 1
            return wrapped_embs, wrapped_atts
        else:
            return img_embeds, atts_img

    def concat_emb_input_output(self, input_embs, input_atts, output_embs, output_atts):
        input_lens = []
        cat_embs = []
        cat_atts = []
        for i in range(input_embs.size(0)):
            input_len = input_atts[i].sum()
            input_lens.append(input_len)
            cat_embs.append(
                torch.cat([
                    input_embs[i][:input_len],
                    output_embs[i],
                    input_embs[i][input_len:]
                ])
            )
            cat_atts.append(
                torch.cat([
                    input_atts[i][:input_len],
                    output_atts[i],
                    input_atts[i][input_len:]
                ])
            )
        cat_embs = torch.stack(cat_embs)
        cat_atts = torch.stack(cat_atts)
        return cat_embs, cat_atts, input_lens

    def get_dpo_input(
        self,
        text: List[str],
        img_embeds: torch.Tensor,
        atts_img: torch.Tensor,
    ):
        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(img_embeds.device)
        
        # begin-of-sentence token
        batch_size = img_embeds.shape[0]
        bos = torch.ones([batch_size, 1],
                         dtype=to_regress_tokens.input_ids.dtype,
                         device=to_regress_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.embed_tokens(bos)
        atts_bos = atts_img[:, :1]
        
        # image-wrapped text token
        to_regress_embeds = self.embed_tokens(to_regress_tokens.input_ids)
        inputs_embeds, attention_mask, input_lens = \
            self.concat_emb_input_output(img_embeds, atts_img, to_regress_embeds, to_regress_tokens.attention_mask)
        inputs_embeds = torch.cat([bos_embeds, inputs_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, attention_mask], dim=1)
        
        # set targets
        part_targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
        )
        targets = (
            torch.ones([inputs_embeds.shape[0], inputs_embeds.shape[1]],
                       dtype=torch.long).to(img_embeds.device).fill_(-100)
        )

        for i, target in enumerate(part_targets):
            targets[i, input_lens[i] + 1:input_lens[i] + len(target) + 1] = target  # plus 1 for bos
            
        return inputs_embeds, attention_mask, targets
    
    def forward(self, inputs):
        
        image = torch.cat([image.unsqueeze(0) for image in inputs['image']], dim=0)
        img_embeds, atts_img = self.encode_img(image)
        _img_embeds = img_embeds

        with self.maybe_autocast():
            _image_features = self.visual_encoder(image).to(image.device)
            # _image_features = _image_features.to(dtype=img_embeds.dtype())
            _image_features = _image_features.type_as(img_embeds)

        instruction = []
        for idx in range(len(inputs["data_type"])):
            if inputs["data_type"][idx] == "pope":
                prompt = "<Img><ImageHere></Img> " + inputs["prompt"][idx]
                prompt = self.prompt_template.format(prompt)
                instruction.append(prompt)
            else:
                instruction.append(random.choice(self.prompt_list))
        
        img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img, instruction)
        
        self.llama_tokenizer.padding_side = "right"
        text = [t + self.end_sym for t in inputs["chosen"]]

        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(image.device)

        batch_size = img_embeds.shape[0]
        bos = torch.ones([batch_size, 1],
                         dtype=to_regress_tokens.input_ids.dtype,
                         device=to_regress_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.embed_tokens(bos)
        atts_bos = atts_img[:, :1]

        to_regress_embeds = self.embed_tokens(to_regress_tokens.input_ids)
        inputs_embeds, attention_mask, input_lens = \
            self.concat_emb_input_output(img_embeds, atts_img, to_regress_embeds, to_regress_tokens.attention_mask)
        inputs_embeds = torch.cat([bos_embeds, inputs_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, attention_mask], dim=1)

        part_targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
        )
        targets = (
            torch.ones([inputs_embeds.shape[0], inputs_embeds.shape[1]],
                       dtype=torch.long).to(image.device).fill_(-100)
        )

        for i, target in enumerate(part_targets):
            targets[i, input_lens[i] + 1:input_lens[i] + len(target) + 1] = target  # plus 1 for bos

        with self.maybe_autocast():
            outputs = self.llama_model(
                image_features=_image_features,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
        loss = outputs.loss

        return {"loss": loss}
    
    def prepare_minigpt4_dpo_inputs(
        self, 
        inputs: Dict[str, Union[torch.Tensor, Any]],
    ) :
        
        images = [image.unsqueeze(0) for image in inputs['image']]
        images = images*2 # chosen + rejected
        
        instruction = []
        for idx in range(len(inputs["data_type"])):
            if inputs["data_type"][idx] == "pope":
                prompt = "<Img><ImageHere></Img> " + inputs["prompt"][idx]
                prompt = self.prompt_template.format(prompt)
                instruction.append(prompt)
            else:
                instruction.append(random.choice(self.prompt_list))
        instruction = instruction * 2
        
        chosen = inputs['chosen']
        rejected = inputs['rejected']
        text_input = chosen + rejected
        
        # encode image
        image_batch = torch.cat(images, dim=0)
        img_embeds, atts_img = self.encode_img(image_batch)
        img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img, instruction)
        
        with self.maybe_autocast():
            _image_features = self.visual_encoder(image_batch).to(image_batch.device)
            # _image_features = _image_features.to(dtype=img_embeds.dtype())
            _image_features = _image_features.type_as(img_embeds)
        
        self.llama_tokenizer.padding_side = "right"
        text_input = [t + self.end_sym for t in text_input]
    
        inputs_embeds, attention_mask, labels = self.get_dpo_input(
            text_input, img_embeds, atts_img,
        )
        
        dpo_inputs = {
            "concatenated_image_features": _image_features,
            "concatenated_input_embeds": inputs_embeds,
            "concatenated_attention_mask": attention_mask,
            "concatenated_labels": labels,
        }
        
        return dpo_inputs

    def embed_tokens(self, token_ids):
        if hasattr(self.llama_model.base_model, 'model'): ## lora wrapped model
            embeds = self.llama_model.base_model.model.model.embed_tokens(token_ids)
        else:
            embeds = self.llama_model.base_model.embed_tokens(token_ids)
        return embeds

    def answer(self, image, max_new_tokens=300, num_beams=1, min_length=1, top_p=0.9,
               repetition_penalty=1.0, length_penalty=1, temperature=1.0, max_length=2000, do_sample=True):
        image_emb, _ = self.encode_img(image)
        prompt = "[INST] <Img><ImageHere></Img> Describe this image in detail. [/INST] "
        embs = self.get_context_emb(prompt, [image_emb])
        current_max_len = embs.shape[1] + max_new_tokens
        if current_max_len - max_length > 0:
            print('Warning: The number of tokens in current conversation exceeds the max length. '
                  'The model will not see the contexts outside the range.')
        begin_idx = max(0, current_max_len - max_length)

        embs = embs[:, begin_idx:]

        stop_words_ids = [torch.tensor([835]).to(self.device),
                          torch.tensor([2277, 29937]).to(self.device)]  # '###' can be encoded in two different ways.
        
        outputs = self.llama_model.generate(
            inputs_embeds=embs,
            max_new_tokens=max_new_tokens,
            stopping_criteria=StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)]),
            num_beams=num_beams,
            do_sample=do_sample,
            min_length=min_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            temperature=temperature,
        )
        output_token = outputs[0]
        if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
            output_token = output_token[1:]
        if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
            output_token = output_token[1:]
        output_text = self.llama_tokenizer.decode(output_token, add_special_tokens=False)
        output_text = output_text.split('###')[0]  # remove the stop sign '###'
        output_text = output_text.split('Assistant:')[-1].strip()
        output_text = output_text.strip('</s>')   # remove the end '</s>'
        output_text = output_text.strip()
        return output_text
    
    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        q_former_model = cfg.get("q_former_model", "/home/cuiruochen/model/minigpt4/blip2_pretrained_flant5xxl/blip2_pretrained_flant5xxl.pth/blip2_pretrained_flant5xxl.pth")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        llama_model = cfg.get("llama_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)
        has_qformer = cfg.get("has_qformer", True)
        freeze_qformer = cfg.get("freeze_qformer", True)
        low_resource = cfg.get("low_resource", False)
        device_8bit = cfg.get("device_8bit", 0)

        prompt_path = cfg.get("prompt_path", "")
        prompt_template = cfg.get("prompt_template", "")
        max_txt_len = cfg.get("max_txt_len", 32)
        end_sym = cfg.get("end_sym", '\n')

        freeze_llama_proj = cfg.get("freeze_llama_proj", False)
        tune_post_interaction_block = cfg.get("tune_post_interaction_block", True)
        
        model = cls(
            vit_model=vit_model,
            q_former_model=q_former_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            has_qformer=has_qformer,
            freeze_qformer=freeze_qformer,
            num_query_token=num_query_token,
            llama_model=llama_model,
            prompt_path=prompt_path,
            prompt_template=prompt_template,
            max_txt_len=max_txt_len,
            end_sym=end_sym,
            low_resource=low_resource,
            device_8bit=device_8bit,
            freeze_llama_proj=freeze_llama_proj,
            tune_post_interaction_block=tune_post_interaction_block,
        )

        ckpt_path = cfg.get("ckpt", "")  # load weights of MiniGPT-4
        if ckpt_path:
            print("Load BLIP2-LLM Checkpoint: {}".format(ckpt_path))
            ckpt = torch.load(ckpt_path, map_location="cpu")
            msg = model.load_state_dict(ckpt['model'], strict=False)

        return model
