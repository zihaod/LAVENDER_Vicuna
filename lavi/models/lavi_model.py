import logging
import random

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn

from lavi.common.registry import registry
from lavi.models.llama_model import LlamaForCausalLM
from lavi.models.lavender_model import EncVideo
from lavi.models.base_model import BaseModel
from transformers import LlamaTokenizer

@registry.register_model("lavi")
class LAVI(BaseModel):
    """
    LAVENDER GPT-LLAMA model.
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_vicuna": "configs/models/lavi.yaml",
    }

    def __init__(
        self,
        max_size_frame=6,
        max_size_patch=14,
        size_img=224,
        vis_backbone_size='base',
        vis_backbone_init='3d',
        kinetics=600,
        vis_hidden_size=768,
        lavender_model="",
        llama_model="",
        prompt_path="",
        prompt_template="",
        max_txt_len=32,
        end_sym='\n',
        low_resource=False,  # use 8 bit and put vit in cpu
        device_8bit=0,  # the device of 8bit model should be set when loading and cannot be changed anymore.
    ):
        super().__init__()

        self.low_resource = low_resource

        print('Loading Vision Encoder')
        self.visual_encoder = EncVideo(
            max_size_frame,
            max_size_patch,
            size_img,
            vis_backbone_size,
            vis_backbone_init,
            kinetics,
            vis_hidden_size, 
        )
        if lavender_model:
            loaded_state_dict = torch.load(lavender_model, map_location=torch.device('cpu'))
            self.visual_encoder.load_vis_ckpt_from_lavender(loaded_state_dict)

        for name, param in self.visual_encoder.named_parameters():
            param.requires_grad = False
        self.visual_encoder = self.visual_encoder.eval()

        print('Loading Vision Encoder Done')

        print('Loading LLAMA')
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model, use_fast=False)
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token

        if self.low_resource:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.float16,
                load_in_8bit=True,
                device_map={'': device_8bit}
            )
        else:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.float16,
            )

        for name, param in self.llama_model.named_parameters():
            param.requires_grad = False
        print('Loading LLAMA Done')

        self.llama_proj = nn.Linear(
            self.visual_encoder.img_feature_dim, self.llama_model.config.hidden_size
        )
        self.max_txt_len = max_txt_len
        self.end_sym = end_sym

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
        self.visual_encoder.to("cpu")
        self.visual_encoder.float()

    def encode_img(self, image):
        device = image.device
        if self.low_resource:
            self.vit_to_cpu()
            image = image.to("cpu")

        #with self.maybe_autocast():
        vis_embeds = self.visual_encoder(image)[0].to(device)
        inputs_llama = self.llama_proj(vis_embeds)
        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(image.device)
        return inputs_llama, atts_llama

    def prompt_wrap(self, img_embeds, atts_img, prompt):
        if prompt:
            batch_size = img_embeds.shape[0]
            p_before, p_after = prompt.split('<ImageHere>')
            p_before_tokens = self.llama_tokenizer(
                p_before, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
            p_after_tokens = self.llama_tokenizer(
                p_after, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
            p_before_embeds = self.llama_model.model.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1)
            p_after_embeds = self.llama_model.model.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1)
            wrapped_img_embeds = torch.cat([p_before_embeds, img_embeds, p_after_embeds], dim=1)
            wrapped_atts_img = atts_img[:, :1].expand(-1, wrapped_img_embeds.shape[1])
            return wrapped_img_embeds, wrapped_atts_img
        else:
            return img_embeds, atts_img

    def forward(self, samples):
        image = samples["image"]
        img_embeds, atts_img = self.encode_img(image)
        if hasattr(samples, 'question_split'):  # VQA dataset
            print('VQA Batch')
            vqa_prompt = '###Human: <Img><ImageHere></Img> '
            img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img, vqa_prompt)
        elif self.prompt_list:
            prompt = random.choice(self.prompt_list)
            img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img, prompt)

        self.llama_tokenizer.padding_side = "right"

        text = [t + self.end_sym for t in samples["text_input"]]

        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(image.device)

        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
        )

        empty_targets = (
            torch.ones([atts_img.shape[0], atts_img.shape[1]+1],
                       dtype=torch.long).to(image.device).fill_(-100)  # plus one for bos
        )
        targets = torch.cat([empty_targets, targets], dim=1)

        batch_size = img_embeds.shape[0]
        bos = torch.ones([batch_size, 1],
                         dtype=to_regress_tokens.input_ids.dtype,
                         device=to_regress_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.llama_model.model.embed_tokens(bos)
        atts_bos = atts_img[:, :1]

        to_regress_embeds = self.llama_model.model.embed_tokens(to_regress_tokens.input_ids)
        inputs_embeds = torch.cat([bos_embeds, img_embeds, to_regress_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, atts_img, to_regress_tokens.attention_mask], dim=1)

        #with self.maybe_autocast():
        outputs = self.llama_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets,
        )
        loss = outputs.loss

        return {"loss": loss}

    @classmethod
    def from_config(cls, cfg):
        max_size_frame = cfg.get("max_size_frame", 6)
        max_size_patch = cfg.get("max_size_patch", 14)
        size_img = cfg.get("size_img", 224)
        vis_backbone_size = cfg.get("vis_backbone_size", 'base')
        vis_backbone_init = cfg.get("vis_backbone_init", '3d')
        kinetics = cfg.get("kinetics", 600)
        vis_hidden_size = cfg.get("vis_hidden_size", 768)
        lavender_model = cfg.get("lavender_model", "")
        llama_model = cfg.get("llama_model")

        low_resource = cfg.get("low_resource", False)
        device_8bit = cfg.get("device_8bit", 0)

        prompt_path = cfg.get("prompt_path", "")
        prompt_template = cfg.get("prompt_template", "")
        max_txt_len = cfg.get("max_txt_len", 32)
        end_sym = cfg.get("end_sym", '\n')

        model = cls(
            max_size_frame=max_size_frame,
            max_size_patch=max_size_patch,
            size_img=size_img,
            vis_backbone_size=vis_backbone_size,
            vis_backbone_init=vis_backbone_init,
            kinetics=kinetics,
            vis_hidden_size=vis_hidden_size,
            lavender_model=lavender_model,
            llama_model=llama_model,
            prompt_path=prompt_path,
            prompt_template=prompt_template,
            max_txt_len=max_txt_len,
            end_sym=end_sym,
            low_resource=low_resource,
            device_8bit=device_8bit,
        )

        ckpt_path = cfg.get("ckpt", "")  # load weights of LAVI
        if ckpt_path:
            print("Load LAVENDER-LLM Checkpoint: {}".format(ckpt_path))
            ckpt = torch.load(ckpt_path, map_location="cpu")
            msg = model.load_state_dict(ckpt['model'], strict=False)

        return model
