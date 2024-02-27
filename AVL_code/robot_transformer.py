"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging
import string
import random
import copy

import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from transformers import T5TokenizerFast


from transformers.modeling_outputs import BaseModelOutput
from timm.models.vision_transformer import PatchEmbed, Block

from transformers import BertTokenizer
# TODO: Allow for better fp16 support since the model is frozen. Right now fp32 is used for the model.
# https://github.com/huggingface/transformers/issues/14189#issuecomment-961571628

from transformers import  AutoTokenizer, AutoModelForCausalLM
import re
# from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
# from huggingface_hub import login

from AVL_code.video_model import  create_vit_b_video


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class RobotTransformer(nn.Module):


    def __init__(
        self,
        num_frames=9,
    ):
        """
        apply_lemmatizer: when set to True, postprocess predict_answers() result with lemmas.
        """
        super().__init__()
        
        self.base_model_name = "facebook/opt-125m"

        self.tokenizer = self.init_tokenizer()
        self.start_token_id = self.tokenizer.bos_token_id
        self.visual_encoder, self.ln_vision = self.init_vision_encoder()
        self.visual_encoder = self.visual_encoder.to(torch.float32)

        self.model = AutoModelForCausalLM.from_pretrained(self.base_model_name ,
                                                           torch_dtype=torch.float32)
        self.model.resize_token_embeddings(len(self.tokenizer))
        # self.model = self.model.half()
        self.num_frames = num_frames

        self.model_size = 768
        self.linear_projection = nn.Linear(768, self.model_size).to(torch.float32)

        self.total_embeddings = None
        self.length = []
        self.instruction_length = 0

    
    def init_vision_encoder(self):
        visual_encoder = create_vit_b_video(224)
        ln_vision = LayerNorm(visual_encoder.num_features)
        return visual_encoder, ln_vision
    
    def init_tokenizer(self):

        base_model_name = self.base_model_name
        tokenizer =  AutoTokenizer.from_pretrained(base_model_name)
      
        bin_sizes = {
            'language_table': 20,
            'calvin': 20,
        }

        language_table_bin_size = bin_sizes['language_table']
        calvin_bin_size = bin_sizes['calvin']

        assert language_table_bin_size >= 0
        assert calvin_bin_size >= 0

        # actions for language table
        for i in range(language_table_bin_size+1):
            tokenizer.add_tokens([f"[ROBOTACTIONX{i}]", f"[ROBOTACTIONY{i}]"])
            tokenizer.add_tokens([f"[ROBOTEETX{i}]", f"[ROBOTEETY{i}]"])
            tokenizer.add_tokens([f"[ROBOTEETTX{i}]", f"[ROBOTEETTY{i}]"])
        
        tokenizer.add_tokens(['[ENDOFACTION]'])
        tokenizer.add_tokens(['[STARTACTION]'])
        tokenizer.add_tokens(['[TERMINAL]'])
        
        tokenizer.add_tokens(['[STARTEET]'])
        tokenizer.add_tokens(['[ENDOFEET]'])

        tokenizer.add_tokens(['[STARTEETT]'])
        tokenizer.add_tokens(['[ENDOFEETT]'])
        tokenizer.pad_token = tokenizer.eos_token

        # actions for minecraft

        NOOP_ACTION = {
            "ESC": 0,
            "back": 0,
            "drop": 0,
            "forward": 0,
            "hotbar.1": 0,
            "hotbar.2": 0,
            "hotbar.3": 0,
            "hotbar.4": 0,
            "hotbar.5": 0,
            "hotbar.6": 0,
            "hotbar.7": 0,
            "hotbar.8": 0,
            "hotbar.9": 0,
            "inventory": 0,
            "jump": 0,
            "left": 0,
            "right": 0,
            "sneak": 0,
            "sprint": 0,
            "swapHands": 0,
            "attack": 0,
            "use": 0,
            "pickItem": 0,
        }
        new_tokens = list(NOOP_ACTION.keys())
        new_tokens = [ f'[{new_token}]' for new_token in new_tokens ]
        tokenizer.add_tokens(new_tokens)
        camera_actions = []
        for i in range(-49, 51):
            camera_actionx = f'[CAMERAX{i}]'
            camera_actiony = f'[CAMERAY{i}]'
            camera_actions.extend([camera_actionx, camera_actiony])
        
        tokenizer.add_tokens(camera_actions)

        # actions for calvin
        for i in range(calvin_bin_size+1):
            tokenizer.add_tokens([f"[ROBOTACTION0_{i}]", f"[ROBOTACTION1_{i}]", f"[ROBOTACTION2_{i}]",
                                       f"[ROBOTACTION3_{i}]", f"[ROBOTACTION4_{i}]", f"[ROBOTACTION5_{i}]"])
        
        for i in range(14):
            for j in range(calvin_bin_size+1):
                tokenizer.add_tokens([f"[ROBOTSTATE{i}_{j}]"])
        
        tokenizer.add_tokens(['[GRIPPER_OPEN]', '[GRIPPER_CLOSE]', '[GRIPPER_OPENED]', '[GRIPPER_CLOSED]'])

        BE_ACTIONS_LIST = ["Evade", "Jump", "LockOn", "Mount", "MeleeAttack", "SpecialAbility1", "SpecialAbility2", "SpecialAbility3", "SuperAbility", "SwitchLockOnTarget", "Taunt"]

        tokenizer.add_tokens([f'[{x}]' for x in BE_ACTIONS_LIST])
        tokenizer.add_tokens([f'[lrot{rot + 1}]' for rot in range(256)])
        tokenizer.add_tokens([f'[lmag{mag + 1}]' for mag in range(4)])
        tokenizer.add_tokens([f'[rrot{rot + 1}]' for rot in range(256)])
        tokenizer.add_tokens([f'[rmag{mag + 1}]' for mag in range(4)])

        return tokenizer


    def set_instructions(self, instructions):
        instruction_embeds = self.model.model.get_input_embeddings()(instructions)
        self.total_embeddings = instruction_embeds
        # self.instruction_length = instruction_embeds.shape[1]
        # self.length = []
    
    # def remove_first_embeddings(self):
    #     if len(self.length) == 0:
    #         return
    #     self.total_embeddings = torch.cat( (self.total_embeddings[:,  :self.instruction_length, :],
    #                                          self.total_embeddings[:,  self.instruction_length + self.length[0]: , :]), dim = 1)
        
    #     self.length.pop(0)

    
    def step(self, image, state_ids):

        actions = []
    
        b, t, c, h, w = image.shape
        video_batch = image.reshape(b * t, c, h, w)

        features, mask, ids_restore =   self.visual_encoder(video_batch.unsqueeze(1))
        image_embeds = self.ln_vision(features)
        image_embeds = image_embeds.reshape( b, t, image_embeds.shape[-2], image_embeds.shape[-1])
        image_embeds = self.linear_projection(image_embeds)

        
        image_embeddings = image_embeds[:, -1, :, :]

        state_embeddings = self.model.model.get_input_embeddings()(state_ids)

        total_embeddings = self.total_embeddings
        prediction_embeddings = torch.cat((total_embeddings, image_embeddings,
                                            state_embeddings ), dim = 1 )

        # Generate the next token
        # TODO: replace image with image special token in label
        action_embeddings = torch.empty((b, 0, 768), dtype=torch.float16).to(image.device)
        action = "[STARTACTION]"
        
        for index in range(8):
            if index == 0:
                next_token = self.tokenizer.encode('[STARTACTION]')
                next_token = torch.tensor(next_token).unsqueeze(0).to(image.device)
                next_token_embedding = self.model.model.get_input_embeddings()(next_token)
                prediction_embeddings = torch.cat((prediction_embeddings, next_token_embedding), dim = 1 )
                action_embeddings = torch.cat((action_embeddings, next_token_embedding), dim = 1 )
                continue
          
            outputs = self.model(
                inputs_embeds = prediction_embeddings,
            )

            next_token_logits =  outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)

            result = self.tokenizer.decode(next_token[0], skip_special_tokens=False)

            next_token_embedding = self.model.model.get_input_embeddings()(next_token)
            action_embeddings = torch.cat((action_embeddings, next_token_embedding), dim = 1 )
            prediction_embeddings = torch.cat((prediction_embeddings, next_token_embedding), dim = 1 )
            action += result
          
 
        actions.append(action)
        to_cat = torch.cat( ( image_embeddings, state_embeddings, action_embeddings), dim = 1 )
        # self.length.append(to_cat.shape[1])

        total_embeddings = torch.cat((total_embeddings, to_cat), dim = 1 )
        
        self.total_embeddings = total_embeddings
        return actions