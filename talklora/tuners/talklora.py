# coding=utf-8
# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.




import math
import re
import warnings
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import List, Optional, Union

import deepspeed

from collections import defaultdict




import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

from ..import_utils import is_bnb_available
from ..utils import (
    TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING,
    ModulesToSaveWrapper,
    PeftConfig,
    PeftType,
    _freeze_adapter,
    _get_submodules,
    transpose,
)

if is_bnb_available():
    import bitsandbytes as bnb


def compute_columnwise_norm(input_tensor: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(input_tensor, dim=1)


@dataclass
class LoraConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`LoraModel`].

    Args:
        r (`int`): Lora attention dimension.
        target_modules (`Union[List[str],str]`): The names of the modules to apply Lora to.
        lora_alpha (`float`): The alpha parameter for Lora scaling.
        lora_dropout (`float`): The dropout probability for Lora layers.
        fan_in_fan_out (`bool`): Set this to True if the layer to replace stores weight like (fan_in, fan_out).
        For example, gpt-2 uses `Conv1D` which stores weights like (fan_in, fan_out) and hence this should be set to `True`.:
        bias (`str`): Bias type for Lora. Can be 'none', 'all' or 'lora_only'
        modules_to_save (`List[str]`):List of modules apart from LoRA layers to be set as trainable
            and saved in the final checkpoint.
    """
    init_a: str = field(default='kaiming')
    init_b: str = field(default='zero')
    rand_R: bool = field(default=False)
    r_ab: int = field(default=8, metadata={"help": "HiRA attention dimension"})
    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with Lora."
                    "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        },
    )
    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with Lora."
                    "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        },
    )
    lora_alpha: int = field(default=32, metadata={"help": "Lora alpha"})
    lora_dropout: float = field(default=0.05, metadata={"help": "Lora dropout"})
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)"},
    )
    bias: str = field(default="none", metadata={"help": "Bias type for Lora. Can be 'none', 'all' or 'lora_only'"})
    modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "List of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint. "
                    "For example, in Sequence Classification or Token Classification tasks, "
                    "the final layer `classifier/score` are randomly initialized and as such need to be trainable and saved."
        },
    )
    init_lora_weights: bool = field(
        default=True,
        metadata={"help": "Whether to initialize the weights of the Lora layers."},
    )

    scale_ab: float = field(
        default=1.0,
        metadata={"help": "The scale factor to scaling the product of A and B"}
    )
    train_a: bool = field(default=True,
                          metadata={"help": "train HiRA A"},
                          )
    train_b: bool = field(default=True,
                          metadata={"help": "train HiRA B"},
                          )


    def __post_init__(self):
        self.peft_type = PeftType.LORA


class LoraModel(torch.nn.Module):

    def __init__(self, model, config, adapter_name):
        super().__init__()
        self.model = model
        self.forward = self.model.forward
        # config = {'default': loraconfig}
        self.peft_config = config
        self.expert_num = 4
        self.add_adapter(adapter_name, self.peft_config[adapter_name])


    def add_adapter(self, adapter_name, config=None):
        if config is not None:

            model_config = self.model.config.to_dict() if hasattr(self.model.config, "to_dict") else self.model.config
            config = self._prepare_lora_config(config, model_config)
            self.peft_config[adapter_name] = config

        self.r = self.peft_config[adapter_name].r_ab
        self.lora_B_q = nn.ModuleDict({})
        self.lora_B_k = nn.ModuleDict({})
        self.lora_B_v = nn.ModuleDict({})
        self.lora_B_up = nn.ModuleDict({})
        self.lora_B_down = nn.ModuleDict({})


        #llama3-8b
        # self.lora_B_q.update({adapter_name: nn.ModuleList([nn.Linear(self.r // self.expert_num, 4096, bias=False) for _ in range(self.expert_num)])})
        # self.lora_B_k.update({adapter_name: nn.ModuleList([nn.Linear(self.r // self.expert_num, 1024, bias=False) for _ in range(self.expert_num)])})
        # self.lora_B_v.update({adapter_name: nn.ModuleList([nn.Linear(self.r // self.expert_num, 1024, bias=False) for _ in range(self.expert_num)])})
        # self.lora_B_up.update({adapter_name: nn.ModuleList([nn.Linear(self.r // self.expert_num, 14336, bias=False) for _ in range(self.expert_num)])})
        # self.lora_B_down.update({adapter_name: nn.ModuleList([nn.Linear(self.r // self.expert_num, 4096, bias=False) for _ in range(self.expert_num)])})


        #llama2-7b
        # self.lora_B_q.update({adapter_name: nn.ModuleList([nn.Linear(self.r // self.expert_num, 4096, bias=False) for _ in range(self.expert_num)])})
        # self.lora_B_k.update({adapter_name: nn.ModuleList([nn.Linear(self.r // self.expert_num, 4096, bias=False) for _ in range(self.expert_num)])})
        # self.lora_B_v.update({adapter_name: nn.ModuleList([nn.Linear(self.r // self.expert_num, 4096, bias=False) for _ in range(self.expert_num)])})
        # self.lora_B_up.update({adapter_name: nn.ModuleList([nn.Linear(self.r // self.expert_num, 11008, bias=False) for _ in range(self.expert_num)])})
        # self.lora_B_down.update({adapter_name: nn.ModuleList([nn.Linear(self.r // self.expert_num, 4096, bias=False) for _ in range(self.expert_num)])})

        #qwen2.5-7b
        self.lora_B_q.update({adapter_name: nn.ModuleList([nn.Linear(self.r // self.expert_num, 3584, bias=False) for _ in range(self.expert_num)])})
        self.lora_B_k.update({adapter_name: nn.ModuleList([nn.Linear(self.r // self.expert_num, 512, bias=False) for _ in range(self.expert_num)])})
        self.lora_B_v.update({adapter_name: nn.ModuleList([nn.Linear(self.r // self.expert_num, 512, bias=False) for _ in range(self.expert_num)])})
        self.lora_B_up.update({adapter_name: nn.ModuleList([nn.Linear(self.r // self.expert_num, 18944, bias=False) for _ in range(self.expert_num)])})
        self.lora_B_down.update({adapter_name: nn.ModuleList([nn.Linear(self.r // self.expert_num, 3584, bias=False) for _ in range(self.expert_num)])})



        self.reset_parameters(adapter_name)



        self._find_and_replace(adapter_name)
        if len(self.peft_config) > 1 and self.peft_config[adapter_name].bias != "none":
            raise ValueError(
                "LoraModel supports only 1 adapter with bias. When using multiple adapters, set bias to 'none' for all adapters."
            )
        mark_only_lora_as_trainable(self.model, self.peft_config[adapter_name].bias, config)
        if self.peft_config[adapter_name].inference_mode:
            _freeze_adapter(self.model, adapter_name)

    def reset_parameters(self, adapter_name):
        
        for i in range(self.expert_num):
            nn.init.zeros_(self.lora_B_q[adapter_name][i].weight)
            nn.init.zeros_(self.lora_B_k[adapter_name][i].weight)
            nn.init.zeros_(self.lora_B_v[adapter_name][i].weight)
            nn.init.zeros_(self.lora_B_up[adapter_name][i].weight)
            nn.init.zeros_(self.lora_B_down[adapter_name][i].weight)



    def _find_and_replace(self, adapter_name):
        lora_config: LoraConfig = self.peft_config[adapter_name]
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)
        if loaded_in_8bit and not is_bnb_available():
            raise ImportError(
                "To use Lora with 8-bit quantization, please install the `bitsandbytes` package. "
                "You can install it with `pip install bitsandbytes`."
            )
        is_target_modules_in_base_model = False
        kwargs = {
            "r_ab": lora_config.r_ab,
            "lora_alpha": lora_config.lora_alpha,
            "lora_dropout": lora_config.lora_dropout,
            "fan_in_fan_out": lora_config.fan_in_fan_out,
            "init_lora_weights": lora_config.init_lora_weights,
            "scale_ab": lora_config.scale_ab,
            "init_a": lora_config.init_a,
            "init_b": lora_config.init_b,
            "train_a": lora_config.train_a,
            "train_b": lora_config.train_b,
            "rand_R": lora_config.rand_R,
            "expert_num":self.expert_num,
        }
        key_list = [key for key, _ in self.model.named_modules()]
        for key in key_list:
            if isinstance(lora_config.target_modules, str):
                target_module_found = re.fullmatch(lora_config.target_modules, key)
            else:
                target_module_found = any(key.endswith(target_key) for target_key in lora_config.target_modules)
            if target_module_found:
                if not is_target_modules_in_base_model:
                    is_target_modules_in_base_model = True
                parent, target, target_name = _get_submodules(self.model, key)
                if hasattr(target, "bias"):
                    bias = target.bias is not None

                if isinstance(target, LoraLayer):
                    print("...............................................................\n")
                    target.update_layer(
                        adapter_name,
                        lora_config.r,
                        lora_config.lora_alpha,
                        lora_config.lora_dropout,
                        lora_config.init_lora_weights,
                    )
                else:
                    if loaded_in_8bit and isinstance(target, bnb.nn.Linear8bitLt):
                        eightbit_kwargs = kwargs.copy()
                        eightbit_kwargs.update(
                            {
                                "has_fp16_weights": target.state.has_fp16_weights,
                                "memory_efficient_backward": target.state.memory_efficient_backward,
                                "threshold": target.state.threshold,
                                "index": target.index,
                            }
                        )
                    elif isinstance(target, torch.nn.Embedding):
                        embedding_kwargs = kwargs.copy()
                        embedding_kwargs.pop("fan_in_fan_out", None)
                        in_features, out_features = target.num_embeddings, target.embedding_dim
                        # new_module = Embedding(adapter_name, in_features, out_features, **embedding_kwargs)
                    else:
                        if isinstance(target, torch.nn.Linear):
                            in_features, out_features = target.in_features, target.out_features
                            if kwargs["fan_in_fan_out"]:
                                warnings.warn(
                                    "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                                    "Setting fan_in_fan_out to False."
                                )
                                kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = False
                        elif isinstance(target, Conv1D):
                            in_features, out_features = (
                                target.weight.ds_shape if hasattr(target.weight, "ds_shape") else target.weight.shape
                            )
                            if not kwargs["fan_in_fan_out"]:
                                warnings.warn(
                                    "fan_in_fan_out is set to False but the target module is `Conv1D`. "
                                    "Setting fan_in_fan_out to True."
                                )
                                kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = True
                        else:
                            raise ValueError(
                                f"Target module {target} is not supported. "
                                f"Currently, only `torch.nn.Linear` and `Conv1D` are supported."
                            )
                        # new_module = Linear(adapter_name, in_features, out_features, bias=bias, **kwargs)

                        if 'q' in target_name:
                            new_module = Linear(adapter_name, in_features, out_features, self.lora_B_q, key, bias=bias, **kwargs)
                        elif 'k' in target_name:
                            new_module = Linear(adapter_name, in_features, out_features, self.lora_B_k, key, bias=bias, **kwargs)
                        elif 'v' in target_name:
                            new_module = Linear(adapter_name, in_features, out_features, self.lora_B_v, key, bias=bias, **kwargs)
                        elif 'up' in target_name:
                            new_module = Linear(adapter_name, in_features, out_features, self.lora_B_up, key, bias=bias, **kwargs)
                        elif 'down' in target_name:
                            new_module = Linear(adapter_name, in_features, out_features, self.lora_B_down, key, bias=bias, **kwargs)

                    self._replace_module(parent, target_name, new_module, target)
        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {lora_config.target_modules} not found in the base model. "
                f"Please check the target modules and try again."
            )

    def _replace_module(self, parent_module, child_name, new_module, old_module):
        setattr(parent_module, child_name, new_module)
        new_module.weight = old_module.weight
        if hasattr(old_module, "bias"):
            if old_module.bias is not None:
                new_module.bias = old_module.bias

        if getattr(old_module, "state", None) is not None:
            new_module.state = old_module.state
            new_module.to(old_module.weight.device)

        # dispatch to correct device
        for name, module in new_module.named_modules():
            if "lora_" in name:
                module.to(old_module.weight.device)

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)

    def get_peft_config_as_dict(self, inference: bool = False):
        config_dict = {}
        for key, value in self.peft_config.items():
            config = {k: v.value if isinstance(v, Enum) else v for k, v in asdict(value).items()}
            if inference:
                config["inference_mode"] = True
        config_dict[key] = config
        return config

    def _set_adapter_layers(self, enabled=True):
        for module in self.model.modules():
            if isinstance(module, LoraLayer):
                module.disable_adapters = False if enabled else True

    def enable_adapter_layers(self):
        self._set_adapter_layers(enabled=True)

    def disable_adapter_layers(self):
        self._set_adapter_layers(enabled=False)

    def set_adapter(self, adapter_name):
        for module in self.model.modules():
            if isinstance(module, LoraLayer):
                if module.merged:
                    warnings.warn("Adapter cannot be set when the model is merged. Unmerging the model first.")
                    module.unmerge()
                module.active_adapter = adapter_name

    def merge_adapter(self):
        for module in self.model.modules():
            if isinstance(module, LoraLayer):
                module.merge()

    def unmerge_adapter(self):
        for module in self.model.modules():
            if isinstance(module, LoraLayer):
                module.unmerge()

    @staticmethod
    def _prepare_lora_config(peft_config, model_config):
        if peft_config.target_modules is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify `target_modules` in `peft_config`")
            peft_config.target_modules = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING[model_config["model_type"]]
        if peft_config.inference_mode:
            peft_config.merge_weights = False
        return peft_config

    def merge_and_unload(self):
        if getattr(self.config, "model_type", None) == "gpt2":
            raise ValueError("GPT2 models are not supported for merging LORA layers")

        if getattr(self.model, "is_loaded_in_8bit", False):
            raise ValueError("Cannot merge LORA layers when the model is loaded in 8-bit mode")

        key_list = [key for key, _ in self.model.named_modules() if "lora" not in key]
        for key in key_list:
            try:
                parent, target, target_name = _get_submodules(self.model, key)
            except AttributeError:
                continue
            if isinstance(target, LoraLayer):
                bias = target.bias is not None
                new_module = torch.nn.Linear(target.in_features, target.out_features, bias=bias)
                target.merge()
                self._replace_module(parent, target_name, new_module, target)

            # save any additional trainable modules part of `modules_to_save`
            if isinstance(target, ModulesToSaveWrapper):
                setattr(parent, target_name, target.modules_to_save[target.active_adapter])

        return self.model

    def add_weighted_adapter(self, adapters, weights, adapter_name):
        if len({self.peft_config[adapter].r_ab for adapter in adapters}) != 1:
            raise ValueError("All adapters must have the same r value")
        self.peft_config[adapter_name] = self.peft_config[adapters[0]]
        self.peft_config[adapter_name].lora_alpha = self.peft_config[adapters[0]].r_ab
        self._find_and_replace(adapter_name)
        mark_only_lora_as_trainable(self.model, self.peft_config[adapter_name].bias, self.peft_config)
        _freeze_adapter(self.model, adapter_name)
        key_list = [key for key, _ in self.model.named_modules() if "lora" not in key]
        for key in key_list:
            _, target, _ = _get_submodules(self.model, key)
            if isinstance(target, LoraLayer):
                if adapter_name in target.lora_A or adapter_name in target.lora_C:
                    # QS: Initialize lora here
                    if self.peft_config.r_ab > 0:
                        target.lora_A[adapter_name].data = target.lora_A[adapter_name].data * 0.0
                        target.lora_B[adapter_name].data = target.lora_B[adapter_name].data * 0.0
                    for adapter, weight in zip(adapters, weights):
                        if adapter not in target.lora_A:
                            continue
                        target.lora_A[adapter_name].data += (
                                target.lora_A[adapter].weight.data * weight * target.scaling[adapter]
                        )
                        target.lora_B[adapter_name].data += target.lora_B[adapter].weight.data * weight

                elif adapter_name in target.lora_embedding_A:
                    # QS: Initialize lora here
                    target.lora_embedding_A[adapter_name].data = target.lora_embedding_A[adapter_name].data * 0.0
                    target.lora_embedding_B[adapter_name].data = target.lora_embedding_B[adapter_name].data * 0.0
                    target.lora_embedding_C[adapter_name].data = target.lora_embedding_C[adapter_name].data * 0.0
                    target.lora_embedding_D[adapter_name].data = target.lora_embedding_D[adapter_name].data * 0.0
                    for adapter, weight in zip(adapters, weights):
                        if adapter not in target.lora_embedding_A:
                            continue
                        target.lora_embedding_A[adapter_name].data += (
                                target.lora_embedding_A[adapter].data * weight * target.scaling[adapter]
                        )
                        target.lora_embedding_B[adapter_name].data += target.lora_embedding_B[adapter].data * weight
                        target.lora_embedding_C[adapter_name].data += (
                                target.lora_embedding_C[adapter].data * weight * target.scaling[adapter]
                        )
                        target.lora_embedding_D[adapter_name].data += (
                                target.lora_embedding_D[adapter].data * weight * target.scaling[adapter]
                        )


# Below code is based on https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
# and modified to work with PyTorch FSDP


#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------


# had to adapt it for `lora_only` to work
def mark_only_lora_as_trainable(model: nn.Module, bias: str = "none", config: LoraConfig = None) -> None:
    for n, p in model.named_parameters():
        if "lora_" not in n:
            p.requires_grad = False
    if bias == "none":
        pass
    elif bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True
    elif bias == "lora_only":
        for m in model.modules():
            if isinstance(m, LoraLayer) and hasattr(m, "bias") and m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError
    if config:
        for n, p in model.named_parameters():
            if "lora_" in n:
                p.requires_grad = False
        if config.train_a:
            for n, p in model.named_parameters():
                if "lora_A" in n:
                    p.requires_grad = True
        if config.train_b:
            for n, p in model.named_parameters():
                if "lora_B" in n:
                    p.requires_grad = True


class LoraLayer:
    def __init__(
            self,
            in_features: int,
            out_features: int,
            expert_num: int,
    ):
        self.r_ab = {}
        self.lora_alpha = {}
        self.scaling_ab = {}
        self.lora_dropout = nn.ModuleDict({})
        self.lora_A = nn.ModuleDict({})

        self.lora_A_talk = nn.ModuleDict({})
 
        self.lora_A_route = nn.ModuleDict({})

        self.lora_A_experts = nn.ModuleDict({})

        self.lora_embedding_A = nn.ParameterDict({})
        self.lora_embedding_B = nn.ParameterDict({})

        self.merged = False
        self.disable_adapters = False
        self.in_features = in_features
        self.out_features = out_features
        self.init_a = None
        self.init_b = None
        self.rand_R = False
        self.expert_num =expert_num
        self.lora_R = nn.ParameterDict({})

    def update_layer(self, adapter_name, r_ab, lora_alpha, lora_dropout, init_lora_weights, init_a,
                     init_b):
        self.r_ab[adapter_name] = r_ab
        self.init_a = init_a
        self.init_b = init_b
        # self.rand_R = rand_R

        self.lora_alpha[adapter_name] = lora_alpha
        self.scaling_ab[adapter_name] = self.lora_alpha[adapter_name] / self.r_ab[adapter_name]
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        # Actual trainable parameters
        # if self.rand_R:
        #     self.lora_R.update(
        #         nn.ParameterDict({adapter_name: nn.Parameter(torch.randn(self.out_features, self.in_features))}))
        if r_ab > 0:
            # self.lora_A.update(nn.ParameterDict({adapter_name: nn.Parameter(torch.randn(r_ab, self.in_features))}))
            # self.lora_B.update(nn.ParameterDict({adapter_name: nn.Parameter(torch.randn(self.out_features, r_ab))}))

            self.lora_A.update({adapter_name: nn.Linear(self.in_features, self.r_ab[adapter_name], bias=False)})
            self.lora_A_talk.update({adapter_name: nn.Linear(self.expert_num, self.expert_num, bias=False)})
            self.lora_A_route.update({adapter_name: nn.Linear(self.r_ab[adapter_name], self.expert_num, bias=False)})
            self.lora_A_experts.update({adapter_name: nn.ModuleList([nn.Linear(self.r_ab[adapter_name] // self.expert_num, 
                                                                               self.r_ab[adapter_name] // self.expert_num, bias=False) for _ in range(self.expert_num)])})
        if init_lora_weights:
            self.reset_lora_parameters(adapter_name)
        self.to(self.weight.device)


    def reset_lora_parameters(self, adapter_name):
        init_mapping = {'kaiming': nn.init.kaiming_uniform_, 'zero': nn.init.zeros_}
        init_kwargs = {'kaiming': {'a': math.sqrt(5)}, 'zero': {}}
        init_a = self.init_a
        init_b = self.init_b
        if adapter_name in self.lora_A.keys() or adapter_name in self.lora_C.keys():
            if self.rand_R:
                nn.init.uniform_(self.lora_R[adapter_name])
            if self.r_ab[adapter_name] > 0:
                # initialize A the same way as the default for nn.Linear and B to zero
                init_mapping[init_a](self.lora_A[adapter_name].weight, **init_kwargs[init_a])
                init_mapping[init_a](self.lora_A_talk[adapter_name].weight, **init_kwargs[init_a])
                init_mapping[init_a](self.lora_A_route[adapter_name].weight, **init_kwargs[init_a])
                for i in range(self.expert_num):
                    init_mapping[init_a](self.lora_A_experts[adapter_name][i].weight, **init_kwargs[init_a])
        if adapter_name in self.lora_embedding_A.keys():
            # initialize a the same way as the default for nn.linear and b to zero
            nn.init.ones_(self.lora_embedding_A[adapter_name])
            nn.init.ones_(self.lora_embedding_B[adapter_name])


class Linear(nn.Linear, LoraLayer):
    # Lora implemented in a dense layer
    def __init__(
            self,
            adapter_name: str,
            in_features: int,
            out_features: int,
            lora_B: nn.ModuleDict,
            key: str,
            r_ab: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
            fan_in_fan_out: bool = False,
            # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
            scale_ab: float = 1.0,
            init_a: str = 'zero',
            init_b: str = 'zero',
            train_a: bool = True,
            train_b: bool = True,
            rand_R: bool = False,
            expert_num: int = 2,
            **kwargs,
    ):
        init_lora_weights = kwargs.pop("init_lora_weights", True)
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoraLayer.__init__(self, in_features=in_features, out_features=out_features, expert_num=expert_num)
        # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False
        self.train_a = train_a
        self.train_b = train_b
        # self.rand_R = rand_R
        self.fan_in_fan_out = fan_in_fan_out
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

        nn.Linear.reset_parameters(self)
        self.update_layer(adapter_name, r_ab, lora_alpha, lora_dropout, init_lora_weights,
                          init_a, init_b)

        self.lora_B = lora_B

        self.active_adapter = adapter_name

   
        self.key = key

    def forward(self, x: torch.Tensor):
        previous_dtype = x.dtype
        result = None
        if self.active_adapter not in self.lora_A.keys() and self.active_adapter not in self.lora_C.keys():
            return F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        if self.disable_adapters:
            if self.r_ab[self.active_adapter] > 0 and self.merged:
                self.unmerge()
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        elif self.r_ab[self.active_adapter] > 0 and not self.merged:


            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
            x = x.to(self.lora_A[self.active_adapter].weight.dtype)

            after_encoder = self.lora_A[self.active_adapter](self.lora_dropout[self.active_adapter](x))

            after_encoder_splits = torch.chunk(after_encoder, self.expert_num, dim=-1)

            processed_splits = []

            for i in range(self.expert_num):
                split = after_encoder_splits[i]  
      
                expert_output = self.lora_A_experts[self.active_adapter][i](split) 

                decoded_output = self.lora_B[self.active_adapter][i](expert_output) 
                processed_splits.append(decoded_output)
            

            processed_stack = torch.stack(processed_splits, dim=-2)  

   
            heads = after_encoder.view(after_encoder.shape[0], after_encoder.shape[1], self.expert_num, self.r_ab[self.active_adapter] // self.expert_num)  

            heads_permuted = heads.permute(0, 1, 3, 2).contiguous()  
            talked_heads = self.lora_A_talk[self.active_adapter](heads_permuted)  

            interacted_heads = talked_heads.permute(0, 1, 3, 2).contiguous() 
            
   
            mixed_after_encoder = interacted_heads.view(after_encoder.shape[0], after_encoder.shape[1], self.r_ab[self.active_adapter])  

            route_weight = self.lora_dropout[self.active_adapter](nn.functional.softmax(self.lora_A_route[self.active_adapter](mixed_after_encoder), dim=-1))

       
            route_weight = route_weight.unsqueeze(-1) 

            after_weight = (processed_stack * route_weight).sum(dim=-2) 




            result += self.lora_dropout[self.active_adapter](after_weight)
           
           


        elif result is None:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

        result = result.to(previous_dtype)

        return result
