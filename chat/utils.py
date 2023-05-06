# coding=utf-8
# Copyright 2023 The HuggingFace Team. All rights reserved.
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

import dataclasses
import os
from dataclasses import dataclass
from typing import List, Optional

from transformers import AutoConfig, HfArgumentParser

from huggingface_hub import login


class StarChatArgumentParser(HfArgumentParser):
    def parse_yaml_and_args(self, yaml_arg: str, other_args: Optional[List[str]] = None) -> List[dataclass]:
        arg_list = self.parse_yaml_file(os.path.abspath(yaml_arg))

        outputs = []
        # strip other args list into dict of key-value pairs
        other_args = {arg.split("=")[0].strip("-"): arg.split("=")[1] for arg in other_args}
        used_args = {}

        # overwrite the default/loaded value with the value provided to the command line
        # adapted from https://github.com/huggingface/transformers/blob/d0b5002378daabf62769159add3e7d66d3f83c3b/src/transformers/hf_argparser.py#L327
        for data_yaml, data_class in zip(arg_list, self.dataclass_types):
            keys = {f.name for f in dataclasses.fields(data_yaml) if f.init}
            inputs = {k: v for k, v in vars(data_yaml).items() if k in keys}
            for arg, val in other_args.items():
                # add only if in keys
                if arg in keys:
                    base_type = data_yaml.__dataclass_fields__[arg].type
                    inputs[arg] = val

                    # cast type for ints, floats, and bools (default to strings)
                    if base_type in [int, float, bool]:
                        inputs[arg] = base_type(val)

                    # add to used-args so we can check if double add
                    if arg not in used_args:
                        used_args[arg] = val
                    else:
                        raise ValueError(f"Duplicate argument provided: {arg}, may cause unexpected behavior")

            obj = data_class(**inputs)
            outputs.append(obj)

        return outputs


def get_model_architecture(model_path_or_name: str, revision: str = "main") -> str:
    config = AutoConfig.from_pretrained(model_path_or_name, revision=revision)
    architectures = config.architectures
    if architectures is None or len(architectures) > 1:
        raise ValueError(
            f"The model architecture is either not defined or not unique. Found architectures: {architectures}"
        )
    return architectures[0]


def hf_login():
    """Login to HuggingFace Hub if HF_TOKEN is defined in the environment"""
    hf_token = os.getenv("HF_TOKEN")
    if hf_token is not None:
        login(token=hf_token)


def init_wandb_training(training_args):
    """
    Helper function for setting up Weights & Biases logging tools.
    """
    os.environ["WANDB_ENTITY"] = training_args.wandb_entity
    os.environ["WANDB_PROJECT"] = training_args.wandb_project
    os.environ["WANDB_RUN_GROUP"] = training_args.wandb_run_group
    if training_args.wandb_run_id is not None:
        os.environ["WANDB_RUN_ID"] = training_args.wandb_run_id
    if training_args.wandb_tags is not None:
        os.environ["WANDB_TAGS"] = ",".join(tag for tag in training_args.wandb_tags)


def get_wandb_kwargs(training_args):
    """
    Helper function for passing off Weights & Biases logging settings to TRL.
    """
    tracker_kwargs = {
        "entity": training_args.wandb_entity,
        # "wandb_project": training_args.wandb_project, # this is and arg now kwarg in TRL init
        "group": training_args.wandb_run_group,
    }
    if training_args.wandb_run_id is not None:
        tracker_kwargs["id"] = training_args.wandb_run_id
    if training_args.wandb_tags is not None:
        tracker_kwargs["tags"] = training_args.wandb_tags

    return {
        "wandb": tracker_kwargs,
    }
