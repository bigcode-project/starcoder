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
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import transformers
from transformers import MODEL_FOR_CAUSAL_LM_MAPPING


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    use_peft: Optional[bool] = field(
        default=False,
        metadata={"help": ("Whether to use PEFT or not for training.")},
    )
    lora_r: Optional[int] = field(
        default=16,
        metadata={"help": ("LoRA R value.")},
    )
    lora_alpha: Optional[int] = field(
        default=32,
        metadata={"help": ("LoRA alpha.")},
    )
    lora_dropout: Optional[float] = field(
        default=0.05,
        metadata={"help": ("LoRA dropout.")},
    )
    lora_target_modules: Optional[List[str]] = field(
        default=None,
        metadata={"help": ("LoRA target modules.")},
    )
    lora_modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={"help": ("Model layers to unfreeze & train")},
    )


@dataclass
class RewardModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to use for the reward model in RLHF.
    """

    rm_model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    rm_model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_mixer: Optional[Dict[str, float]] = field(
        default=None,
        metadata={"help": ("Datasets and their proportions to be used for training ift/rl.")},
    )
    data_filter_fn: Optional[str] = field(
        default=None, metadata={"help": "option to include non-default data filtering (e.g. for toxicity example)."}
    )
    data_tokenize_fn: Optional[str] = field(
        default=None,
        metadata={"help": "option to include non-default data tokenization (for datasets not in the format)."},
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    dataset_split_name: Optional[str] = field(
        default="train", metadata={"help": "The dataset split to use (via the datasets library)."}
    )
    prompt_column: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the column in the datasets containing the model prompt (usually an instruction)."
        },
    )
    completion_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the completions."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    pre_tokenized: Optional[bool] = field(
        default=False,
        metadata={"help": "If the training dataset is pre-tokenized."},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": (
                "The maximum total sequence length for target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total sequence length for validation target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
                "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
                "during ``evaluate`` and ``predict``."
            )
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to model maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
            )
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                "which is used during ``evaluate`` and ``predict``."
            )
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default="", metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )

    forced_bos_token: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The token to force as the first generated token after the decoder_start_token_id."
                "Useful for multilingual models like mBART where the first generated token"
                "needs to be the target language token (Usually it is the target language token)"
            )
        },
    )
    prompt_template: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the prompt template to use for conditioning the model. Deprecated in favour of `dialogue_template`"
        },
    )
    dialogue_template: Optional[str] = field(
        default="no_system",
        metadata={
            "help": "The name of the dialogue template to use for conditioning the model. See h4.training.dialogues for choices."
        },
    )

    def __post_init__(self):
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    """
    Arguments related to the training process itself. For all parameters, see: https://huggingface.co/docs/transformers/v4.26.1/en/main_classes/trainer#transformers.TrainingArguments
    """

    logging_first_step: Optional[bool] = field(
        default=True,
        metadata={"help": ("Whether to log and evaluate the first global_step or not.")},
    )
    optim: Optional[str] = field(default="adamw_torch")
    reward_loss_fn: Optional[str] = field(
        default="NegLogSigmoid",
        metadata={"help": ("Loss function for reward model.")},
    )
    wandb_tags: Optional[List[str]] = field(
        default=None,
        metadata={"help": ("Tags to group and filter runs on Weights and Biases.")},
    )
    wandb_enabled: Optional[bool] = field(
        default=True,
        metadata={"help": ("Whether to enable or disable WandB.")},
    )
    wandb_project: Optional[str] = field(
        default="h4",
        metadata={"help": ("The project to store runs under.")},
    )
    wandb_entity: Optional[str] = field(
        default="huggingface",
        metadata={"help": ("The entity to store runs under.")},
    )
    wandb_run_group: Optional[str] = field(
        default="tr_00_some-descriptor",
        metadata={"help": ("Group multiple runs under this group name.")},
    )
    wandb_run_id: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Set this to a globally unique string (per project) corresponding to a single run of your script."
            )
        },
    )


@dataclass
class RLTrainingArguments:
    """
    Arguments related to the TRL training process.
    """

    decode_skip_special_tokens: Optional[bool] = field(
        default=True,
        metadata={"help": ("Whether to skip special tokens when decoding. Used rarely for debugging.")},
    )
    disable_eos_token_id: Optional[bool] = field(
        default=False,
        metadata={"help": ("Whether to set `eos_token_id = -1` or not")},
    )
    early_stopping: Optional[bool] = field(
        default=False,
        metadata={"help": ("Whether to apply early stopping during PPO")},
    )
    gradient_accumulation_steps: Optional[int] = field(
        default=1, metadata={"help": "The number of gradient accumulation steps"}
    )
    hub_model_id: Optional[str] = field(
        default=None,
        metadata={"help": ("Hub model ID to push model to.")},
    )
    learning_rate: Optional[float] = field(
        default=1e-5,
        metadata={"help": ("Learning rate for RL training.")},
    )
    max_generation_length: Optional[int] = field(
        default=128,
        metadata={"help": ("The max_length length the generated tokens can have.")},
    )
    min_generation_length: Optional[int] = field(
        default=5,
        metadata={"help": ("The minimum length the generated tokens can have.")},
    )
    num_train_epochs: Optional[int] = field(
        default=4,
        metadata={"help": ("Number of epochs for RL training.")},
    )
    num_shared_layers: Optional[int] = field(
        default=None,
        metadata={"help": ("Number of shared layers in the reference model.")},
    )
    optim: Optional[str] = field(
        default=None,
        metadata={"help": ("The optimiser to use for gradient descent")},
    )
    output_dir: Optional[str] = field(
        default="data",
        metadata={"help": ("The directory to save models to")},
    )
    per_device_batch_size: Optional[int] = field(
        default=32,
        metadata={"help": ("Batch size for RL training -- PPO optimizer.")},
    )
    per_device_mini_batch_size: Optional[int] = field(
        default=1,
        metadata={"help": ("Batch size for RL training -- inference per step on GPU.")},
    )
    ppo_epochs: Optional[int] = field(
        default=4,
        metadata={"help": ("Number of PPO optimisation epochs per batch of samples.")},
    )
    push_to_hub: Optional[bool] = field(
        default=False,
        metadata={"help": ("Whether to push the model to the Hub.")},
    )
    push_best_reward: Optional[bool] = field(
        default=True,
        metadata={"help": ("Whether to push the model with the best reward or not to the hub.")},
    )
    push_best_reward_period: Optional[int] = field(
        default=5,
        metadata={"help": ("Number of steps in between model save for best reward.")},
    )
    return_prompt: Optional[bool] = field(
        default=False,
        metadata={"help": ("Whether to return the prompt in the output from PPO model.")},
    )
    reward_baseline: Optional[float] = field(
        default=0,
        metadata={
            "help": (
                "Mean reward on training set for reward model substracted to make reward during RL training 0 mean."
            )
        },
    )
    save_steps: Optional[int] = field(
        default=50,
        metadata={"help": ("Number of steps in between model save (RL).")},
    )
    seed: Optional[int] = field(
        default=42,
        metadata={"help": ("The seed for training.")},
    )
    target_kl: Optional[float] = field(default=0.1, metadata={"help": "KL target for early stopping"})
    temperature: Optional[float] = field(
        default=1.0,
        metadata={"help": ("The value used to modulate the next token probabilities.")},
    )
    top_k: Optional[int] = field(
        default=50,
        metadata={"help": ("The number of highest probability vocabulary tokens to keep for top-k-filtering.")},
    )
    top_p: Optional[float] = field(
        default=1.0,
        metadata={
            "help": (
                "If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation."
            )
        },
    )
    wandb_tags: Optional[List[str]] = field(
        default=None,
        metadata={"help": ("Tags to group and filter runs on Weights and Biases.")},
    )
    wandb_enabled: Optional[bool] = field(
        default=True,
        metadata={"help": ("Whether to enable or disable WandB.")},
    )
    wandb_project: Optional[str] = field(
        default="h4",
        metadata={"help": ("The project to store runs under.")},
    )
    wandb_entity: Optional[str] = field(
        default="huggingface",
        metadata={"help": ("The entity to store runs under.")},
    )
    wandb_run_group: Optional[str] = field(
        default="tr_00_some-descriptor",
        metadata={"help": ("Group multiple runs under this group name.")},
    )
    wandb_run_id: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Set this to a globally unique string (per project) corresponding to a single run of your script."
            )
        },
    )


@dataclass
class Seq2SeqTrainingArguments(transformers.Seq2SeqTrainingArguments):
    """
    Arguments related to the training process itself. For all parameters, see: https://huggingface.co/docs/transformers/v4.26.1/en/main_classes/trainer#transformers.TrainingArguments
    """

    wandb_tags: Optional[List[str]] = field(
        default=None,
        metadata={"help": ("Tags to group and filter runs on Weights and Biases.")},
    )
    wandb_enabled: Optional[bool] = field(
        default=True,
        metadata={"help": ("Whether to enable or disable WandB.")},
    )
    wandb_project: Optional[str] = field(
        default="h4",
        metadata={"help": ("The project to store runs under.")},
    )
    wandb_entity: Optional[str] = field(
        default="huggingface",
        metadata={"help": ("The entity to store runs under.")},
    )
    wandb_run_group: Optional[str] = field(
        default="tr_00_some-descriptor",
        metadata={"help": ("Group multiple runs under this group name.")},
    )
    wandb_run_id: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Set this to a globally unique string (per project) corresponding to a single run of your script."
            )
        },
    )
