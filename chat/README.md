# Fine-tuning StarCoder for chat-based applications


## Getting started

### Installation

### Launch training

We use DeepSpeed ZeRO-3 to shard the model and optimizer across 8 x A100 (80GB) GPUs. To train StarChat run:

```
TRANSFORMERS_VERBOSITY=info torchrun --nproc_per_node=8 chat/train.py chat/config.yaml --deepspeed=chat/deepspeed_z3_config_bf16.json
```