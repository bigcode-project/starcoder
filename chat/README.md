# Fine-tuning StarCoder for chat-based applications

## Getting started

```
TRANSFORMERS_VERBOSITY=info torchrun --nproc_per_node=2 chat/train.py chat/config.yaml --deepspeed=chat/deepspeed_z3_config_bf16.json
```