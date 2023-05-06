# Fine-tuning StarCoder for chat-based applications

This is a fully-working example to fine-tune StarCoder on multi-turn dialogues like those curated from OpenAssistant. 

## Getting started

To run the `train.py` script, first create a Python virtual environment using e.g. Conda:

```shell
conda create -n starchat python=3.10 && conda activate starchat
```

Next, install PyTorch - since this hardware-dependent, we
direct you to the [PyTorch Installation Page](https://pytorch.org/get-started/locally/) for this step (make sure you install v1.13.1).

```shell
pip install -r requirements.txt
```

You'll also need to be logged into both your Hugging Face and Weights and Biases accounts. To do so, run:

```shell
huggingface-cli login

wandb login
```

Finally, install Git LFS with:

```shell
sudo apt-get install git-lfs
```

### Launch training

We use DeepSpeed ZeRO-3 to shard the model and optimizer across 8 x A100 (80GB) GPUs. To train StarChat run:

```
TRANSFORMERS_VERBOSITY=info torchrun --nproc_per_node=8 chat/train.py chat/config.yaml --deepspeed=chat/deepspeed_z3_config_bf16.json
```