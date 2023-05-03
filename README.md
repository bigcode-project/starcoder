# 💫 StarCoder

# What is this about?
💫 StarCoder is a language model (LM) trained on source code and natural language text. Its training data incorporates more that 80 different programming languages as well as text extracted from github issues and commits and from notebooks. This repository showcases how we get an overview of this LM's capabilities.

# Table of Contents
1. [Fine-tuning](#fine-tuning)
  - [Step by step installation with conda](#step-by-step-installation-with-conda)
  - [Datasets](#datasets)
    - [Stack Exchange](#stack-exchange-se)
  - [Merging PEFT adapter layers](#merging-peft-adapter-layers)
# Fine-tuning

Here, we showcase how we can fine-tune this LM on a specific downstream task.

## Step by step installation with conda 

Create a new conda environment and activate it
```bash
conda create -n env
conda activate env
```
Install the `pytorch` version compatible with your version of cuda [here](https://pytorch.org/get-started/previous-versions/), for example the following command works with cuda 11.6
```bash
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
```
Install `transformers` and `peft`
```bash
conda install -c huggingface transformers 
pip install  git+https://github.com/huggingface/peft.git
```
Note that you can install the latest stable version of transformers by using

```bash
pip install git+https://github.com/huggingface/transformers
```

Install `datasets`, `accelerate` and `huggingface_hub`

```bash
conda install -c huggingface -c conda-forge datasets
conda install -c conda-forge accelerate
conda install -c conda-forge huggingface_hub
```

Finally, install `bitsandbytes` and `wandb`
```bash
pip install bitsandbytes
pip install wandb
```
To get the full list of arguments with descriptions you can run the following command on any script:
```
python scripts/some_script.py --help
```
Before you run any of the scripts make sure you are logged in and can push to the hub:
```bash
huggingface-cli login
```
Make sure you are logged in `wandb`:
```bash
wandb login
```
Now that everything is done, you can clone the repository and get into the corresponding directory.

## Datasets
💫 StarCoder can be fine-tuned to achieve multiple downstream tasks. Our interest here is to fine-tune StarCoder in order to make it follow instructions. [Instruction fine-tuning](https://arxiv.org/pdf/2109.01652.pdf) has gained a lot of attention recently as it proposes a simple framework that teaches language models to align their outputs with human needs. That procedure requires the availability of quality instruction datasets, which contain multiple `instruction - answer` pairs. Unfortunately such datasets are not ubiquitous but thanks to Hugging Face 🤗's [datasets](https://github.com/huggingface/datasets) library we can have access to some good proxies. To fine-tune cheaply and efficiently, we use Hugging Face 🤗's [PEFT](https://github.com/huggingface/peft) as well as Tim Dettmers' [bitsandbytes](https://github.com/TimDettmers/bitsandbytes).


### Stack Exchange SE
[Stack Exchange](https://en.wikipedia.org/wiki/Stack_Exchange) is a well-known network of Q&A websites on topics in diverse fields. It is a place where a user can ask a question and obtain answers from other users. Those answers are scored and ranked based on their quality. [Stack exchange instruction](https://huggingface.co/datasets/ArmelR/stack-exchange-instruction) is a dataset that was obtained by scrapping the site in order to build a collection of Q&A pairs. A language model can then be fine-tuned on that dataset to make it elicit strong and diverse question-answering skills.

To execute the fine-tuning script run the following command:
```bash
python finetune/finetune.py \
  --model_path="bigcode/large-model"\
  --dataset_name="ArmelR/stack-exchange-instruction"\
  --subset="data/finetune"\
  --split="train"\
  --size_valid_set 10000\
  --streaming\
  --seq_length 2048\
  --max_steps 1000\
  --batch_size 1\
  --input_column_name="question"\
  --output_column_name="response"\ 
  --gradient_accumulation_steps 16\
  --learning_rate 1e-4\
  --lr_scheduler_type="cosine"\
  --num_warmup_steps 100\
  --weight_decay 0.05\
  --output_dir="./checkpoints" \
```
The command is quite similar to the what we use on alpaca code. However, the size of the SE dataset is better manageable when using streaming. We also have to precise the split of the dataset that is used. For more details, check the [dataset's page](https://huggingface.co/datasets/ArmelR/stack-exchange-instruction) on 🤗. Similarly we can modify the command to account for the availability of GPUs

```bash
python -m torch.distributed.launch \
  --nproc_per_node number_of_gpus finetune/finetune.py \
  --model_path="bigcode/large-model"\
  --dataset_name="ArmelR/stack-exchange-instruction"\
  --subset="data/finetune"\
  --split="train"\
  --size_valid_set 10000\
  --streaming \
  --seq_length 2048\
  --max_steps 1000\
  --batch_size 1\
  --input_column_name="question"\
  --output_column_name="response"\ 
  --gradient_accumulation_steps 16\
  --learning_rate 1e-4\
  --lr_scheduler_type="cosine"\
  --num_warmup_steps 100\
  --weight_decay 0.05\
  --output_dir="./checkpoints" \
```
## Merging PEFT adapter layers
If you train a model with PEFT, you'll need to merge the adapter layers with the base model if you want to run inference / evaluation. To do so, run:
```bash
python finetune/merge_peft_adapters.py --model_name_or_path model_to_merge --peft_model_path model_checkpoint

# Push merged model to the Hub
python finetune/merge_peft_adapters.py --model_name_or_path model_to_merge --peft_model_path model_checkpoint --push_to_hub
```
For example

```bash
python finetune/merge_peft_adapters.py --model_name_or_path bigcode/large-model --peft_model_path checkpoints/checkpoint-1000 --push_to_hub
```

## How to do text-generation with StarCoder

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "bigcode/large-model"
device = "cuda" # for GPU usage or "cpu" for CPU usage

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint, trust_remote_code=True).to(device)

inputs = tokenizer.encode("def print_hello_world():", return_tensors="pt").to(device)
outputs = model.generate(inputs)
print(tokenizer.decode(outputs[0]))
```
## Text-inference
