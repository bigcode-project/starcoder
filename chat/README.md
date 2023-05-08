# Fine-tuning StarCoder for chat-based applications

This is an educational example to fine-tune `StarCoderBase` on a corpus of multi-turn dialogues and thus create a coding assistant that is chatty and helpful. Check out our [blog post](ADD LINK) for more details.

**Note:** For training and inference, we use _dialogue templates_ to format each message in a conversation. For example, a typical dialogue between a human user and AI assistant takes the form:

```json
{
    "messages": [
        {"content": "Is it possible to imagine a society without law?", "role": "user"},
        {
            "content": "It is difficult to imagine a society that is able to be maintained without any semblance of Law. Laws exists for the purpose of maintaining society and past human behavior suggests that there would people that have no interest in promoting social cohesion and norms so it's unlikely that complex social and economic systems could develop without a legal system.",
            "role": "assistant",
        },
        {
            "content": "It seems like you consider the absence of law equal to the absence of anything that could guide the behaviour of the individual, however there are many other such things, like religion, conscience, tradition to mention few important patterns which are known to have crucial influence on society even today, but also we could imagine more, like universal love if everyone suddenly became 'awaken' in some spiritual sense, or simply just having an omnipotent leader who maintains a certain order without ever codifying the principles in law. Is it still difficult to imagine?",
            "role": "user",
        },
        {
            "content": "You are correct that there are other factors that can guide behavior in a society and play a role in shaping individuals' behavior and interactions with each other. However, even in societies where these factors are present, laws still serve an important role in maintaining social order and resolving conflicts.",
            "role": "assistant",
        }
    ]
}
```

Make sure you convert your dataset according to this schema. You can adjust the model, dataset, and hyperparamters in the `config.yaml` file.

## Getting started

To run the `train.py` script, first create a Python virtual environment using e.g. Conda:

```shell
conda create -n chat python=3.10 && conda activate chat
```

Next, install PyTorch v1.13.1. Since this hardware-dependent, we direct you to the [PyTorch Installation Page](https://pytorch.org/get-started/previous-versions/#v1131) for this step. Once you've installed it, install the rest of the project dependencies:

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

## Launch training

We use DeepSpeed ZeRO-3 to shard the model and optimizer across 8 x A100 (80GB) GPUs. To fine-tune run:

```
TRANSFORMERS_VERBOSITY=info torchrun --nproc_per_node=8 train.py config.yaml --deepspeed=deepspeed_z3_config_bf16.json
```

By default, this will save the model checkpoint in the `data/` directory and also push it to the Hugging Face Hub.


## Generate samples

To generate a few examples from your model, run:

