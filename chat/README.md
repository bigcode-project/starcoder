# Fine-tuning StarCoder for chat-based applications

This is a fully-working example to fine-tune `StarCoder` on a corpus of multi-turn dialogues and thus create a coding assistant that is chatty and helpful. The resulting model is quite good at generating code for plots and other programming tasks. For example, given the prompt:

```
Draw me a map of the world using geopandas. Make it so that only Germany and Spain are colored red.
```

the model will produce:

```python
import pandas as pd
from matplotlib import pyplot as plt
import geopandas as gpd
from shapely.geometry import Point, Polygon
%matplotlib inline
# Load data from URL or local file
url = 'https://raw.githubusercontent.com/datasets/geo-countries/master/data/countries.geojson'
df = gpd.read_file(url)
 
# Create plot
fig, ax = plt.subplots()
ax.set_aspect('equal')
world = gpd.GeoDataFrame({'name': ['World'],
                          'geometry': [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])]})
germany = df[df['ADMIN'] == "Germany"]
spain = df[df['ADMIN'] == "Spain"]
other = df[(df['ADMIN']!= "Germany") & (df['ADMIN']!= "Spain")]
world.plot(color='lightgrey', edgecolor='white', ax=ax)
germany.plot(color="red", ax=ax)
spain.plot(color="red", ax=ax)
other.plot(color="skyblue", ax=ax)
plt.title("European Countries")
plt.show()
```

Check out our [blog post](https://huggingface.co/blog/starchat-alpha) for more details.

## Getting started

To run the `train.py` script, first create a Python virtual environment using e.g. Conda:

```shell
conda create -n chat python=3.10 && conda activate chat
```

Next, install PyTorch v1.13.1. Since this is hardware-dependent, we direct you to the [PyTorch Installation Page](https://pytorch.org/get-started/previous-versions/#v1131) for this step. Next, install the rest of the project dependencies:

```shell
pip install -r requirements.txt
```

You'll also need to be logged into both your Hugging Face account. To do so, run:

```shell
huggingface-cli login
```

Finally, install Git LFS with:

```shell
sudo apt-get install git-lfs
```

## Prepare your dataset

For training and inference, we use _dialogue templates_ to format each message in a conversation. For example, a typical dialogue between a human user and AI assistant takes the form:

```json
{
    "messages": [
        {
            "content": "Is it possible to imagine a society without law?", 
            "role": "user"},
        {
            "content": "It is difficult to imagine a society that is able to be maintained without any semblance of Law.",
            "role": "assistant",
        },
        {
            "content": "It seems like you consider the absence of law equal to the absence of anything that could guide the behaviour of the individual.",
            "role": "user",
        },
        {
            "content": "You are correct that there are other factors that can guide behavior in a society and play a role in shaping individuals' behavior and interactions with each other. However, even in societies where these factors are present, laws still serve an important role in maintaining social order and resolving conflicts.",
            "role": "assistant",
        }
    ]
}
```

Make sure you convert your dataset according to this schema, in particular you need to include a `messages` column like the above. You can adjust the model, dataset, and hyperparamters in the `config.yaml` file.

## Launch training

We use DeepSpeed ZeRO-3 to shard the model and optimizer across 8 x A100 (80GB) GPUs. To fine-tune run:

```
TRANSFORMERS_VERBOSITY=info torchrun --nproc_per_node=8 train.py config.yaml --deepspeed=deepspeed_z3_config_bf16.json
```

By default, this will save the model checkpoint in the `data/` directory and also push it to the Hugging Face Hub.


## Generate samples

To generate a few coding examples from your model, run:

```shell
python generate.py --model_id path/to/your/model
```

