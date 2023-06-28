# coding=utf-8
# Copyright 2023 The BigCode and HuggingFace teams. All rights reserved.
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
#
"""A simple script to quickly check the model outputs of a generative model"""
import argparse

import torch
from dialogues import DialogueTemplate, get_dialogue_template
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          GenerationConfig, set_seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        type=str,
        help="Name of model to generate samples with",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="The model repo's revision to use",
    )
    parser.add_argument(
        "--system_prompt", type=str, default=None, help="Overrides the dialogue template's system prompt"
    )
    args = parser.parse_args()

    # Set seed for reproducibility
    set_seed(42)

    prompts = [
        [
            {
                "role": "user",
                "content": "Develop a C++ program that reads a text file line by line and counts the number of occurrences of a specific word in the file.",
            }
        ],
        [
            {
                "role": "user",
                "content": "Implement a Python function to find the longest common subsequence of two input strings using dynamic programming.",
            }
        ],
        [{"role": "user", "content": "Implement a regular expression in Python to validate an email address."}],
        [
            {
                "role": "user",
                "content": "Write a program to find the nth Fibonacci number using dynamic programming.",
            }
        ],
        [
            {
                "role": "user",
                "content": "Implement a binary search algorithm to find a specific element in a sorted array.",
            }
        ],
        [{"role": "user", "content": "Implement a queue data structure using two stacks in Python."}],
        [
            {
                "role": "user",
                "content": "Implement a program to find the common elements in two arrays without using any extra data structures.",
            }
        ],
    ]

    try:
        dialogue_template = DialogueTemplate.from_pretrained(args.model_id, revision=args.revision)
    except Exception:
        print("No dialogue template found in model repo. Defaulting to the `no_system` template.")
        dialogue_template = get_dialogue_template("no_system")

    if args.system_prompt is not None:
        dialogue_template.system = args.system_prompt
    formatted_prompts = []
    for prompt in prompts:
        dialogue_template.messages = [prompt] if isinstance(prompt, dict) else prompt
        formatted_prompts.append(dialogue_template.get_inference_prompt())

    print("=== SAMPLE PROMPT ===")
    print(formatted_prompts[0])
    print("=====================")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, revision=args.revision)
    print(f"Special tokens: {tokenizer.special_tokens_map}")
    print(f"EOS token ID for generation: {tokenizer.convert_tokens_to_ids(dialogue_template.end_token)}")
    generation_config = GenerationConfig(
        temperature=0.2,
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.2,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.convert_tokens_to_ids(dialogue_template.end_token),
        min_new_tokens=32,
        max_new_tokens=256,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, revision=args.revision, load_in_8bit=True, device_map="auto", torch_dtype=torch.float16
    )
    outputs = ""
    for idx, prompt in enumerate(formatted_prompts):
        batch = tokenizer(prompt, return_tensors="pt", return_token_type_ids=False).to(device)
        generated_ids = model.generate(**batch, generation_config=generation_config)
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=False).lstrip()
        outputs += generated_text + "\n\n"
        print(f"=== EXAMPLE {idx} ===")
        print()
        print(generated_text)
        print()
        print("======================")
        print()

    raw_model_name = args.model_id.split("/")[-1]
    model_name = f"{raw_model_name}"
    if args.revision is not None:
        model_name += f"-{args.revision}"

    with open(f"data/samples-{model_name}.txt", "w", encoding="utf-8") as f:
        f.write(outputs)


if __name__ == "__main__":
    main()
