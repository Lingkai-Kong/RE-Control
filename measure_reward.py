from transformers import AutoTokenizer, AutoModelForSequenceClassification, LlamaTokenizer, LlamaForSequenceClassification
import argparse
import torch
import json
import re
import os

parser = argparse.ArgumentParser()
parser.add_argument("--out_file", type=str)
parser.add_argument("--rm", type=str, default="argsearch/llama-7b-rm-float32")
parser.add_argument("--rm_gpu", type=str, default="cuda:3")
parser.add_argument("--tokenizer", type=str, default="AlekseyKorshuk/vicuna-7b")
parser.add_argument("--npout", type=str, default="")
parser.add_argument("--experiment", type=str, default="hhrlhf")

args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

path = os.path.join("response_github", f"{args.out_file}.json")
with open(path, "r") as out_f:
    lines = json.load(out_f)

rm_model = AutoModelForSequenceClassification.from_pretrained(args.rm, num_labels=1, torch_dtype=torch.float16, cache_dir='./huggingface_cache').to(args.rm_gpu)


def extract_out(output_data):
    if "response" in output_data:
        output = output_data["response"]
    elif "output" in output_data:
        output = output_data["output"]

    if args.experiment == "hhrlhf":
        output_np = output.removeprefix(output_data["prompt"])
        if output_np.startswith(": "): output = output_np[2:]
        output_np = re.split("human:", output_np, flags=re.IGNORECASE)[0]
        return output_data["prompt"]+output_np


def get_rm(text):
    tokens = tokenizer(text, return_tensors="pt").input_ids.to(args.rm_gpu)
    print(f"{tokens.shape=}")
    rm_out = rm_model(tokens)
    rm_val = rm_out.logits.flatten().item()

    del rm_out
    del tokens
    return rm_val


def get_rm_from_tokens(tokens):
    return rm_model(torch.tensor(tokens).unsqueeze(0).to(args.rm_gpu)).logits.flatten().item()

from tqdm import tqdm

rm_scores = []
num_skip = 0
for line in tqdm(lines):
    outp = extract_out(line)
    if len(outp) == 0: rm_scores.append(0.)
    rm_score = get_rm(outp)
    if rm_score == None: 
        print("skipped one")
        num_skip += 1
        continue
    else: rm_scores.append(rm_score)

import numpy as np
print(f"{np.mean(rm_scores)=}")
print(f"{num_skip=}")
if not os.path.exists("reward"):
    os.makedirs("reward")
with open(f"reward/{args.out_file}.json", "w") as out_f:
    json.dump({"reward": np.mean(rm_scores), "num_skip": num_skip}, out_f)
    out_f.write('\n')
    json.dump({"all reward": rm_scores}, out_f)