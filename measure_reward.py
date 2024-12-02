from transformers import AutoTokenizer, AutoModelForSequenceClassification, LlamaTokenizer, LlamaForSequenceClassification
import argparse
import torch
from typing import Optional, List
import json
import re
import os
from transformers import PreTrainedModel, LlamaConfig, LlamaModel, LlamaTokenizer
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
parser = argparse.ArgumentParser()
parser.add_argument("--out_file", type=str)
parser.add_argument("--model_name", type=str)
parser.add_argument("--reward_model", type=str, default="openbmb/UltraRM-13b")
parser.add_argument("--rm_gpu", type=str, default="cuda:1")
parser.add_argument("--tokenizer", type=str, default="AlekseyKorshuk/vicuna-7b")
parser.add_argument("--npout", type=str, default="")
parser.add_argument("--dataset_name", type=str, default="shp")

args = parser.parse_args()


class LlamaRewardModel(PreTrainedModel):
    config_class = LlamaConfig
    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.regression_head = nn.Linear(self.config.hidden_size, 1, bias=False)

    def forward( # args are the same as LlamaForCausalLM
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):

        transformer_outputs = self.model(
                                input_ids,
                                attention_mask=attention_mask,
                                position_ids=position_ids,
                                past_key_values=past_key_values,
                                inputs_embeds=inputs_embeds,                               
                            )

        hidden_states = transformer_outputs[0]
        rewards = self.regression_head(hidden_states).squeeze(-1)
        
        ends = attention_mask.cumsum(dim=1).argmax(dim=1).view(-1,1)
        rewards = torch.gather(rewards, 1, ends)
        
        return rewards

#load response here
path = os.path.join("response_value", f"{args.out_file}.json")
with open(path, "r") as out_f:
    lines = json.load(out_f)


if args.reward_model == 'openbmb/UltraRM-13b':
    reward_model = LlamaRewardModel.from_pretrained(args.reward_model, device_map=device, 
                            trust_remote_code=True, torch_dtype=torch.bfloat16)
    tokenizer = LlamaTokenizer.from_pretrained(args.reward_model, use_fast=True)
else:
    reward_model = AutoModelForSequenceClassification.from_pretrained(args.reward_model, num_labels=1, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(args.reward_model)

tokenizer.pad_token = tokenizer.eos_token
reward_model.config.pad_token_id = tokenizer.pad_token_id

reward_model = reward_model.to(args.rm_gpu)

def data_process(text):

    text = text.replace('User:', "Human:")
    text = text.replace('ASSISTANT:', "Assistant:")
    return text   

def extract_out(output_data):
    if "response" in output_data:
        output = output_data["response"]
    elif "output" in output_data:
        output = output_data["output"]

    if args.dataset_name == "hh_rlhf":
        output_np = output.removeprefix(output_data["prompt"])
        if output_np.startswith(": "): output = output_np[2:]
        output_np = re.split("human:", output_np, flags=re.IGNORECASE)[0]
        return output_data["prompt"]+output_np
    elif args.dataset_name == "shp":
        output_np = output.removeprefix(output_data["prompt"])
        if output_np.startswith(": "): output = output_np[2:]
        if args.model_name == 'vicuna_7B':
            output_np = re.split("Human:", output_np, flags=re.IGNORECASE)[0]
        elif args.model_name == 'llama3_8B':
            output_np = re.split("User:", output_np, flags=re.IGNORECASE)[0]
        all_text = output_data["prompt"]+output_np
        all_text = data_process(all_text)
        return all_text


def get_rm(text):
    #tokens = tokenizer(text, return_tensors="pt").input_ids.to(args.rm_gpu)
    tokens = tokenizer(text, return_tensors="pt", padding=True).to(args.rm_gpu)
    #print(f"{tokens.shape=}")
    with torch.no_grad():
        output = reward_model(**tokens).cpu().item()

    del tokens
    return output


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
if not os.path.exists("final_reward"):
    os.makedirs("final_reward")
with open(f"final_reward/{args.out_file}.json", "w") as out_f:
    json.dump({"average reward": np.mean(rm_scores), "num_skip": num_skip}, out_f)
    out_f.write('\n')
    json.dump({"all reward": rm_scores}, out_f)