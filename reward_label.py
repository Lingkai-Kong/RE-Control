import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
#import intervented_model.llama as llama
from datasets import load_dataset
from torch.utils.data import DataLoader
import re
from transformers import LlamaTokenizer, LlamaForCausalLM, AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer, PreTrainedModel
import os
import numpy as np
from tqdm import tqdm
from peft import PeftModel, PeftConfig
import json
from tqdm import tqdm

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

class ListDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
        

def data_collactor(data):

    responses = []
    for sample in data:
        text = sample['response']
        text = text.replace('USER:', "Human:")
        text = text.replace('ASSISTANT:', "Assistant:")
        responses.append(text)
    return responses   



def get_rm(text, rm_model, tokenizer, args):
    encoded_input = tokenizer(text, return_tensors="pt", padding=True)
    inputd_ids = encoded_input['input_ids'].to(args.device)
    attention_mask = encoded_input['attention_mask'].to(args.device)

    with torch.no_grad():
        rm_out = rm_model(inputd_ids, attention_mask=attention_mask)

    rm_val = rm_out.logits # shape: batch_size x 1

    return rm_val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='vicuna_7B', choices=["vicuna_7B", "falcon_7B", "llama3_8B"])
    parser.add_argument('--dataset_name', type=str, default='hh_rlhf', choices=["hh_rlhf", "shp"])
    parser.add_argument('--reward_model', type=str, default='openbmb/UltraRM-13b')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--num_samples', type=int, default=1)

    parser.add_argument('--device', type=int, default=1)
    args = parser.parse_args()

    ## load the off-the-self reward model
    if args.reward_model == 'openbmb/UltraRM-13b':
        reward_model = LlamaRewardModel.from_pretrained(path, device_map=device, 
                               trust_remote_code=True, torch_dtype=torch.bfloat16)
        tokenizer = LlamaTokenizer.from_pretrained(path, use_fast=True)
    else:
        reward_model = AutoModelForSequenceClassification.from_pretrained(args.reward_model, num_labels=1, torch_dtype=torch.bfloat16)
        tokenizer = AutoTokenizer.from_pretrained(args.reward_model)
    
    device = args.device
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    reward_model.config.pad_token_id = tokenizer.pad_token_id

    reward_model = reward_model.to(args.device)
    

    if args.mode == 'train':
        out_file = 'features/'+ str(args.model_name) + '_' + str(args.dataset_name) + '_' +'response_train.json'
    elif args.mode == 'test':
        out_file = 'features/'+ str(args.model_name) + '_' + str(args.dataset_name) + '_' +'response_test.json'

    with open(out_file, "r") as out_f:
        lines = json.load(out_f)

    dataset = ListDataset(lines)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=data_collactor)
    

    rm_scores = []
    
    for i, data in enumerate(tqdm(data_loader)):
        # print(f"{get_rm(outp)}")
        rm_score = get_rm(data, reward_model, tokenizer, args)

        rm_scores.append(rm_score)

    rm_scores = torch.cat(rm_scores, dim=0)

    storage_path = None
    if args.mode == 'train':
        storage_path = 'features/'+ str(args.model_name) + '_' + str(args.dataset_name) + '_' +'labels_train.json'
    elif args.mode == 'test':
        storage_path = 'features/'+ str(args.model_name) + '_' + str(args.dataset_name) + '_' +'labels_test.json'

    torch.save(rm_scores, storage_path)  

if __name__ == "__main__":
    main()