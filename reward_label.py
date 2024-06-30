# We first need to train a value function 
# The value function is a MLP on top of the llama features
# We need to first load the HH-RLHF dataset from huggingface and initialize the model

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
#import intervented_model.llama as llama
from datasets import load_dataset
from torch.utils.data import DataLoader
import re
from transformers import LlamaTokenizer, LlamaForCausalLM, AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
import os
import numpy as np
from tqdm import tqdm
from peft import PeftModel, PeftConfig
import json
from tqdm import tqdm


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
    #print(f"{tokens.shape=}")
    # 1966 1819 1813
    #if tokens.shape[1] >= 1334: return None
    with torch.no_grad():
        rm_out = rm_model(inputd_ids, attention_mask=attention_mask)

    rm_val = rm_out.logits # shape: batch_size x 1

    return rm_val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='llama_7B')
    parser.add_argument('--dataset_name', type=str, default='hhrlhf')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--num_samples', type=int, default=1)

    parser.add_argument('--device', type=int, default=1)
    args = parser.parse_args()
    
    MODEL_NAMES = {
        'vicuna_7B': 'lmsys/vicuna-7b-v1.5', 
        'falcon_7B': 'tiiuae/falcon-7b-instruct'
    }
    MODEL = MODEL_NAMES[args.model_name]
    ## load the base llm model
    tokenizer = AutoTokenizer.from_pretrained('argsearch/llama-7b-rm-float32')
 
    device = args.device
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id


    ## load the off-the-self reward model
    reward_model = AutoModelForSequenceClassification.from_pretrained('argsearch/llama-7b-rm-float32', num_labels=1, torch_dtype=torch.bfloat16)
    
    #from peft import PeftConfig, PeftModel
    #from transformers import AutoModelForCausalLM
    #from peft import AutoPeftModelForCausalLM
    #peft_config = PeftConfig.from_pretrained("/localscratch/haorui/control/LMFlow/output_models/llama2-7b-onlyrm-hh-lora-lr-5e-6") 
    #reward_model.load_adapter('/localscratch/haorui/control/LMFlow/output_models/llama2-7b-onlyrm-hh-lora-lr-5e-6')
    #peft_model = PeftModel.from_pretrained('meta-llama/Llama-2-7b-hf', peft_config)
    reward_model.config.pad_token_id = tokenizer.pad_token_id

    reward_model = reward_model.to(args.device)
    
    ## load the value function model, 
    #value_model = llama.LLaMAForSequenceClassification.from_pretrained(MODEL, num_labels=1)
    #value_model = ValueFunction(input_dim=4096, hidden_dim=4096, output_dim=1)

    # for param in value_model.model.parameters():
    #     param.requires_grad = False
    if args.mode == 'train':
        out_file = 'features/response_train.json'
    elif args.mode == 'test':
        out_file = 'features/response_test.json'

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
    # for line in tqdm(lines):
    #     outp = extract_out(line, args)
    #     if len(outp) == 0: rm_scores.append(0.)
    #     # print(f"{get_rm(outp)}")
    #     rm_score = get_rm(outp, args)
    #     if rm_score == None: 
    #         print("skipped one")
    #         num_skip += 1
    #         continue
    #     else: rm_scores.append(rm_score)




    # create the features directory if no
    storage_path = None
    if args.mode == 'train':
        storage_path = 'features/labels_train.pth'
    elif args.mode == 'test':
        storage_path = 'features/labels_test.pth'
    
    #if not os.path.exists('features_phi-2'):
        #os.makedirs('features_phi-2')
    # get the file path to save the results
    #save the tensor rm_scores to the file

    torch.save(rm_scores, storage_path)  

if __name__ == "__main__":
    main()