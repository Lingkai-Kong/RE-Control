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


class DataCollatorReward:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, data):
        batch = {}
        data_batch = []
        for sample in data:
            data_batch.append({"input_ids": sample['input_ids'], "attention_mask": sample["attention_mask"]})
        batch_data= self.tokenizer.pad(data_batch, padding=True, return_tensors="pt")
        batch['input_ids'] = batch_data['input_ids']
        batch['attention_mask'] = batch_data['attention_mask']
        return batch  

def get_llm_activations(model_name, model, dataloader, tokenizer, device, num_samples):
    
    hidden_activations = []
    mask_list = []
    responses = []
    model = model.to(device)
    for s, batch_encoded_input in enumerate(tqdm(dataloader)):
        input_ids = batch_encoded_input['input_ids'].to(device)
        attention_mask = batch_encoded_input['attention_mask'].to(device)
        prompt = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        with torch.no_grad():
            if num_samples > 1:
                outputs = model.generate(input_ids, attention_mask=attention_mask, output_hidden_states=True, return_dict_in_generate=True, num_return_sequences=num_samples, temperature=0.7, top_k=50, top_p=0.95, max_length=1000)

            else:
                outputs = model.generate(input_ids, attention_mask=attention_mask, output_hidden_states=True, return_dict_in_generate=True, max_new_tokens=128)
        
        generated_response = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)

        for prompt, generated_response in zip(prompt, generated_response):
            responses.append({'prompt': prompt, 'response': generated_response, 'results': generated_response.removeprefix(prompt)})

        if num_samples > 1:
            input_ids_repeated = input_ids.repeat_interleave(num_samples, dim=0)
        else:
            input_ids_repeated = input_ids

        hidden_states = process_hidden_states(outputs, input_ids_repeated) # shape: batch_size x length x hidden_dim

        #for falcon and phi-2
        if model_name == 'falcon_7B':
            length_of_prompts_padding = (input_ids == 11)  ## llama's padding id is 2, the same as the eos token id; falcon is 11; phi-2 is 50256
            length_of_prompts_padding = length_of_prompts_padding.sum(dim=1)
            padding_len_answer = (outputs.sequences == 11)
            padding_length = padding_len_answer.sum(dim=1)
            padding_length = padding_length - length_of_prompts_padding
        elif model_name == 'vicuna_7B':
            length_of_prompts_padding = (input_ids == 2)  ## llama's padding id is 2, the same as the eos token id; falcon is 11; phi-2 is 50256
            length_of_prompts_padding = length_of_prompts_padding.sum(dim=1)
       
            padding_len_answer = (outputs.sequences == 0) ## each sequence in the batch can have different lengths, it is padded with 0, so we need to mask the loss
            padding_length = padding_len_answer.sum(dim=1)
            padding_length = padding_length

        range_tensor = torch.arange(len(hidden_states)).expand(hidden_states[0].shape[0], -1)

        thresholds = (len(hidden_states) - padding_length).unsqueeze(1)
        thresholds = thresholds.to(device)
        range_tensor = range_tensor.to(device)
        mask = range_tensor < thresholds

        #mask = mask.int()

        cut = []
        for d in range (outputs.sequences.shape[0]):
            outid = outputs.sequences[d]
            pattern = torch.tensor(tokenizer(['User:'])['input_ids'][0]).to(device)
            pattern_length = pattern.size(0)

            windows = outid.unfold(0, pattern_length, 1)

            matches = (windows == pattern.unsqueeze(0)).all(1)

            matching_indices = torch.where(matches)[0]
            if matching_indices.shape[0] == 0:
                cut.append(torch.tensor(outid.shape[0]).to(device))
            else:
                cut.append(matching_indices[0].to(device))
        cut = torch.stack(cut)
        cut = cut.reshape(-1,1)

        cut_mask = range_tensor < cut

        all_mask = cut_mask & mask

        all_mask= all_mask.int()

        hidden_activations.append([h.cpu() for h in hidden_states])
        mask_list.append(mask.cpu())

    max_length = max(len(hidden) for hidden in hidden_activations)
    padded_hiddens = [F.pad(torch.stack(hidden, dim=0), (0, 0, 0, 0, 0, max_length - torch.stack(hidden, dim=0).shape[0])).transpose(0,1) for hidden in hidden_activations]
    hidden_activations = torch.cat(padded_hiddens, dim=0)
    #print(max_length)
    #print(hidden_activations.shape)
    padded_mask = [F.pad(mask, (0, max_length - mask.shape[1])) for mask in mask_list]

    mask = torch.cat(padded_mask, dim=0)
    #print(mask.shape)
    #print(labels_list[0].shape)

    #print(hidden_activations.shape)
    #print(labels.shape)
    #print(mask.shape)

    return hidden_activations, mask, responses

def process_hidden_states(outputs, input_ids_repeated):
    last_hidden_states = []
    for idx, hidden_state in enumerate(outputs.hidden_states):
        last_hidden_states.append(hidden_state[-1][:, -1, :])
    return last_hidden_states



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='llama_7B')
    parser.add_argument('--dataset_name', type=str, default='tqa_mc2')
    parser.add_argument('--num_samples', type=int, default=1)

    parser.add_argument('--device', type=int, default=1)
    args = parser.parse_args()
    
    MODEL_NAMES = { 
        'vicuna_7B': 'lmsys/vicuna-7b-v1.5', 
        'falcon_7B': 'tiiuae/falcon-7b-instruct',
        'phi-2': "microsoft/phi-2"
    }
    MODEL = MODEL_NAMES[args.model_name]
    ## load the base llm model
    tokenizer = AutoTokenizer.from_pretrained(MODEL, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(MODEL, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16)
    device = args.device
    model = model.to(device)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    ## load the off-the-self reward model
    ## load the value function model, 
    #value_model = llama.LLaMAForSequenceClassification.from_pretrained(MODEL, num_labels=1)
    #value_model = ValueFunction(input_dim=4096, hidden_dim=4096, output_dim=1)

    # for param in value_model.model.parameters():
    #     param.requires_grad = False

    dataset = load_dataset("Anthropic/hh-rlhf")
    dataset = dataset.remove_columns("rejected")
    #dataset['train'] = dataset['train'].select(range(100))
    #dataset['test'] = dataset['test'].select(range(100))
    for split in dataset.keys():
        dataset[split] = dataset[split].rename_column('chosen', 'prompt')

    #dataset_test = load_dataset("Anthropic/hh-rlhf", split="test")
    #def preprocessing(example):
            
        #parts = example['prompt'].rsplit("Assistant:", 1)  # Split the string at the last occurrence of "Assistant:"
        #result = parts[0] + "Assistant:"  # Append "Assistant:" back to the first part if needed    
        #return result
    
    #dataset = dataset.map(preprocessing)
    
    ## first we need to 

    def tokenize(example):
        replaced_text = example['prompt'].replace("Human:", "User:")
        parts = replaced_text.rsplit("Assistant:", 1)  # Split the string at the last occurrence of "Assistant:"
        result = parts[0] + "Assistant:"  # Append "Assistant:" back to the first part if needed   
        tokenized = tokenizer(result, truncation=True)
    
        example["input_ids"] = tokenized["input_ids"]
        example["attention_mask"] = tokenized["attention_mask"]

        return example
    
    dataset = dataset.map(tokenize, batched=False)
    dataset = dataset.filter(lambda x: len(x["input_ids"]) <= 512)
    data_collator = DataCollatorReward(tokenizer=tokenizer)

    train_dataloader = DataLoader(dataset['train'], batch_size=32, collate_fn=data_collator)
    test_dataloader = DataLoader(dataset['test'], batch_size=32, collate_fn=data_collator)
    
    #dataset = dataset.map(tokenize, batched=False)
    #train_dataset = dataset['train']
    #test_dataset = dataset['test']



    #train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    #test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)




    
    token_wise_activations_train, mask_train, response_train = get_llm_activations(args.model_name, model,train_dataloader,tokenizer, device, args.num_samples)
    
    if not os.path.exists('features'):
        os.makedirs('features')
    # save the activations
    with open('features/response_train', 'w') as f:
        json.dump(response_train, f, ensure_ascii=False)
    token_wise_activations_test, mask_test, response_test  = get_llm_activations(args.model_name, model, test_dataloader,tokenizer, device, args.num_samples)
    
    # create the features directory if no

    torch.save(token_wise_activations_train, 'features/token_wise_activations_train.pth')
    torch.save(mask_train, 'features/mask_train.pth')
    #torch.save(labels_train, 'features/labels_train.pth')

    torch.save(token_wise_activations_test, 'features/token_wise_activations_test.pth')
    torch.save(mask_test, 'features/mask_test.pth')
    #torch.save(labels_test, 'features/labels_test.pth')

    with open('features/response_train', 'w') as f:
        json.dump(response_train, f, ensure_ascii=False)

    with open('features/response_test', 'w') as f:
        json.dump(response_test, f, ensure_ascii=False)


if __name__ == "__main__":
    main()