import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
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

        hidden_states = process_hidden_states(outputs) # shape: batch_size x length x hidden_dim

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
        elif model_name == 'llama3_8B':
            length_of_prompts_padding = (input_ids == 128001)  ## llama3's padding id is 2128001, the same as the eos token id; falcon is 11; phi-2 is 50256
            length_of_prompts_padding = length_of_prompts_padding.sum(dim=1)
       
            padding_len_answer = (outputs.sequences == 128001) ## each sequence in the batch can have different lengths, it is padded with 0, so we need to mask the loss
            padding_length = padding_len_answer.sum(dim=1)
            padding_length = padding_length - length_of_prompts_padding

        if model_name == 'vicuna_7B':
            begin_word = 'Human: '
        elif model_name == 'llama3_8B':
            begin_word = 'User: '
        elif model_name == 'falcon_7B':
            begin_word = 'User: '
        
        range_tensor = torch.arange(len(hidden_states)).expand(hidden_states[0].shape[0], -1)

        thresholds = (len(hidden_states) - padding_length).unsqueeze(1)
        thresholds = thresholds.to(device)
        range_tensor = range_tensor.to(device)
        mask = range_tensor < thresholds


        cut = []
        for d in range (outputs.sequences.shape[0]):
            outid = outputs.sequences[d]
            pattern = torch.tensor(tokenizer([begin_word])['input_ids'][0]).to(device)
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

def process_hidden_states(outputs):
    last_hidden_states = []
    for idx, hidden_state in enumerate(outputs.hidden_states):
        last_hidden_states.append(hidden_state[-1][:, -1, :])
    return last_hidden_states



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='vicuna_7B', choices=["vicuna_7B", "falcon_7B", "llama3_8B"])
    parser.add_argument('--dataset_name', type=str, default='hh_rlhf', choices=["hh_rlhf", "shp"])
    parser.add_argument('--num_samples', type=int, default=1)

    parser.add_argument('--device', type=int, default=1)
    args = parser.parse_args()
    
    MODEL_NAMES = { 
        'vicuna_7B': 'lmsys/vicuna-7b-v1.5', 
        'falcon_7B': 'tiiuae/falcon-7b-instruct',
        'llama3_8B': 'meta-llama/Meta-Llama-3-8B'
    }

    DATASET_NAMES = { 
        'hh_rlhf': 'Anthropic/hh-rlhf', 
        'shp': 'stanfordnlp/SHP'
    }
    MODEL = MODEL_NAMES[args.model_name]
    DATASET = DATASET_NAMES[args.dataset_name]
    ## load the base llm model
    tokenizer = AutoTokenizer.from_pretrained(MODEL, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(MODEL, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16)
    device = args.device
    model = model.to(device)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id


    dataset = load_dataset(DATASET)

    if args.dataset_name == 'hh_rlhf':
        dataset = dataset.remove_columns("rejected")
        for split in dataset.keys():
            dataset[split] = dataset[split].rename_column('chosen', 'prompt')
    

    if args.model_name == 'vicuna_7B':
        begin_word = 'Human: '
    elif args.model_name == 'llama3_8B':
        begin_word = 'User: '
    elif args.model_name == 'falcon_7B':
        begin_word = 'User: '

    def tokenize(example):
        if args.dataset_name == 'hh_rlhf':
            replaced_text = example['prompt'].replace("Human:", begin_word)
            parts = replaced_text.rsplit("Assistant:", 1)  # Split the string at the last occurrence of "Assistant:"
            result = parts[0] + "Assistant:"  # Append "Assistant:" back to the first part if needed
        elif args.dataset_name == 'shp':
            text = example['history']
            result = begin_word + text + "\nAssistant:"    
        tokenized = tokenizer(result, truncation=True)
    
        example["input_ids"] = tokenized["input_ids"]
        example["attention_mask"] = tokenized["attention_mask"]

        return example
 
    dataset = dataset.map(tokenize, batched=False)
    dataset = dataset.filter(lambda x: len(x["input_ids"]) <= 512)
    data_collator = DataCollatorReward(tokenizer=tokenizer)

    train_dataloader = DataLoader(dataset['train'], batch_size=32, collate_fn=data_collator)
    test_dataloader = DataLoader(dataset['test'], batch_size=32, collate_fn=data_collator)
     
    token_wise_activations_train, mask_train, response_train = get_llm_activations(args.model_name, model,train_dataloader,tokenizer, device, args.num_samples)
    token_wise_activations_test, mask_test, response_test  = get_llm_activations(args.model_name, model, test_dataloader,tokenizer, device, args.num_samples)    
    
    if not os.path.exists('features'):
        os.makedirs('features')
    
    # create the features directory if no

    torch.save(token_wise_activations_train, 'features/'+ str(args.model_name) + '_' + str(args.dataset_name) + '_' +'token_wise_activations_train.pth')
    torch.save(mask_train, 'features/'+ str(args.model_name) + '_' + str(args.dataset_name) + '_' +'mask_train.pth')
    #torch.save(labels_train, 'features/labels_train.pth')

    torch.save(token_wise_activations_test, 'features/'+ str(args.model_name) + '_' + str(args.dataset_name) + '_' +'token_wise_activations_test.pth')
    torch.save(mask_test, 'features/'+ str(args.model_name) + '_' + str(args.dataset_name) + '_' +'mask_test.pth')
    #torch.save(labels_test, 'features/labels_test.pth')

    with open('features/'+ str(args.model_name) + '_' + str(args.dataset_name) + '_' +'response_train.json', 'w') as f:
        json.dump(response_train, f, ensure_ascii=False)

    with open('features/'+ str(args.model_name) + '_' + str(args.dataset_name) + '_' +'response_test.json', 'w') as f:
        json.dump(response_test, f, ensure_ascii=False)


if __name__ == "__main__":
    main()