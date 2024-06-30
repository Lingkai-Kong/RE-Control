from intervented_model.llama import Intervented_LlamaForCausalLM
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
import intervented_model.llama as llama
import intervented_model.falcon as falcon
from datasets import load_dataset
from torch.utils.data import DataLoader, Subset
import re
from transformers import LlamaTokenizer, LlamaForCausalLM, AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
import os
import json
from tqdm import tqdm

class ValueFunction(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ValueFunction, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DataCollatorReward:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, data):
        batch = {}
        data_batch = []
        for sample in data:
            data_batch.append({"input_ids": sample['input_ids'], "attention_mask": sample["attention_mask"]})
        batch_data = self.tokenizer.pad(data_batch, padding=True, return_tensors="pt")
        batch['input_ids'] = batch_data['input_ids']
        batch['attention_mask'] = batch_data['attention_mask']
        return batch   

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', type=str, default='vicuna_7B')
    parser.add_argument('--device', type=int, default=2)
    parser.add_argument('--use_intervention', default=False)
    parser.add_argument('--value_lr', default=0.0001)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--bellman', default=False)
    args = parser.parse_args()

    if args.use_intervention:

        value_model = ValueFunction(input_dim=4096, hidden_dim=4096, output_dim=1)
        ##load weights
        value_model.load_state_dict(torch.load(f'./trained_model/value_model_vicuna_{args.value_lr}.pth'))
 

    MODEL_NAMES = {
        'vicuna_7B': 'lmsys/vicuna-7b-v1.5',
        'falcon_7B': 'tiiuae/falcon-7b-instruct',
    }
    MODEL = MODEL_NAMES[args.model_name]
    tokenizer = LlamaTokenizer.from_pretrained(MODEL, padding_side='left')

    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"

    if args.use_intervention:
        model = Intervented_LlamaForCausalLM.from_pretrained(MODEL, low_cpu_mem_usage=True, torch_dtype=torch.float16, device_map=device, cache_dir='./huggingface_cache')
        model.set_value_model(value_model)
        model.set_lr_and_epochs(args.lr, args.epochs)
    else:
        model = LlamaForCausalLM.from_pretrained(MODEL, low_cpu_mem_usage=True, torch_dtype=torch.float16, device_map=device, cache_dir='./huggingface_cache')
    
    model.generation_config.temperature = None
    model.generation_config.top_k = None
    model.to(device)

    dataset = load_dataset("Anthropic/hh-rlhf", cache_dir='./huggingface_cache')
    dataset = dataset.remove_columns("rejected")
    for split in dataset.keys():
        dataset[split] = dataset[split].rename_column('chosen', 'prompt')

    def preprocessing(example):
            
        replaced_text = example['prompt'].replace("Human:", "USER:")
        replaced_text = replaced_text.replace("Assistant:", "ASSISTANT:")

        parts = replaced_text.rsplit("ASSISTANT:", 1)  # Split the string at the last occurrence of "Assistant:"
        result = parts[0] + "ASSISTANT:"  # Append "Assistant:" back to the first part if needed   
        # result = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user’s questions." + parts[0] + "Assistant:"  # Append "Assistant:" back to the first part if needed 
       
        return {'prompt': result} # return the modified example dictionary
    
    
    dataset = dataset.map(preprocessing)
    dataloader = DataLoader(dataset['test'], batch_size=15) # only use the test set for now
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id    

    generated_responses = []
    for batch_prompts in tqdm(dataloader):
        encoded_inputs = tokenizer(batch_prompts['prompt'], return_tensors="pt", padding=True) # tokenize the prompt
        input_ids = encoded_inputs['input_ids'].to(device)
        attention_mask = encoded_inputs['attention_mask'].to(device)
        outputs = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=128, num_return_sequences=1, return_dict_in_generate=True)
        outputs_text = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
        for prompt, output in zip(batch_prompts['prompt'], outputs_text):
            generated_responses.append({'prompt': prompt, 'result': output.removeprefix(prompt), 'response': output})
    
    if not os.path.exists('response'):
        os.makedirs('response')
    if args.use_intervention:
            file_name = os.path.join('response', f"{args.model_name}_{args.value_lr}_{args.epochs}_{args.lr}.json")
    else:
        file_name = os.path.join('response', f"{args.model_name}_base.json")

    with open(file_name, 'w') as f:
        json.dump(generated_responses, f, ensure_ascii=False)

if __name__ == '__main__':
    main()







