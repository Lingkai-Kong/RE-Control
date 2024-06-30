# We first need to train a value function 
# The value function is a MLP on top of the llama features
# We need to first load the HH-RLHF dataset from huggingface and initialize the model

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
import intervented_model.llama as llama
from datasets import load_dataset
from torch.utils.data import DataLoader
import re
from transformers import LlamaTokenizer, LlamaForCausalLM, AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
import os
from torch.utils.data import Dataset
import wandb

wandb.init(project="llm-control",
           name="train_value_bellman")

class ValueFunction(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ValueFunction, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train_epoch(value_model, train_dataloader, optimizer, device):
    value_model.train()  # Set the model to training mode
    total_loss = 0
    count_batches = 0
    for batch_input in train_dataloader:
        # move batch input in the dataloader to device 
        optimizer.zero_grad()
        hidden_states, labels, mask = batch_input
        hidden_states = hidden_states.to(device)
        labels = labels.to(device)
        mask = mask.to(device)
        batch_size, length, hidden_dim = hidden_states.shape
        all_predictions = value_model(hidden_states.view(-1, hidden_dim)).view(batch_size, length, -1)
        
        # Creating pairs of consecutive predictions where mask is 1
        valid_mask = (mask[:, :-1] * mask[:, 1:])  # Only consider pairs where both are 1
        valid_pred1 = all_predictions[:, :-1][valid_mask.bool()]
        valid_pred2 = all_predictions[:, 1:][valid_mask.bool()]
        valid_pred3 = all_predictions[:, :-1]
        valid_preds = all_predictions[:, :-1][valid_mask.bool()]
        next_valid_preds = all_predictions[:, 1:][valid_mask.bool()]

        # Calculate MSE for valid consecutive predictions
        pairwise_loss = F.mse_loss(valid_preds, next_valid_preds, reduction='sum')

        # Find the index of the last 1 in each mask row
        last_indices = mask.float().argmax(dim=1, keepdim=True)
        last_indices[mask.sum(dim=1) == 0] = -1  # Handling cases where there are no 1s in mask

        # Gathering the last valid predictions according to the mask
        batch_indices = torch.arange(batch_size).to(last_indices.device)
        last_valid_preds = all_predictions[batch_indices, last_indices.squeeze()]
        final_loss = F.mse_loss(last_valid_preds, labels, reduction='sum')
        
        # Combine losses and normalize
        loss = (pairwise_loss + final_loss) / batch_size
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        count_batches += 1

    average_loss = total_loss / count_batches
    return average_loss   


def test_epoch(value_model, test_dataloader, device):
    value_model.eval()  
    total_loss = 0
    count_batches = 0
    for batch_input in test_dataloader:
        # move batch input in the dataloader to device 
        hidden_states, labels, mask = batch_input
        hidden_states = hidden_states.to(device)
        labels = labels.to(device)
        mask = mask.to(device)
        batch_size, length, hidden_dim = hidden_states.shape
        all_predictions = value_model(hidden_states.view(-1, hidden_dim)).view(batch_size, length, -1)
        
        # Creating pairs of consecutive predictions where mask is 1
        
        valid_mask = (mask[:, :-1] * mask[:, 1:])  # Only consider pairs where both are 1
        valid_preds = all_predictions[:, :-1][valid_mask.bool()]
        next_valid_preds = all_predictions[:, 1:][valid_mask.bool()].detach()

        # Calculate MSE for valid consecutive predictions
        pairwise_loss = F.mse_loss(valid_preds, next_valid_preds, reduction='sum')

        # Find the index of the last 1 in each mask row
        last_indices = mask.float().argmax(dim=1, keepdim=True)
        last_indices[mask.sum(dim=1) == 0] = -1  # Handling cases where there are no 1s in mask

        # Gathering the last valid predictions according to the mask
        batch_indices = torch.arange(batch_size).to(last_indices.device)
        last_valid_preds = all_predictions[batch_indices, last_indices.squeeze()]
        final_loss = F.mse_loss(last_valid_preds, labels, reduction='sum')
        
        # Combine losses and normalize
        loss = (pairwise_loss + final_loss) / batch_size
        total_loss += loss.item()
        count_batches += 1

    average_loss = total_loss / count_batches
    return average_loss  


class TensorDataset(Dataset):
    def __init__(self, data, labels, mask):
        """
        Args:
            data_tensor (Tensor): A tensor containing the data with shape (num_data, hidden_dim).
        """
        self.data = data
        self.labels = labels
        self.mask = mask

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (self.data[idx], self.labels[idx], self.mask[idx])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', type=str, default='llama_7B')
    parser.add_argument('dataset_name', type=str, default='tqa_mc2')
    parser.add_argument('--device', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()

    wandb.config.update(args)
    
    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"

    ## load the value function model, 
    value_model = ValueFunction(input_dim=4096, hidden_dim=4096, output_dim=1)

    hidden_states_train = torch.load('features/token_wise_activations_train.pth')
    labels_train = torch.load('features/labels_train.pth')
    mask_train = torch.load('features/mask_train.pth')

    hidden_states_test = torch.load('features/token_wise_activations_test.pth')
    labels_test = torch.load('features/labels_test.pth')
    mask_test = torch.load('features/mask_test.pth')


    train_dataset = TensorDataset(hidden_states_train, labels_train, mask_train)
    test_dataset = TensorDataset(hidden_states_test, labels_test, mask_test)
    
 
    train_dataloader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=512, shuffle=False)


    ## train the value function model
    optimizer = optim.Adam(value_model.parameters(), lr=args.lr)
    value_model = value_model.to(dtype=torch.bfloat16)
    value_model.to(device)


    for epoch in range(args.epochs):
        train_loss = train_epoch(value_model, train_dataloader, optimizer, device)
        test_loss = test_epoch(value_model, test_dataloader, device)
        print(f"Epoch {epoch+1}, Training Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

        wandb.log({"Training Loss": train_loss, "Test Loss": test_loss, "Epoch": epoch})

        if not os.path.exists('trained_model'):
            os.makedirs('trained_model')
        torch.save(value_model.state_dict(), f'trained_model/value_model_{args.model_name}_{args.lr}.pth')
        if (epoch+1) % 10 == 0:
            torch.save(value_model.state_dict(), f'trained_model/value_model_{args.model_name}_{args.lr}_{epoch+1}.pth')

if __name__ == '__main__':
    main()