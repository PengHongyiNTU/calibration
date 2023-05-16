import torch
import torchmetrics as metrics
from box import Box
import numpy as np
import random
from tqdm import tqdm


def fed_avg(global_model, 
            clients_params_dict):
    clients_params_list = list(clients_params_dict.values())
    for name, param in global_model.named_parameters():
       param.data = torch.mean(torch.stack(
           [clients_params_list[i][name] for i in range(len(clients_params_list))]), dim=0)
    return global_model.state_dict()


def select_clients(num_clients, num_clients_per_round):
    if num_clients_per_round > num_clients:
        raise ValueError(
            'num_selected_clients should be less than num_clients')
    else:
        return random.sample(range(num_clients), num_clients_per_round)
        

def run(model, 
        num_clients, num_clients_per_round,
        local_round, global_round, 
        train_idx_mapping, test_idx_mapping, val_idx_mapping,
        algorithm, optimizer, device):
    
    # Initialization
    loss = torch.nn.CrossEntropyLoss()
    optimizer = optimizer(model.parameters())
    clients_params_dict = dict.fromkeys(range(num_clients), model.state_dict())
    # Prepare dataset 
    
    for e in range(global_round):
        selected_clients = select_clients(num_clients, num_clients_per_round)
        for client in select_clients:
            model.load_state_dict(clients_params_dict[client])
            loader = torch.utils.data.DataLoader(
                
            )
            
            
            
        
        
    
   
    
    
     
    
        