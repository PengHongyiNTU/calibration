import torch
import torchmetrics as metrics
from box import Box
import numpy as np
import random
from tqdm import tqdm
from utils import get_best_gpu

def random_selection(num_clients, num_selected_clients):
    assert num_selected_clients <= num_clients, 'num_selected_clients should be less than num_clients'
    return random.sample(range(num_clients), num_selected_clients)

class FedAvg:
    def __init__(self, model, 
                 num_clients, num_clients_per_round, 
                 local_rounds, epochs, 
                 optimizer, criterion, device,
                 metrics):
        self.num_clients = num_clients 
        self.num_clients_per_round = num_clients_per_round
        self.local_rounds = local_rounds
        self.epochs = epochs
        self.model = model
        self.optimizer = optimizer or torch.optim.SGD(model.parameters(), lr=0.1)
        self.criterion = criterion or torch.nn.CrossEntropyLoss()
        self.device = get_best_gpu() if device == 'cuda' else torch.device('cpu')
        self.metric = metrics
        self.prev_global_model_param = None
        
        
    def select_clients(self):
        return random_selection(self.num_clients, self.num_clients_per_round)
    
    def local_train(self, train_loader):
        self.model.train()
        self.model.to(self.device)
        loss_value = 0
        num_batches = 0
        pbar = tqdm(train_loader, dynamic_ncols=True)
        for x, y in pbar:
            x = x.to(self.device)
            y = y.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(x)
            loss = self.criterion(output, y)
            loss_value += loss.item()
            num_batches += 1
            loss.backward()
            self.optimizer.step()
            pbar.set_postfix('Running Average Loss: {:.4f}'.format(loss_value/num_batches))
            
    def aggregate(self):
        clients_params_list = list(self.clients_params.values())
        for name, param in self.model.named_parameters():
            param.data = torch.mean(torch.stack(
           [clients_params_list[i][name] for i in range(len(clients_params_list))]), dim=0)
        return self.model.state_dict().copy()
            
    def local_eval(self, data_loader):
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total = 0
        with torch.no_grad():
            for x, y in data_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                outputs = self.model(x)
                loss = self.criterion(outputs, y)
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                total_correct += (predicted == y).sum().item()

        average_loss = total_loss / total
        accuracy = total_correct / total
        return {'average_loss': average_loss, 'accuracy': accuracy}
    
    def global_eval(self, test_loader):
        results = {metric.__class__.__name__: []for metric in self.metrics}
        self.model.eval()
        with torch.no_grad():
            for x, y in test_loader:
                y_pred = self.model(x)
                for metric in self.metrics:
                    results[metric.__class__.__name__].append(metric(y_pred, y).item())
        for metric in results:
            results[metric] = np.mean(results[metric])
            print(f"{metric}: {results[metric]}")
        return results
            
   
    def on_global_round_start(self, e, selected_clients):
        print("Global round {} started".format(e+1))
        self.selected_clients = selected_clients
        if not self.prev_global_model_param:
            self.prev_global_model_param = self.model.state_dict().copy()
        self.clients_params = dict.fromkeys(selected_clients, None)
        
    def on_local_round_start(self, client_id, train_loader):
        print(f"Client {client_id} started local training")
        self.model.load_state_dict(self.prev_global_model_param)
        for i in range(self.local_rounds):
             print(f"Client {client_id} local round {i+1}/{self.local_rounds}")
             self.local_train(train_loader)
        self.clients_params[client_id] = self.model.state_dict().copy()
        
    def on_local_round_end(self, client_id, train_loader):
        print(f"Client {client_id} finished local training")
        self.model.load_state_dict(self.clients_params[client_id])
        results = self.local_eval(train_loader)
        return results
    
    def on_global_round_end(self, test_loader, val_loader):
        global_params = self.aggregate()
        self.model.load_state_dict(global_params)
        val_results = self.global_eval(val_loader)
        test_results = self.global_eval(test_loader)
        return val_results, test_results
    
        
    
        
    
        
        
    
   
    
    
     
    
        