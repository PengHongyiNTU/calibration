import torch
import torchmetrics as metrics
from box import Box
import numpy as np
import random
from tqdm import tqdm
from utils import get_best_gpu
from torch.utils.data import DataLoader, Subset, BatchSampler
from torchmetrics.classification import MulticlassAccuracy, MulticlassCalibrationError

def random_selection(num_clients, num_selected_clients):
    assert num_selected_clients <= num_clients, 'num_selected_clients should be less than num_clients'
    return random.sample(range(num_clients), num_selected_clients)

class FedAvg:
    def __init__(self, model, 
                 num_clients, num_clients_per_round, 
                 local_rounds, epochs, 
                 optimizer, lr,  
                 device,
                 trainset, testset, valset,
                 local_test, local_val, 
                 train_data_map, test_data_map, valid_data_map,
                 train_batch_size, eval_batch_size
                 ):
        self.num_clients = num_clients 
        self.num_clients_per_round = num_clients_per_round
        self.local_rounds = local_rounds
        self.epochs = epochs
        self.model = model
        if optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(self.model.parameters(),
                                             lr=lr, momentum=0.9)
        elif optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), 
                                              lr=0.01)
        else:
            raise ValueError(f'optimizer {optimizer} not supported')
        self.criterion = torch.nn.CrossEntropyLoss()
        self.device = get_best_gpu() if device == 'cuda' else torch.device('cpu')
        self.prev_global_model_param = None
        num_classes = len(trainset.classes)
        self.metrics = [
    MulticlassAccuracy(num_classes).to(self.device), 
    MulticlassCalibrationError(num_classes).to(self.device)]
        self.trainset = trainset
        self.testset = testset
        self.valset = valset
        self.local_test = local_test
        self.local_val = local_val if valset is not None else False
        self.train_data_map = train_data_map
        self.test_data_map = test_data_map
        self.val_data_map = valid_data_map
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        
        
    def __repr__(self):
        # print all attributes
        return f'{self.__class__.__name__}({self.__dict__})'
        
        
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
            pbar.set_postfix({"Running Average Loss": '{:.4f}'.format(loss_value/num_batches)})

            
    def aggregate(self):
        weights = torch.tensor([len(self.train_data_map['clients_idx'][client_id]) for client_id in self.selected_clients], dtype=torch.float32).to(self.device)
        total_weights = torch.sum(weights)
        weights = weights / total_weights
        clients_params_list = list(self.clients_params.values())
        for name, param in self.model.named_parameters():
            param.data = torch.sum(torch.stack(
            [weights[i] * clients_params_list[i][name] for i in range(len(clients_params_list))]), dim=0)
        return self.model.state_dict().copy()
            
    def after_train_eval(self, data_loader):
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
    
    def eval(self, test_loader):
        results = {metric.__class__.__name__: []for metric in self.metrics}
        self.model.eval()
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                y_pred = self.model(x)
                for metric in self.metrics:
                    results[metric.__class__.__name__].append(metric(y_pred, y).item())
        for metric in results:
            results[metric] = np.mean(results[metric])
            print(f"{metric}: {results[metric]}")
        return results
            
    
    def run(self):
        for e in range(self.epochs):
            print("Global round {} started".format(e+1))
            self.selected_clients = self.select_clients()
            if e == 0:
                self.prev_global_model_param = self.model.state_dict().copy()
            self.clients_params = dict.fromkeys(self.selected_clients, None)
            for client_id in self.selected_clients:
                print(f"Client {client_id} started local training")
                self.model.load_state_dict(self.prev_global_model_param)
                local_train_idx = self.train_data_map['clients_idx'][client_id]
                local_train_set = Subset(
                    self.trainset, local_train_idx)
                local_train_loader = DataLoader(
                    local_train_set, batch_size=self.train_batch_size, 
                    shuffle=True)
                for i in range(self.local_rounds):
                    print(f"Client {client_id} local round {i+1}/{self.local_rounds}")
                    self.local_train(local_train_loader)
                print(f"Client {client_id} finished local training")
                self.clients_params[client_id] = self.model.state_dict().copy()
                self.model.load_state_dict(self.clients_params[client_id])
                local_train_loader = DataLoader(
                    local_train_set, batch_size=self.eval_batch_size, 
                    shuffle=True
                )
                results = self.after_train_eval(local_train_loader)
                print(f'Local Training {results}')
                if self.local_test:
                    if client_id in self.test_data_map['clients_idx']:
                        local_test_set = Subset(
                            self.testset, 
                            self.test_data_map['clients_idx'][client_id])
                        local_test_loader = DataLoader(
                            local_test_set, batch_size=self.eval_batch_size,
                            shuffle=True
                        )
                        results = self.eval(local_test_loader)
                        print(f'Local Test: {results}')
                    else:
                        print('no local test data')
                if self.local_val and self.valset:
                    if client_id in self.test_data_map['clients_idx']:
                        local_val_set = Subset(
                            self.valset, 
                            self.val_data_map['clients_idx'][client_id])
                        local_val_loader = DataLoader(
                            local_val_set, batch_size=self.eval_batch_size,
                            shuffle=True
                        )
                        results = self.eval(local_val_loader)
                        print(f'Local Validation: {results}')
                    else:
                        print('no local validation data')
            print('All clients finished local training')
            print('Aggregating global model')
            global_params = self.aggregate()
            self.model.load_state_dict(global_params)
            if self.test_data_map['global_idx']:
                global_test_subset = Subset(
                    self.trainset, self.test_data_map['global_idx'])
                global_test_loader = DataLoader(
                    global_test_subset, batch_size=self.eval_batch_size,
                    shuffle=True
                )
                results = self.eval(global_test_loader)
                print(f'Global Test: {results}')
            else:
                print('no global test data')
            if self.valset:
                if self.val_data_map['global_idx']:
                    global_val_subset = Subset(
                        self.valset, self.val_data_map['global_idx'])
                    global_val_loader = DataLoader(
                        global_val_subset, batch_size=self.eval_batch_size,
                        shuffle=True
                    )
                    results = self.eval(global_val_loader)
                    print(f'Global Validation: {results}')
                
                
            
                
    
    
        
    
        
        
    
   
    
    
     
    
        