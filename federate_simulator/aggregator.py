from abc import ABC, abstractmethod
from box import Box
from modelfactory import ModelFactory
import torch
from utils import get_best_gpu
from tqdm import tqdm


class FederatedAggregator(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def on_initialization(self, config:Box, **kwargs):
        self.config = config
        model_factory = ModelFactory()
        self.model = model_factory.create_model(config)
        
    @abstractmethod
    def on_global_round_start(self, contexts, **kwargs):
        pass
    @abstractmethod
    def on_global_round_end(self, contexts, **kwargs):
        pass
    @abstractmethod
    def on_local_round_start(self, contexts, **kwargs):
        pass
    @abstractmethod
    def on_local_round_end(self, contexts, **kwargs):
        pass
    

class SingleGPULocalFedAvg(FederatedAggregator):
    def __init__(self):
        super().__init__()
    
    def on_initialization(self, config: Box, **kwargs):
        super().on_initialization(config, **kwargs)
        self.local_rounds = config.local_rounds
        device = config.device
        if device == 'cuda':
            if torch.cuda.is_available():
                self.device = get_best_gpu()
            else:
                raise ValueError('cuda is not available')
        elif device == 'cpu':
            print('cpu is used')
            self.device = torch.device('cpu')
        optimizer_type = config.optimizer.type
        if optimizer_type == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(), 
                                             lr=config.optimizer.lr)
        elif optimizer_type == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(),
                                              lr=config.optimizer.lr)
        self.criterion = torch.nn.CrossEntropyLoss()
        
    def on_global_round_start(self, contexts, selected_clients, **kwargs):
        try:
            self.prev_global_model = contexts.global_round_end.FederatedAggregator.global_model_params
        except KeyError:
            self.prev_global_model = self.model.state_dict().copy()
        self.clients_params = dict.fromkeys(selected_clients, None)
       
        
                 
    
    def local_train(self, local_train_loader):
        self.model.train()
        self.model.to(self.device)
        loss_value = 0
        num_batches = 0
        pbar = tqdm(local_train_loader, dynamic_ncols=True)
        for x, y in pbar:
            x = x.to(self.device)
            y = y.to(self.device)
            # print(x.shape, y.shape)
            self.optimizer.zero_grad()
            output = self.model(x)
            loss = self.criterion(output, y)
            loss_value += loss.item()
            num_batches += 1
            loss.backward()
            self.optimizer.step()
            pbar.set_postfix({'Running Average Lossoss': loss_value/num_batches})
        
     
     # local evaluation on training data
    def local_eval(self, local_test_loader):
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total = 0
        with torch.no_grad():
            for x, y in local_test_loader:
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
        
    
    def on_local_round_start(self, contexts, local_train_loader, client_id,
                             **kwargs):
        print(f"Client {client_id} started local training")
        self.model.load_state_dict(self.prev_global_model)
        for i in range(self.local_rounds):
            print(f"Client {client_id} local round {i+1}/{self.local_rounds}")
            self.local_train(local_train_loader)
        self.clients_params[client_id] = self.model.state_dict().copy()
        return {'local_model_params': self.clients_params[client_id],
                'model': self.model}
        
    
    def on_local_round_end(self, contexts, client_id, **kwargs):
        print(f"Client {client_id} finished local training")
        self.model.load_state_dict(self.clients_params[client_id])
        results = self.local_eval(contexts['local_round_start']['local_train_loader'])
        return results
        
    
    def aggregate(self):
        clients_params_list = list(self.clients_params.values())
        for name, param in self.model.named_parameters():
            param.data = torch.mean(torch.stack(
           [clients_params_list[i][name] for i in range(len(clients_params_list))]), dim=0)
        return self.model.state_dict().copy()
    
    def on_global_round_end(self, contexts, **kwargs):
        global_params = self.aggregate()
        return {'global_model_params': global_params}
    


    
    
class SingleGPULocalFedProx(FederatedAggregator):
    def __init__(self, config):
        super().__init__(config)
        pass
    def on_global_round_start(self):
        pass
    def on_global_round_end(self):
        pass
    def on_local_round_start(self):
        pass
    def on_local_round_end(self):
        pass
    
class SingleGPULocalDitto(FederatedAggregator):
    def __init__(self, config):
        super().__init__(config)
        pass
    def on_global_round_start(self):
        pass
    def on_global_round_end(self):
        pass
    def on_local_round_start(self):
        pass
    def on_local_round_end(self):
        pass
    
    
    
if __name__ == "__main__":
    from federateloader import FederatedDatasetLoader
    from clientselector import RandomClientSelector
    from box import Box
    config = Box({
        
         'local_rounds': 5,
         'device': 'cuda',
         'num_clients': 2,
         'num_selected_clients': 2,
         'global_rounds': 1,
         'model':'MLP',

         "dataset": {
                "name": "mnist",
                "num_clients": 10,
                "split_type": 'lda',
                'require_local_test': True,
                'require_local_val': True,
                'global_local_ratio': 0.5,
                'val': True,
                'val_portion': 0.2,
                "alpha": 0.5,
            },
            
        "loader": {
                "batch_size": 64,
                "test_batch_size": 1000,
            }, 
        
        "optimizer": {
            'lr': 0.01,
            'type': 'sgd'
        }      
    })
    loader = FederatedDatasetLoader()
    client_selector = RandomClientSelector()
    aggregator = SingleGPULocalFedAvg()
    # Initialization
    client_selector.on_initialization(config)
    loader.on_initialization(config)
    aggregator.on_initialization(config)
    contexts = Box({})
    for global_round in range(config.global_rounds):
        client_selector_return = client_selector.on_global_round_start(contexts, global_round)
        selected_clients = client_selector_return['selected_clients']
        _ = loader.on_global_round_start(contexts)
        _ = aggregator.on_global_round_start(contexts, 
                                             selected_clients=selected_clients)
        print(selected_clients)
        for client_id in selected_clients:
            _ = client_selector.on_local_round_start(contexts, client_id=client_id)
            dataloader_return = loader.on_local_round_start(contexts, client_id=client_id)
            local_train_loader = dataloader_return['local_train_loader']
            aggregator_return = aggregator.on_local_round_start(contexts,
                                                                local_train_loader=local_train_loader, 
                                                                client_id=client_id)
            if 'on_local_round_start' not in contexts:
                contexts['local_round_start'] = {}
                contexts['local_round_start']['local_train_loader'] = local_train_loader
            local_eval_return = aggregator.on_local_round_end(contexts, client_id=client_id)
            print(local_eval_return)
        client_selector.on_global_round_end(contexts)
        loader_return = loader.on_global_round_end(contexts)
        print(loader_return)
        local_test_loader = loader_return['global_test_loader']
        local_val_loader = loader_return['global_val_loader']
        aggregator_return = aggregator.on_global_round_end(contexts)
        global_param = aggregator_return['global_model_params']
        aggregator.model.load_state_dict(global_param)
        print(aggregator.local_eval(local_test_loader))
        print(aggregator.local_eval(local_val_loader))
                                                           
        
            
            
            
            
        
        
        
    