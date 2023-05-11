import torch
from torch.utils.data import Dataset, DataLoader, Subset, BatchSampler
from split import LDASplitter, IIDSplitter, ClientDataIdxMap
from abc import ABC, abstractmethod
from box import Box
from .utils import load_dataset



class BaseFederateDataLoader(ABC):
    def __init__(self, config:Box, **kwargs):
        self.config = config
        pass
    @abstractmethod
    def on_local_round_start(self):
        pass
    @abstractmethod
    def on_local_round_end(self):
        pass
    @abstractmethod
    def on_global_round_start(self):
        pass
    @abstractmethod
    def on_global_round_end(self): 
        pass
    


class FederatedDatasetLoader(BaseFederateDataLoader):
    def __init__(self, config:Box, **kwargs):
        self.cfg = config.dataset
        self.dataset = self.cfg.dataset
        trainset, testset, valset = load_dataset(self.dataset, config.dataset)
       
            
    def on_global_round_start(self):
        pass
    
    def on_local_round_start(self, client_id):
        trainset = Subset(self.dataset, self.train_idx_mapping[client_id])
        self.client_loader = DataLoader(trainset, batch_size=self.batch_size, shuffle=True)
        return self.client_loader
    
    def on_local_round_end(self):
        pass

    def split_dataset(self):
        self.client_indices = self.splitter.get_clients_dataidx_map()
        
    def build_dataloader(self, **kwargs):
        self.client_indices = self.split_dataset
        return self.client_loaders
        