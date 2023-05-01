import torch
from torch.utils.data import Dataset, DataLoader, Subset, BatchSampler
from .split import LDASplitter, IIDSplitter
from abc import ABC, abstractmethod
from box import Box
from .utils import load_dataset

class BaseFederateDataWrapper(ABC):
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
    


class FederatedDatasetWrapper(BaseFederateDataWrapper):
    def __init__(self, config:Box, **kwargs):
        self.cfg = config.dataset
        self.dataset = self.cfg.dataset
        self.trainset, self.testset, self.valset = load_dataset(self.dataset)
        if (type(self.trainset) == torch.utils.data.Dataset 
            and type(self.testset) == torch.utils.data.Datase 
            and type(self.valset) == torch.utils.data.Dataset):
            self.num_clients = self.cfg.num_clients
            self.batch_size = self.cfg.batch_size
            self.split_type = self.cfg.split_type
            supported_split_types = ['iid', 'lda']
            if self.split_type not in supported_split_types:
                raise ValueError(f'split_type must be one of {supported_split_types}')
            elif self.split_type == 'lda':
                self.splitter = LDASplitter(self.num_clients, **kwargs)
            elif self.split_type == 'iid':
                self.splitter = IIDSplitter(self.num_clients, **kwargs)
            self.train_idx_mapping = self.splitter(self.trainset).get_clients_dataidx_map()
            # In case, the setting require local test and local val dataset
            if self.cfg.require_local_test:
                self.test_idx_mapping = self.splitter(self.testset).get_clients_dataidx_map()
            if self.cfg.require_local_val:
                self.val_idx_mapping = self.splitter(self.valset).get_clients_dataidx_map()
            
     
            
            
        

    def split_dataset(self):
        self.client_indices = self.splitter.get_clients_dataidx_map()
        
    def build_dataloader(self, **kwargs):
        self.client_indices = self.split_dataset
        return self.client_loaders
        