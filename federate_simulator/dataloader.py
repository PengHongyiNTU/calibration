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
        # if the dataset return is centralized, we need to split it
        if (type(trainset) == torch.utils.data.Dataset 
            and type(testset) == torch.utils.data.Dataset 
            and type(valset) == torch.utils.data.Dataset):
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
            self.train_idx_mapping = self.splitter(trainset).get_clients_dataidx_map()
            self.test_idx_mapping = ClientDataIdxMap(self.num_clients)
            self.text_idx_mapping.server = list(range(len(testset)))
            self.val_idx_mapping = ClientDataIdxMap(self.num_clients)
            self.val_idx_mapping.server = list(range(len(valset)))
            # In case, the setting require local test and local val dataset
            if self.cfg.require_local_test:
                self.test_idx_mapping = self.splitter(testset).get_clients_dataidx_map()
            if self.cfg.require_local_val:
                self.val_idx_mapping = self.splitter(valset).get_clients_dataidx_map()
        # Real dataset is already splitted into clients idx maps
        elif (type(trainset)== ClientDataIdxMap 
              and type(testset) == ClientDataIdxMap
              and type(valset) == ClientDataIdxMap):
            self.train_idx_mapping = trainset
            self.test_idx_mapping = testset
            self.val_idx_mapping = valset
            
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
        