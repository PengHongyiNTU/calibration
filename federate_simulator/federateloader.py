
from torch.utils.data import Dataset, DataLoader, Subset, BatchSampler
from datasets import load_dataset, split_dataset
from abc import ABC, abstractmethod
from box import Box




class FederatedDataLoader(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def on_initilization(self, config: Box, **kwargs):
        self.config = config
    
        
    
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
    


class FederatedDatasetLoader(FederatedDataLoader):
    def __init__(self):
        super().__init__()

    def on_initilization(self, config: Box, **kwargs):
        self.data_cfg = config.dataset
        self.loader_cfg = config.loader
        self.trainset, self.valset, self.testset = load_dataset(self.data_cfg.name, 
                                                                config.dataset
                                                                **kwargs)
        train_idx_mapping, test_idx_mapping, val_idx_mapping = split_dataset(self.trainset, 
                                                                             self.valset, 
                                                                             self.testset, 
                                                                             data_cfg=self.data_cfg,
                                                                             **kwargs)
        self.train_idx_mapping = train_idx_mapping
        self.test_idx_mapping = test_idx_mapping
        self.val_idx_mapping = val_idx_mapping
        
    
    def on_global_round_start(self):
        pass

    def on_local_round_start(self, client_id):
        local_trainset = Subset(self.trainset, 
                                self.train_idx_mapping['clients_idx'][client_id])
        local_train_loader = DataLoader(local_trainset,
                                        batch_size=self.loader_cfg.batch_size,
                                        shuffle=True)
        return {'local_train_loader':local_train_loader}

    def on_local_round_end(self, client_id):
        local_test_loader = self.get_client_loader(self.testset, self.test_idx_mapping, client_id)
        local_val_loader = self.get_client_loader(self.valset, self.val_idx_mapping, client_id)
        return {'local_test_loader':local_test_loader or None, 
                'local_val_loader':local_val_loader or None}

    def get_client_loader(self, dataset, index_mapping, client_id):
        if client_id in index_mapping["clients_idx"]:
            local_subset = Subset(dataset, index_mapping["clients_idx"][client_id])
            local_loader = DataLoader(local_subset,
                                      batch_size=self.loader_cfg.test_batch_size,
                                      shuffle=True)
            return local_loader
        return None

    def on_global_round_end(self):
        global_test_loader = self.get_global_loader(self.testset, self.test_idx_mapping)
        global_val_loader = self.get_global_loader(self.valset, self.val_idx_mapping)
        return {'global_test_loader':global_test_loader or None,
                'global_val_loader':global_val_loader or None}

    def get_global_loader(self, dataset, index_mapping):
        if index_mapping["global_idx"]:
            global_subset = Subset(dataset, index_mapping["global_idx"])
            global_loader = DataLoader(global_subset,
                                       batch_size=self.loader_cfg.test_batch_size,
                                       shuffle=True)
            return global_loader
        return None
            
    
    
if __name__ == "__main__":
    cfg = Box(
        {
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
            }        
        }
    )
    fedloader = FederatedDatasetLoader(cfg)
    for client_id in range(cfg.dataset.num_clients):
        print(f"\nClient {client_id}:")
        local_train_loader = fedloader.on_local_round_start(client_id)
        local_test_loader, local_val_loader = fedloader.on_local_round_end(
            client_id)
        print(f"Local Train Loader: {len(local_train_loader.dataset)} samples")
        if local_test_loader is not None:
            print(
                f"Local Test Loader: {len(local_test_loader.dataset)} samples")
        if local_val_loader is not None:
            print(
                f"Local Validation Loader: {len(local_val_loader.dataset)} samples")

    print("\nGlobal Loaders:")
    global_test_loader, global_val_loader = fedloader.on_global_round_end()
    if global_test_loader is not None:
        print(f"Global Test Loader: {len(global_test_loader.dataset)} samples")
    if global_val_loader is not None:
        print(
            f"Global Validation Loader: {len(global_val_loader.dataset)} samples")
