from box import Box 
from typing import Tuple, Union
from torch.utils.data import Dataset
from .split import LDASplitter, IIDSplitter
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split


def load_dataset(dataset:str, data_cfg:Box):
    if dataset in ['mnist', 'cifar10', 'cifar100']:
        trainset, testset, valset = load_centralized_dataset(dataset)
        train_idx_mapping, test_idx_mapping, val_idx_mapping = split_dataset(trainset, testset, valset, data_cfg)
    elif dataset in ['some_real_fl_dataset']:
        train_idx_mapping, test_idx_mapping, val_idx_mapping = load_federated_dataset(dataset, data_cfg)
    else:
        raise ValueError(f'Unknown dataset {dataset}')
    return train_idx_mapping, test_idx_mapping, val_idx_mapping
        
def split_dataset(trainset:Dataset, 
                  testset:Dataset,
                  valset:Dataset,
                  data_cfg:Box, 
                  **kwargs):
    num_clients = data_cfg.num_clients
    split_type = data_cfg.split_type
    supported_split_types = ['iid', 'lda']
    if split_type in supported_split_types:
        if split_type == 'lda':
            splitter = LDASplitter(num_clients, **kwargs)
        elif split_type == 'iid':
            splitter = IIDSplitter(num_clients, **kwargs)
        train_idx_mapping = splitter(trainset).get_clients_dataidx_map()
        test_idx_mapping = splitter(testset).get_clients_dataidx_map()
        val_idx_mapping = splitter(valset).get_clients_dataidx_map()
        return train_idx_mapping, test_idx_mapping, val_idx_mapping
    else:
        raise ValueError(f'split_type must be one of {supported_split_types}')
    

def load_centralized_dataset(dataset:str, data_cfg:Box) -> Tuple[Dataset, Dataset, Union[Dataset, None]]:
    if dataset == 'mnist':
        data_class = torchvision.datasets.MNIST
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    elif dataset == 'cifar10':
        data_class = torchvision.datasets.CIFAR10
        transform = transforms.Compose([transforms.RandomHorizaontalFlip(), transforms.RandomCrop(32, 4),
                                        transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                                  (0.247, 0.243, 0.261))])
    elif dataset == 'cifar100':
        data_class = torchvision.datasets.CIFAR100
        transform = transforms.Compose([transforms.RandomHorizaontalFlip(), transforms.RandomCrop(32, 4),
                                        transform.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                                  (0.247, 0.243, 0.261))])
    trainset = data_class(root='./data', train=True, download=True, transform=transform)
    testset = data_class(root='./data', train=False, download=True, transform=transform)
    if data_cfg.val:
        if data_cfg.val_portion is None:
            raise ValueError('val_portion must be set when val is True')
        elif data_cfg.val_portion < 0 or data_cfg.val_portion > 1:
            raise ValueError('val_portion must be in [0, 1]')
        else:
            val_size = int(len(trainset) * data_cfg.val_portion)
            trainset, valset = random_split(trainset, [len(trainset) - val_size, val_size])
        return trainset, testset, valset
    else:
        return trainset, testset, None
        
    
   
    

def load_federated_dataset(dataset:str, data_cfg:Box):
    pass




if __name__ == "__main__":
    pass
    
        
    

