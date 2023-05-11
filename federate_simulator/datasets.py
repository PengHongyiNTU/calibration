from box import Box
from typing import Tuple, Union
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split
from split import LDASplitter, IIDSplitter


def load_dataset(dataset: str, data_cfg: Box):
    if dataset in ['mnist', 'cifar10', 'cifar100']:
        trainset, testset, valset = load_centralized_dataset(dataset, data_cfg)
        train_idx_mapping, test_idx_mapping, val_idx_mapping = split_dataset(
            trainset, testset, valset, data_cfg)
    elif dataset in ['some_real_fl_dataset']:
        train_idx_mapping, test_idx_mapping, val_idx_mapping = load_federated_dataset(
            dataset, data_cfg)
    else:
        raise ValueError(f'Unknown dataset {dataset}')
    return train_idx_mapping, test_idx_mapping, val_idx_mapping


def split_dataset(trainset: Dataset,
                  testset: Dataset,
                  valset: Dataset,
                  data_cfg: Box,
                  **kwargs):
    num_clients = data_cfg.num_clients
    split_type = data_cfg.split_type
    supported_split_types = ['iid', 'lda']
    if split_type not in supported_split_types:
        raise ValueError(f'split_type must be one of {supported_split_types}')
    else:
        if split_type == 'lda':
            assert data_cfg.alpha, f'alpha must be provided for LDA split'
            alpha = data_cfg.alpha
            splitter = LDASplitter(num_clients, alpha, **kwargs)
        elif split_type == 'iid':
            splitter = IIDSplitter(num_clients, **kwargs)
        train_mapping = splitter.split(trainset, train=True)
        test_mapping = splitter.split(testset, train=False,
                                          local=data_cfg.require_local_test,
                                          global_local_ratio=data_cfg.global_local_ratio)
        if valset is not None:
            val_mapping = splitter.split(valset, train=False,
                                            local=data_cfg.require_local_val,
                                            global_local_ratio=data_cfg.global_local_ratio)
    return train_mapping, test_mapping, val_mapping


def load_centralized_dataset(dataset: str, data_cfg: Box) -> Tuple[Dataset, Dataset, Union[Dataset, None]]:
    if dataset == 'mnist':
        data_class = torchvision.datasets.MNIST
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    elif dataset == 'cifar10':
        data_class = torchvision.datasets.CIFAR10
        transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, 4),
                                        transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                                    (0.2023, 0.1994, 0.2010))])
    elif dataset == 'cifar100':
        data_class = torchvision.datasets.CIFAR100
        transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, 4),
                                        transforms.ToTensor(), transforms.Normalize((0.5071, 0.4867, 0.4408),
    
    # To be include in the future
    # TINY IMAGENET
    # FEMNIST
    # SVHN
    # SHAKESPEARE

                                                                                    (0.2675, 0.2565, 0.2761))])
    trainset = data_class(root='./data', train=True,
                          download=True, transform=transform)
    testset = data_class(root='./data', train=False,
                         download=True, transform=transform)
    if data_cfg.val:
        if data_cfg.val_portion is None:
            raise ValueError('val_portion must be set when val is True')
        elif data_cfg.val_portion < 0 or data_cfg.val_portion > 1:
            raise ValueError('val_portion must be in [0, 1]')
        else:
            val_size = int(len(trainset) * data_cfg.val_portion)
            trainset, valset = random_split(
                trainset, [len(trainset) - val_size, val_size])
        return trainset, testset, valset
    else:
        return trainset, testset, None


def load_federated_dataset(dataset: str, data_cfg: Box):
    pass


if __name__ == "__main__":

    data_cfg = Box({
        'num_clients': 10,
        'split_type': 'lda',
        'require_local_test': True,
        'require_local_val': True,
        'global_local_ratio': 0.5,
        'val': True,
        'val_portion': 0.1,
        'alpha': 0.5
    }, dafault_box=True)

    train_idx_mapping, test_idx_mapping, val_idx_mapping = load_dataset(
        'mnist', data_cfg)

    # print(train_idx_mapping)
    print("Train index mapping:")
    for client_id, idxs in train_idx_mapping['clients_idx'].items():
        print(f"Client {client_id}: {len(idxs)} samples")

    print("\nTest index mapping:")
    print(f'Global : {len(test_idx_mapping["global_idx"])}')
    for client_id, idxs in test_idx_mapping['clients_idx'].items():
        print(f"Client {client_id}: {len(idxs)} samples")

    print("\nValidation index mapping:")
    print(f'Global : {len(val_idx_mapping["global_idx"])}')
    for client_id, idxs in val_idx_mapping['clients_idx'].items():
        print(f"Client {client_id}: {len(idxs)} samples")
