from box import Box
from typing import Tuple, Union
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split
from split import LDASplitter, IIDSplitter


def load_dataset(data_cfg: Box):
    dataset = data_cfg.name
    if dataset in ['mnist', 'cifar10', 'cifar100', 'tiny-imagenet-200']:
        trainset, testset, valset = load_centralized_dataset(dataset, data_cfg)
    elif dataset in ['some_real_fl_dataset']:
        trainset, testset, valset = load_federated_dataset(dataset, data_cfg)
    else:
        raise ValueError(f'Unknown dataset {dataset}')
    return trainset, testset, valset


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
        else:
            val_mapping = None
    return train_mapping, test_mapping, val_mapping


def load_centralized_dataset(dataset: str, data_cfg: Box) -> Tuple[Dataset, Dataset, Union[Dataset, None]]:
    UNIFORM_SIZE = (32, 32)
    if dataset == 'mnist':
        data_class = torchvision.datasets.MNIST
        transform = transforms.Compose(
            [transforms.Resize(UNIFORM_SIZE),
             transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        trainset = data_class(root='./data', train=True,
                              download=True, transform=transform)
        testset = data_class(root='./data', train=False,
                             download=True, transform=transform)
    elif dataset == 'cifar10':
        data_class = torchvision.datasets.CIFAR10
        transform = transforms.Compose([
            transforms.Resize(UNIFORM_SIZE),
            transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, 4),
                                        transforms.ToTensor(), 
                                        transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                            (0.2023, 0.1994, 0.2010))])
        trainset = data_class(root='./data', train=True,
                          download=True, transform=transform)
        testset = data_class(root='./data', train=False,
                         download=True, transform=transform)
    elif dataset == 'cifar100':
        data_class = torchvision.datasets.CIFAR100
        transform = transforms.Compose([
            transforms.Resize(UNIFORM_SIZE),
            transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, 4),
                                        transforms.ToTensor(), 
                                        transforms.Normalize((0.5071, 0.4867, 0.4408),
                                                            (0.2675, 0.2565, 0.2761))])
        trainset = data_class(root='./data', train=True,
                          download=True, transform=transform)
        testset = data_class(root='./data', train=False,
                         download=True, transform=transform)
    elif dataset == 'tiny-imagenet-200':
        train_dir = './data/tiny-imagenet-200/train/'
        test_dir = './data/tiny-imagenet-200/val/'
        # val_dir = './data/tiny-imagenet-200/val/'
        transform = transforms.Compose([
            transforms.Resize(UNIFORM_SIZE),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225])  # Normalize using ImageNet norms
        ])
        trainset = torchvision.datasets.ImageFolder(
            train_dir, transform=transform)
        testset = torchvision.datasets.ImageFolder(
            test_dir, transform=transform)
        # valset = torchvision.datasets.ImageFolder(val_dir, transform=transform)
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
        'val': False,
        'val_portion': 0.1,
        'alpha': 0.5
    }, dafault_box=True)

    train_set, test_set, val_set = load_dataset(
        'tiny-imagenet', data_cfg)
    train_idx_mapping, test_idx_mapping, val_idx_mapping = split_dataset(
        trainset=train_set, testset=test_set, valset=val_set, data_cfg=data_cfg
    )

    # print(train_idx_mapping)
    print("Train index mapping:")
    for client_id, idxs in train_idx_mapping['clients_idx'].items():
        print(f"Client {client_id}: {len(idxs)} samples")

    print("\nTest index mapping:")
    print(f'Global : {len(test_idx_mapping["global_idx"])}')
    for client_id, idxs in test_idx_mapping['clients_idx'].items():
        print(f"Client {client_id}: {len(idxs)} samples")

    if val_idx_mapping is not None:
        print("\nValidation index mapping:")
        print(f'Global : {len(val_idx_mapping["global_idx"])}')
        for client_id, idxs in val_idx_mapping['clients_idx'].items():
            print(f"Client {client_id}: {len(idxs)} samples")
