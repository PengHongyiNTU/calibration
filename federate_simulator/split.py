import numpy as np
from abc import ABC, abstractmethod

class BaseSplitter(ABC):
    def __init__(self, num_clients):
        self.num_clients = num_clients
        pass
    @abstractmethod
    def __call__(self, dataset):
        pass
    @abstractmethod
    def get_clients_dataidx_map(self):
        pass
    


def dirichlet_distribution_non_iid_slice(label, num_clients, alpha, min_size=10):
    """
    Arguments:
        label (np.array): Label list to be split.
        num_clients (int): Split label into num_clients parts.
        alpha (float): alpha of LDA.
        min_size (int): min number of sample in each client
    Returns:
        idx_slice (List): List of splited label index slice.
    """
    if len(label.shape) != 1:
        raise ValueError("label must be a 1-D array")
    num_samples = len(label)
    num_classes = len(np.unique(label))
    assert num_samples >= num_clients * min_size, f"num_samples must be larger than {num_clients * min_size}"
    size = 0
    while size < min_size:
        idx_slice = [[] for _ in range(num_clients)]
        for k in range(num_classes):
            idx_k = np.where(label == k)[0]
            np.random.shuffle(idx_k)
            p = np.random.dirichlet(np.repeat(alpha, num_clients))
            p = np.array(
                [
                    p * (len(idx_j) < num_samples / num_clients)
                    for p, idx_j in zip(p, idx_slice)
                ]
            )
            p = p / p.sum(axis=0)
            p = (np.cumsum(p, axis=0)*len(idx_k)).astype(int)[:-1]
            idx_slice = [
                idx_j + idx.tolist()
                for idx_j, idx in zip(idx_slice, np.array_split(idx_k, p))
            ]
            size = min(len(idx_j) for idx_j in idx_slice)
    for i in range(num_clients):
        np.random.shuffle(idx_slice[i])
    return idx_slice


class LDASplitter(BaseSplitter):
    def __init__(self, num_clients, alpha=0.5, min_size=10):
        self.num_clients = num_clients
        self.alpha = alpha
        self.min_size = min_size
        self.clients_dataidx_map = dict.fromkeys(
            list(range(num_clients))
        )
    
    def __call__(self, dataset):
        label = np.array([y for x, y in dataset])
        idx_slice = dirichlet_distribution_non_iid_slice(label, 
                                                         self.num_clients,
                                                         self.alpha,
                                                         self.min_size
                                                         )
        for i, idx in enumerate(idx_slice):
            self.clients_dataidx_map[i] = idx
        print('Splitting dataset into {} clients.'.format(self.num_clients))
        print({id: len(idxs) for id, idxs in self.clients_dataidx_map.items()})
        return self
    
    def get_clients_dataidx_map(self):
        return self.clients_dataidx_map
    
    def __repr__(self):
        return f'{self.__class__.__name__}(num_clients={self.num_clients}, alpha={self.alpha})'
    


class IIDSplitter(BaseSplitter):
    def __init__(self, num_clients):
        self.num_clients = num_clients
        self.clients_dataidx_map = dict.fromkeys(
            list(range(num_clients))
        )
        
    def __call__(self, dataset):
        idxs = np.arange(len(dataset))
        idxs_slice = np.array_split(idxs, self.num_clients)
        for i, idx in enumerate(idxs_slice):
            self.clients_dataidx_map[i] = idx
        print('Splitting dataset into {} clients.'.format(self.num_clients))
        print({id: len(idxs) for id, idxs in self.clients_dataidx_map.items()})
        return self
    
    def get_clients_dataidx_map(self):
        return self.clients_dataidx_map
        
    
    def __repr__(self):
        return f'{self.__class__.__name__}(num_clients={self.num_clients})'
    


if __name__ == "__main__":
    pass
   
