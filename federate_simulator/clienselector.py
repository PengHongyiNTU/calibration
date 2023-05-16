from abc import ABC, abstractmethod
import random


class ClientSelector(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def on_initialization(self, config):
        pass

    @abstractmethod
    def on_global_round_start(self, contexts, **kwargs):
        pass

    @abstractmethod
    def on_local_round_start(self, contexts, **kwargs):
        pass

    @abstractmethod
    def on_local_round_end(self, contexts, **kwargs):
        pass

    @abstractmethod
    def on_global_round_end(self, contexts, **kwargs):
        pass


class RandomClientSelector(ClientSelector):
    def __init__(self):
        super().__init__()

    def on_initialization(self, config):
        self.num_clients = config.num_clients
        self.num_selected_clients = config.num_selected_clients

    def select_clients(self, num_clients, num_selected_clients):
        if num_selected_clients > num_clients:
            raise ValueError(
                'num_selected_clients should be less than num_clients')
        else:
            return random.sample(range(num_clients), num_selected_clients)

    def on_global_round_start(self, contexts, global_round):
        selected_clients = self.select_clients(self.num_clients,
                                               self.num_selected_clients)
        return {'selected_clients': selected_clients,
                'global_round': global_round}
        
        
    def on_local_round_start(self,  contexts, **kwargs):
        pass
    
    def on_local_round_end(self,  contexts, **kwargs):
        pass
    
    def on_global_round_end(self, contexts, **kwargs):
        pass
    


if __name__ == '__main__':
    from box import Box
    config = Box({
        "num_clients": 10,
        "num_selected_clients": 5
    })
    random_selector = RandomClientSelector()
    random_selector.on_initialization(config)
    result = random_selector.on_global_round_start(contexts=None, global_round=1)
    print(f"Selected clients for global round 1: {result['selected_clients']}")
    
    