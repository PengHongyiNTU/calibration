from abc import ABC, abstractmethod
from box import Box
from modelfactory import ModelFactory

class FederatedAggregator(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def on_initialization(self, config:Box, **kwargs):
        self.config = config
        model_factory = ModelFactory()
        self.model = model_factory.create_model(config.model.name, config.model)
        
        
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
    

class LocalFedAvg(FederatedAggregator):
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
    
class FedProx(FederatedAggregator):
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
    
class Ditto(FederatedAggregator):
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
    