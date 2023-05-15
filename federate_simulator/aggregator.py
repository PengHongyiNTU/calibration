from abc import ABC, abstractmethod



class BaseAggregation(ABC):
    def __init__(self, config):
        self.config = config
        pass
    @abstractmethod
    def on_global_round_start(self):
        pass
    @abstractmethod
    def on_global_round_end(self):
        pass
    @abstractmethod
    def on_local_round_start(self):
        pass
    @abstractmethod
    def on_local_round_end(self):
        pass
    

class FedAvg(BaseAggregation):
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
    
class FedProx(BaseAggregation):
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
    
class Ditto(BaseAggregation):
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
    