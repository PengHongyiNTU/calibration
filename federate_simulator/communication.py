from abc import ABC, abstractmethod

class BaseCommunicationChannel(ABC):
    def __init__(self, config):
        self.config = config
        pass
    def on_global_round_start(self):
        pass
    def on_global_round_end(self):
        pass
    def on_local_round_start(self):
        pass
    def on_local_round_end(self):
        pass
    


class PefectChannel(BaseCommunicationChannel):
    def __init__(self, config):
        super().__init__(config)
        pass
