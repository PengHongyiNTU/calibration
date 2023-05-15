from abc import ABC, abstractmethod
import os 

class BaseLogger(ABC):
    def __init__(self):
        super().__init__()
        
    @abstractmethod
    def on_initialization(self, config):
        pass
    
    @abstractmethod
    def on_global_round_start(self, results):
        pass
    
    @abstractmethod
    def on_local_round_start(self, results):
        pass
    
    @abstractmethod
    def on_local_round_end(self, results):
        pass
    
    @abstractmethod
    def on_global_round_end(self, results):
        pass
    
    
class LocalJsonLogger(BaseLogger):
    def __init__(self):
        super().__init__()
        
    def on_initialization(self, config):
        self.log_dir = config.logging.log_dir
        log_file_name = config.project_name + '.json'
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.log_file_path = os.path.join(self.log_dir, log_file_name)
        self.log_file = open(self.log_file, 'w')
        print(f'Logging saved to {self.log_file_path}')
    
    def on_global_round_start(self, global_round, results):
        print(f'Global round {global_round} start')
    
    def on_local_round_start(self,  client_id, results):
        print(f'Local round  start: Client {client_id} selected')

        
    def on_local_round_end(self, client_id, results):
        print(f'Local round  end: Client {client_id} finished training')
        
        
    def on_global_round_end(self, global_round, results):
        pass
        
        
            