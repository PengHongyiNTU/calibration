from abc import ABC, abstractmethod
import os 
from torchmetrics.classification import Accuracy, ECE

class Evaluator(ABC):
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
    
    
class AccuracyCalibrationEvaluator(Evaluator):
    def __init__(self):
        super().__init__()
        
    def on_initialization(self, config):
        self.num_classes = config.num_classes
        self.top_k = config.evaluation.top_k
        self.n_bins = config.evaluation.n_bins
        self.ece = ECE(num_classes=self.num_classes, n_bins=self.n_bins)
        self.accuracy = Accuracy(num_classes=self.num_classes, top_k=self.top_k)
        
    def on_global_round_start(self, contexts, **kwargs):
        pass
    
    def on_local_round_start(self, contexts, **kwargs):
        pass
    
    def on_local_round_end(self, contexts, **kwargs):
        pass
        
        
    
        
      
        
            
           