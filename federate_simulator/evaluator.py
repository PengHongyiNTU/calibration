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
    
    def on_local_round_end(self, contexts, client_id, 
                           local_test_loader):
        try:
            model = contexts['local_round_end']['FederatedAggregator']['model']
            model_params = contexts['local_round_end']['FederatedAggregator']['clients_params'][client_id]
            model.load_state_dict(model_params)
        except Exception as e:
            print(f'Error when retrieving models from contexts: {e}')  
            return {
                'client_id': client_id, 'ece': None, 'accuracy': None
            }
        ece_values = []
        accuracy_values = []
        if local_test_loader is None:
            return {
                'client_id': client_id, 'ece': None, 'accuracy': None
            }
        else:
            for x, y in local_test_loader:
                y_pred = model(x)
                ece = self.ece(y_pred, y)
                acc = self.accuracy(y_pred, y)
                ece_values.append(ece.item())
                accuracy_values.append(acc.item())
            avg_ece = sum(ece_values) / len(ece_values)
            avg_accuracy = sum(accuracy_values) / len(accuracy_values)
            return {
                'client_id': client_id, 
                'ece': avg_ece, 
                'avg_accuracy': avg_accuracy
            }
        
        
    def on_global_round_end(self, contexts,
                            global_test_loader):
        try:
            global_model = contexts['global_round_end']['FederatedAggregator']['model']
            model_params = contexts['global_round_end']['FederatedAggregator']['global_model_params']
            global_model.load_state_dict(model_params)
        except Exception as e:
            print(f'Error when retrieving global models from contexts: {e}')
            return {
                'ece': None, 'accuracy': None
            }
        ece_values = []
        accuracy_values = []
        assert global_test_loader is not None, 'global_test_loader is Empty'
        for x, y in global_test_loader:
            y_pred = global_model(x)
            ece = self.ece(y_pred, y)
            acc = self.accuracy(y_pred, y)
            ece_values.append(ece.item())
            accuracy_values.append(acc.item())
        avg_ece = sum(ece_values) / len(ece_values)
        avg_accuracy = sum(accuracy_values) / len(accuracy_values)
        return {
            'client_id': 'global',
            'ece': avg_ece, 
            'accuracy': avg_accuracy
        }
        
            
        
            
        
        
    
        
      
        
            
           