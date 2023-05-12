from box import Box

class BaseFederatedLearningSimulator:
    def __init__(self, config: Box):
        self.config = config
        self.observers = []
    
    def add_observer(self, observer):
        self.observers.append(observer)
        
    def notify_observers(self, method_name, *args, **kwargs):
        for observer in self.observers:
            if hasattr(observer, method_name):
                method = getattr(observer, method_name)
                method(*args, **kwargs)
    
    
class FederatedLearningSimulator:
    def __init__(self, config):
        self.observers = []
        self.config = config

    def register_observer(self, observer):
        self.observers.append(observer)

    def notify_observers(self, method_name, **kwargs):
        results = {}
        for observer in self.observers:
            method = getattr(observer, method_name, None)
            if method:
                results[observer.__class__.__name__] = method(**kwargs)
        return results

    def run(self):
        for global_round in range(self.config.global_rounds):
            self.notify_observers('on_global_round_start')

            local_models = []
            for client_id in range(self.config.num_clients):
                self.notify_observers('on_local_round_start', client_id=client_id)

                # Get the local train loader from the FederatedDatasetLoader observer
                loader_results = self.notify_observers('on_local_round_start', client_id=client_id)
                local_train_loader = loader_results.get('FederatedDatasetLoader')

                # Train the local model using the FedAvgAlgorithm observer
                algorithm_results = self.notify_observers('on_local_round_start', client_id=client_id)
                local_model = algorithm_results.get('FedAvgAlgorithm')
                
                if local_train_loader and local_model:
                    algorithm = [observer for observer in self.observers if isinstance(observer, FedAvgAlgorithm)][0]
                    trained_local_model = algorithm.local_train(local_model, local_train_loader, client_id)
                    local_models.append(trained_local_model)
                
                self.notify_observers('on_local_round_end', client_id=client_id)

            # Aggregate local models
            algorithm = [observer for observer in self.observers if isinstance(observer, FedAvgAlgorithm)][0]
            client_weights = [len(local_train_loader.dataset) for local_train_loader in local_train_loaders]
            aggregated_model = algorithm.aggregate_models(local_models, client_weights)
            
            self.notify_observers('on_global_round_end', aggregated_model=aggregated_model)