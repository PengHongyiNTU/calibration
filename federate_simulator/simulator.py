from box import Box
from abc import ABC, abstractmethod


class BaseFederatedLearningSimulator(ABC):
    def __init__(self, config: Box, observers):
        self.config = config
        self.observers = {}
        self.stages = ['initialization',
                        'local_round_start',
                       'local_round_end',
                       'global_round_start',
                       'global_round_end']
        self.contexts = Box(dict.fromkeys(self.stages, {}), 
                            default_box=True)
        self.observers_priority = {'ClientSelector': 1, 
                                   'FederatedDataLoader': 2,
                                   'FederatedAggregator': 3,
                                   'CommunicationChannel': 4,
                                   'Evaluator': 5,
                                   'Logger': 6}
        try:
            self.register_observers(observers)
        except Exception as e:
            print(f'Error when registering observers: {e}')
        
    
    def register_observer(self, observers):
        for observer in observers:
            base_class_name = observer.__class__.__bases__[0].__name__
            self.observers[base_class_name] = observer
        # Sort the observers by their priority
        # the priority is 1. ClientSelector, 
        # 2. FederatedDataLoader, 
        # 3. FederatedAggregator, 
        # 4. Evaluator,
        # 5. Logger
        # Sort the dict of observers by the key value following the above order
        self.observers = dict(sorted(self.observers.items(), 
                                 key=lambda x: self.observer_priority[x[0]]))
    
         

        
    def notify_observer(self, observer_name, stage, *args, **kwargs):
        if observer_name not in self.observers:
            raise ValueError(f'{observer_name} is not registered in the simulator')
        else: 
            observer = self.observers[observer_name]
            if hasattr(observer, stage):
                method = getattr(observer, stage)
                result = method(self.contexts, *args, **kwargs)
                self.contexts[stage][observer.__class__.__name__] = result
            
            
    def notify_observers(self, observer_names, stage, *args, **kwargs):
        for observer_name in observer_names:
            self.notify_observer(observer_name, stage, *args, **kwargs)
            
    def notify_all(self, stage, *args, **kwargs):
        for observer_name in self.observers:
            self.notify_observer(observer_name, stage, *args, **kwargs)
            
            
    def flush_contexts(self):
        self.contexts = dict.fromkeys(self.stages, {})

        

    # Need to implement the main runing loop of the simulator here
    @abstractmethod
    def run(self):
        pass


class FederatedLearningSimulator(BaseFederatedLearningSimulator):
    def __init__(self, config: Box, Observers):
        super().__init__(config, Observers)

    def run(self):
        # Initialize all observers
        stage = 'initialization'
        self.notify_all(f'on_{stage}', self.config)
        for global_round in self.config.global_rounds:
            stage = 'global_round_start'
            self.notify_observer('ClientSelector', f'on_{stage}', global_round)
            selected_clients = self.contexts.global_round_start.ClientSelector.selected_clients
            for client_id in selected_clients:
                stage = 'local_round_start'
                self.notify_observer('FederatedDataLoader', 
                                      f'on_{stage}', client_id)
                local_train_loader = self.contexts.local_round_start.FederatedDatasetLoader.local_train_loader
                if local_train_loader:
                    self.notify_observers(
                        ['FederatedAggregator', 
                         'Logger'],
                        'on_local_round_start',
                        local_train_loader,
                        client_id
                    )
                stage = 'local_round_end'
                self.notify_observer('FederatedDataLoader', f'on_{stage}', client_id)
                local_test_loader = self.contexts.local_round_end.FederatedDatasetLoader.local_test_loader
                local_val_loader = self.contexts.local_round_end.FederatedDatasetLoader.local_val_loader
                if local_test_loader or local_val_loader:
                    self.notify_observer(['FederatedAggregator',
                                          'Evaluator', 
                                          'Logger'], 
                                         f'on_{stage}', 
                                         local_test_loader, 
                                         local_val_loader, 
                                         client_id)
            stage = 'global_round_end'
            self.notify_observer(
                'FederatedDataLoader',
                f'on_{stage}',
            )
            global_val_loader = self.contexts.global_round_end.FederatedDatasetLoader.global_val_loader
            global_test_loader = self.contexts.global_round_end.FederatedDatasetLoader.global_test_loader
            self.notify_observers(
                ['FederatedAggregator', 'Evaluator', 'Logger'],
                f'on_{stage}',
                global_val_loader,
                global_test_loader,
            )
                
                
            
            
   