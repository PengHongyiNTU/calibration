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
    