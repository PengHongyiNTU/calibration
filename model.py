import torch
from torchvision import models
from torch import nn
from box import Box
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_size=1024, hidden_sizes=[512, 512], num_classes=10):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
        for i in range(len(hidden_sizes)-1):
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(hidden_sizes[-1], num_classes))  # Corrected parentheses placement
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        for layer in self.layers:
            x = layer(x)
        return x 

class CNN(nn.Module):
    def __init__(self, num_channels=3, hidden_sizes=[32, 64, 128], 
                 num_classes=10, input_size=(32, 32)):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Conv2d(num_channels, hidden_sizes[0], kernel_size=3, stride=1, padding=1))
        for i in range(len(hidden_sizes)-1):
            self.layers.append(nn.ReLU())
            self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2))  # Corrected 'kernal' to 'kernel'
            self.layers.append(nn.Conv2d(hidden_sizes[i], hidden_sizes[i+1], kernel_size=3, stride=1, padding=1))
        self.layers.append(nn.ReLU())  # Corrected 'lauyers' to 'layers'
        self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc1 = nn.Linear(2048, 256)
        self.fc2 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x



def convert_bn_to_gn(module, num_groups):
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            return torch.nn.GroupNorm(num_groups, module.num_features,
                                      eps=module.eps, affine=module.affine)
        else:
            for name, child in module.named_children():
                module.add_module(name, convert_bn_to_gn(child, num_groups))
            return module
        
        
class ModelFactory:
    def __init__(self):
        self.supported = {
            'mnist': ['MLP', 'CNN'],
            'cifar10': ['CNN', 'resnet18', 'resnet34', 'resnet50', 'densenet121', 'resnext50'],
            'cifar100': ['CNN', 'resnet18', 'resnet34', 'resnet50', 'densenet121', 'resnext50'],
            'tiny-imagenet-200': ['CNN', 'resnet18', 'resnet34', 'resnet50', 'densenet121', 'resnext50']
        }
        
    
    def create_model(self, config:Box):
        dataset = config.dataset.name
        model_type = config.model
        if dataset not in self.supported.keys():
            raise ValueError(f'Unknown dataset {dataset}')
        elif model_type not in self.supported[dataset]:
            raise ValueError(f'Not supported model {model_type}')
        if dataset == "mnist":
            num_classes = 10
            num_channels = 1  
        elif dataset == "cifar10":
            num_classes = 10
            num_channels = 3
        elif dataset == "cifar100":
            num_classes = 100
            num_channels = 3
        elif dataset == "tiny-imagenet-200":
            num_channels = 3
            num_classes = 200

        if model_type == 'MLP':
           model = MLP(num_classes=num_classes)  # Here you need to pass in the relevant parameters
        elif model_type == 'CNN':
            model = CNN(num_classes=num_classes, num_channels=num_channels)  # Here you need to pass in the relevant parameters
        elif model_type == 'resnet18':
            model = models.resnet18(pretrained=False)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif model_type == 'resnet34':
            model = models.resnet34(pretrained=False)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif model_type == 'resnet50':
            model = models.resnet50(pretrained=False)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif model_type == 'densenet121':
            model = models.densenet121(pretrained=False)
            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        elif model_type == 'resnext50_32x4d':
            model = models.resnext50_32x4d(pretrained=False)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        else:
            raise ValueError(f'Unsupported model type {model_type}')   
        model = convert_bn_to_gn(model, 32)
        return model  # Return the model

if __name__ == "__main__":
    config_mnist = Box({
        'dataset': {'name': 'mnist'},
        'model': 'MLP'
    })
    factory = ModelFactory()
    model_mnist = factory.create_model(config_mnist)
    dummy_input = torch.randn(1, 1, 32, 32)
    output = model_mnist(dummy_input)
    print(output.shape)