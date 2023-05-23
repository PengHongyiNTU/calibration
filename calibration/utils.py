import torch


def get_best_gpu():
    best_gpu = None
    max_mem = 0
    for  i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)
        allocated = torch.cuda.memory_allocated(i)
        cached = torch.cuda.memory_reserved()
        total = torch.cuda.get_device_properties(i).total_memory
        available = total - max(allocated, cached)
        if available > max_mem:
             max_mem = available
             best_gpu = i
    return torch.device(f'cuda:{best_gpu}')


if  __name__ == '__main__':
    print(get_best_gpu())