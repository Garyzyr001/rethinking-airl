import torch
import numpy as np


def copy_model_params_from_to(source, target):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def fanin_init(tensor):
    size = tensor.size()
    if len(size) == 2:
        fan_in = size[0]
    elif len(size) > 2:
        fan_in = np.prod(size[1:])
    else:
        raise Exception("Shape must be have dimension at least 2.")
    bound = 1. / np.sqrt(fan_in)
    return tensor.data.uniform_(-bound, bound)


def fanin_init_weights_like(tensor):
    size = tensor.size()
    if len(size) == 2:
        fan_in = size[0]
    elif len(size) > 2:
        fan_in = np.prod(size[1:])
    else:
        raise Exception("Shape must be have dimension at least 2.")
    bound = 1. / np.sqrt(fan_in)
    new_tensor = FloatTensor(tensor.size())
    new_tensor.uniform_(-bound, bound)
    return new_tensor


"""
GPU wrappers
"""

_use_gpu = True

device='cuda'

_gpu_id = 0


def set_gpu_mode(mode, gpu_id=0):
    global _use_gpu
    global device
    global _gpu_id
    _gpu_id = gpu_id
    _use_gpu = mode
    device = torch.device("cuda:" + str(gpu_id) if _use_gpu else "cpu")
    torch.cuda.set_device(device)


def gpu_enabled():
    return _use_gpu


def set_device(gpu_id):
    torch.cuda.set_device(gpu_id)


# noinspection PyPep8Naming
def FloatTensor(*args, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.FloatTensor(*args, **kwargs, device=torch_device)


# 把数组np_array转换成张量，且二者共享内存，对张量进行修改，比如重新赋值，那么原始数组也会相应发生改变
# 相当于torch.from_numpy(np_array)函数
def from_numpy(*args, **kwargs):
    return torch.from_numpy(*args, **kwargs).float().to(device)


# 将张量转换为numpy，存在cpu设备中
def get_numpy(tensor):
    return tensor.to('cpu').detach().numpy()


# 返回一个形状为为size,类型为torch.dtype，里面的每一个值都是0的张量
# 参数可就写size，如两个参数，相当于torch.zeros()函数
def zeros(*sizes, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.zeros(*sizes, **kwargs, device=torch_device)


def ones(*sizes, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.ones(*sizes, **kwargs, device=torch_device)


# 把矩阵张量x的每个元素全部变为1，形成新的矩阵张量，数据类型和维度都不变
# 参数可就写一个张量，相当于torch.ones_like()函数
def ones_like(*args, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.ones_like(*args, **kwargs, device=torch_device)


def randn(*args, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.randn(*args, **kwargs, device=torch_device)


# 表示从区间[0,1)的均匀分布中抽取一组随机数，形成张量的规模由参数size定义
# 参数可就写size，如两个参数，相当于torch.rand()函数
def rand(*args, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.rand(*args, **kwargs, device=torch_device)


# 把矩阵张量x的每个元素全部变为0，形成新的矩阵张量，数据类型和维度都不变
# 参数可就写一个张量，相当于torch.zeros_like()函数
def zeros_like(*args, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.zeros_like(*args, **kwargs, device=torch_device)


def tensor(*args, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.tensor(*args, **kwargs, device=torch_device)


def normal(*args, **kwargs):
    return torch.normal(*args, **kwargs).to(device)


# 用于对策略梯度更新的clip操作
def fast_clip_grad_norm(parameters, max_norm):
    r"""Clips gradient norm of an iterable of parameters.
    Only support norm_type = 2
    max_norm = 0, skip the total norm calculation and return 0 
    https://pytorch.org/docs/stable/_modules/torch/nn/utils/clip_grad.html#clip_grad_norm_
    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    max_norm = float(max_norm)
    if abs(max_norm) < 1e-6:  # max_norm = 0
        return 0
    else:
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = list(filter(lambda p: p.grad is not None, parameters))
        total_norm = torch.stack([(p.grad.detach().pow(2)).sum() for p in parameters]).sum().sqrt().item()
        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for p in parameters:
                p.grad.detach().mul_(clip_coef)
        return total_norm
